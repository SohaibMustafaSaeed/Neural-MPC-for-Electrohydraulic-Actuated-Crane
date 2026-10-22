import os,sys
from pathlib import Path
for _thread in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[_thread]='1'
sys.dont_write_bytecode=True
from preprocess import resolve_array,same_recording,INPUT_COLS,OUTPUT_COLS,COMMAND_INDICES,STATE_INDICES
ROOT=Path(__file__).resolve().parents[2]
RAW_DATA_DIR=ROOT.parent
PREPARED_DIR=RAW_DATA_DIR/'PATU_prepared_cache'
import json
from pathlib import Path
import numpy as np
DEFAULT_DATASET_DIR=Path(os.environ.get('PATU_PREPARED_ROOT',Path(__file__).resolve().parents[3]/'PATU_prepared_cache'))
INPUT_COLS = ['ulift[V]', 'speedref[rad/s]', 'speedpmsm[rad/s]', 'iqref[A]', 'udref[V]', 'uqref[V]', 'id[A]', 'iq[A]', 'psid[Wb]', 'psiq[Wb]', 'pumppressure[Pa]', 'plinepistonlift[Pa]', 'plinerodlift[Pa]', 'ppistonsidelift[Pa]', 'prodsidelift[Pa]', 'thetalift[rad]', 'ujib[V]', 'plinepistonjib[Pa]', 'plinerodjib[Pa]', 'ppistonjib[Pa]', 'prodjib[Pa]', 'thetajib[rad]', 'thetaliftdot[rad/s]', 'thetajibdot[rad/s]']
COMMAND_INDICES = (0, 16)
STATE_INDICES = tuple((i for i in range(24) if i not in COMMAND_INDICES))
OUTPUT_COLS = [INPUT_COLS[i] for i in STATE_INDICES]

def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def load_training_records(dataset_dir):
    training_dir = dataset_dir / 'training'
    manifest_path = dataset_dir / 'training_manifest.json'
    scaler_path = dataset_dir / 'scaler.json'
    if not training_dir.is_dir() or not manifest_path.is_file() or (not scaler_path.is_file()):
        raise FileNotFoundError(f'Prepared training data not found under {dataset_dir}; run Data Preprocessing/preprocess_dataset.py first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    scaler = json.loads(scaler_path.read_text(encoding='utf-8'))
    if scaler.get('input_columns') != INPUT_COLS or scaler.get('output_columns') != OUTPUT_COLS:
        raise ValueError('Prepared dataset feature order does not match this model')
    for entry in manifest:
        source = Path(entry['source_csv'])
        if source.parent.name.lower() != 'train':
            raise ValueError(f'Training manifest must contain only Train recordings: {source}')
    for source in scaler.get('fitted_files', []):
        if Path(source).parent.name.lower() != 'train':
            raise ValueError(f'Scaler was not fitted exclusively on Train: {source}')
    records = []
    for entry in manifest:
        array_path = resolve_array(entry, dataset_dir)
        if not array_path.is_absolute():
            array_path = dataset_dir / array_path
        array = np.load(array_path, mmap_mode='r')
        if array.ndim != 2 or array.shape != (entry['rows'], 24) or array.dtype != np.float32:
            raise ValueError(f'Invalid prepared array: {array_path}')
        train_rows = int(entry['train_rows'])
        validation_start = int(entry['validation_start'])
        if not 0 < train_rows == validation_start < len(array):
            raise ValueError(f'Invalid train/validation boundaries in {array_path}')
        records.append(dict(array=array, train_rows=train_rows, validation_start=validation_start, rows=len(array)))
    if not records:
        raise ValueError(f'No prepared training recordings in {manifest_path}')
    return (records, scaler, manifest)

import sys
from pathlib import Path
from dataclasses import asdict, dataclass
import argparse
import math
import time
import shutil
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn
from numba import njit
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / 'results'
ARCHITECTURE = 'lstm4'
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / 'checkpoints'
FORMAT = 'lstm4_encoded_residual_numba_24x22_v1'
ANGLE_COLS = ('thetalift[rad]', 'thetajib[rad]')
VELOCITY_COLS = ('thetaliftdot[rad/s]', 'thetajibdot[rad/s]')
STATE_ARRAY = np.asarray(STATE_INDICES, dtype=np.int64)

@dataclass
class ModelConfig:
    hidden: int = 16
    latent_dim: int = 12
    decoder_dim: int = 16
    use_skip: bool = True

class EncodedResidualLSTM(nn.Module):

    def __init__(self, config, scale):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(22, 2 * config.hidden)
        self.feature_encoder = nn.Linear(48, config.latent_dim)
        self.cell = nn.LSTMCell(config.latent_dim, config.hidden)
        self.post_encoder = nn.Linear(config.hidden, config.decoder_dim)
        self.decoder = nn.Linear(config.decoder_dim, 22)
        self.skip = nn.Linear(48, 22, bias=False) if config.use_skip else None
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        nn.init.normal_(self.decoder.weight, std=0.001)
        nn.init.zeros_(self.decoder.bias)
        if self.skip is not None:
            nn.init.zeros_(self.skip.weight)

    def encode(self, current):
        return torch.tanh(self.encoder(current[:, STATE_INDICES]))

    def step(self, previous, current, latent):
        features = torch.cat((current, current - previous), -1)
        compact = torch.tanh(self.feature_encoder(features))
        hidden, cell = self.cell(compact, latent.chunk(2, -1))
        memory = torch.cat((hidden, cell), -1)
        decoded = torch.tanh(self.post_encoder(hidden))
        update = self.decoder(decoded)
        if self.skip is not None:
            update = update + self.skip(features)
        delta = update * self.scale
        return (current[:, STATE_INDICES] + delta, memory)

    def rollout(self, initial, future_commands):
        previous, current = (initial[:, 0], initial[:, 1])
        memory = self.encode(current)
        predictions = []
        for t in range(future_commands.shape[1]):
            state, memory = self.step(previous, current, memory)
            following = torch.empty_like(current)
            following[:, COMMAND_INDICES] = future_commands[:, t]
            following[:, STATE_INDICES] = state
            previous, current = (current, following)
            predictions.append(state)
        return torch.stack(predictions, 1)

def export_weights(model):
    return tuple((np.ascontiguousarray(p.detach().cpu().numpy()) for p in (model.encoder.weight, model.encoder.bias, model.feature_encoder.weight, model.feature_encoder.bias, model.cell.weight_ih, model.cell.bias_ih, model.cell.weight_hh, model.cell.bias_hh, model.post_encoder.weight, model.post_encoder.bias, model.decoder.weight, model.decoder.bias, model.skip.weight if model.skip is not None else model.scale.new_zeros((22, 48)), model.scale)))

@njit(cache=True)
def native_encode(current, ew, eb):
    z = np.empty(len(eb), np.float32)
    for i in range(len(z)):
        value = eb[i]
        for j in range(22):
            value += ew[i, j] * current[STATE_ARRAY[j]]
        z[i] = np.tanh(value)
    return z

@njit(cache=True)
def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + np.exp(-value))
    e = np.exp(value)
    return e / (1.0 + e)

@njit(cache=True)
def native_step(previous, current, z, ew, eb, fw, fb, iw, ib, rw, rb, pw, pb, dw, db, skip, scale):
    features = np.empty(48, np.float32)
    for j in range(24):
        features[j] = current[j]
        features[j + 24] = current[j] - previous[j]
    compact = np.empty(len(fb), np.float32)
    for i in range(len(fb)):
        value = fb[i]
        for j in range(48):
            value += fw[i, j] * features[j]
        compact[i] = np.tanh(value)
    H = rw.shape[1]
    xi = np.empty(len(ib), np.float32)
    hh = np.empty(len(rb), np.float32)
    for i in range(len(ib)):
        a, b = (ib[i], rb[i])
        for j in range(len(compact)):
            a += iw[i, j] * compact[j]
        for j in range(H):
            b += rw[i, j] * z[j]
        xi[i], hh[i] = (a, b)
    memory = np.empty_like(z)
    for i in range(H):
        ingate = sigmoid(xi[i] + hh[i])
        forget = sigmoid(xi[H + i] + hh[H + i])
        candidate = np.tanh(xi[2 * H + i] + hh[2 * H + i])
        outgate = sigmoid(xi[3 * H + i] + hh[3 * H + i])
        cell = forget * z[H + i] + ingate * candidate
        memory[H + i] = cell
        memory[i] = outgate * np.tanh(cell)
    decoded = np.empty(len(pb), np.float32)
    for i in range(len(pb)):
        value = pb[i]
        for j in range(H):
            value += pw[i, j] * memory[j]
        decoded[i] = np.tanh(value)
    state = np.empty(22, np.float32)
    for i in range(22):
        value = db[i]
        for j in range(len(decoded)):
            value += dw[i, j] * decoded[j]
        for j in range(48):
            value += skip[i, j] * features[j]
        state[i] = current[STATE_ARRAY[i]] + value * scale[i]
    return (state, memory)

@njit(cache=True)
def native_chunk(rows, previous, current, z, recursive, mean, std, ew, eb, fw, fb, iw, ib, rw, rb, pw, pb, dw, db, skip, scale):
    previous = previous.copy()
    current = current.copy()
    z = z.copy()
    output = np.empty((len(rows), 22), np.float64)
    for k in range(len(rows)):
        if recursive:
            current[0] = rows[k, 0]
            current[16] = rows[k, 16]
        else:
            current = rows[k].copy()
        state, z = native_step(previous, current, z, ew, eb, fw, fb, iw, ib, rw, rb, pw, pb, dw, db, skip, scale)
        previous = current.copy()
        for j in range(22):
            current[STATE_ARRAY[j]] = state[j]
            output[k, j] = np.float64(state[j]) * std[STATE_ARRAY[j]] + mean[STATE_ARRAY[j]]
    return (output, previous, current, z)

@njit(cache=True)
def physical_step(previous, current, z, mean, std, ew, eb, fw, fb, iw, ib, rw, rb, pw, pb, dw, db, skip, scale):
    p = np.empty(24, np.float32)
    c = np.empty(24, np.float32)
    for j in range(24):
        p[j] = (previous[j] - mean[j]) / std[j]
        c[j] = (current[j] - mean[j]) / std[j]
    state, next_z = native_step(p, c, z, ew, eb, fw, fb, iw, ib, rw, rb, pw, pb, dw, db, skip, scale)
    result = np.empty(22, np.float64)
    for j in range(22):
        result[j] = np.float64(state[j]) * std[STATE_ARRAY[j]] + mean[STATE_ARRAY[j]]
    return (result, next_z)

def load_bundle(path=DEFAULT_ARTIFACT_DIR / 'best_model.pt'):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('format') != FORMAT:
        raise ValueError('Checkpoint architecture/format mismatch')
    config = dict(ck['model_config'])
    model = EncodedResidualLSTM(ModelConfig(**config), ck['scale'])
    model.load_state_dict(ck['model'])
    model.eval()
    return (model, ck)

def batch(records, pairs, horizon):
    return torch.from_numpy(np.stack([r['array'][s - 1:s + horizon + 1] for r, s in ((records[i], s) for i, s in pairs)]).copy())

def state_loss_weights(angle_weight=4.0, velocity_weight=4.0):
    weights = np.ones(22, dtype=np.float32)
    for col in ANGLE_COLS:
        weights[OUTPUT_COLS.index(col)] = angle_weight
    for col in VELOCITY_COLS:
        weights[OUTPUT_COLS.index(col)] = velocity_weight
    return weights

def weighted_mse(error, weights):
    return (error.square() * weights).sum(dim=-1).mean() / weights.sum()

def sample_horizon(rng, short, medium, long, medium_probability=0.6, long_probability=0.2):
    choice = rng.random()
    if choice < medium_probability:
        return medium
    if choice < medium_probability + long_probability:
        return long
    return short

def train_batch(model, optimizer, rows, block, initial_latent=None, loss_weights=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    previous, current = (rows[:, 0], rows[:, 1])
    z = model.encode(current) if initial_latent is None else initial_latent
    horizon = rows.shape[1] - 2
    total = 0.0
    loss_scale = model.scale.clamp_min(0.01)
    weights = rows.new_ones(22) if loss_weights is None else torch.as_tensor(loss_weights, dtype=rows.dtype, device=rows.device)
    for start in range(0, horizon, block):
        loss = rows.new_zeros(())
        for t in range(start, min(start + block, horizon)):
            state, z = model.step(previous, current, z)
            loss += weighted_mse((state - rows[:, t + 2, STATE_INDICES]) / loss_scale, weights) / horizon
            following = torch.empty_like(current)
            following[:, COMMAND_INDICES] = rows[:, t + 2, COMMAND_INDICES]
            following[:, STATE_INDICES] = state
            previous, current = (current, following)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        total += float(loss.detach())
        previous, current, z = (previous.detach(), current.detach(), z.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return total

def native_validation(model, rows, horizons, loss_weights=None, return_weighted=False):
    weights = export_weights(model)
    mean = np.zeros(24)
    std = np.ones(24)
    scores = {str(h): 0.0 for h in horizons}
    weighted = {str(h): 0.0 for h in horizons}
    state_weights = np.ones(22) if loss_weights is None else np.asarray(loss_weights, dtype=float)
    for window in rows:
        z = native_encode(window[1], weights[0], weights[1])
        prediction, _, _, _ = native_chunk(window[1:-1], window[0], window[1], z, True, mean, std, *weights)
        for h in horizons:
            errors = (prediction[:h] - window[2:2 + h, STATE_ARRAY]) ** 2
            scores[str(h)] += float(np.mean(errors)) / len(rows)
            weighted[str(h)] += float(np.mean(np.sum(errors * state_weights, axis=1) / state_weights.sum())) / len(rows)
    return (scores, weighted) if return_weighted else scores

def burn_in_batch(model, rows, steps):
    if not steps:
        return (rows, None)
    weights = export_weights(model)
    mean = np.zeros(24)
    std = np.ones(24)
    suffix = rows[:, steps:].clone()
    latents = []
    for i, window in enumerate(rows.numpy()):
        z = native_encode(window[1], weights[0], weights[1])
        _, previous, current, z = native_chunk(window[1:steps + 1], window[0], window[1], z, True, mean, std, *weights)
        current[list(COMMAND_INDICES)] = window[steps + 1, list(COMMAND_INDICES)]
        suffix[i, 0] = torch.from_numpy(previous)
        suffix[i, 1] = torch.from_numpy(current)
        latents.append(z)
    return (suffix, torch.from_numpy(np.stack(latents)))

def historical_main():
    p = argparse.ArgumentParser(description='iMotion residual dynamics')
    p.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--overwrite', action='store_true', help='Replace existing checkpoints and history with a fresh training run')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--steps-per-epoch', type=int, default=20)
    p.add_argument('--hidden', type=int, default=16, help='Recurrent units; try 8 for smaller or 32 for more capacity')
    p.add_argument('--latent-dim', type=int, default=12, help='Feature encoding width before LSTM')
    p.add_argument('--decoder-dim', type=int, default=16, help='Encoding width after GRU')
    p.add_argument('--angle-weight', type=float, default=4.0, help='Loss weight for lift/jib angles; other states use 1')
    p.add_argument('--velocity-weight', type=float, default=4.0, help='Loss weight for lift/jib angular velocities')
    p.add_argument('--medium-horizon', type=int, default=500, help='Medium rollout steps at dataset sample interval')
    p.add_argument('--medium-probability', type=float, default=0.6)
    p.add_argument('--long-probability', type=float, default=0.2)
    p.add_argument('--selection-horizon', type=int, help='Select best checkpoint on weighted error at this horizon; default medium horizon')
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--warmup-epochs', type=int, default=2)
    p.add_argument('--short-horizon', type=int, default=100)
    p.add_argument('--long-horizon', type=int, default=1000)
    p.add_argument('--burn-in-steps', type=int, default=10000, help='Maximum unrefreshed prefix before differentiable training suffix')
    p.add_argument('--grad-block', type=int, default=50)
    p.add_argument('--val-horizons', type=int, nargs='+', default=[100, 500, 1000])
    p.add_argument('--val-windows', type=int, default=4)
    p.add_argument('--scale-samples-per-file', type=int, default=10000)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    a.selection_horizon = a.medium_horizon if a.selection_horizon is None else a.selection_horizon
    a.val_horizons = sorted(set(a.val_horizons + [a.selection_horizon]))
    if min(a.epochs, a.batch_size, a.steps_per_epoch, a.hidden, a.latent_dim, a.decoder_dim, a.short_horizon, a.medium_horizon, a.long_horizon, a.selection_horizon, a.grad_block, a.val_windows, a.threads, a.scale_samples_per_file, *a.val_horizons) < 1 or min(a.warmup_epochs, a.burn_in_steps) < 0:
        p.error('Invalid positive counts')
    if not math.isfinite(a.learning_rate) or a.learning_rate <= 0 or a.short_horizon > a.long_horizon:
        p.error('Invalid learning rate/horizons')
    if not a.short_horizon <= a.medium_horizon <= a.long_horizon:
        p.error('Require short <= medium <= long horizons')
    if not all((math.isfinite(v) and v > 0 for v in (a.angle_weight, a.velocity_weight))):
        p.error('Loss weights must be finite and positive')
    if not all((math.isfinite(v) and 0 <= v <= 1 for v in (a.medium_probability, a.long_probability))) or a.medium_probability + a.long_probability > 1:
        p.error('Horizon probabilities must be in [0,1] and sum to at most 1')
    if a.epochs <= a.warmup_epochs:
        p.error('epochs must exceed warmup-epochs so recursive curriculum is trained')
    if not a.overwrite and a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Output folder is nonempty; choose a new --output-dir or use --overwrite')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    records, sc, manifest = load_training_records(a.dataset_dir.resolve())
    sums = np.zeros(22)
    count = 0
    for r in records:
        indices = np.unique(np.linspace(1, r['train_rows'] - 1, min(r['train_rows'] - 1, a.scale_samples_per_file), dtype=int))
        if not len(indices):
            continue
        delta = np.asarray(r['array'][indices][:, STATE_INDICES], float) - r['array'][indices - 1][:, STATE_INDICES]
        sums += (delta ** 2).sum(axis=0)
        count += len(delta)
    if count == 0:
        raise ValueError('Insufficient training rows for delta scale')
    scale = np.clip(np.sqrt(sums / count), 1e-05, 1.0)
    max_train = max(5, a.long_horizon) + a.burn_in_steps
    max_val = max(a.val_horizons)
    counts = np.array([max(0, r['train_rows'] - max_train - 1) for r in records])
    if counts.sum() == 0:
        raise ValueError('No training windows')
    val_counts = [max(0, (r['rows'] - r['validation_start'] - 2 - max_val) // max_val + 1) for r in records]
    total_val = sum(val_counts)
    if not total_val:
        raise ValueError('No validation windows')
    cumulative = np.cumsum(val_counts)
    pairs = []
    for flat in np.unique(np.linspace(0, total_val - 1, min(total_val, a.val_windows), dtype=int)):
        i = int(np.searchsorted(cumulative, flat, side='right'))
        before = cumulative[i - 1] if i else 0
        pairs.append((i, records[i]['validation_start'] + 1 + int(flat - before) * max_val))
    cfg = ModelConfig(a.hidden, a.latent_dim, a.decoder_dim)
    model = EncodedResidualLSTM(cfg, scale)
    loss_weights = state_loss_weights(a.angle_weight, a.velocity_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=1e-06)
    config = dict(format=FORMAT, model=asdict(cfg), training_manifest=manifest, scaler_fitted_rows=sc['fitted_rows'], delta_scale_training_pairs=count, loss_weights=dict(zip(OUTPUT_COLS, loss_weights.tolist())), checkpoint_selection=f'weighted normalized MSE at {a.selection_horizon} steps', command_alignment='u[t],x[t],x[t-1] -> x[t+1]', arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.overwrite:
        previous = [a.output_dir / name for name in ('best_model.pt', 'latest_model.pt', 'best_numba_weights.npz', 'config.json', 'training_history.json', 'training_source.txt') if (a.output_dir / name).is_file()]
        if previous:
            backup = a.output_dir / 'previous_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            backup.mkdir(parents=True)
            for path in previous:
                shutil.copy2(path, backup / path.name)
            for path in previous:
                if path.parent.resolve() != a.output_dir.resolve():
                    raise ValueError('Overwrite target is outside output folder')
                path.unlink()
            print(f'Previous LSTM4 artifacts copied to: {backup}', flush=True)
    write_json(a.output_dir / 'config.json', config)
    (a.output_dir / 'training_source.txt').write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    dt = sc['sampling_interval_seconds']
    print(f'CPU training; parameters={sum((p.numel() for p in model.parameters())):,}; dt={dt * 1000000.0:g} us; validation horizons={[h * dt for h in a.val_horizons]} s; Numba inference exported after each epoch', flush=True)
    print(f'Curriculum after warmup: short={a.short_horizon * dt:g}s ({1 - a.medium_probability - a.long_probability:.0%}), medium={a.medium_horizon * dt:g}s ({a.medium_probability:.0%}), long={a.long_horizon * dt:g}s ({a.long_probability:.0%}); checkpoint selection={a.selection_horizon * dt:g}s weighted error', flush=True)
    validation_rows = batch(records, pairs, max_val).numpy()
    compile_started = time.perf_counter()
    native_validation(model, validation_rows[:1, :3], [1])
    print(f'Numba validation warmup: {time.perf_counter() - compile_started:.2f} s (excluded from epoch timings)', flush=True)
    best = math.inf
    history = []
    for epoch in range(1, a.epochs + 1):
        started = time.perf_counter()
        total = 0.0
        prefix_total = 0
        horizon_total = 0
        horizon_counts = {}
        for step in range(a.steps_per_epoch):
            h = 5 if epoch <= a.warmup_epochs else sample_horizon(rng, a.short_horizon, a.medium_horizon, a.long_horizon, a.medium_probability, a.long_probability)
            prefix = 0 if epoch <= a.warmup_epochs or rng.random() < 0.25 else int(rng.integers(0, a.burn_in_steps + 1))
            ids = rng.choice(len(records), size=a.batch_size, p=counts / counts.sum())
            selected = [(int(i), int(rng.integers(1, counts[i] + 1))) for i in ids]
            rows, z = burn_in_batch(model, batch(records, selected, h + prefix), prefix)
            total += train_batch(model, opt, rows, a.grad_block, z, loss_weights)
            prefix_total += prefix
            horizon_total += h
            horizon_counts[str(h)] = horizon_counts.get(str(h), 0) + 1
        training_seconds = time.perf_counter() - started
        model.eval()
        validation_started = time.perf_counter()
        scores, weighted_scores = native_validation(model, validation_rows, a.val_horizons, loss_weights, True)
        validation_seconds = time.perf_counter() - validation_started
        score = weighted_scores[str(a.selection_horizon)]
        if not all((math.isfinite(value) for value in list(scores.values()) + list(weighted_scores.values()))):
            raise RuntimeError('Nonfinite validation rollout')
        ck = dict(format=FORMAT, model_config=asdict(cfg), model=model.state_dict(), scale=scale.tolist(), scaler=sc, training_config=config, epoch=epoch, validation_loss=score, optimizer=opt.state_dict())
        for name in ['latest_model.pt', 'best_model.pt'] if score < best else ['latest_model.pt']:
            temp = a.output_dir / (name + '.tmp')
            torch.save(ck, temp)
            temp.replace(a.output_dir / name)
        if score < best:
            with (a.output_dir / 'best_numba_weights.npz.tmp').open('wb') as f:
                np.savez(f, **{f'w{i}': w for i, w in enumerate(export_weights(model))})
            (a.output_dir / 'best_numba_weights.npz.tmp').replace(a.output_dir / 'best_numba_weights.npz')
        best = min(best, score)
        row = dict(epoch=epoch, train_loss=total / a.steps_per_epoch, validation=scores, weighted_validation=weighted_scores, validation_loss=score, training_seconds=training_seconds, validation_seconds=validation_seconds, seconds=time.perf_counter() - started, mean_burn_in_steps=prefix_total / a.steps_per_epoch, mean_gradient_rollout_steps=horizon_total / a.steps_per_epoch, horizon_counts=horizon_counts)
        history.append(row)
        write_json(a.output_dir / 'training_history.json', history)
        print(f"Epoch {epoch}/{a.epochs}: train={row['train_loss']:.7g}; validation={scores}; best={best:.7g}; train_s={training_seconds:.2f}; validation_s={validation_seconds:.3f}; seconds={row['seconds']:.1f}", flush=True)
    print(f'Artifacts: {a.output_dir}', flush=True)

def main():
    parser=argparse.ArgumentParser(description='Historical lstm training; frozen checkpoints are never overwritten',add_help=False)
    parser.add_argument('--data-root',type=Path,default=Path(os.environ.get('PATU_DATA_ROOT',ROOT.parent)))
    parser.add_argument('--prepared-root','--dataset-dir',dest='prepared',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--smoke-test',action='store_true');parser.add_argument('--verify',action='store_true')
    args,remaining=parser.parse_known_args()
    if '--help' in remaining:
        print('Additional options: --data-root --prepared-root --seed --smoke-test --verify; historical options follow.')
        sys.argv=[sys.argv[0],'--help'];return historical_main()
    if args.verify:
        model,ck=load_bundle(SCRIPT_DIR/'checkpoints/lstm_original.pt')
        count=sum(p.numel() for p in model.parameters());assert count==dict(gru=4098,lstm=4946,transformer=4710,compact=4246)['lstm']
        print('lstm frozen checkpoint verified:',count,'parameters');return
    prepared=(args.prepared or args.data_root/'PATU_prepared_cache').resolve()
    if not (prepared/'training_manifest.json').exists():
        import subprocess
        subprocess.run([sys.executable,str(SCRIPT_DIR/'preprocess.py'),'--data-root',str(args.data_root),'--prepared-root',str(prepared),'--training-only'],check=True)
    from preprocess import verify_prepared
    verify_prepared(prepared)
    label=('smoke_' if args.smoke_test else '')+'s'+str(args.seed)
    output=SCRIPT_DIR/'results'/('training_'+label)/'artifacts'
    destination=SCRIPT_DIR/'checkpoints'/('lstm_retrained_'+label+'.pt')
    if destination.exists():raise FileExistsError('New checkpoint already exists; preserve it and choose another seed/output: '+str(destination))
    extra=[]
    if args.smoke_test:
        extra=['--epochs','3','--steps-per-epoch','1','--batch-size','2','--warmup-epochs','2','--short-horizon','5','--medium-horizon','10','--long-horizon','20','--burn-in-steps','0','--grad-block','5','--val-horizons','5','10','20','--selection-horizon','10','--val-windows','2','--scale-samples-per-file','16','--threads','1']
    records,sc,manifest=load_training_records(prepared)
    print('lstm TRAINING FROM SCRATCH; seed=',args.seed,'training_recordings=',len(records),'validation_recordings=',len(records),'training_rows=',sum(r['train_rows'] for r in records),'validation_rows=',sum(r['rows']-r['validation_start'] for r in records),'smoke=',args.smoke_test,flush=True)
    print('Original split: 90/10 within each of43Train recordings. AdamW, lr=.001, wd=1e-6, gradclip1, no scheduler/early stopping. Full curriculum100/500/1000, best weighted500-step validation.',flush=True)
    sys.argv=[sys.argv[0],'--dataset-dir',str(prepared),'--output-dir',str(output),'--seed',str(args.seed),*remaining,*extra]
    started=time.perf_counter();historical_main();duration=time.perf_counter()-started
    ck=torch.load(output/'best_model.pt',map_location='cpu',weights_only=False)
    torch.save(ck,destination);model,reloaded=load_bundle(destination);assert sum(p.numel() for p in model.parameters())==dict(gru=4098,lstm=4946,transformer=4710,compact=4246)['lstm']
    destination.with_suffix('.scaler.json').write_text(json.dumps(ck['scaler'],indent=2))
    import pandas as pd,matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    history=json.loads((output/'training_history.json').read_text());frame=pd.DataFrame([dict(epoch=r['epoch'],training_loss=r['train_loss'],validation_loss=r['validation_loss']) for r in history]);frame.to_csv(output.parent/'training_loss.csv',index=False)
    fig,ax=plt.subplots();ax.plot(frame.epoch,frame.training_loss,label='Training');ax.plot(frame.epoch,frame.validation_loss,label='Validation');ax.set(xlabel='Epoch',ylabel='Historical weighted loss',yscale='log');ax.legend();fig.tight_layout();fig.savefig(output.parent/'training_loss.png',dpi=150);plt.close(fig)
    report=dict(model='lstm',checkpoint=str(destination.relative_to(SCRIPT_DIR)),seed=args.seed,epochs=len(history),smoke_only=args.smoke_test,training_seconds=duration,checkpoint_reloaded=True,normalization='Saved checkpoint scaler; Train prefix only')
    (output.parent/'training_summary.json').write_text(json.dumps(report,indent=2));print('SELECTED CHECKPOINT',destination,'total_training_seconds=',duration,flush=True)
if __name__=='__main__':main()
