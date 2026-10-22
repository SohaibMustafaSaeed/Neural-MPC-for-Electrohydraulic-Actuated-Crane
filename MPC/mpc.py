import os,sys
from pathlib import Path
for _thread in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[_thread]='1'
sys.dont_write_bytecode=True
import json,hashlib,time,argparse,math
from dataclasses import dataclass,asdict
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from torch import nn
from numba import njit,set_num_threads
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent
import numpy as np
INPUT_COLS = ['ulift[V]', 'speedref[rad/s]', 'speedpmsm[rad/s]', 'iqref[A]', 'udref[V]', 'uqref[V]', 'id[A]', 'iq[A]', 'psid[Wb]', 'psiq[Wb]', 'pumppressure[Pa]', 'plinepistonlift[Pa]', 'plinerodlift[Pa]', 'ppistonsidelift[Pa]', 'prodsidelift[Pa]', 'thetalift[rad]', 'ujib[V]', 'plinepistonjib[Pa]', 'plinerodjib[Pa]', 'ppistonjib[Pa]', 'prodjib[Pa]', 'thetajib[rad]', 'thetaliftdot[rad/s]', 'thetajibdot[rad/s]']
COMMAND_INDICES = (0, 16)
STATE_INDICES = tuple((i for i in range(24) if i not in COMMAND_INDICES))
OUTPUT_COLS = [INPUT_COLS[i] for i in STATE_INDICES]

SI=np.asarray(STATE_INDICES,np.int64)
CI=np.asarray(COMMAND_INDICES,np.int64)
ANGLES=[OUTPUT_COLS.index(n) for n in ('thetalift[rad]','thetajib[rad]')]
VELOCITIES=[OUTPUT_COLS.index(n) for n in ('thetaliftdot[rad/s]','thetajibdot[rad/s]')]
PRESSURES=[i for i,n in enumerate(OUTPUT_COLS) if '[Pa]' in n]
MECH=np.asarray(ANGLES+VELOCITIES)

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from numba import njit
FORMAT = 'compact_latent_recurrent_numba_24x22_v1'
ANGLE_COLS = ('thetalift[rad]', 'thetajib[rad]')
VELOCITY_COLS = ('thetaliftdot[rad/s]', 'thetajibdot[rad/s]')
STATE_ARRAY = np.asarray(STATE_INDICES, dtype=np.int64)

@dataclass
class backbone_ModelConfig:
    latent_dim: int = 8
    hidden: int = 32

class backbone_CompactLatent(nn.Module):

    def __init__(self, config, scale):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(22, config.latent_dim)
        self.transition = nn.Linear(48 + config.latent_dim, config.latent_dim)
        self.hidden = nn.Linear(48 + config.latent_dim, config.hidden)
        self.decoder = nn.Linear(config.hidden, 22)
        self.skip = nn.Linear(48, 22, bias=False)
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        nn.init.normal_(self.decoder.weight, std=0.001)
        nn.init.zeros_(self.decoder.bias)
        nn.init.zeros_(self.skip.weight)

    def encode(self, current):
        return torch.tanh(self.encoder(current[:, STATE_INDICES]))

    def step(self, previous, current, latent):
        features = torch.cat((current, current - previous), -1)
        latent = torch.tanh(self.transition(torch.cat((features, latent), -1)))
        hidden = torch.relu(self.hidden(torch.cat((features, latent), -1)))
        state = current[:, STATE_INDICES] + (self.decoder(hidden) + self.skip(features)) * self.scale
        return (state, latent)

    def rollout(self, initial, future_commands):
        previous, current = (initial[:, 0], initial[:, 1])
        latent = self.encode(current)
        predictions = []
        for t in range(future_commands.shape[1]):
            state, latent = self.step(previous, current, latent)
            following = torch.empty_like(current)
            following[:, COMMAND_INDICES] = future_commands[:, t]
            following[:, STATE_INDICES] = state
            previous, current = (current, following)
            predictions.append(state)
        return torch.stack(predictions, 1)

def backbone_export_weights(model):
    return tuple((np.ascontiguousarray(p.detach().cpu().numpy()) for p in (model.encoder.weight, model.encoder.bias, model.transition.weight, model.transition.bias, model.hidden.weight, model.hidden.bias, model.decoder.weight, model.decoder.bias, model.skip.weight, model.scale)))

@njit(cache=True)
def backbone_native_encode(current, ew, eb):
    z = np.empty(len(eb), np.float32)
    for i in range(len(z)):
        value = eb[i]
        for j in range(22):
            value += ew[i, j] * current[STATE_ARRAY[j]]
        z[i] = np.tanh(value)
    return z

@njit(cache=True)
def backbone_native_step(previous, current, z, ew, eb, tw, tb, hw, hb, dw, db, skip, scale):
    features = np.empty(48, np.float32)
    for j in range(24):
        features[j] = current[j]
        features[j + 24] = current[j] - previous[j]
    next_z = np.empty(len(z), np.float32)
    for i in range(len(z)):
        value = tb[i]
        for j in range(48):
            value += tw[i, j] * features[j]
        for j in range(len(z)):
            value += tw[i, 48 + j] * z[j]
        next_z[i] = np.tanh(value)
    hidden = np.empty(len(hb), np.float32)
    for i in range(len(hidden)):
        value = hb[i]
        for j in range(48):
            value += hw[i, j] * features[j]
        for j in range(len(z)):
            value += hw[i, 48 + j] * next_z[j]
        hidden[i] = max(value, 0.0)
    state = np.empty(22, np.float32)
    for i in range(22):
        value = db[i]
        for j in range(len(hidden)):
            value += dw[i, j] * hidden[j]
        for j in range(48):
            value += skip[i, j] * features[j]
        state[i] = current[STATE_ARRAY[i]] + value * scale[i]
    return (state, next_z)

@njit(cache=True)
def backbone_native_chunk(rows, previous, current, z, recursive, mean, std, ew, eb, tw, tb, hw, hb, dw, db, skip, scale):
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
        state, z = backbone_native_step(previous, current, z, ew, eb, tw, tb, hw, hb, dw, db, skip, scale)
        previous = current.copy()
        for j in range(22):
            current[STATE_ARRAY[j]] = state[j]
            output[k, j] = np.float64(state[j]) * std[STATE_ARRAY[j]] + mean[STATE_ARRAY[j]]
    return (output, previous, current, z)

@njit(cache=True)
def backbone_physical_step(previous, current, z, mean, std, ew, eb, tw, tb, hw, hb, dw, db, skip, scale):
    p = np.empty(24, np.float32)
    c = np.empty(24, np.float32)
    for j in range(24):
        p[j] = (previous[j] - mean[j]) / std[j]
        c[j] = (current[j] - mean[j]) / std[j]
    state, next_z = backbone_native_step(p, c, z, ew, eb, tw, tb, hw, hb, dw, db, skip, scale)
    result = np.empty(22, np.float64)
    for j in range(22):
        result[j] = np.float64(state[j]) * std[STATE_ARRAY[j]] + mean[STATE_ARRAY[j]]
    return (result, next_z)

def backbone_load_bundle(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('format') != FORMAT:
        raise ValueError('Expected compact latent checkpoint')
    model = backbone_CompactLatent(backbone_ModelConfig(**ck['model_config']), ck['scale'])
    model.load_state_dict(ck['model'])
    model.eval()
    return (model, ck)

compact_backbone=SimpleNamespace(ModelConfig=backbone_ModelConfig,CompactLatent=backbone_CompactLatent,export_weights=backbone_export_weights,native_encode=backbone_native_encode,native_step=backbone_native_step,native_chunk=backbone_native_chunk)
import importlib
import numpy as np
import torch
from torch import nn
from numba import njit
rh_VARIANTS = {'A': (True, True, True), 'B': (False, True, True), 'C': (True, False, True), 'D': (False, False, True), 'E': (True, True, False), 'F': (True, True, True)}
rh__NATIVE_KERNELS = {}
rh_MODELS = {'gru': ('GRU', 'gru', 'EncodedResidualGRU'), 'compact': ('Compact recurrent', 'compact', 'CompactLatent')}

def rh_descriptors(raw):
    """Every raw command is used. Six temporal descriptors/channel/interval."""
    if torch.is_tensor(raw):
        n = raw.shape[2]
        w = torch.linspace(-1, 1, n, device=raw.device) if n > 1 else raw.new_zeros(1)
        return torch.cat((raw.mean(2), raw[:, :, 0], raw[:, :, -1], raw.amin(2), raw.amax(2), (raw * w[None, None, :, None]).mean(2)), -1)
    n = raw.shape[2]
    w = np.linspace(-1, 1, n, dtype=np.float32) if n > 1 else np.zeros(1, np.float32)
    return np.concatenate((raw.mean(2), raw[:, :, 0], raw[:, :, -1], raw.min(2), raw.max(2), (raw * w[None, None, :, None]).mean(2)), -1)

class rh_Ablation(nn.Module):

    def __init__(self, architecture, variant, scale, backbone_config):
        super().__init__()
        self.architecture, self.variant = (architecture, variant)
        self.difference, self.residual, self.use_skip = rh_VARIANTS[variant]
        self.recurrent = architecture != 'feedforward'
        module = rh_backbone_module(architecture if self.recurrent else 'compact')
        cls = getattr(module, rh_MODELS[architecture if self.recurrent else 'compact'][2])
        self.backbone = cls(module.ModelConfig(**backbone_config), scale)
        if not self.recurrent:
            del self.backbone.encoder
            del self.backbone.transition
            self.backbone.hidden = nn.Linear(48, 32)
        if not self.use_skip:
            self.backbone.skip = None
        self.context = nn.Linear(70, 16)
        self.controls = nn.Linear(240, 16, bias=False)
        self.queries = nn.Parameter(torch.randn(20, 16) * 0.02)
        self.output = nn.Linear(16, 22)
        nn.init.normal_(self.output.weight, std=0.001)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('mask', torch.tril(torch.ones(20, 20)).repeat_interleave(12, 1))
        self.register_buffer('lead', torch.arange(1, 21, dtype=torch.float32)[None, :, None])
        if not self.residual:
            with torch.no_grad():
                self.backbone.skip.weight.zero_()
                self.backbone.skip.weight[:, list(STATE_INDICES)] = torch.diag(1 / self.backbone.scale)

    def features(self, p, c):
        return torch.cat((c, c - p if self.difference else torch.zeros_like(c)), -1)

    def base(self, p, c):
        b = self.backbone
        f = self.features(p, c)
        if not self.recurrent:
            hidden = torch.relu(b.hidden(f))
        elif self.architecture == 'gru':
            memory = b.encode(p if self.difference else c)
            past = self.features(p, p) if self.difference else self.features(c, c)
            memory = b.cell(torch.tanh(b.feature_encoder(past)), memory)
            memory = b.cell(torch.tanh(b.feature_encoder(f)), memory)
            hidden = torch.tanh(b.post_encoder(memory))
        else:
            memory = b.encode(p if self.difference else c)
            past = self.features(p, p) if self.difference else self.features(c, c)
            memory = torch.tanh(b.transition(torch.cat((past, memory), -1)))
            memory = torch.tanh(b.transition(torch.cat((f, memory), -1)))
            hidden = torch.relu(b.hidden(torch.cat((f, memory), -1)))
        update = b.decoder(hidden)
        if b.skip is not None:
            update = update + b.skip(f)
        return (f, update)

    def core(self, previous, current, control_features):
        current = current.clone()
        current[:, [0, 16]] = control_features[:, 0, 2:4]
        f, base = self.base(previous, current)
        context_base = base if self.residual else base * self.backbone.scale
        context = torch.tanh(self.context(torch.cat((f, context_base), -1)))
        controls = control_features.flatten(1)[:, None, :] * self.mask[None]
        correction = self.output(torch.tanh(context[:, None, :] + self.controls(controls) + self.queries[None]))
        if self.residual:
            return current[:, STATE_INDICES][:, None, :] + self.lead * (base[:, None, :] + correction) * self.backbone.scale
        return (base[:, None, :] + self.lead * correction) * self.backbone.scale

    def forward(self, previous, current, raw_future_commands):
        return self.core(previous, current, rh_descriptors(raw_future_commands))

@njit(cache=True)
def rh_native_descriptors(raw):
    h, n, _ = raw.shape
    out = np.empty((h, 12), np.float32)
    for k in range(h):
        for j in range(2):
            total, moment = (np.float32(0), np.float32(0))
            low, high = (raw[k, 0, j], raw[k, 0, j])
            for q in range(n):
                v = raw[k, q, j]
                total += v
                moment += v * np.float32(2 * q / (n - 1) - 1 if n > 1 else 0)
                low, high = (min(low, v), max(high, v))
            out[k, j], out[k, 2 + j], out[k, 4 + j] = (total / n, raw[k, 0, j], raw[k, n - 1, j])
            out[k, 6 + j], out[k, 8 + j], out[k, 10 + j] = (low, high, moment / n)
    return out

def rh_native_factory(model):
    arch, difference, residual = (model.architecture, model.difference, model.residual)
    normalized = bool(getattr(model, 'normalized_differences', False))

    def arr(t):
        return np.ascontiguousarray(t.detach().cpu().numpy())
    b = model.backbone

    def linear(layer):
        return (arr(layer.weight), arr(layer.bias))
    empty = (np.zeros((1, 1), np.float32), np.zeros(1, np.float32))
    enc = linear(b.encoder) if model.recurrent else empty
    feature = linear(b.feature_encoder) if arch == 'gru' else linear(b.transition) if model.recurrent else empty
    cell = (arr(b.cell.weight_ih), arr(b.cell.bias_ih), arr(b.cell.weight_hh), arr(b.cell.bias_hh)) if arch == 'gru' else (*empty, *empty)
    hidden = linear(b.post_encoder if arch == 'gru' else b.hidden)
    dec = linear(b.decoder)
    skip = arr(b.skip.weight) if b.skip is not None else np.zeros((22, 48), np.float32)
    weights = (*enc, *feature, *cell, *hidden, *dec, skip, *linear(model.context), arr(model.controls.weight), arr(model.queries), *linear(model.output), arr(b.scale))
    if normalized:
        weights = (*weights, arr(model.difference_scale))
    key = (arch, difference, residual, normalized, tuple((v.shape for v in weights)))
    if key in rh__NATIVE_KERNELS:
        run, core = rh__NATIVE_KERNELS[key]
        return (run, core, weights)

    @njit(cache=False)
    def cell_step(f, z, w):
        if arch == 'gru':
            compact = np.tanh(w[2] @ f + w[3])
            xi = w[4] @ compact + w[5]
            hh = w[6] @ z + w[7]
            n = len(z)
            out = np.empty(n, np.float32)
            for j in range(n):
                r = 1 / (1 + np.exp(-(xi[j] + hh[j])))
                u = 1 / (1 + np.exp(-(xi[n + j] + hh[n + j])))
                out[j] = (1 - u) * np.tanh(xi[2 * n + j] + r * hh[2 * n + j]) + u * z[j]
            return out
        return np.tanh(w[2] @ np.concatenate((f, z)) + w[3])

    @njit(cache=False)
    def core(p, current, desc, w):
        c = current.copy()
        c[0], c[16] = (desc[0, 2], desc[0, 3])
        f = np.zeros(48, np.float32)
        f[:24] = c
        if difference:
            f[24:] = (c - p) / w[20] if normalized else c - p
        if arch == 'feedforward':
            hidden = np.maximum(w[8] @ f + w[9], np.float32(0))
        else:
            initial = p if difference else c
            z = np.tanh(w[0] @ initial[SI] + w[1])
            past = np.zeros(48, np.float32)
            past[:24] = initial
            z = cell_step(past, z, w)
            z = cell_step(f, z, w)
            if arch == 'gru':
                hidden = np.tanh(w[8] @ z + w[9])
            else:
                hidden = np.maximum(w[8] @ np.concatenate((f, z)) + w[9], np.float32(0))
        base = w[10] @ hidden + w[11] + w[12] @ f
        context_base = base if residual else base * w[19]
        context = np.tanh(w[13] @ np.concatenate((f, context_base)) + w[14])
        out = np.empty((20, 22), np.float32)
        flat = desc.flatten()
        for k in range(20):
            prefix = np.zeros(240, np.float32)
            prefix[:12 * (k + 1)] = flat[:12 * (k + 1)]
            q = np.tanh(context + w[15] @ prefix + w[16][k])
            corr = w[17] @ q + w[18]
            for j in range(22):
                if residual:
                    out[k, j] = c[SI[j]] + np.float32(k + 1) * (base[j] + corr[j]) * w[19][j]
                else:
                    out[k, j] = (base[j] + np.float32(k + 1) * corr[j]) * w[19][j]
        return out

    @njit(cache=False)
    def run(p, c, raw, w):
        return core(p, c, rh_native_descriptors(raw), w)
    rh__NATIVE_KERNELS[key] = (run, core)
    return (run, core, weights)

def rh_backbone_module(name):
    if name!='compact':raise ValueError('This frozen control checkpoint uses Compact only')
    return compact_backbone
residualhead=SimpleNamespace(Ablation=rh_Ablation,descriptors=rh_descriptors,native_factory=rh_native_factory)
import json
from pathlib import Path
import numpy as np
import torch
from numba import njit
_SELECTED_KERNELS = {}

class ImprovedDynamics(residualhead.Ablation):

    def __init__(self, architecture, variant, scale, backbone_config, difference_scale, scaler, stride):
        super().__init__(architecture, 'A', scale, backbone_config)
        self.variant = variant
        self.normalized_differences = variant.startswith('norm_')
        self.angle_mode = variant.split('_', 1)[1]
        if self.angle_mode not in ('neural', 'kinematic', 'integrated'):
            raise ValueError(variant)
        self.stride = stride
        self.register_buffer('difference_scale', torch.tensor(difference_scale, dtype=torch.float32))
        self.register_buffer('state_std', torch.tensor(np.asarray(scaler['std'])[SI], dtype=torch.float32))
        self.register_buffer('state_mean', torch.tensor(np.asarray(scaler['mean'])[SI], dtype=torch.float32))

    def features(self, p, c):
        difference = (c - p) / self.difference_scale if self.normalized_differences else c - p
        return torch.cat((c, difference), -1)

    def core(self, p, c, control_features):
        prediction = super().core(p, c, control_features)
        if self.angle_mode == 'neural':
            return prediction
        initial_velocity = c[:, SI[VELOCITIES]] * self.state_std[VELOCITIES] + self.state_mean[VELOCITIES]
        dt = self.stride * 1e-05
        if self.angle_mode == 'kinematic':
            correction = self.lead * dt * initial_velocity[:, None, :] / self.state_std[ANGLES]
            prediction = prediction.clone()
            prediction[:, :, ANGLES] = prediction[:, :, ANGLES] + correction
            return prediction
        velocity = prediction[:, :, VELOCITIES] * self.state_std[VELOCITIES] + self.state_mean[VELOCITIES]
        previous = torch.cat((initial_velocity[:, None, :], velocity[:, :-1]), 1)
        displacement = torch.cumsum((previous + velocity) * (0.5 * dt), 1)
        prediction = prediction.clone()
        prediction[:, :, ANGLES] = c[:, SI[ANGLES]][:, None, :] + displacement / self.state_std[ANGLES]
        return prediction

    def recursive(self, p, c, raw_commands):
        """Diagnostic recursion of the first lead; re-encodes two-row history."""
        desc = residualhead.descriptors(raw_commands)
        output = []
        for k in range(20):
            suffix = desc[:, k:]
            if k:
                suffix = torch.cat((suffix, desc[:, -1:].expand(-1, k, -1)), 1)
            current = c.clone()
            current[:, [0, 16]] = desc[:, k, 2:4]
            state = self.core(p, current, suffix)[:, 0]
            following = current.clone()
            following[:, SI] = state
            output.append(state)
            p, c = (current, following)
        return torch.stack(output, 1)

def load_improved(path):
    """Load a final checkpoint with its shared, adjacent normalization snapshot."""
    path = Path(path)
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck['config'].get('experiment') != 'physics_normalization_v1':
        raise ValueError('Not an improved-dynamics checkpoint')
    scaler = json.loads((path.parent / 'scaler.json').read_text())
    info = ck['training']
    model = ImprovedDynamics(info['architecture'], info['variant'], ck['scale'], ck['backbone_config'], ck['model']['difference_scale'].numpy(), scaler, info['stride'])
    model.load_state_dict(ck['model'])
    model.eval()
    np.testing.assert_array_equal(model.state_std.numpy(), np.asarray(scaler['std'], np.float32)[SI])
    np.testing.assert_array_equal(model.state_mean.numpy(), np.asarray(scaler['mean'], np.float32)[SI])
    return (model, ck, scaler)

def selected_native_factory(model):
    _, base, w = residualhead.native_factory(model)
    key = (model.architecture, model.normalized_differences, model.angle_mode, model.stride, tuple(model.state_std.tolist()), tuple(model.state_mean.tolist()))
    if key in _SELECTED_KERNELS:
        native, core, recursive = _SELECTED_KERNELS[key]
        return (native, core, w, recursive)
    mode = model.angle_mode
    dt = np.float32(model.stride * 1e-05)
    std = np.ascontiguousarray(model.state_std.numpy())
    mean = np.ascontiguousarray(model.state_mean.numpy())
    angles = np.asarray(ANGLES, np.int64)
    velocities = np.asarray(VELOCITIES, np.int64)

    @njit(cache=False)
    def core(p, c, desc, weights):
        out = base(p, c, desc, weights)
        if mode != 'neural':
            for axis in range(2):
                a, v = (angles[axis], velocities[axis])
                initial = c[SI[v]] * std[v] + mean[v]
                previous = initial
                integral = np.float32(0)
                for k in range(20):
                    if mode == 'kinematic':
                        out[k, a] += np.float32(k + 1) * dt * initial / std[a]
                    else:
                        velocity = out[k, v] * std[v] + mean[v]
                        integral += np.float32(0.5) * dt * (previous + velocity)
                        out[k, a] = c[SI[a]] + integral / std[a]
                        previous = velocity
        return out

    @njit(cache=False)
    def native(p, c, raw, weights):
        return core(p, c, rh_native_descriptors(raw), weights)

    @njit(cache=False)
    def recursive(p, current, raw, weights):
        desc = rh_native_descriptors(raw)
        c = current.copy()
        p = p.copy()
        result = np.empty((20, 22), np.float32)
        for k in range(20):
            future = np.empty_like(desc)
            for j in range(20):
                future[j] = desc[min(k + j, 19)]
            c[0], c[16] = (desc[k, 2], desc[k, 3])
            result[k] = core(p, c, future, weights)[0]
            p = c.copy()
            c[SI] = result[k]
        return result
    _SELECTED_KERNELS[key] = (native, core, recursive)
    return (native, core, w, recursive)
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT = ROOT.parent
FOUR=PROJECT/'Results'
NEURAL = PROJECT / 'Neural_MPC'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

native_factory=selected_native_factory
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

def schedule(moves, cfg):
    per_block = round(cfg['control_interval_s'] / cfg['prediction_interval_s'])
    plan = np.repeat(np.asarray(moves, float), per_block, axis=0)
    if plan.shape != (cfg['horizon_steps'], 2):
        raise ValueError('Move blocks do not cover the prediction horizon')
    return plan

def rate_slacks(moves, previous, limit, interval):
    change = np.diff(np.vstack((previous, moves)), axis=0)
    maximum = np.asarray(limit, float) * interval
    return np.r_[(maximum - change).ravel(), (maximum + change).ravel()]

class Predictor:

    def __init__(self, path):
        self.model, self.ck, self.sc = load_improved(path)
        self.native, self.core, self.weights, _ = native_factory(self.model)
        self.mean = np.asarray(self.sc['mean'])
        self.std = np.asarray(self.sc['std'])
        self.path = Path(path)
        if self.model.stride != 1000 or self.model.angle_mode != 'integrated' or (not self.model.normalized_differences):
            raise ValueError('Pilot requires the trained 10-ms normalized/integrated model')

    def commands(self, plan):
        commands = ((plan - self.mean[CI]) / self.std[CI]).astype(np.float32)
        return np.ascontiguousarray(np.repeat(commands[:, None, :], self.model.stride, axis=1))

    def predict(self, p, c, raw_commands):
        values = self.native(p, c, raw_commands, self.weights)
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite model forecast')
        return values.astype(float) * self.std[SI] + self.mean[SI]

class Objective:

    def __init__(self, predictor, p, c, previous, reference, tip_reference, geometry, cfg, bounds):
        self.predictor, self.p, self.c = (predictor, p, c)
        self.previous = previous
        self.reference = reference
        self.tip_reference = tip_reference
        self.geometry, self.cfg = (geometry, cfg)
        self.low, self.high = bounds
        self.center = (self.low + self.high) / 2
        self.half = (self.high - self.low) / 2
        self.best_cost = np.inf
        self.best_moves = None
        self.calls = 0

    def physical(self, z):
        return self.center + np.asarray(z).reshape(-1, 2) * self.half

    def feasible(self, moves):
        if np.any(moves < self.low - 1e-07) or np.any(moves > self.high + 1e-07):
            return False
        limit = self.cfg['command_rate_limit_V_per_s']
        return limit is None or np.min(rate_slacks(moves, self.previous, limit, self.cfg['control_interval_s'])) >= -1e-07

    def __call__(self, z):
        self.calls += 1
        moves = self.physical(z)
        if not self.feasible(moves):
            return 1000000000000.0
        try:
            prediction = self.predictor.predict(self.p, self.c, self.predictor.commands(schedule(moves, self.cfg)))
            mechanical = prediction[:, MECH]
            tip = self.geometry.end_effector(mechanical)
            weights = self.cfg['cost_weights']
            scales = self.predictor.std[SI[MECH]]
            position = (tip[:, :2] - self.tip_reference[:, :2]) / self.cfg['position_scale_m']
            angles = (mechanical[:, :2] - self.reference[:, :2]) / scales[:2]
            velocity = (mechanical[:, 2:] - self.reference[:, 2:]) / scales[2:]
            effort = (moves / self.half) ** 2
            change = (np.diff(np.vstack((self.previous, moves)), axis=0) / self.half) ** 2
            cost = float(weights['position'] * np.mean(position ** 2) + weights['joint_angles'] * np.mean(angles ** 2) + weights['angular_velocity'] * np.mean(velocity ** 2) + weights['effort'] * np.mean(effort) + weights['command_change'] * np.mean(change) + weights['terminal_position'] * np.mean(position[-1] ** 2))
        except FloatingPointError:
            return 1000000000000.0
        if not np.isfinite(cost):
            return 1000000000000.0
        if cost < self.best_cost:
            self.best_cost, self.best_moves = (cost, moves.copy())
        return cost

def solve(objective, guess, cfg):
    z = np.clip((guess - objective.center) / objective.half, -1, 1).ravel()
    constraints = []
    if cfg['command_rate_limit_V_per_s'] is not None:
        constraints = [{'type': 'ineq', 'fun': lambda decision: rate_slacks(objective.physical(decision), objective.previous, cfg['command_rate_limit_V_per_s'], cfg['control_interval_s'])}]
    started = time.perf_counter()
    result = minimize(objective, z, method='SLSQP', bounds=[(-1, 1)] * len(z), constraints=constraints, options={'maxiter': cfg['maxiter'], 'ftol': cfg['ftol'], 'eps': cfg['finite_difference_step']})
    elapsed = time.perf_counter() - started
    if objective.best_moves is None:
        raise RuntimeError('Optimizer produced no finite feasible candidate')
    return (result, objective.best_moves, elapsed)

_pilot=SimpleNamespace(SI=SI,CI=CI,MECH=MECH,Objective=Objective,Predictor=Predictor,solve=solve,native_factory=native_factory)
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class PlanarGeometry:
    lift_length: float
    jib_length: float
    jib_angle_mode: str
    base_x: float = 0.0
    base_z: float = 0.0
    lift_offset_deg: float = 0.0
    jib_offset_deg: float = 0.0
    lift_sign: float = 1.0
    jib_sign: float = 1.0
    lift_vector: tuple | None = None
    jib_vector: tuple | None = None

    def __post_init__(self):
        numbers = [value for key, value in asdict(self).items() if key not in ('jib_angle_mode', 'lift_vector', 'jib_vector')]
        if not np.isfinite(numbers).all() or min(self.lift_length, self.jib_length) <= 0:
            raise ValueError('Boom lengths must be positive metres; geometry values must be finite')
        if self.jib_angle_mode not in ('relative', 'absolute'):
            raise ValueError('jib_angle_mode must be relative or absolute')
        if self.lift_sign not in (-1, 1) or self.jib_sign not in (-1, 1):
            raise ValueError('Angle signs must be +1 or -1')
        for name in ('lift', 'jib'):
            vector = getattr(self, name + '_vector')
            if vector is not None:
                values = np.asarray(vector, float)
                if values.shape != (2,) or not np.isfinite(values).all() or np.linalg.norm(values) <= 0:
                    raise ValueError('Local link vectors require two finite coordinates and nonzero length')
                object.__setattr__(self, name + '_vector', tuple(values.tolist()))
                object.__setattr__(self, name + '_length', float(np.linalg.norm(values)))

    def evaluate(self, mechanics):
        state = np.asarray(mechanics, dtype=float)
        if state.shape[-1] != 4:
            raise ValueError('Expected lift angle, jib angle, lift velocity, jib velocity')
        lift = self.lift_sign * state[..., 0] + np.deg2rad(self.lift_offset_deg)
        jib = self.jib_sign * state[..., 1] + np.deg2rad(self.jib_offset_deg)
        lift_rate = self.lift_sign * state[..., 2]
        jib_rate = self.jib_sign * state[..., 3]
        if self.jib_angle_mode == 'relative':
            jib = lift + jib
            jib_rate = lift_rate + jib_rate

        def rotate(angle, local):
            return np.stack((local[0] * np.cos(angle) - local[1] * np.sin(angle), local[0] * np.sin(angle) + local[1] * np.cos(angle)), axis=-1)
        lift_vector = rotate(lift, self.lift_vector or (self.lift_length, 0.0))
        jib_vector = rotate(jib, self.jib_vector or (self.jib_length, 0.0))
        lift_velocity = lift_rate[..., None] * np.stack((-lift_vector[..., 1], lift_vector[..., 0]), axis=-1)
        jib_velocity = jib_rate[..., None] * np.stack((-jib_vector[..., 1], jib_vector[..., 0]), axis=-1)
        elbow = lift_vector + np.asarray((self.base_x, self.base_z))
        tip = elbow + jib_vector
        return (np.concatenate((elbow, lift_velocity), axis=-1), np.concatenate((tip, lift_velocity + jib_velocity), axis=-1))

    def end_effector(self, mechanics):
        return self.evaluate(mechanics)[1]
import numpy as np

class Prediction:
    """Same frozen weights, explicitly selected direct or native recursive operator."""

    def __init__(self, base, mode):
        self.base = base
        self.mode = mode
        self.recursive = _pilot.native_factory(base.model)[3]

    def __getattr__(self, name):
        return getattr(self.base, name)

    def predict(self, p, c, raw):
        kernel = self.base.native if self.mode == 'direct' else self.recursive
        result = kernel(p, c, raw, self.weights)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite MPC forecast')
        return result.astype(float) * self.std[_pilot.SI] + self.mean[_pilot.SI]
import numpy as np

class ExtendedPredictor:
    """Original 20-step recursive operator, causally continued beyond20 steps."""

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def predict(self, p, c, raw):
        if len(raw) == 20:
            return self.base.predict(p, c, raw)
        p, c = (p.copy(), c.copy())
        pieces = []
        for start in range(0, len(raw), 20):
            part = raw[start:start + 20]
            n = len(part)
            padded = part if n == 20 else np.concatenate((part, np.repeat(part[-1:], 20 - n, axis=0)))
            values = self.base.recursive(p, c, padded, self.base.weights)[:n]
            pieces.append(values)
            previous = c.copy()
            following = c.copy()
            previous[SI] = values[-2]
            following[SI] = values[-1]
            previous[CI] = part[-1, 0]
            following[CI] = part[-1, -1]
            p, c = (previous, following)
        values = np.concatenate(pieces)
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite extended forecast')
        return values.astype(float) * self.std[SI] + self.mean[SI]

def plan_from_moves(moves, horizon):
    if len(moves) == 2:
        return np.concatenate((np.repeat(moves[:1], 10, axis=0), np.repeat(moves[1:], horizon - 10, axis=0)))
    if len(moves) == 4 and horizon == 40:
        return np.repeat(moves, 10, axis=0)
    raise ValueError('Unsupported blocking')

class HorizonObjective(_pilot.Objective):

    def __call__(self, z):
        self.calls += 1
        moves = self.physical(z)
        if not self.feasible(moves):
            return 1000000000000.0
        try:
            prediction = self.predictor.predict(self.p, self.c, self.predictor.commands(plan_from_moves(moves, self.cfg['horizon_steps'])))
            mechanical = prediction[:, MECH]
            tip = self.geometry.end_effector(mechanical)
            weights = self.cfg['cost_weights']
            scales = self.predictor.std[SI[MECH]]
            position = (tip[:, :2] - self.tip_reference[:, :2]) / self.cfg['position_scale_m']
            angles = (mechanical[:, :2] - self.reference[:, :2]) / scales[:2]
            velocity = (mechanical[:, 2:] - self.reference[:, 2:]) / scales[2:]
            effort = (moves / self.half) ** 2
            change = (np.diff(np.vstack((self.previous, moves)), axis=0) / self.half) ** 2
            cost = float(weights['position'] * np.mean(position ** 2) + weights['joint_angles'] * np.mean(angles ** 2) + weights['angular_velocity'] * np.mean(velocity ** 2) + weights['effort'] * np.mean(effort) + weights['command_change'] * np.mean(change) + weights['terminal_position'] * np.mean(position[-1] ** 2))
        except FloatingPointError:
            return 1000000000000.0
        if not np.isfinite(cost):
            return 1000000000000.0
        if cost < self.best_cost:
            self.best_cost, self.best_moves = (cost, moves.copy())
        return cost

def make_objective(ctx, p, c, previous, reference, horizon):
    settings = dict(ctx['cfg'], horizon_steps=horizon)
    cls = _pilot.Objective if horizon == 20 else HorizonObjective
    predictor = ctx['predictor'] if horizon == 20 else ExtendedPredictor(ctx['predictor'])
    return cls(predictor, p, c, previous, reference[:, MECH], ctx['geometry'].end_effector(reference[:, MECH]), ctx['geometry'], settings, ctx['bounds'])

_horizon=SimpleNamespace(make_objective=make_objective,plan_from_moves=plan_from_moves,ExtendedPredictor=ExtendedPredictor)
import numpy as np
import torch
from numba import set_num_threads
EXPECTED = '590028f9aa7c1e660f4a09ec4b7563619ddcb9ebcee34eb694b589d33dc942f5'

def context(configuration='mpc400'):
    torch.set_num_threads(1)
    set_num_threads(1)
    path = ROOT / 'checkpoints/compact_norm_integrated_s44_dt1000.pt'
    if digest(path) != EXPECTED:
        raise ValueError('Frozen dynamics hash mismatch')
    cfg = read_json(ROOT / 'configs' / (configuration + '.json'))
    base = Predictor(path)
    return dict(base=base, predictor=Prediction(base, 'recursive'), cfg=cfg['settings'], geometry=PlanarGeometry(**cfg['geometry']), bounds=(np.asarray(cfg['bounds']['low']), np.asarray(cfg['bounds']['high'])))
import time
import numpy as np

class NeuralPlant:

    def __init__(self, predictor, previous, current, correct_command_history=False):
        self.predictor = predictor
        self.previous = previous.copy()
        self.current = current.copy()
        self.steps = 0
        self.correct_command_history = correct_command_history

    def physical(self):
        return self.current[_pilot.SI].astype(float) * self.predictor.std[_pilot.SI] + self.predictor.mean[_pilot.SI]

    def advance(self, commands, raw_intervals=None):
        """Each transition uses output 1 of the same direct model, with causal hold padding."""
        states = []
        elapsed = []
        for index, command in enumerate(commands):
            tick = time.perf_counter()
            raw = self.predictor.commands(np.tile(command, (20, 1)))
            if raw_intervals is not None:
                raw[0] = ((raw_intervals[index] - self.predictor.mean[_pilot.CI]) / self.predictor.std[_pilot.CI]).astype(np.float32)
            history_current = self.current.copy()
            if self.correct_command_history:
                history_current[_pilot.CI] = raw[0, 0]
            prediction = self.predictor.native(self.previous, self.current, raw, self.predictor.weights)
            if not np.isfinite(prediction[0]).all():
                raise FloatingPointError(f'Nonfinite autonomous transition at step {self.steps}')
            nxt = self.current.copy()
            nxt[_pilot.SI] = prediction[0]
            nxt[_pilot.CI] = raw[0, -1] if self.correct_command_history else raw[0, 0]
            before = self.physical()
            after = nxt[_pilot.SI].astype(float) * self.predictor.std[_pilot.SI] + self.predictor.mean[_pilot.SI]
            expected = before[_pilot.MECH[:2]] + 0.005 * (before[_pilot.MECH[2:]] + after[_pilot.MECH[2:]])
            np.testing.assert_allclose(after[_pilot.MECH[:2]], expected, rtol=0, atol=2e-06)
            self.previous, self.current = (history_current, nxt)
            self.steps += 1
            states.append(after)
            elapsed.append(time.perf_counter() - tick)
        return (np.asarray(states), elapsed)

def stats(values, prefix):
    values = np.asarray(values) * 1000
    return {prefix + '_' + key + '_ms': float(fun(values)) for key, fun in [('mean', np.mean), ('median', np.median), ('p95', lambda x: np.percentile(x, 95)), ('p99', lambda x: np.percentile(x, 99)), ('max', np.max)]}
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
FORMAT = 'imotion_lstm_normalized_raw_csv_v1'

def read_chunks(path: Path, skip_rows: int, chunksize: int):
    seen = 0
    for frame in pd.read_csv(path, usecols=INPUT_COLS, chunksize=chunksize, dtype=np.float64):
        values = frame[INPUT_COLS].to_numpy(dtype=np.float64, copy=False)
        first = max(0, skip_rows - seen)
        seen += len(values)
        if first < len(values):
            retained = values[first:]
            if not np.isfinite(retained).all():
                raise ValueError(f'Nonfinite values found in {path}')
            yield retained

def convert_recording(path, destination_dir, rows, skip_rows, chunksize, scaler):
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / f'{path.stem}.npy'
    metadata_path = destination_dir / f'{path.stem}.json'
    stat = path.stat()
    signature = dict(format=FORMAT, source=str(path), source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns, rows=rows, skip_rows=skip_rows, mean=scaler['mean'], std=scaler['std'], smooth_columns=list(COMMAND_INDICES), smooth_order=3, smooth_window=51)
    if target.exists() and metadata_path.exists():
        try:
            if json.loads(metadata_path.read_text(encoding='utf-8')) == signature:
                cached = np.load(target, mmap_mode='r')
                if cached.shape == (rows, 24) and cached.dtype == np.float32:
                    del cached
                    print(f'Already prepared: {target.name}', flush=True)
                    return target
                del cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    temporary = target.with_name(target.stem + '.incomplete.npy')
    mapped = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(rows, 24))
    mean = np.asarray(scaler['mean'], dtype=np.float64)
    std = np.asarray(scaler['std'], dtype=np.float64)
    offset = 0
    for chunk in read_chunks(path, skip_rows, chunksize):
        end = offset + len(chunk)
        mapped[offset:end] = ((chunk - mean) / std).astype(np.float32)
        offset = end
    if offset != rows:
        del mapped
        raise RuntimeError(f'CSV row count changed during conversion: {path}')
    for column in COMMAND_INDICES:
        smoothed = savgol_filter(np.asarray(mapped[:, column], dtype=np.float64), window_length=51, polyorder=3)
        mapped[:, column] = smoothed.astype(np.float32)
    mapped.flush()
    del mapped
    temporary.replace(target)
    write_json(metadata_path, signature)
    print(f'Prepared {target}: {rows:,} x 24 float32', flush=True)
    return target
import json
from pathlib import Path
import numpy as np
import pandas as pd

def raw_controls(source, scaler, rows, cache_dir, prepare=False):
    """Normalize actual command samples, without any future-looking filter."""
    source = Path(source).resolve()
    split = source.parent.name.lower()
    if split not in ('train', 'test'):
        raise ValueError(f'Expected source in Train or Test: {source}')
    cache_location = Path(cache_dir).resolve()
    if any((cache_location.is_relative_to(source.parent.parent / name) for name in ('Train', 'Test'))):
        raise ValueError('Control caches must be outside raw Train and Test folders')
    stat = source.stat()
    mean = np.asarray(scaler['mean'], float)[[0, 16]]
    std = np.asarray(scaler['std'], float)[[0, 16]]
    skip = int(scaler['startup_rows_removed'])
    signature = dict(source=str(source), source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns, rows=int(rows), skip_rows=skip, mean=mean.tolist(), std=std.tolist(), commands=['ulift[V]', 'ujib[V]'], filter='none: raw physical command samples')
    target = Path(cache_dir) / split / (source.stem + '.npy')
    meta = target.with_suffix('.json')
    if target.is_file() and meta.is_file():
        if json.loads(meta.read_text(encoding='utf-8')) == signature:
            result = np.load(target, mmap_mode='r')
            if result.shape == (rows, 2) and result.dtype == np.float32:
                return result
    if not prepare:
        raise FileNotFoundError(f'Raw-control cache missing/stale for {source.name}. Run prepare_direct_controls.py (Train only).')
    if Path(cache_dir).resolve().is_relative_to(source.parent):
        raise ValueError('Raw-control cache must be outside the raw source folder')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.incomplete.npy')
    mapped = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(rows, 2))
    offset, seen = (0, 0)
    print(f'Preparing RAW controls only, no smoothing: {source.name}', flush=True)
    for frame in pd.read_csv(source, usecols=['ulift[V]', 'ujib[V]'], chunksize=200000, dtype=np.float64):
        first = max(0, skip - seen)
        seen += len(frame)
        values = frame[['ulift[V]', 'ujib[V]']].to_numpy()[first:]
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite raw command: {source}')
        if offset + len(values) > rows:
            raise ValueError(f'Raw/prepared row count mismatch: {source}')
        mapped[offset:offset + len(values)] = ((values - mean) / std).astype(np.float32)
        offset += len(values)
    if offset != rows:
        raise ValueError(f'Raw/prepared row count mismatch: {source}: {offset} != {rows}')
    after = source.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Source changed during raw-control preparation: {source}')
    mapped.flush()
    del mapped
    temporary.replace(target)
    write_json(meta, signature)
    return np.load(target, mmap_mode='r')
import os
from pathlib import Path
import numpy as np

def inventory(split='validation'):
    rows = read_json(ROOT / 'configs/fixtures/splits.json')
    selected = [s for s in rows if s['source_split'] == 'Test'] if split == 'historical-test' else [s for s in rows if s['source_split'] == 'Train' and (split == 'all-train' or s['controller_split'] == split)]
    for s in selected:
        s['matched_updates'] = min(s['updates'], (s['rows'] - 31001) // 10000)
    return selected

class Dataset:

    def __init__(self, data_root=None, prepared_root=None, control_cache=None, allow_reused_test=False):
        self.allow_reused_test = allow_reused_test
        self.raw = Path(data_root or os.environ.get('PATU_DATA_ROOT', ROOT / 'configs/fixtures/raw')).resolve()
        self.prepared = Path(prepared_root or os.environ.get('PATU_PREPARED_ROOT', self.raw / 'PATU_prepared_cache')).resolve()
        self.controls = Path(control_cache or self.prepared / 'controls').resolve()
        for p in [self.prepared, self.controls]:
            if any((p.is_relative_to(self.raw / s) for s in ['Train', 'Test'])):
                raise ValueError('Caches must be outside raw Train/Test')

    def array(self, spec, scaler):
        if spec['source_split'] != 'Train' and (not self.allow_reused_test):
            raise ValueError('Default pipeline prohibits new Test evaluations')
        path = self.prepared / ('training' if spec['source_split'] == 'Train' else 'testing') / (Path(spec['recording']).stem + '.npy')
        source = self.raw / spec['source_split'] / spec['recording']
        if not source.is_file():
            raise FileNotFoundError(f'Provide --data-root containing Train/: {source}')
        if source.stat().st_size != spec['source_size']:
            raise ValueError('Recording size differs from frozen manifest: ' + source.name)
        if not path.exists():
            convert_recording(source, path.parent, spec['rows'], int(scaler['startup_rows_removed']), 100000, scaler)
        metadata = read_json(path.with_suffix('.json'))
        stat = source.stat()
        if (metadata['source_size'], metadata['source_mtime_ns']) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError('Prepared cache source signature changed; use a fresh --prepared-root')
        for key in ['mean', 'std']:
            np.testing.assert_array_equal(metadata[key], scaler[key])
        arr = np.load(path, mmap_mode='r')
        if arr.shape != (spec['rows'], 24) or arr.dtype != np.float32:
            raise ValueError('Prepared shape/dtype mismatch')
        return arr

    def record(self, spec, ctx):
        if spec['source_split'] != 'Train':
            raise ValueError('No new Test controller simulations in this package')
        base = ctx['base']
        x = self.array(spec, base.sc)
        n = spec['matched_updates']
        source = self.raw / 'Train' / spec['recording']
        controls = raw_controls(source, base.sc, len(x), self.controls, prepare=True)
        rows = 1000 + np.arange((n - 1) * 10 + 41) * 1000
        reference = x[rows][:, SI].astype(float) * base.std[SI] + base.mean[SI]
        return dict(reference=reference, previous=x[0].copy(), current=x[1000].copy(), previous_command=controls[999].astype(float) * base.std[CI] + base.mean[CI], raw_controls=controls, name=source.name, start=18.01, duration=n * 0.1)

    def verify_hashes(self):
        import hashlib
        expected = read_json(ROOT / 'configs/fixtures/required_recording_sha256.json')
        for name, sha in expected.items():
            path = self.raw / 'Train' / name
            with path.open('rb') as handle:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
            if actual != sha:
                raise ValueError('Raw recording content hash mismatch: ' + name)
            print('VERIFIED Train/' + name, flush=True)
import torch

class MLP(torch.nn.Module):

    def __init__(self, dimension, width, second, outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(dimension, width), torch.nn.ReLU(), torch.nn.Linear(width, second), torch.nn.ReLU(), torch.nn.Linear(second, outputs), torch.nn.Tanh())

    def forward(self, x):
        return self.layers(x)

full=SimpleNamespace(MLP=MLP)
import numpy as np
import torch

def feature(base, p, c, previous, ref):
    current = c[SI].astype(float) * base.std[SI] + base.mean[SI]
    past = p[SI].astype(float) * base.std[SI] + base.mean[SI]
    f = np.r_[current, past, previous, current[MECH] - ref[0, MECH], ref[1:41, MECH].ravel()].astype(np.float32)
    assert f.shape == (210,) and np.isfinite(f).all()
    return f

class Policy:

    def __init__(self, path):
        d = torch.load(path, map_location='cpu', weights_only=False)
        self.d = d
        self.memory = None
        self.mean = d['mean']
        self.std = d['std']
        self.center = d['center']
        self.half = d['half']
        self.weights = []
        model = full.MLP(d['dimension'], d['width'], d['second'], d['outputs'])
        model.load_state_dict(d['weights'])
        for l in model.layers:
            if isinstance(l, torch.nn.Linear):
                self.weights.append((l.weight.detach().numpy().copy(), l.bias.detach().numpy().copy()))

    def reset(self, previous):
        self.memory = np.asarray(previous, np.float32).copy()

    def __call__(self, f):
        x = np.r_[f, self.memory].astype(np.float32) if self.d['outputs'] == 4 else f
        value = (x - self.mean) / self.std
        for k, (w, b) in enumerate(self.weights):
            value = w @ value + b
            value = np.maximum(value, 0) if k < 2 else np.tanh(value)
        result = self.center + self.half * value
        if self.d['outputs'] == 4:
            self.memory = result[2:].copy()
        return result[:2].astype(float)
import time
import numpy as np
from types import SimpleNamespace
d = SimpleNamespace(feature=feature, horizon=_horizon, mpc=_pilot)

def online(ctx, pa, cu, previous, ref, shifted, method, budget, old=None, policy=None):
    tick = time.perf_counter()
    warm = shifted.copy() if method == 'H0' else np.tile(previous, (2, 1))
    if method in ['H2', 'H3']:
        neural = np.clip(old(d.feature(ctx['base'], pa, cu, previous, ref)), *ctx['bounds'])
        warm = np.tile(neural, (2, 1)) if method == 'H2' else np.vstack((neural, shifted[1]))
    if method in ['H2', 'H3']:
        warm = np.clip(warm, *ctx['bounds'])
    obj = d.horizon.make_objective(ctx, pa, cu, previous, ref[1:], 40)
    res, plan, solve_s = d.mpc.solve(obj, warm, dict(ctx['cfg'], maxiter=budget))
    command = plan[0].copy()
    following = np.repeat(plan[1:], 2, axis=0)
    elapsed = time.perf_counter() - tick
    return (command, following, dict(controller_s=elapsed, solve_s=solve_s, status=int(res.status), success=bool(res.success), iterations=int(res.nit), nfev=int(res.nfev), njev=int(res.njev), calls=obj.calls, cost=float(obj.best_cost)))
import time
from pathlib import Path
import numpy as np
import pandas as pd
CONTROLLERS = ['replay', 'mpc200', 'mpc400', 'neural', 'conventional3', 'hybrid']

def decision(ctx, plant, previous, ref, shifted, name, policy):
    if name == 'neural':
        tick = time.perf_counter()
        command = policy(feature(ctx['base'], plant.previous, plant.current, previous, ref))
        return (command, shifted, dict(controller_s=time.perf_counter() - tick, solve_s=0.0, status=-1, success=False, iterations=0, nfev=0, njev=0, calls=0, cost=0.0))
    if name == 'mpc200':
        tick = time.perf_counter()
        obj = _horizon.make_objective(ctx, plant.previous, plant.current, previous, ref[1:21], 20)
        result, plan, seconds = _pilot.solve(obj, shifted, dict(ctx['cfg'], horizon_steps=20, maxiter=5))
        command = plan[0].copy()
        following = np.repeat(plan[1:], 2, axis=0)
        return (command, following, dict(controller_s=time.perf_counter() - tick, solve_s=seconds, status=int(result.status), success=bool(result.success), iterations=int(result.nit), nfev=int(result.nfev), njev=int(result.njev), calls=obj.calls, cost=float(obj.best_cost)))
    return online(ctx, plant.previous, plant.current, previous, ref, shifted, 'H2' if name == 'hybrid' else 'H0', 3 if name in ['hybrid', 'conventional3'] else 5, policy)

def rollout(ctx, dataset, spec, name, output, limit=None, policy_override=None, force=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / (name + '_' + Path(spec['recording']).stem)
    if prefix.with_suffix('.json').exists() and not force:
        saved = read_json(prefix.with_suffix('.json'))
        expected = min(spec['matched_updates'], limit or spec['matched_updates']) * 10
        if saved['samples'] != expected:
            raise ValueError('Existing run has a different duration; choose a separate --output directory')
        return saved
    record = dataset.record(spec, ctx)
    plant = NeuralPlant(ctx['base'], record['previous'], record['current'], True)
    previous = record['previous_command'].copy()
    shifted = np.tile(previous, (2, 1))
    policy = Policy(ROOT.parent / 'Neural_MPC/checkpoints/wide_s44.pt') if name in ['neural','hybrid'] else None
    if policy_override is not None:policy=policy_override
    n = min(spec['matched_updates'], limit or spec['matched_updates'])
    states = [plant.physical()]
    commands = []
    rows = []
    raw = ctx['base'].commands(np.zeros((20, 2)))
    ctx['base'].native(plant.previous, plant.current, raw, ctx['base'].weights)
    ctx['predictor'].predict(plant.previous, plant.current, raw)
    for k in range(n):
        ref = record['reference'][k * 10:k * 10 + 41]
        if name == 'replay':
            tick = time.perf_counter()
            normalized = record['raw_controls'][1000 + k * 10000:1000 + (k + 1) * 10000].reshape(10, 1000, 2)
            intervals = normalized.astype(float) * ctx['base'].std[CI] + ctx['base'].mean[CI]
            applied = intervals[:, -1, :]
            command = applied[0]
            row = dict(controller_s=time.perf_counter() - tick, solve_s=0.0, status=-1, success=False, iterations=0, nfev=0, njev=0, calls=0, cost=0.0)
        else:
            command, shifted, row = decision(ctx, plant, previous, ref, shifted, name, policy)
            applied = np.tile(command, (10, 1))
            intervals = None
        feasible = bool(np.isfinite(applied).all() and np.all(applied >= ctx['bounds'][0] - 1e-07) and np.all(applied <= ctx['bounds'][1] + 1e-07))
        if not feasible:
            raise FloatingPointError('Infeasible command')
        tick = time.perf_counter()
        generated, _ = plant.advance(applied, intervals)
        row['plant_s'] = time.perf_counter() - tick
        if not np.isfinite(generated).all() or abs(plant.current[SI]).max() > 1000000.0:
            raise FloatingPointError('Numerical state guard')
        sat = abs(applied) > 0.999
        row.update(update=k, time_s=k * 0.1, feasible=feasible, lift_saturated=bool(sat[:, 0].any()), jib_saturated=bool(sat[:, 1].any()))
        rows.append(row)
        states.extend(generated)
        commands.extend(applied)
        previous = applied[-1].copy()
    states = np.asarray(states)
    commands = np.asarray(commands)
    df = pd.DataFrame(rows)
    actual = ctx['geometry'].end_effector(states[1:, MECH])
    wanted = ctx['geometry'].end_effector(record['reference'][1:len(states), MECH])
    error = np.linalg.norm(actual[:, :2] - wanted[:, :2], axis=1) * 1000
    summary = dict(controller=name, recording=spec['recording'], samples=len(error), duration_s=len(error) * 0.01, RMSE_mm=float(np.sqrt(np.mean(error ** 2))), MAE_mm=float(error.mean()), final_mm=float(error[-1]), max_mm=float(error.max()), squared_error_mm2=float(np.sum(error ** 2)), deadline_miss_percent=float(df.controller_s.gt(0.1).mean() * 100), feasible_percent=float(df.feasible.mean() * 100), lift_saturation_percent=float(df.lift_saturated.mean() * 100), jib_saturation_percent=float(df.jib_saturated.mean() * 100), solver_success_percent=float(df.success.mean() * 100), **stats(df.controller_s, 'controller'), **stats(df.plant_s, 'plant'))
    for j, key in enumerate(['lift_angle', 'jib_angle', 'lift_velocity', 'jib_velocity']):
        summary[key + '_RMSE'] = float(np.sqrt(np.mean((states[1:, MECH[j]] - record['reference'][1:len(states), MECH[j]]) ** 2)))
    np.savez_compressed(prefix.with_suffix('.npz'), states=states, commands=commands)
    df.to_csv(prefix.with_suffix('.csv'), index=False)
    write_json(prefix.with_suffix('.json'), summary)
    print(f"{name} {spec['recording']}: {n} updates, RMSE {summary['RMSE_mm']:.5f} mm, mean {summary['controller_mean_ms']:.3f} ms", flush=True)
    return summary

def verify():
    z=np.load(ROOT/'configs/fixtures/matched_contexts_136.npz');ctx=context();base=ctx['base']
    assert sum(p.numel() for p in base.model.parameters())==9916
    expected=np.load(ROOT/'configs/fixtures/cost_regression.npz')['cost']
    for i in [0,17,68,135]:
        ref=np.zeros((41,22));ref[:,MECH]=z['reference'][i]
        obj=make_objective(ctx,z['previous_input'][i],z['current_input'][i],z['previous_command'][i],ref[1:],40)
        assert obj(((z['warm'][i]-obj.center)/obj.half).ravel())==expected[i]
    ref=np.zeros((41,22));ref[:,MECH]=z['reference'][0]
    command,*_=online(ctx,z['previous_input'][0],z['current_input'][0],z['previous_command'][0],ref,z['warm'][0],'H0',5)
    np.testing.assert_array_equal(command,z['target'][0])
    geometry=np.load(ROOT/'configs/fixtures/geometry_regression.npz');np.testing.assert_array_equal(ctx['geometry'].end_effector(geometry['mechanical']),geometry['cartesian'])
    plan=plan_from_moves(z['plan'][0],40);prediction=ExtendedPredictor(ctx['predictor']).predict(z['previous_input'][0],z['current_input'][0],base.commands(plan));plant=NeuralPlant(base,z['previous_input'][0],z['current_input'][0],True);actual,_=plant.advance(plan);np.testing.assert_array_equal(prediction,actual)
    policy=Policy(ROOT.parent/'Neural_MPC/checkpoints/wide_s44.pt');actions=np.load(ROOT/'configs/fixtures/policy_regression.npz')['actions']
    for i in range(136):
        ref[:,MECH]=z['reference'][i];np.testing.assert_array_equal(policy(feature(base,z['previous_input'][i],z['current_input'][i],z['previous_command'][i],ref)),actions[i])
    ref[:,MECH]=z['reference'][0];command,_,_=online(ctx,z['previous_input'][0],z['current_input'][0],z['previous_command'][0],ref,z['warm'][0],'H2',3,policy)
    assert np.isfinite(command).all();print('MPC parity PASSED: native prediction, geometry, original objective/action, 136 historical policy outputs, hybrid initialization; selected model9916 parameters',flush=True)
    return True

def check_saved(name,spec,folder):
    key=name+'_'+Path(spec['recording']).stem;expected=read_json(ROOT/'configs/trajectory_regression.json')[key]
    with np.load(Path(folder)/(key+'.npz')) as z:
        for signal in ['states','commands']:
            if list(z[signal].shape)!=expected[signal]['shape'] or hashlib.sha256(z[signal].tobytes()).hexdigest()!=expected[signal]['sha256']:raise AssertionError('Frozen trajectory differs: '+key+' '+signal)
    return True

def results_summary(names,folders,output,dataset,ctx,limit=None):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True);specs=inventory();detail=[];summary=[];traces={};references={}
    for s in specs:references[Path(s['recording']).stem]=dataset.record(s,ctx)['reference']
    for name in names:
        rows=[];diagnostics=[];traces[name]={}
        for s in specs:
            key=name+'_'+Path(s['recording']).stem;folder=Path(folders[name]);rows.append(read_json(folder/(key+'.json')));diagnostics.append(pd.read_csv(folder/(key+'.csv')))
            with np.load(folder/(key+'.npz')) as z:traces[name][Path(s['recording']).stem]=dict(z)
        f=pd.DataFrame(rows);diag=pd.concat(diagnostics,ignore_index=True);detail.append(f)
        summary.append(dict(controller=name,recordings=len(f),mean_tip_RMSE_mm=float(f.RMSE_mm.mean()),pooled_tip_RMSE_mm=float(np.sqrt(f.squared_error_mm2.sum()/f.samples.sum())),worst_recording_RMSE_mm=float(f.RMSE_mm.max()),mean_final_mm=float(f.final_mm.mean()),max_error_mm=float(f.max_mm.max()),lift_saturation_percent=float(diag.lift_saturated.mean()*100),jib_saturation_percent=float(diag.jib_saturated.mean()*100),solver_success_percent=float(diag.success.mean()*100),max_iteration_percent=float(diag.status.eq(9).mean()*100),feasible_percent=float(diag.feasible.mean()*100),deadline_miss_percent=float(diag.controller_s.gt(.1).mean()*100),**stats(diag.controller_s,'controller'),**stats(diag.plant_s,'plant')))
    f=pd.DataFrame(summary);f.to_csv(output/'summary.csv',index=False);pd.concat(detail,ignore_index=True).to_csv(output/'per_recording.csv',index=False)
    for kind in ['trajectory','tip_error','commands']:
        fig,axes=plt.subplots(3,3,figsize=(15,11))
        for ax,s in zip(axes.flat,specs):
            stem=Path(s['recording']).stem;ref=references[stem];n=len(traces[names[0]][stem]['states']);desired=ctx['geometry'].end_effector(ref[:n,MECH])
            if kind=='trajectory':ax.plot(desired[:,0],desired[:,1],'k',label='Reference',lw=1.5)
            for name in names:
                z=traces[name][stem];tip=ctx['geometry'].end_effector(z['states'][:,MECH])
                if kind=='trajectory':ax.plot(tip[:,0],tip[:,1],label=name,lw=.8);ax.set(xlabel='Tip x [m]',ylabel='Tip z [m]')
                elif kind=='tip_error':ax.plot(np.arange(len(tip))*.01,np.linalg.norm(tip[:,:2]-desired[:len(tip),:2],axis=1)*1000,label=name,lw=.8);ax.set(xlabel='Time [s]',ylabel='Tip error [mm]')
                else:
                    u=z['commands'][::10];u=np.vstack((u,u[-1]));line=ax.step(np.arange(len(u))*.1,u[:,0],where='post',label=name+' lift',lw=.7)[0];ax.step(np.arange(len(u))*.1,u[:,1],where='post',label=name+' jib',lw=.7,linestyle='--',color=line.get_color());ax.set(xlabel='Time [s]',ylabel='Command [V]')
            ax.set_title(stem);ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=6);fig.tight_layout();fig.savefig(output/(kind+'.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True);stem=Path(specs[0]['recording']).stem;n=len(traces[names[0]][stem]['states']);ref=references[stem][:n]
    for i,ax in enumerate(axes):
        ax.plot(np.arange(n)*.01,ref[:,MECH[i]],'k',label='Reference')
        for name in names:ax.plot(np.arange(n)*.01,traces[name][stem]['states'][:,MECH[i]],label=name,lw=.8)
        ax.set_ylabel(OUTPUT_COLS[MECH[i]])
    axes[0].legend();axes[-1].set_xlabel('Time [s]');fig.tight_layout();fig.savefig(output/'mechanical_states.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots()
    for row in summary:ax.scatter(row['controller_mean_ms'],row['mean_tip_RMSE_mm']);ax.annotate(row['controller'],(row['controller_mean_ms'],row['mean_tip_RMSE_mm']))
    ax.set(xlabel='Mean online controller time [ms]',ylabel='Mean recording tip RMSE [mm]');fig.tight_layout();fig.savefig(output/'accuracy_computation.png',dpi=150);plt.close(fig)
    print(f.to_string(index=False),flush=True);return f

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Self-contained frozen PATU dynamics, plant and MPC implementation; use run_mpc.py for recordings')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--verify',action='store_true');parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    if args.verify or args.smoke_test:verify()
    else:parser.print_help()
