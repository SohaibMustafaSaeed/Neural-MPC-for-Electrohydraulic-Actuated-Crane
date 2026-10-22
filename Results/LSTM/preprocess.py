import os,sys
from pathlib import Path
for _thread in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[_thread]='1'
sys.dont_write_bytecode=True
import sys
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
ROOT=Path(__file__).resolve().parents[2]
RAW_DATA_DIR=Path(os.environ.get('PATU_DATA_ROOT',ROOT.parent))
PREPARED_DIR=Path(os.environ.get('PATU_PREPARED_ROOT',RAW_DATA_DIR/'PATU_prepared_cache'))
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = RAW_DATA_DIR
DEFAULT_OUTPUT = PREPARED_DIR
FORMAT = 'imotion_lstm_normalized_raw_csv_v1'
INPUT_COLS = ['ulift[V]', 'speedref[rad/s]', 'speedpmsm[rad/s]', 'iqref[A]', 'udref[V]', 'uqref[V]', 'id[A]', 'iq[A]', 'psid[Wb]', 'psiq[Wb]', 'pumppressure[Pa]', 'plinepistonlift[Pa]', 'plinerodlift[Pa]', 'ppistonsidelift[Pa]', 'prodsidelift[Pa]', 'thetalift[rad]', 'ujib[V]', 'plinepistonjib[Pa]', 'plinerodjib[Pa]', 'ppistonjib[Pa]', 'prodjib[Pa]', 'thetajib[rad]', 'thetaliftdot[rad/s]', 'thetajibdot[rad/s]']
COMMAND_INDICES = (0, 16)
STATE_INDICES = tuple((i for i in range(24) if i not in COMMAND_INDICES))
OUTPUT_COLS = [INPUT_COLS[i] for i in STATE_INDICES]

def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

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

def merge_moments(count, mean, m2, values):
    n = len(values)
    if not n:
        return (count, mean, m2)
    batch_mean = values.mean(axis=0)
    batch_m2 = np.square(values - batch_mean).sum(axis=0)
    if count == 0:
        return (n, batch_mean, batch_m2)
    delta = batch_mean - mean
    total = count + n
    return (total, mean + delta * n / total, m2 + batch_m2 + np.square(delta) * count * n / total)

def inspect_rows(paths, skip_rows, chunksize):
    retained = {}
    for path in paths:
        count = sum((len(chunk) for chunk in read_chunks(path, skip_rows, chunksize)))
        if count < 51:
            raise ValueError(f'Need at least 51 retained rows after startup trim in {path}')
        retained[path] = count
        print(f'{path.parent.name}: {path.name}: {count:,} retained rows', flush=True)
    return retained

def fit_training_scaler(training_paths, retained_rows, train_fraction, skip_rows, chunksize):
    count = 0
    mean = np.zeros(24, dtype=np.float64)
    m2 = np.zeros(24, dtype=np.float64)
    command_count = 0
    command_mean = np.zeros(2, dtype=np.float64)
    command_m2 = np.zeros(2, dtype=np.float64)
    for path in training_paths:
        rows = retained_rows[path]
        train_rows = max(1, int(math.floor(rows * train_fraction)))
        if train_rows < 51:
            raise ValueError(f'Training portion of {path.name} has fewer than 51 rows')
        commands = []
        position = 0
        for chunk in read_chunks(path, skip_rows, chunksize):
            end = min(position + len(chunk), train_rows)
            if position < train_rows:
                training_chunk = chunk[:end - position]
                count, mean, m2 = merge_moments(count, mean, m2, training_chunk)
            commands.append(chunk[:, COMMAND_INDICES].copy())
            position += len(chunk)
        smoothed_commands = savgol_filter(np.concatenate(commands, axis=0), window_length=51, polyorder=3, axis=0)
        command_train = smoothed_commands[:train_rows]
        command_count, command_mean, command_m2 = merge_moments(command_count, command_mean, command_m2, command_train)
        print(f'Scaler fit: {path.name}: {train_rows:,} training rows', flush=True)
    if count < 2:
        raise ValueError('Insufficient training rows for normalization')
    mean[list(COMMAND_INDICES)] = command_mean
    m2[list(COMMAND_INDICES)] = command_m2
    raw_std = np.sqrt(m2 / (count - 1))
    std = np.where(raw_std < 1e-08, 1.0, raw_std)
    return dict(format=FORMAT, input_columns=INPUT_COLS, output_columns=OUTPUT_COLS, command_columns=[INPUT_COLS[i] for i in COMMAND_INDICES], mean=mean.tolist(), std=std.tolist(), raw_std=raw_std.tolist(), fitted_rows=count, fitted_files=[str(path) for path in training_paths], startup_rows_removed=skip_rows, train_fraction=train_fraction, sampling_interval_seconds=1e-05, normalized_dtype='float32')

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

def prepare_split(name, paths, retained_rows, skip_rows, chunksize, scaler, output_dir, train_fraction=None):
    split_dir = output_dir / name
    entries = []
    for path in paths:
        rows = retained_rows[path]
        data_path = convert_recording(path, split_dir, rows, skip_rows, chunksize, scaler)
        entry = dict(file=str(data_path.relative_to(output_dir)), source_csv=str(path.resolve()), rows=rows)
        if train_fraction is not None:
            train_rows = max(1, int(math.floor(rows * train_fraction)))
            entry.update(train_rows=train_rows, validation_start=train_rows, validation_rows=rows - train_rows)
        entries.append(entry)
    write_json(output_dir / f'{name}_manifest.json', entries)
    return entries

def main():
    parser = argparse.ArgumentParser(description='iMotion residual dynamics')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--prepared-root',type=Path)
    parser.add_argument('--frozen-scaler',type=Path)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--train-dir', type=Path, default=WORKSPACE / 'Train')
    parser.add_argument('--test-dir', type=Path, default=WORKSPACE / 'Test')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--skip-initial-rows', type=int, default=1800000)
    parser.add_argument('--train-fraction', type=float, default=0.9, help='Per-recording prefix used for training and scaler fitting')
    parser.add_argument('--chunksize', type=int, default=100000)
    parser.add_argument('--training-only', action='store_true', help='Prepare only Train; do not inspect or open Test')
    args = parser.parse_args()
    if args.data_root:args.train_dir,args.test_dir=args.data_root/'Train',args.data_root/'Test'
    if args.prepared_root:args.output_dir=args.prepared_root
    if args.verify or args.smoke_test:
        return verify_prepared(args.output_dir)
    if args.skip_initial_rows < 0 or args.chunksize < 1:
        parser.error('skip-initial-rows must be nonnegative and chunksize positive')
    if not 0.5 <= args.train_fraction < 1:
        parser.error('train-fraction must be in [0.5,1)')
    train_dir = args.train_dir.resolve()
    test_dir = args.test_dir.resolve()
    if train_dir == test_dir:
        parser.error('Training and testing directories must be different')
    train_paths = sorted(train_dir.glob('*.csv'))
    test_paths = [] if args.training_only else sorted(test_dir.glob('*.csv'))
    if not train_paths or (not args.training_only and (not test_paths)):
        raise FileNotFoundError('Expected CSV recordings in both --train-dir and --test-dir')
    if set(train_paths) & set(test_paths):
        raise ValueError('A CSV cannot be in both training and testing splits')
    if {path.name for path in train_paths} & {path.name for path in test_paths}:
        raise ValueError('Train and Test recording filenames must be distinct')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Inspecting {len(train_paths)} training and {len(test_paths)} testing recordings', flush=True)
    retained_rows = inspect_rows(train_paths + test_paths, args.skip_initial_rows, args.chunksize)
    scaler = json.loads(args.frozen_scaler.read_text()) if args.frozen_scaler else fit_training_scaler(train_paths, retained_rows, args.train_fraction, args.skip_initial_rows, args.chunksize)
    write_json(output_dir / 'scaler.json', scaler)
    write_json(output_dir/'feature_schema.json',dict(current=INPUT_COLS,difference=['delta_'+s for s in INPUT_COLS],features=48,definition='concat(current,current-previous); constructed lazily'))
    training = prepare_split('training', train_paths, retained_rows, args.skip_initial_rows, args.chunksize, scaler, output_dir, args.train_fraction)
    testing = [] if args.training_only else prepare_split('testing', test_paths, retained_rows, args.skip_initial_rows, args.chunksize, scaler, output_dir)
    manifest = dict(format=FORMAT, training_records=len(training), testing_records=len(testing), scaler='scaler.json', test_used_for_fitting=False, test_used_for_training=False)
    write_json(output_dir / 'dataset_manifest.json', manifest)
    print(f'Preprocessing complete: {output_dir}', flush=True)

def resolve_array(entry,dataset_dir):
    name=Path(str(entry['file']).replace('\\','/')).name
    for candidate in [dataset_dir/entry['file'],dataset_dir/'training'/name,dataset_dir/name]:
        if candidate.is_file():return candidate
    raise FileNotFoundError(name)
def same_recording(a,b):return Path(str(a).replace('\\','/')).name==Path(str(b).replace('\\','/')).name
def verify_prepared(directory):
    directory=Path(directory);sc=json.loads((directory/'scaler.json').read_text());manifest=json.loads((directory/'training_manifest.json').read_text())
    assert sc['input_columns']==INPUT_COLS and sc['output_columns']==OUTPUT_COLS and sc['sampling_interval_seconds']==1e-5
    for entry in manifest:
        assert Path(entry['source_csv']).parent.name=='Train';x=np.load(resolve_array(entry,directory),mmap_mode='r');assert x.shape==(entry['rows'],24) and x.dtype==np.float32
        assert entry['validation_start']==entry['train_rows']==int(entry['rows']*.9)
    schema=dict(current=INPUT_COLS,difference=['delta_'+s for s in INPUT_COLS],features=48,definition='concat(current,current-previous); constructed lazily, not duplicated as large arrays',train='first90% of each Train recording',validation='last10% of each Train recording',test='Test43..53; never used for fitting')
    print('Prepared training verified:',len(manifest),'recordings;',sum(e['train_rows'] for e in manifest),'training rows; 48 features constructed by model.step')
    return schema
if __name__=='__main__':
    main()
