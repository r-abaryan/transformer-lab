"""Read-only structural/scalar supplement, not a checkpoint or training audit."""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('cross_only', 'both')
STAGES = (6500, 7000, 7500, 8000)
GROUPS = ('all', 'distinct', 'repeated')
KEYS = ('arm', 'step', 'pool', 'group')
METRICS = ('sequences', 'targets', 'exact_accuracy', 'generated_content_exact',
           'token_accuracy', 'content_accuracy', 'loss', 'early_eos', 'on_time_eos',
           'late_eos', 'missing_eos', 'first_accuracy', 'outranked_fraction', 'near_tie_fraction')
TOLERANCE = 1e-10
# Study-specific frozen cohort counts, not a general copying benchmark schema.
COUNTS = {
    'train_3': (32, 32), 'train_4': (128, 128), 'train_5': (128, 128), 'train_6': (128, 128),
    'validation_3': (8, 8), 'validation_4': (32, 32), 'validation_5': (32, 32), 'validation_6': (32, 32),
    'test_3': (8, 8), 'test_4': (64, 64), 'test_5': (64, 64), 'test_6': (64, 64),
    'test_8': (64, 64), 'test_10': (0, 128),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite or nonnumeric metric.')
    return value


def close(a, b, context):
    require(abs(number(a) - number(b)) <= TOLERANCE, 'Numeric mismatch: ' + context)


def exposure_benefit(repair, gain):
    """Original decision expression, without rounding or a tolerance."""
    return repair and gain >= .10


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode_json(raw):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid)


def validate_metrics(metrics):
    require(isinstance(metrics, dict), 'Metrics must be an object.')
    require(set(metrics) == set(COUNTS), 'Missing or unexpected report pools.')
    for pool, sizes in COUNTS.items():
        groups = metrics[pool]
        require(isinstance(groups, dict) and set(groups) == set(GROUPS), 'Missing or unexpected group: ' + pool)
        length = int(pool.rsplit('_', 1)[1])
        for group, count in zip(GROUPS, (sum(sizes), *sizes)):
            values = groups[group]
            if count == 0:
                require(values is None, 'Absent group must be explicit null: ' + pool + '/' + group)
                continue
            require(isinstance(values, dict) and set(values) == set(METRICS), 'Wrong metric fields: ' + pool + '/' + group)
            for key in METRICS:
                value = number(values[key])
                if key in ('sequences', 'targets'):
                    expected = count if key == 'sequences' else count * (length + 1)
                    require(type(value) is int and value == expected, 'Wrong count: ' + pool + '/' + group + '/' + key)
                elif key == 'loss':
                    require(value >= 0, 'Negative loss.')
                else:
                    require(0 <= value <= 1, 'Rate outside [0, 1]: ' + key)
            close(sum(values[k] for k in ('early_eos', 'on_time_eos', 'late_eos', 'missing_eos')), 1, pool + '/EOS partition')
        # There are no distinct length-10 sequences in this fixed cohort.
        if pool == 'test_10':
            for key in METRICS:
                close(groups['all'][key], groups['repeated'][key], 'length10 all/repeated ' + key)


def validate(plan, report, csv_text):
    require(isinstance(plan, dict) and isinstance(report, dict), 'Plan/report must be objects.')
    require(plan.get('start') == 6000 and plan.get('stages') == list(STAGES)
            and plan.get('arms') == list(ARMS), 'Plan differs from the frozen study contract.')
    require(report.get('status') == 'complete', 'Report is incomplete.')
    validate_metrics(report['initial']['metrics'])
    expected_runs = {(arm, step) for arm in ARMS for step in STAGES}
    runs, expected = {}, {}
    require(isinstance(report['runs'], list), 'Runs must be a list.')
    for run in report['runs']:
        require(isinstance(run, dict), 'Run must be an object.')
        require(type(run['step']) is int, 'Noninteger run step.')
        key = (run['arm'], run['step'])
        require(key in expected_runs and key not in runs, 'Unexpected or duplicate run: ' + str(key))
        metrics = run['evaluation']['metrics']
        validate_metrics(metrics)
        runs[key] = metrics
        for pool, groups in metrics.items():
            for group, values in groups.items():
                if values is not None:
                    expected[(*key, pool, group)] = values
    require(set(runs) == expected_runs, 'Missing expected runs.')
    reader = csv.DictReader(io.StringIO(csv_text, newline=''), strict=True)
    fields = reader.fieldnames or []
    require(len(fields) == len(set(fields)) and set(fields) == set(KEYS + METRICS), 'Missing, duplicate or unexpected CSV columns.')
    observed = set()
    for row in reader:
        require(set(row) == set(fields) and all(v is not None for v in row.values()), 'Ragged CSV row.')
        require(re.fullmatch(r'[0-9]+', row['step']) is not None, 'Invalid CSV step.')
        step = int(row['step'])
        require(row['step'] == str(step), 'Noncanonical CSV step.')
        key = (row['arm'], step, row['pool'], row['group'])
        require(key not in observed, 'Duplicate CSV record: ' + str(key))
        require(key in expected, 'Unexpected CSV record: ' + str(key))
        observed.add(key)
        for field in METRICS:
            text = row[field]
            if field in ('sequences', 'targets'):
                require(re.fullmatch(r'[0-9]+', text) is not None, 'Invalid CSV count: ' + field)
                value = int(text)
                require(value == expected[key][field], 'CSV count mismatch: ' + str(key))
            else:
                value = number(float(text))
                require(value >= 0 and (field == 'loss' or value <= 1), 'CSV metric out of range.')
                close(value, expected[key][field], str(key) + '/' + field)
    require(observed == set(expected), 'Missing CSV records: ' + str(sorted(set(expected) - observed)))
    final = {arm: runs[(arm, STAGES[-1])] for arm in ARMS}
    own_gates = {}
    for arm, metrics in final.items():
        short = all(metrics[p][g]['exact_accuracy'] >= .99 for p in COUNTS
                    if int(p.rsplit('_', 1)[1]) <= 6 for g in ('distinct', 'repeated'))
        long = all(metrics[p][g][k] >= .95 for p, g in
                   (('test_8', 'distinct'), ('test_8', 'repeated'), ('test_10', 'repeated'))
                   for k in ('exact_accuracy', 'on_time_eos'))
        own_gates[arm] = dict(short_preserved=short, long_recovered=long, repair=short and long)
    require(report['gates'] == own_gates
            and all(type(v) is bool for gs in report['gates'].values() for v in gs.values()), 'Stored repair gates differ.')
    gains = {g: final['both']['test_10'][g]['exact_accuracy'] - final['cross_only']['test_10'][g]['exact_accuracy']
             for g in ('all', 'repeated')}
    close(gains['all'], gains['repeated'], 'producer/auditor gain reconciliation')
    for group, gain in gains.items():
        close(report['length10_gain'], gain, 'reported gain versus ' + group)
    benefit = exposure_benefit(own_gates['both']['repair'], gains['repeated'])
    require(type(report['exposure_specific_benefit']) is bool
            and report['exposure_specific_benefit'] == benefit, 'Stored benefit gate differs.')
    return dict(passed=True, scope='Report/CSV structure and scalar consistency only; no checkpoint audit.',
                runs=len(runs), csv_rows=len(observed), metrics_per_row=len(METRICS),
                length10_gain=gains, gain_basis='test_10 distinct is null; all and repeated have identical cohort counts/metrics.',
                gates=own_gates, exposure_specific_benefit=benefit, training_updates=0)


def validate_directory(directory):
    raw = {name: (directory / name).read_bytes() for name in ('plan.json', 'report.json', 'metrics.csv')}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    plan, report = decode_json(raw['plan.json']), decode_json(raw['report.json'])
    require(isinstance(plan, dict) and isinstance(report, dict), 'Plan/report must be objects.')
    require(report['plan_sha256'] == hashes['plan.json'], 'Plan hash mismatch.')
    require(report['metrics_sha256'] == hashes['metrics.csv'], 'Metrics hash mismatch.')
    result = validate(plan, report, raw['metrics.csv'].decode('utf-8'))
    result['input_sha256'] = hashes
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'results/copy_exposure_continuation')
    args = parser.parse_args(argv)
    try:
        result = validate_directory(args.directory)
    except (ValueError, KeyError, TypeError, OSError, OverflowError, csv.Error) as error:
        print(json.dumps(dict(passed=False, error=type(error).__name__, message=str(error))), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
