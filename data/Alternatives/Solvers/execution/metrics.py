"""Strict schedule validation and reproducible official objective metrics."""
import contextlib
import importlib.util
import json
import math
import re
from pathlib import Path


def parse_schedule(path, instance):
    starts = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        fields = line.split()
        if len(fields) != 2 or not re.fullmatch(r'[0-9]+', fields[1]):
            raise ValueError('Malformed schedule line')
        name, value = fields[0], int(fields[1])
        if name not in instance['Interventions'] or name in starts:
            raise ValueError('Unknown or duplicate intervention')
        if not 1 <= value <= instance['T']:
            raise ValueError('Start outside horizon')
        starts[name] = value
    if set(starts) != set(instance['Interventions']):
        raise ValueError('Missing interventions')
    return starts


def evaluate(instance_path, solution_path, checker_path, log_path, details_path=None):
    spec = importlib.util.spec_from_file_location('execution_checker', checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    instance = json.loads(Path(instance_path).read_text())
    starts = parse_schedule(solution_path, instance)
    with Path(log_path).open('x') as log, contextlib.redirect_stdout(log):
        for name, start in starts.items():
            instance['Interventions'][name][checker.START_STR] = start
        checker.check_all_constraints(instance)
        log.flush()
        if 'ERROR:' in Path(log_path).read_text():
            raise ValueError('Official checker rejected schedule')
        risk=checker.compute_risk_distribution(instance['Interventions'],instance['T'],instance['Scenarios_number'])
        mean=checker.compute_mean_risk(risk,instance['T'],instance['Scenarios_number'])
        quantile=checker.compute_quantile([list(row) for row in risk],instance['T'],instance['Scenarios_number'],instance['Quantile'])
        resources=checker.compute_resources(instance)
    obj1 = sum(float(x) for x in mean) / len(mean)
    obj2 = sum(max(float(q)-float(m), 0) for m, q in zip(mean, quantile)) / len(mean)
    alpha = float(instance['Alpha'])
    result = {'mean_risk': obj1, 'expected_excess': obj2,
              'objective': alpha*obj1 + (1-alpha)*obj2, 'alpha': alpha}
    if not all(math.isfinite(x) for x in result.values()):
        raise ValueError('Nonfinite objective')
    with Path(log_path).open('a') as log:
        log.write(f"\nObjective 1 (mean risk): {obj1!r}\nObjective 2 (expected excess): {obj2!r}\n")
        log.write(f"Alpha: {alpha!r}\nWeighted mean: {alpha*obj1!r}\nWeighted excess: {(1-alpha)*obj2!r}\nTotal objective: {result['objective']!r}\n")
    if details_path is not None:
        from .diagnostics import retain
        retain(details_path,instance,starts,risk,mean,quantile,resources,result)
    return result
