"""Validate configuration and bind job identity to input and executable content."""
import hashlib
import itertools
import json
from contextlib import nullcontext
from pathlib import Path
from .storage import digest
from .runtime import identity

REPO = Path(__file__).resolve().parents[4]


def load(path, feedback=False):
    config = json.loads(Path(path).read_text())
    for key in ('instances','alphas','seeds','seconds','solvers'):
        if not config.get(key): raise ValueError('Empty axis: '+key)
    if any(type(x) not in (int,float) or not 0 < x < 1 for x in config['alphas']):
        raise ValueError('Alpha must lie strictly between 0 and 1')
    for key in ('seconds','seeds'):
        if any(type(x) != int or x < (1 if key=='seconds' else 0) for x in config[key]):
            raise ValueError('Invalid '+key)
    if set(config['solvers']) - {'team1','team2','team3'}: raise ValueError('Unknown solver')
    for key in ('alphas','seeds','seconds','solvers'):
        if len(set(config[key])) != len(config[key]): raise ValueError('Duplicate '+key)
    threads=config.get('omp_num_threads',1)
    if type(threads)!=int or threads<1:raise ValueError('omp_num_threads must be a positive integer')
    reserve=config.get('memory_reserve_mib',3072)
    if type(reserve)!=int or reserve<1:raise ValueError('memory_reserve_mib must be a positive integer')
    checker = REPO/'RTE_ChallengeROADEF2020_checker.py'
    identities = {'checker':digest(checker),'instances':{},'solvers':{}}
    for name, value in config['instances'].items():
        if not name.isalnum(): raise ValueError('Unsafe instance name')
        p = (REPO/value).resolve()
        from .progress import stage
        with stage(f'Checking original input {name} ({p.stat().st_size/1024**2:.0f} MiB)') if feedback else nullcontext():
            identities['instances'][name] = digest(p)
    for solver in config['solvers']:
        identities['solvers'][solver] = digest(REPO/'data/Alternatives/Solvers'/solver)
    runtimes={solver:identity(solver,REPO) for solver in config['solvers'] if identity(solver,REPO) is not None}
    if runtimes:identities['runtimes']=runtimes
    bound = dict(config=config, identities=identities, schema=1,
                 input_preparation='alpha_only_preserve_member_order_v2')
    bound['fingerprint'] = hashlib.sha256(json.dumps(bound,sort_keys=True).encode()).hexdigest()
    jobs = []
    for name,alpha,seed,seconds,solver in itertools.product(config['instances'],config['alphas'],config['seeds'],config['seconds'],config['solvers']):
        params = dict(instance=name,alpha=alpha,seed=seed,seconds=seconds,solver=solver)
        key = hashlib.sha256(json.dumps(dict(params=params,fingerprint=bound['fingerprint']),sort_keys=True).encode()).hexdigest()
        jobs.append(dict(params,id=key))
    return bound, jobs


def command(binary, job, instance, solution):
    problem = instance.with_suffix('') if job['solver']=='team2' else instance
    args = [str(binary),'-t',str(job['seconds']),'-p',str(problem),'-o',str(solution),'-s',str(job['seed'])]
    if job['solver'] != 'team3': args += ['-name',job['solver'].capitalize()]
    return args
