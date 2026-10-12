"""Fetch pinned official X instances without replacing existing data."""
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from .config import REPO, load
from .storage import atomic_json, digest

COMMIT='052def4d47271f913f89a68169083059732360f3'
SOURCE='https://roadef.org/challenge/2020/en/instances.php'
ARCHIVES={
    'X_compact_1_1.7z':('9d16aae54446127d46e873b536880249c1c6ae4c90cabd40c789c373afa1d570',['X_03.json']),
    'X_compact_1_2.7z':('05bf7c7b9da03fcb968a9a198d49c0cce1ff1b96f65fcf672c0ad4a105f2063f',['X_12.json','X_13.json'])}
INSTANCES={
    'X_03.json':'968c1e732c1c0e70cae04da78441cbfcf99bcf7c4b7b783a572ddc7bd0700c36',
    'X_12.json':'db05807ca1088c53d3e9d0688ccacea7d32aa0d6ccc3f3ec86dde3b505ee12b8',
    'X_13.json':'e731b356660aba83815e1b60dfca4b868aae22fd3733dbc8ab770c2bc46bb0a9'}

# These exact releases were checked for alpha and native invocation compatibility.
VERIFIED_SOLVERS={
    'team1':'12f619a709b805e859d7d47a4a56e51e1633ecf243abb5360126c67e2806b3d4',
    'team2':'1b82a9c88b08dd4e061043a91794ac1644cfc960efe9d8e96ef92983d4d5b731',
    'team3':'4ea99fc98aae3090a74cb1e47b2222832283c208428d61a826664919512fdf2a'}


def download():
    destination=REPO/'data/Alternatives/downloaded';destination.mkdir(parents=True,exist_ok=True)
    for name,expected in INSTANCES.items():
        path=destination/name
        if path.exists() and digest(path)!=expected: raise ValueError('Existing instance differs; refusing to overwrite '+name)
    for name,(expected,names) in ARCHIVES.items():
        missing=[x for x in names if not (destination/x).exists()]
        if not missing: continue
        if not shutil.which('7z'): raise ValueError('7z is required for targeted extraction')
        with tempfile.TemporaryDirectory(prefix='.download-',dir=destination) as temp:
            temp=Path(temp);archive=temp/name
            url=f'https://raw.githubusercontent.com/rte-france/challenge-roadef-2020/{COMMIT}/{name}'
            print('Downloading pinned official archive: '+name,flush=True)
            with urllib.request.urlopen(url,timeout=60) as response,archive.open('xb') as output:
                shutil.copyfileobj(response,output);output.flush();os.fsync(output.fileno())
            if digest(archive)!=expected: raise ValueError('Archive checksum mismatch')
            subprocess.run(['7z','x','-aos',str(archive),*missing,'-o'+str(temp/'extract')],check=True,stdout=subprocess.DEVNULL)
            for instance in missing:
                source=temp/'extract'/instance
                if digest(source)!=INSTANCES[instance]: raise ValueError('Extracted instance checksum mismatch')
                # Hard-link publication is atomic and fails if another downloader published first.
                source.chmod(0o400);os.link(source,destination/instance)
    manifest=destination/'download-manifest.json'
    if not manifest.exists(): atomic_json(manifest,dict(commit=COMMIT,official_source=SOURCE,instances=INSTANCES,archives=ARCHIVES))
    print('Verified X03, X12, X13; existing files preserved.')


def prepare():
    """Download inputs and check dependencies without executing a solver."""
    from .runner import preflight
    import importlib.util
    download()
    if importlib.util.find_spec('numpy') is None:
        raise ValueError('NumPy is required for official calendar validation')
    bound, _ = load(REPO/'data/Alternatives/Solvers/config-target.json', feedback=True)
    readiness=preflight(bound)
    for solver,result in readiness.items():
        print(f'{solver}: libraries '+('ready' if result['ready'] else 'missing'),flush=True)
    if not all(result['ready'] for result in readiness.values()):
        raise ValueError('Missing solver libraries; configure local Gurobi 9.1 before running')
    if bound['identities']['solvers'] != VERIFIED_SOLVERS:
        raise ValueError('Solver release differs from the verified releases')
    if bound['identities']['checker'] != '6f6799e020665682d19ec97bfcbabbdea008cae9f1b9d41e728889d53ec6125f':
        raise ValueError('Official checker differs from the verified version')
    path=REPO/'data/Alternatives/Solvers/config/compatibility.json'
    evidence=dict(input_preparation=bound['input_preparation'],
                  checker_sha256=bound['identities']['checker'],
                  omp_num_threads=bound['config'].get('omp_num_threads',1),
                  scope='Verified solver releases; local shared-library preflight only. No solver or license test performed.')
    for solver,sha in VERIFIED_SOLVERS.items():
        evidence[solver]=dict(sha256=sha,alpha_verified=True,
                             runtime=bound['identities'].get('runtimes',{}).get(solver))
    if path.exists():
        existing=json.loads(path.read_text())
        for key in ('input_preparation','checker_sha256','omp_num_threads'):
            if existing.get(key)!=evidence[key]:raise ValueError('Existing local compatibility settings differ: '+key)
        for solver in VERIFIED_SOLVERS:
            if any(existing.get(solver,{}).get(key)!=evidence[solver][key]
                   for key in ('sha256','alpha_verified','runtime')):
                raise ValueError('Existing local compatibility settings differ: '+solver)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(path,evidence)
    print('Preparation complete. Solver execution and license availability are not tested by this command.',flush=True)


if __name__=='__main__': prepare()
