"""Local per-solver runtimes, without changing other solvers or the shell."""
import json
import os
from pathlib import Path
from .storage import digest


def settings(solver, repo):
    path=Path(repo)/'data/Alternatives/Solvers/config/runtime.json'
    if not path.exists():return {}
    return json.loads(path.read_text()).get(solver,{})


def environment(solver, repo, omp_num_threads=1):
    env=dict(os.environ,OMP_NUM_THREADS=str(omp_num_threads))
    local=settings(solver,repo)
    if local:
        home=Path(local['gurobi_home']).expanduser().resolve()
        library=home/'lib/libgurobi91.so'
        license_path=Path(local['license_file']).expanduser().resolve()
        if not library.is_file() or not license_path.is_file():raise ValueError('Missing configured Gurobi runtime or license file')
        env['GUROBI_HOME']=str(home)
        env['LD_LIBRARY_PATH']=str(home/'lib')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        env['GRB_LICENSE_FILE']=str(license_path)
    return env


def identity(solver, repo):
    local=settings(solver,repo)
    if not local:return None
    home=Path(local['gurobi_home']).expanduser().resolve()
    return {'gurobi_home':str(home),'library_sha256':digest(home/'lib/libgurobi91.so'),'license_path':str(Path(local['license_file']).expanduser().resolve())}
