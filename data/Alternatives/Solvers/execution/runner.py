"""Sequential native runner with exclusive ownership, recovery, and safe cancellation."""
import argparse
import fcntl
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .config import REPO, command, load
from .metrics import evaluate
from .runtime import environment
from .outcomes import classify
from .logs import observations
from .storage import atomic_json, digest, ledger, record, seal, execution_manifest
from .progress import stage
from .notifications import update as update_notifications


def utc(): return datetime.now(timezone.utc).isoformat()


def preflight(bound):
    results = {}
    for solver in bound['config']['solvers']:
        result = subprocess.run(['ldd',str(REPO/'data/Alternatives/Solvers'/solver)],capture_output=True,text=True,env=environment(solver,REPO))
        results[solver] = {'ready':result.returncode==0 and 'not found' not in result.stdout,
                           'dependencies':result.stdout.strip()}
    return results


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'start':fields[19], 'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except FileNotFoundError: return None


def stop_group(identity):
    if not identity: return
    pid = identity['pid']
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    current=process_identity(pid)
    if boot!=identity['boot'] or (current is not None and current!=identity): return
    def members():
        found=[]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit(): continue
            try:
                fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
                if fields[0]!='Z' and int(fields[2])==pid and int(fields[3])==pid and int(fields[19])>=int(identity['start']):
                    found.append(int(entry.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError): pass
        return found
    if not members(): return
    try: os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError: return
    deadline = time.monotonic()+2
    while members() and time.monotonic()<deadline: time.sleep(.05)
    if members():
        try: os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError: pass
        deadline=time.monotonic()+2
        while members() and time.monotonic()<deadline: time.sleep(.05)
        if members(): raise RuntimeError('Cannot stop orphan process group; artifacts left unsealed')


def recover(db, root):
    for attempt,job,state,value in db.execute('SELECT * FROM attempts').fetchall():
        data = json.loads(value)
        if state == 'running':
            stop_group(data.get('process'))
            data.update(finished=utc(), reason='Recovered unfinished attempt; never counted', metrics=None)
            seal(db,root,attempt,job,'interrupted',data)
        else:
            path = root/'attempts'/attempt/'final.json'
            if not path.exists():
                path.parent.chmod(0o700)
                atomic_json(path,data);path.chmod(0o400);path.parent.chmod(0o500)
            if json.loads(path.read_text())!=data: raise ValueError('Final manifest disagrees with ledger')
            verify(root,attempt,data)


def verify(root, attempt, data):
    for relative, expected in data['hashes'].items():
        if digest(root/'attempts'/attempt/relative) != expected:
            raise ValueError('Sealed artifact changed: '+attempt+'/'+relative)


def prepare(root, bound, readiness=None):
    started=time.monotonic()
    total=len(bound['config']['instances'])*len(bound['config']['alphas']);prepared=0
    print(f'Preparation: {total} input variants; results={root}',flush=True)
    manifest = execution_manifest(root)
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if any(saved.get(key)!=value for key,value in bound.items()):
            raise ValueError('Execution set configuration/content changed; use a new execution directory')
    else:
        atomic_json(manifest,dict(bound,host=platform.platform(),python=platform.python_version(),created=utc(),software={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},dependencies=readiness,environment={'OMP_NUM_THREADS':str(bound['config'].get('omp_num_threads',1)),'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH',''),'license_path_set':bool(os.environ.get('GRB_LICENSE_FILE'))}))
        manifest.chmod(0o400)
    assets = root/'assets'; assets.mkdir(exist_ok=True)
    for solver, expected in bound['identities']['solvers'].items():
        p = assets/solver
        if not p.exists():
            temp=assets/(solver+'.pending');shutil.copyfile(REPO/'data/Alternatives/Solvers'/solver,temp)
            temp.chmod(0o500);os.replace(temp,p)
        with stage(f'Checking executable {solver}'):
            if digest(p)!=expected: raise ValueError('Executable content mismatch')
    checker=assets/'checker.py'
    if not checker.exists(): shutil.copyfile(REPO/'RTE_ChallengeROADEF2020_checker.py',checker);checker.chmod(0o400)
    if digest(checker)!=bound['identities']['checker']: raise ValueError('Checker content mismatch')
    for name,value in bound['config']['instances'].items():
        instance=None
        for alpha in bound['config']['alphas']:
            path = assets/f'{name}-a{alpha}.json'
            if not path.exists():
                if instance is None:
                    with stage(f'Loading JSON {name}'):
                        instance=json.loads((REPO/value).read_text())
                instance['Alpha']=alpha
                # Team1 reads resource min/max by member position, not by name.
                with stage(f'Preparing input {prepared+1}/{total}: {name} alpha={alpha}'):
                    atomic_json(path,instance,compact=True,sort_keys=False);path.chmod(0o400)
            else:
                print(f'Preparation {prepared+1}/{total}: reusing {name} alpha={alpha}; hash verification follows',flush=True)
            prepared+=1
            print(f'Preparation {prepared}/{total} inputs ready | elapsed={time.monotonic()-started:.0f}s',flush=True)
        # Release large instances before parsing the next one; reuse across alphas.
        del instance
    inventory=assets/'inventory.json'
    expected_names=set(bound['identities']['solvers'])|{'checker.py'}|{
        f'{name}-a{alpha}.json' for name in bound['config']['instances'] for alpha in bound['config']['alphas']}
    legacy_names={f'{name}-a{alpha}{seconds}_sol.txt' for name in bound['config']['instances']
                  for alpha in bound['config']['alphas'] for seconds in bound['config']['seconds']}
    extras=[]
    actual={}
    for p in sorted(assets.iterdir()):
        if p.is_file() and p.name!='inventory.json':
            if p.name not in expected_names:
                if p.name not in legacy_names:raise ValueError('Unexpected execution asset: '+p.name)
                extras.append(p);continue
            with stage(f'Verifying prepared asset {p.name}'):
                actual[p.name]=digest(p)
    if inventory.exists():
        if json.loads(inventory.read_text())!=actual: raise ValueError('Execution set assets changed')
    else:
        atomic_json(inventory,actual);inventory.chmod(0o400)
    from .legacy_outputs import preserve
    preserve(root,extras)
    print(f'Preparation complete | elapsed={time.monotonic()-started:.1f}s',flush=True)
    return assets


def wait_for_memory(reserve, cancelled):
    from .telemetry import system_memory
    started=time.monotonic();last=-5;waiting=False
    while not cancelled[0]:
        available=system_memory()['available_ram_plus_free_swap_bytes']
        if available>=reserve:
            if waiting:print('Memory reserve recovered; continuing with the next job',flush=True)
            return True
        waiting=True;elapsed=time.monotonic()-started
        if elapsed-last>=5:
            print(f'Waiting for memory recovery | available RAM + free swap={available/1024**3:.2f} GiB | reserve={reserve/1024**3:.2f} GiB | elapsed={elapsed:.0f}s | Ctrl+C to stop',flush=True)
            last=elapsed
        time.sleep(1)
    return False


def run(root, bound, jobs, pilot=False, limit=None, compatibility_test=False, stop_on_error=False):
    root=Path(root).resolve()
    allowed=(REPO/'data/Alternatives/Executions').resolve()
    if allowed not in root.parents: raise ValueError('Execution set must be a new subdirectory of data/Alternatives/Executions')
    with stage('Checking solver libraries'):
        readiness=preflight(bound)
    if not all(v['ready'] for v in readiness.values()): raise ValueError('Unresolved dependencies; use --preflight')
    selected=jobs[:limit] if limit else jobs
    if compatibility_test:
        if len(selected)!=1 or selected[0]['seconds']>300:
            raise ValueError('Compatibility test requires exactly one job, max 300 seconds')
    elif pilot:
        if sum(j['seconds'] for j in selected)>120 or any(j['seconds']>30 for j in selected):
            raise ValueError('Pilot is limited to 120 nominal seconds, max 30 per job')
    else:
        approval=REPO/'data/Alternatives/Solvers/config/compatibility.json'
        if not approval.exists(): raise ValueError('Compatibility has not been confirmed; only short --pilot permitted')
        evidence=json.loads(approval.read_text())
        blocked=evidence.get('blocked_instances',{})
        for instance in sorted({job['instance'] for job in selected} & blocked.keys()):
            raise ValueError('Instance readiness blocked: '+instance+'; '+blocked[instance]['reason'])
        if (evidence.get('input_preparation')!=bound['input_preparation'] or
            evidence.get('checker_sha256')!=bound['identities']['checker'] or
            evidence.get('omp_num_threads')!=bound['config'].get('omp_num_threads',1)):
            raise ValueError('Compatibility evidence does not match input preparation, checker or thread setting')
        for solver,sha in bound['identities']['solvers'].items():
            if evidence.get(solver,{}).get('sha256')!=sha or not evidence[solver].get('alpha_verified'):
                raise ValueError('Missing alpha/compatibility evidence for '+solver)
            if evidence[solver].get('runtime')!=bound['identities'].get('runtimes',{}).get(solver):
                raise ValueError('Compatibility runtime changed for '+solver)
    if root.exists() and not execution_manifest(root).exists() and any(root.iterdir()):
        raise ValueError('Refusing to use an unrelated nonempty output directory')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assets=prepare(root,bound,readiness);db=ledger(root)
        with stage('Recovering ledger and verifying previous attempts'):
            recover(db,root)
        update_notifications(root,db)
        print(f'Incident report: {root/"notifications.json"}',flush=True)
        completed={x[0] for x in db.execute("SELECT job FROM attempts WHERE state='completed'")}
        print(f"Execution set: {len(completed)}/{len(jobs)} completed; {sum(j['seconds'] for j in jobs if j['id'] not in completed)} nominal seconds remaining",flush=True)
        cancelled=[False];cancel_signal=[None]
        def cancel(signum,frame): cancelled[0]=True;cancel_signal[0]=signal.Signals(signum).name
        previous={s:signal.signal(s,cancel) for s in (signal.SIGINT,signal.SIGTERM)}
        session_start=time.monotonic()
        reserve=bound['config'].get('memory_reserve_mib',3072)*1024**2
        try:
            for job in selected:
                if job['id'] in completed: continue
                if cancelled[0]: break
                if not wait_for_memory(reserve,cancelled):break
                attempt=uuid.uuid4().hex;directory=root/'attempts'/attempt;directory.mkdir(parents=True)
                engine=directory/'engine';engine.mkdir()
                for source in Path(__file__).parent.glob('*.py'):
                    shutil.copyfile(source,engine/source.name)
                source_instance=assets/f"{job['instance']}-a{job['alpha']}.json"
                input_directory=directory/'inputs';input_directory.mkdir()
                instance=input_directory/source_instance.name
                # Native solvers derive auxiliary output names from the input path.
                # A private hard link preserves input bytes without duplicating large JSONs.
                os.link(source_instance,instance)
                solution=directory/'solution.txt'
                args=command(assets/job['solver'],job,instance,solution)
                data=dict(job=job,attempt=attempt,command=args,started=utc(),process=None,
                          input_sha256=digest(instance),metrics=None,execution_environment={'OMP_NUM_THREADS':str(bound['config'].get('omp_num_threads',1))},engine_sha256={p.name:digest(p) for p in engine.glob('*.py')})
                from .telemetry import system_memory
                data['resource_policy']={'reserve_bytes':reserve,'measure':'available RAM plus free swap','initial_system_memory':system_memory()}
                atomic_json(directory/'start.json',data)
                record(db,attempt,job['id'],'running',data)
                process=None;begin=time.monotonic();state='failed'
                try:
                    from .telemetry import sample
                    usage_before=resource.getrusage(resource.RUSAGE_CHILDREN)
                    peak_sampled_rss=0
                    with (directory/'stdout.log').open('xb') as stdout, (directory/'stderr.log').open('xb') as stderr, (directory/'process-samples.jsonl').open('x') as samples:
                        gate=directory/'launch.json'
                        launch=[sys.executable,str(engine/'worker.py'),str(os.getpid()),str(gate)]+args
                        process=subprocess.Popen(launch,cwd=directory,stdout=stdout,stderr=stderr,start_new_session=True,
                                                 env=environment(job['solver'],REPO,bound['config'].get('omp_num_threads',1)))
                        data['process']=process_identity(process.pid)
                        record(db,attempt,job['id'],'running',data)
                        atomic_json(gate,{'authorized_process':data['process']})
                        last=-10;last_sample=-1
                        while process.poll() is None:
                            elapsed=time.monotonic()-begin
                            if elapsed-last_sample>=1:
                                snapshot=sample(process.pid);snapshot['elapsed_seconds']=elapsed
                                peak_sampled_rss=max(peak_sampled_rss,snapshot['group_rss_bytes'])
                                samples.write(json.dumps(snapshot)+'\n');samples.flush();last_sample=elapsed
                                memory=snapshot['system_memory']
                                if memory['available_ram_plus_free_swap_bytes']<reserve:
                                    data['resource_stop']=memory
                                    stop_group(data['process']);process.wait()
                                    state='resource_limited';break
                            if elapsed-last>=5:
                                remaining=sum(j['seconds'] for j in jobs if j['id'] not in completed)
                                print(f"{len(completed)}/{len(jobs)} | {job['instance']} {job['solver']} alpha={job['alpha']} seed={job['seed']} budget={job['seconds']}s | attempt={elapsed:.0f}s session={time.monotonic()-session_start:.0f}s | nominal ETA={max(0,remaining-elapsed):.0f}s",flush=True);last=elapsed
                            if cancelled[0] or elapsed>job['seconds']+30:
                                stop_group(data['process']);process.wait()
                                state='interrupted' if cancelled[0] else 'timed_out';break
                            time.sleep(.1)
                        data['returncode']=process.returncode
                    data['solver_wall_seconds']=time.monotonic()-begin
                    usage_after=resource.getrusage(resource.RUSAGE_CHILDREN)
                    data['telemetry']={'peak_sampled_group_rss_bytes':peak_sampled_rss,'sampling_interval_seconds':1,'reaped_cpu_user_seconds':usage_after.ru_utime-usage_before.ru_utime,'reaped_cpu_system_seconds':usage_after.ru_stime-usage_before.ru_stime,'warning':'Samples can miss transient peaks; reaped CPU does not include unadopted orphan descendants.'}
                    stop_group(data.get('process'))
                    validation={'valid':False,'status':'missing','reason':'Solver did not export a schedule'}
                    validation_begin=time.monotonic()
                    if solution.is_file():
                        try:
                            values=evaluate(instance,solution,assets/'checker.py',directory/'checker.log',directory/'diagnostics.json')
                            validation={'valid':True,'status':'valid','metrics':values}
                        except Exception as error:
                            validation={'valid':False,'status':'invalid','reason':type(error).__name__+': '+str(error)}
                    data['solution_validation']=validation
                    data['validation_wall_seconds']=time.monotonic()-validation_begin
                    termination=state if state in ('interrupted','timed_out','resource_limited') else 'process_exit'
                    state,warnings=classify(job,process.returncode,data['solver_wall_seconds'],termination,validation,directory)
                    data['warnings']=warnings
                    data['process_outcome']={'termination':termination,'returncode':process.returncode,
                                             'requested_seconds':job['seconds'],'watchdog_grace_seconds':30}
                    if process.returncode is not None and process.returncode<0:
                        data['process_outcome']['signal']=signal.Signals(-process.returncode).name
                    if state=='completed': data['metrics']=validation['metrics']
                    elif state=='resource_limited':
                        data['reason']='Available RAM plus free swap fell below the configured memory reserve; job remains pending'
                    elif state in ('interrupted','timed_out'):
                        data['reason']='Cancellation requested by '+str(cancel_signal[0]) if state=='interrupted' else 'External watchdog exceeded budget plus grace'
                    else:
                        data['reason']=f"Process returncode={process.returncode}; solution={validation['status']}"
                except Exception as error:
                    stop_group(data.get('process'))
                    if process and process.poll() is None: process.wait()
                    data['reason']=type(error).__name__+': '+str(error)
                finally:
                    if cancelled[0]:
                        state='interrupted';data['metrics']=None;data['reason']='Cancellation requested by '+str(cancel_signal[0])
                        data['cancellation_signal']=cancel_signal[0]
                        if 'process_outcome' in data: data['process_outcome']['termination']='interrupted'
                    atomic_json(directory/'validation.json',data.get('solution_validation',{'valid':False,'status':'not_checked','reason':data.get('reason')}))
                    atomic_json(directory/'log-observations.json',observations(directory))
                    data.update(finished=utc(),elapsed_seconds=time.monotonic()-begin)
                    seal(db,root,attempt,job['id'],state,data)
                update_notifications(root,db)
                if state=='completed': completed.add(job['id'])
                validity=data.get('solution_validation',{}).get('status','not_checked')
                print(f"{len(completed)}/{len(jobs)} finalized {attempt}: {state}; solution={validity}; returncode={data.get('returncode')}; warnings={len(data.get('warnings',[]))}",flush=True)
                if state=='interrupted':break
                if state!='completed':
                    if stop_on_error:break
                    print(f'Job {job["id"]} remains pending ({state}); continuing with the next job',flush=True)
        finally:
            for sig,handler in previous.items(): signal.signal(sig,handler)
            db.close()
        print(f"Stopped: {len(completed)}/{len(jobs)} completed",flush=True)
        return 130 if cancelled[0] else (0 if all(j['id'] in completed for j in selected) else 1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=REPO/'data/Alternatives/Solvers/config-target.json')
    parser.add_argument('--results',type=Path,default=REPO/'data/Alternatives/Executions/ejor-target')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run',action='store_true');modes.add_argument('--preflight',action='store_true')
    modes.add_argument('--run',action='store_true');modes.add_argument('--pilot',action='store_true')
    modes.add_argument('--compatibility-test',action='store_true')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--stop-on-error',action='store_true',help='Stop after an isolated solver failure instead of continuing with other jobs')
    args=parser.parse_args()
    if args.limit is not None and args.limit<1: parser.error('limit must be positive')
    try:
        executing=args.run or args.pilot or args.compatibility_test
        if executing:print(f'Starting execution set | config={args.config}',flush=True)
        bound,jobs=load(args.config,feedback=executing)
        if args.preflight: print(json.dumps(preflight(bound),indent=2));return 0
        if not (args.run or args.pilot or args.compatibility_test):
            print(json.dumps(dict(fingerprint=bound['fingerprint'],total=len(jobs),nominal_seconds=sum(j['seconds'] for j in jobs),jobs=jobs),indent=2));return 0
        return run(args.results,bound,jobs,args.pilot,args.limit,args.compatibility_test,args.stop_on_error)
    except (ValueError,OSError) as error: parser.exit(2,str(error)+'\n')


if __name__=='__main__': raise SystemExit(main())
