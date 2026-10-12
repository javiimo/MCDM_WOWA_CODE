"""Export only unequivocal ledger finals, checking immutable raw artifacts."""
import argparse
import csv
import json
import fcntl
import itertools
import os
import shutil
import sqlite3
import statistics
import tempfile
import uuid
from pathlib import Path
from .metrics import evaluate
from .runner import verify
from .storage import atomic_json, digest, execution_manifest
from .config import REPO
from .logs import public_text


def export(root):
    root=Path(root).resolve()
    inventory=json.loads((root/'assets/inventory.json').read_text())
    for name,expected in inventory.items():
        if digest(root/'assets'/name)!=expected: raise ValueError('Execution set asset changed: '+name)
    manifest=execution_manifest(root)
    execution=json.loads(manifest.read_text())
    if digest(root/'assets/checker.py')!=execution['identities']['checker']: raise ValueError('Checker content mismatch')
    db=sqlite3.connect(f'file:{root}/ledger.sqlite?mode=ro',uri=True)
    rows=db.execute('SELECT id,job,state,data FROM attempts ORDER BY job,id').fetchall();db.close()
    dest=root/'exports'/uuid.uuid4().hex;dest.mkdir(parents=True)
    metrics=[];attempts=[]
    for attempt,job,state,value in rows:
        data=json.loads(value)
        if state=='running': continue
        verify(root,attempt,data)
        attempts.append(dict(attempt=attempt,job=job,state=state,returncode=data.get('returncode'),
                             solution_validation=data.get('solution_validation'),warnings=data.get('warnings',[]),reason=data.get('reason')))
        if state!='completed': continue
        instance=root/'assets'/f"{data['job']['instance']}-a{data['job']['alpha']}.json"
        if digest(instance)!=data['input_sha256']: raise ValueError('Input modified')
        values=evaluate(instance,root/'attempts'/attempt/'solution.txt',root/'assets/checker.py',dest/(attempt+'-checker.log'),dest/(attempt+'-diagnostics.json'))
        if values!=data['metrics']: raise ValueError('Recomputed metrics do not match sealed final')
        details=json.loads((dest/(attempt+'-diagnostics.json')).read_text())
        original=root/'attempts'/attempt/'diagnostics.json'
        if original.exists():
            recorded=json.loads(original.read_text())
            shared=('objectives','instance','risk','schedule','resources','bottleneck_slack_by_period')
            if any(recorded.get(k)!=details.get(k) for k in shared): raise ValueError('Detailed diagnostics mismatch')
            if recorded.get('schema')==details.get('schema') and recorded!=details: raise ValueError('Diagnostics schema mismatch')
        metrics.append(dict(data['job'],attempt=attempt,elapsed_seconds=data['elapsed_seconds'],solver_wall_seconds=data.get('solver_wall_seconds'),validation_wall_seconds=data.get('validation_wall_seconds'),peak_sampled_group_rss_bytes=data.get('telemetry',{}).get('peak_sampled_group_rss_bytes'),reaped_cpu_user_seconds=data.get('telemetry',{}).get('reaped_cpu_user_seconds'),reaped_cpu_system_seconds=data.get('telemetry',{}).get('reaped_cpu_system_seconds'),schedule_sha256=details['schedule']['fingerprint_sha256'],weighted_mean_component=values['alpha']*values['mean_risk'],weighted_excess_component=(1-values['alpha'])*values['expected_excess'],returncode=data.get('returncode'),warning_codes=';'.join(w['code'] for w in data.get('warnings',[])),omp_num_threads=data.get('execution_environment',{}).get('OMP_NUM_THREADS'),**values))
    columns=['id','instance','solver','alpha','seed','seconds','attempt','elapsed_seconds','mean_risk','expected_excess','objective','weighted_mean_component','weighted_excess_component','schedule_sha256','solver_wall_seconds','validation_wall_seconds','peak_sampled_group_rss_bytes','reaped_cpu_user_seconds','reaped_cpu_system_seconds','returncode','warning_codes','omp_num_threads']
    with (dest/'results.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,columns);writer.writeheader();writer.writerows(metrics)
    groups={}
    for row in metrics:
        key=tuple(row[x] for x in ('instance','solver','alpha','seconds'));groups.setdefault(key,[]).append(row['objective'])
    summary=[dict(zip(('instance','solver','alpha','seconds'),key),n=len(vals),mean=statistics.mean(vals),best=min(vals),stdev=statistics.stdev(vals) if len(vals)>1 else None) for key,vals in sorted(groups.items())]
    duplicates={}
    for row in metrics:duplicates.setdefault((row['instance'],row['schedule_sha256']),[]).append(row['attempt'])
    atomic_json(dest/'summary.json',dict(groups=summary,attempts=attempts,identical_schedule_groups=[dict(instance=k[0],schedule_sha256=k[1],attempts=v) for k,v in duplicates.items() if len(v)>1]))
    lines=[r'\begin{tabular}{llrrrrrr}',r'Instance & Solver & Alpha & Seed & Seconds & Mean risk & Excess & Objective \\',r'\hline']
    for row in metrics:
        cells=[row['instance'],row['solver'],row['alpha'],row['seed'],row['seconds'],f"{row['mean_risk']:.6f}",f"{row['expected_excess']:.6f}",f"{row['objective']:.6f}"]
        lines.append(' & '.join(str(x) for x in cells)+r' \\')
    lines.append(r'\end{tabular}')
    (dest/'results.tex').write_text('\n'.join(lines)+'\n')
    atomic_json(dest/'provenance.json',{'execution_sha256':digest(manifest),'final_attempts':[r['attempt'] for r in metrics],
               'exporter_engine_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
               'files':{p.name:digest(p) for p in dest.iterdir() if p.is_file()}})
    print(dest)
    return dest


def publish(root, destination):
    """Publish validated non-cancelled calendars and one reconstructed log, without replacing results."""
    root=Path(root).resolve();destination=Path(destination).resolve()
    if root==destination or root in destination.parents or destination in root.parents:
        raise ValueError('Published results must be separate from the execution ledger')
    with (root/'owner.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=sqlite3.connect(f'file:{root}/ledger.sqlite?mode=ro',uri=True)
        try:
            db.execute('PRAGMA query_only=ON')
            attempts=[(a,j,s,json.loads(v)) for a,j,s,v in
                      db.execute('SELECT id,job,state,data FROM attempts ORDER BY rowid')]
        finally:
            db.close()
        if any(s=='running' for a,j,s,d in attempts):
            raise ValueError('Unfinished attempt: resume/recover the execution set before extraction')
        manifest=execution_manifest(root);metadata=json.loads(manifest.read_text())
        cache={}
        def checked_hash(path):
            stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
            if key not in cache:cache[key]=digest(path)
            return cache[key]
        inventory=json.loads((root/'assets/inventory.json').read_text())
        for name,expected in inventory.items():
            if checked_hash(root/'assets'/name)!=expected:raise ValueError('Execution asset changed: '+name)
        if inventory.get('checker.py')!=metadata['identities']['checker']:
            raise ValueError('Checker identity mismatch')
        grouped={}
        for index,(aid,jid,state,data) in enumerate(attempts,1):
            directory=root/'attempts'/aid
            if json.loads((directory/'final.json').read_text())!=data:
                raise ValueError('Ledger and final manifest disagree: '+aid)
            for relative,expected in data['hashes'].items():
                path=directory/relative
                if directory not in path.resolve().parents:
                    raise ValueError('Unsafe sealed artifact path')
                if checked_hash(path)!=expected:raise ValueError('Sealed artifact changed: '+aid+'/'+relative)
            job=data['job'];input_name=f"{job['instance']}-a{job['alpha']}.json"
            if inventory.get(input_name)!=data['input_sha256']:
                raise ValueError('Attempt input identity mismatch: '+aid)
            grouped.setdefault((job['instance'],job['solver'],job['alpha'],job['seed'],job['seconds']),[]).append((aid,state,data))
            if index%25==0 or index==len(attempts):
                print(f'Verifying sealed attempts {index}/{len(attempts)}',flush=True)
        destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.results-pending-',dir=destination.parent) as temp:
            staged=Path(temp)/'Results';staged.mkdir()
            config=metadata['config'];rows=[];calendar_count=0
            columns=['instance','solver','alpha','seed','seconds','state','attempt','attempt_count',
                     'calendar','calendar_sha256','calendar_status','early_resource_stop','mean_risk','expected_excess','objective',
                     'weighted_mean_component','weighted_excess_component','solver_wall_seconds',
                     'validation_wall_seconds','elapsed_seconds','returncode','warnings','reason',
                     'input_sha256','solver_sha256','checker_sha256','execution_manifest_sha256']
            for instance,alpha,seed,seconds,solver in itertools.product(
                    config['instances'],config['alphas'],config['seeds'],config['seconds'],config['solvers']):
                history=grouped.get((instance,solver,alpha,seed,seconds),[])
                finals=[item for item in history if item[1]=='completed']
                if len(finals)>1:raise ValueError('Multiple completed attempts for one configuration')
                valid_candidates=[item for item in history if item[1]!='interrupted'
                                  and item[2].get('solution_validation',{}).get('valid')]
                selected=finals[0] if finals else (valid_candidates[-1] if valid_candidates
                                                   else (history[-1] if history else None))
                row=dict(instance=instance,solver=solver,alpha=alpha,seed=seed,seconds=seconds,
                         state='not_attempted',attempt_count=len(history),
                         solver_sha256=metadata['identities']['solvers'][solver],
                         checker_sha256=metadata['identities']['checker'],
                         input_sha256=inventory[f'{instance}-a{alpha}.json'],
                         execution_manifest_sha256=checked_hash(manifest))
                if selected:
                    aid,state,data=selected
                    row.update(attempt=aid,state=state,returncode=data.get('returncode'),
                               reason=data.get('reason',''),warnings=';'.join(w['code'] for w in data.get('warnings',[])))
                    for field in ('solver_wall_seconds','validation_wall_seconds','elapsed_seconds'):
                        row[field]=data.get(field)
                    validation=data.get('solution_validation',{})
                    if state=='completed' and (not validation.get('valid') or data.get('metrics')!=validation.get('metrics')):
                        raise ValueError('Completed attempt lacks matching official validation: '+aid)
                    if state!='interrupted' and validation.get('valid'):
                        values=validation['metrics']
                        row.update(calendar_status='completed' if state=='completed' else 'valid_with_incident',
                                   early_resource_stop=state=='resource_limited' and data.get('solver_wall_seconds',0)<seconds)
                        row.update({key:values[key] for key in ('mean_risk','expected_excess','objective')})
                        row.update(weighted_mean_component=alpha*values['mean_risk'],
                                   weighted_excess_component=(1-alpha)*values['expected_excess'])
                        relative=Path(instance)/solver.capitalize()/f'alpha{alpha}_seed{seed}_time{seconds}.txt'
                        target=staged/relative;target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(root/'attempts'/aid/'solution.txt',target)
                        row.update(calendar=relative.as_posix(),calendar_sha256=checked_hash(target))
                        calendar_count+=1
                row={key:public_text(value).strip() if isinstance(value,str) else value for key,value in row.items()}
                rows.append(row)
            with (staged/'results.csv').open('x',newline='') as stream:
                writer=csv.DictWriter(stream,columns);writer.writeheader();writer.writerows(rows)
            with (staged/'execution.log').open('xb') as stream:
                stream.write(b'Reconstructed execution log from immutable per-attempt artifacts.\n'
                             b'Includes every attempt, including retries and interruptions; validated calendars are published; cancelled attempts are excluded.\n'
                             b'Timestamps and requested/actual durations are retained; stdout/stderr interleaving is unavailable.\n'
                             b'Sealed checker metrics are exported without rerunning the checker.\n'
                             b'Local paths, email addresses and license/host metadata are redacted. Source SHA256 values refer to the unredacted logs.\n')
                for aid,jid,state,data in sorted(attempts,key=lambda item:(item[3].get('started',''),item[0])):
                    header=dict(attempt=aid,state=state,job=data['job'],started=data.get('started'),
                                finished=data.get('finished'),returncode=data.get('returncode'),
                                solver_wall_seconds=data.get('solver_wall_seconds'),
                                reason=public_text(data.get('reason') or '').strip(),
                                warnings=[{key:public_text(value).strip() if isinstance(value,str) else value
                                           for key,value in warning.items()} for warning in data.get('warnings',[])],valid=data.get('solution_validation',{}).get('valid'))
                    stream.write(('\n'+'='*80+'\n'+json.dumps(header,sort_keys=True)+'\n').encode())
                    for relative in sorted(data['hashes']):
                        if relative.endswith('.log'):
                            stream.write(f'\n--- {relative} | sha256={data["hashes"][relative]} ---\n'.encode())
                            with (root/'attempts'/aid/relative).open('r',errors='replace') as source:
                                for line in source:stream.write(public_text(line).encode())
                            stream.write(b'\n')
            if destination.exists():
                def content(directory):
                    return {p.relative_to(directory).as_posix():digest(p) for p in directory.rglob('*') if p.is_file()}
                if content(destination)!=content(staged):
                    raise ValueError('Results already exist with different content; use --output for a new directory. Nothing overwritten.')
                print('Existing published results are identical; preserved.',flush=True)
            else:
                os.rename(staged,destination)
            print(f'Published {calendar_count} valid calendars; {len(rows)-calendar_count} without a retained calendar: {destination}',flush=True)
    return destination


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('results',nargs='?',type=Path,default=REPO/'data/Alternatives/Executions/ejor-target')
    parser.add_argument('--output',type=Path,default=REPO/'data/Alternatives/Results')
    parser.add_argument('--audit-export',action='store_true',help='Recompute checker metrics in a separate internal export')
    args=parser.parse_args()
    if args.audit_export:export(args.results)
    else:publish(args.results,args.output)


if __name__=='__main__': main()
