"""Rebuild a separate incident report from the authoritative, immutable ledger."""
import json
from pathlib import Path
from .storage import atomic_json


def update(root, db):
    rows=db.execute('SELECT id,job,state,data FROM attempts ORDER BY rowid').fetchall()
    completed={job:attempt for attempt,job,state,_ in rows if state=='completed'}
    incidents=[];pending=set()
    for attempt,job,state,value in rows:
        data=json.loads(value)
        if state=='running' or (state=='completed' and not data.get('warnings')):continue
        resolved=completed.get(job) if state!='completed' else None
        if state!='completed' and not resolved:pending.add(job)
        validation=data.get('solution_validation') or {}
        candidate=state!='completed' and validation.get('valid',False)
        incidents.append({'attempt':attempt,'job':data['job'],'state':state,
                          'level':'error' if state=='failed' else ('info' if state=='interrupted' else 'warning'),
                          'finished':data.get('finished'),'returncode':data.get('returncode'),
                          'reason':data.get('reason'),'warnings':data.get('warnings',[]),
                          'solution_validation':data.get('solution_validation'),
                          'candidate_available':candidate,
                          'candidate_metrics':validation.get('metrics') if candidate else None,
                          'resolved_by_completed_attempt':resolved,
                          'logs':f'attempts/{attempt}/',
                          'final_manifest':f'attempts/{attempt}/final.json'})
    report={'schema':1,'source':'ledger.sqlite; derived report, not completion authority',
            'completed_jobs':len(completed),'incident_attempts':len(incidents),
            'pending_incident_jobs':len(pending),'incidents':incidents}
    path=Path(root)/'notifications.json'
    try:
        atomic_json(path,report)
    except OSError as error:
        print(f'Warning: could not update incident report {path}: {error}; incidents remain recorded in the ledger',flush=True)
        return False
    return True
