"""Preserve auxiliary files older solvers wrote beside shared input assets."""
import shutil
import uuid
from .storage import atomic_json, digest


def preserve(root, paths):
    if not paths:return
    destination=root/'legacy-asset-outputs'/uuid.uuid4().hex
    destination.mkdir(parents=True)
    records=[]
    for source in paths:
        expected=digest(source);target=destination/source.name
        shutil.copyfile(source,target)
        if digest(target)!=expected:raise ValueError('Legacy auxiliary copy differs from its source')
        target.chmod(0o400)
        records.append({'source':str(source),'copy':source.name,'sha256':expected})
    atomic_json(destination/'provenance.json',{'files':records,
                'note':'Legacy shared auxiliary outputs; seed/attempt attribution is unknown. Never count as completed results.'})
    (destination/'provenance.json').chmod(0o400);destination.chmod(0o500)
    print(f'Preserved {len(paths)} legacy auxiliary files at {destination}; originals unchanged',flush=True)
