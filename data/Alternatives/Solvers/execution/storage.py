"""A transactional ledger is authoritative; JSON manifests are sealed projections."""
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def atomic_json(path, data, compact=False, sort_keys=True):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as file:
            json.dump(data, file, indent=None if compact else 2, separators=(',', ':') if compact else None, sort_keys=sort_keys)
            file.write('\n'); file.flush(); os.fsync(file.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def ledger(root):
    db = sqlite3.connect(Path(root)/'ledger.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS attempts (
      id TEXT PRIMARY KEY, job TEXT NOT NULL, state TEXT NOT NULL,
      data TEXT NOT NULL);
    CREATE UNIQUE INDEX IF NOT EXISTS one_final ON attempts(job)
      WHERE state='completed';
    CREATE TABLE IF NOT EXISTS events (
      seq INTEGER PRIMARY KEY AUTOINCREMENT, attempt TEXT,
      time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      state TEXT NOT NULL, data TEXT NOT NULL);
    CREATE TRIGGER IF NOT EXISTS immutable_events_update BEFORE UPDATE ON events
      BEGIN SELECT RAISE(ABORT, 'immutable event'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_events_delete BEFORE DELETE ON events
      BEGIN SELECT RAISE(ABORT, 'immutable event'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_terminal BEFORE UPDATE ON attempts
      WHEN OLD.state != 'running' BEGIN SELECT RAISE(ABORT, 'sealed attempt'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_attempt_delete BEFORE DELETE ON attempts
      BEGIN SELECT RAISE(ABORT, 'immutable attempt'); END;
    ''')
    return db


def record(db, attempt, job, state, data):
    value = json.dumps(data, sort_keys=True)
    with db:
        if db.execute('SELECT 1 FROM attempts WHERE id=?', (attempt,)).fetchone():
            db.execute('UPDATE attempts SET state=?,data=? WHERE id=?', (state,value,attempt))
        else:
            db.execute('INSERT INTO attempts VALUES (?,?,?,?)',(attempt,job,state,value))
        db.execute('INSERT INTO events(attempt,state,data) VALUES (?,?,?)',(attempt,state,value))


def seal(db, root, attempt, job, state, data):
    directory = Path(root)/'attempts'/attempt
    hashes = {}
    for file in directory.rglob('*'):
        if file.is_file() and file.name != 'final.json':
            with file.open('rb') as stream: os.fsync(stream.fileno())
            hashes[str(file.relative_to(directory))] = digest(file)
            file.chmod(0o400)
    data = dict(data, hashes=hashes, state=state)
    record(db, attempt, job, state, data)
    atomic_json(directory/'final.json', data)
    (directory/'final.json').chmod(0o400)
    for folder in sorted((p for p in directory.rglob('*') if p.is_dir()),reverse=True): folder.chmod(0o500)
    directory.chmod(0o500)
    return data


def execution_manifest(root):
    root=Path(root)
    current=root/'execution.json'
    legacy=root/'campaign.json'
    return legacy if legacy.exists() and not current.exists() else current
