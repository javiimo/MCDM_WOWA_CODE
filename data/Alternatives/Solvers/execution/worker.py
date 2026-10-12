"""Do not execute a solver until its process identity is durably journaled."""
import os
import sys
import time
from pathlib import Path

parent=int(sys.argv[1]);gate=Path(sys.argv[2])
while not gate.exists():
    if os.getppid()!=parent: raise SystemExit(125)
    time.sleep(.02)
if os.getppid()!=parent: raise SystemExit(125)
os.execv(sys.argv[3],sys.argv[3:])
