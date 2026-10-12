"""Native LMDE entry point. Defaults to dry-run; historical results are never used as outputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.Alternatives.Solvers.execution.runner import main

if __name__ == '__main__':
    raise SystemExit(main())
