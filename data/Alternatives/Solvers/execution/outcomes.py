"""Keep process termination, solution validity, and completion independent."""
from pathlib import Path
import re


def budget_exit_traceback(stderr):
    """Recognize only the packaged exit() bug and its observed close-file cleanup."""
    parts=stderr.split('Traceback (most recent call last):')
    if len(parts) not in (2,3) or parts[0].strip():return False
    primary=parts[1].split("NameError: name 'exit' is not defined",1)
    if len(primary)!=2 or 'in whatsleft' not in primary[0]:return False
    frames=r'(?:\s*File "solve\.py", line \d+, in [^\n]+\n(?:    [^\n]+\n)?)+\s*'
    if not re.fullmatch(frames,primary[0]):return False
    if not re.fullmatch(r'\s*(?:\[\d+\] Failed to execute script solve\s*)?',primary[1]):return False
    if len(parts)==3:
        cleanup=r'\s*File "<string>", line 1, in <module>\s*ValueError: I/O operation on closed file\.\s*'
        if not re.fullmatch(cleanup,parts[2]):return False
    return True


def classify(job, returncode, elapsed, termination, validation, directory):
    if termination in ('interrupted', 'timed_out', 'resource_limited'):
        return termination, []
    if not validation.get('valid'):
        return 'failed', []
    approximation = Path(directory) / 'stdout.log'
    approximation_warnings = []
    if job['solver']=='team1' and approximation.is_file() and 'some risks have been ignored' in approximation.read_text(errors='replace'):
        approximation_warnings=[{'code':'team1_reduced_scenarios',
            'message':'Solver reports ignoring some risks internally; exported objectives use the full official instance.',
            'evidence':['stdout.log','checker.log','validation.json']}]
    if returncode == 0:
        return 'completed', approximation_warnings
    directory = Path(directory)
    def read(name):
        path = directory / name
        return path.read_text(errors='replace') if path.is_file() else ''
    stderr, stdout, main_log = read('stderr.log'), read('stdout.log'), read('solution.log')
    known_budget_exit = (
        job['solver'] == 'team3' and returncode == 1 and elapsed >= job['seconds']
        and budget_exit_traceback(stderr)
        and ('Step 3 (solving main problem)' in stdout or
             ('Updating model took' in stdout and re.search(r'Solution count [1-9]\d*:',main_log)))
        and 'Time limit reached' in main_log
    )
    if known_budget_exit:
        return 'completed', [{'code': 'team3_budget_exit_nameerror',
            'message': 'Valid final schedule after budget exhaustion; packaged solver exit() raises NameError.',
            'evidence': ['stderr.log', 'stdout.log', 'solution.log', 'validation.json'],
            'returncode': returncode}]
    return 'failed', []
