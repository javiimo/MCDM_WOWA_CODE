"""Extract labelled log observations; raw files remain authoritative."""
import math
import re

NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def public_text(text):
    """Redact local identity metadata in published text; raw evidence stays unchanged."""
    lines=[]
    for line in text.splitlines(keepends=True):
        if re.search(r'license|grbgetkey|hostid|host.id|machine.id|boot.id|\bHOSTNAME\b',line,re.I):
            lines.append('[Local identity/license metadata omitted]\n')
            continue
        # WSL shell errors can contain unquoted paths with embedded spaces.
        line=re.sub(r'/mnt/[a-z]/[^\r\n]*?(?=:\s*line\s+\d|["\x27`\r\n]|$)',
                    '[LOCAL_PATH]',line)
        line=re.sub(r'(["\x27`])(?:/home/|/Users/|[A-Za-z]:[\\/](?:Users|Documents)[\\/])[^\r\n]*?\1',
                    lambda match:match[1]+'[LOCAL_PATH]'+match[1],line)
        line=re.sub(r'(?:/home/|/Users/)[^\s"\x27`<>]+','[LOCAL_PATH]',line)
        line=re.sub(r'[A-Za-z]:[\\/](?:Users|Documents)[\\/][^\s"\x27`<>]+','[LOCAL_PATH]',line)
        line=re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}','[EMAIL]',line)
        lines.append(line)
    return ''.join(lines)


def observations(directory):
    records = []
    patterns = [
        ('mip_final', re.compile(r'Best objective ('+NUMBER+r'), best bound ('+NUMBER+r'), gap ('+NUMBER+r')%'), ('objective', 'bound', 'gap_percent')),
        ('mip_progress', re.compile(r'Explored (\d+) nodes \((\d+) simplex iterations\) in ('+NUMBER+r') seconds'), ('nodes', 'simplex_iterations', 'seconds')),
        ('solution_count', re.compile(r'Solution count (\d+):'), ('count',)),
        ('team2_iteration', re.compile(r'Tiempo ('+NUMBER+r') Iter (\d+)'), ('seconds', 'iteration')),
        ('team2_thread', re.compile(r'Hilo\s+(\d+)\s+GraspF\s+(\d+)\s+Tiempo\s+('+NUMBER+r')'), ('thread', 'grasp_type', 'seconds')),
        ('team2_best', re.compile(r'Mejor\s+(\d+)\s+('+NUMBER+r')'), ('thread', 'reported_objective')),
    ]
    for path in sorted(directory.rglob('*.log')):
        if path.name == 'checker.log': continue
        try:
            lines=path.read_text(errors='replace').splitlines()
        except OSError as error:
            records.append({'file':str(path.relative_to(directory)),'kind':'unreadable_log','reason':str(error)})
            continue
        for line_number, line in enumerate(lines, 1):
            source = {'file': str(path.relative_to(directory)), 'line': line_number}
            if line.strip() in ('Time limit reached', 'Optimal solution found (tolerance 1.00e-04)'):
                records.append(dict(source, kind='termination_message', message=line.strip()))
            for kind, pattern, keys in patterns:
                for match in pattern.finditer(line):
                    values = {key: float(value) for key, value in zip(keys, match.groups())}
                    if all(math.isfinite(value) for value in values.values()):
                        records.append(dict(source, kind=kind, values=values))
    return {'schema': 1, 'observations': records,
            'note': 'Reported log values may describe intermediate subproblems. Official checker metrics are authoritative; time limits and gaps are not infeasibility proofs.'}
