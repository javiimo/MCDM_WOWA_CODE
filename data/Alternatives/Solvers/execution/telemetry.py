"""Best-effort process-group samples; raw sampling limits remain explicit."""
import os
from pathlib import Path


def system_memory():
    wanted=('MemTotal','MemAvailable','SwapTotal','SwapFree')
    values={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name,rest=line.split(':',1)
        if name in wanted:values[name]=int(rest.split()[0])*1024
    values['available_ram_plus_free_swap_bytes']=values['MemAvailable']+values['SwapFree']
    return values


def sample(session):
    rows=[];ticks=os.sysconf('SC_CLK_TCK');page=os.sysconf('SC_PAGE_SIZE')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])!=session or int(fields[3])!=session:continue
            status=(entry/'status').read_text().splitlines()
            swap=next((int(line.split()[1])*1024 for line in status if line.startswith('VmSwap:')),0)
            rows.append({'pid':int(entry.name),'state':fields[0],'user_seconds':int(fields[11])/ticks,'system_seconds':int(fields[12])/ticks,'threads':int(fields[17]),'rss_bytes':int(fields[21])*page,'swap_bytes':swap})
        except (OSError,ValueError):pass
    return {'processes':rows,'group_rss_bytes':sum(p['rss_bytes'] for p in rows),'group_threads':sum(p['threads'] for p in rows),
            'group_swap_bytes':sum(p['swap_bytes'] for p in rows),'system_memory':system_memory()}
