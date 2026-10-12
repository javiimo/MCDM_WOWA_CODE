"""Console feedback for preparation operations without changing stored artifacts."""
from contextlib import contextmanager
import threading
import time


@contextmanager
def stage(label):
    started=time.monotonic();stop=threading.Event()
    print(f'{label} | starting',flush=True)
    def heartbeat():
        while not stop.wait(5):
            print(f'{label} | working, elapsed={time.monotonic()-started:.0f}s',flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    status='done'
    try:
        yield
    except BaseException:
        status='stopped';raise
    finally:
        stop.set();thread.join()
        print(f'{label} | {status}, elapsed={time.monotonic()-started:.1f}s',flush=True)
