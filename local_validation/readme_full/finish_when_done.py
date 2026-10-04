"""Finalize only after both existing suite processes finish successfully."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
out=Path(__file__).resolve().parent
while True:
    done=True
    for group in ('main','uncensored'):
        d=json.loads((out/f'{group}_manifest.json').read_text())
        if 'finished' not in d:
            done=False
            try:os.kill(d['pid'],0)
            except ProcessLookupError:raise SystemExit(f'{group} driver stopped before completion')
        elif d.get('failures'):raise SystemExit(f'{group} has failed tasks; inspect logs')
    if done:break
    time.sleep(10)
raise SystemExit(subprocess.call([sys.executable,str(out/'audit_results.py')]))
