"""Use a working GPU environment, or provision a project-only CUDA runtime."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

PROBE = '''import json
try:
 import torch
 from torch.nn import functional as F
 if not torch.cuda.is_available():
  print(json.dumps({"ready":False,"reason":"CUDA unavailable","runtime":torch.version.cuda}))
 else:
  x=torch.randn(8,8,device="cuda");q=torch.randn(1,2,8,16,device="cuda")
  assert torch.isfinite(x@x).all() and torch.isfinite(F.scaled_dot_product_attention(q,q,q)).all()
  torch.cuda.synchronize()
  print(json.dumps({"ready":True,"gpu":torch.cuda.get_device_name(0),"runtime":torch.version.cuda}))
except Exception as exc:
 print(json.dumps({"ready":False,"reason":str(exc),"runtime":None}))
'''


def probe(python):
    result=subprocess.run([str(python),'-c',PROBE],capture_output=True,text=True,encoding='utf-8',errors='replace')
    for line in reversed(result.stdout.splitlines()):
        try:return json.loads(line)
        except (ValueError,TypeError):continue
    return {'ready':False,'reason':(result.stderr or result.stdout)[-1500:],'runtime':None}


def ensure_environment(root,cfg,offline=False,prepare_only=False):
    """Re-execute only when an isolated runtime is required; never train here."""
    root=Path(root).resolve()
    if not cfg.get('auto_project_environment',False) or prepare_only:
        return
    if os.environ.get('CS2_ONECLICK_CHILD')==str(root):
        return
    if probe(sys.executable).get('ready'):
        return
    environment=root/'.venv_run'
    python=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if offline and (not python.exists() or not probe(python).get('ready')):
        raise RuntimeError('Offline mode requires an already working CUDA environment. Run python main.py once with internet access to prepare the project environment.')
    if not python.exists():
        if sys.version_info[:2] not in {(3,10),(3,11),(3,12)}:
            raise RuntimeError('Use Python 3.11 (also supports 3.10/3.12) to prepare this CUDA package.')
        print('Preparing isolated project environment: '+str(environment),flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    status=probe(python)
    if not status.get('ready'):
        print('Installing the official CUDA 12.8 PyTorch runtime in .venv_run.',flush=True)
        subprocess.check_call([str(python),'-m','pip','install','torch==2.11.0','--index-url','https://download.pytorch.org/whl/cu128'])
        status=probe(python)
        if not status.get('ready'):
            raise RuntimeError('CUDA preflight failed in the project environment. Check the NVIDIA driver and GPU availability: '+str(status))
    env=dict(os.environ,CS2_ONECLICK_CHILD=str(root),PYTHONUNBUFFERED='1')
    entry=Path(sys.argv[0]).resolve()
    if entry.parent != root.resolve() or not entry.is_file(): entry=root/'main.py'
    result=subprocess.run([str(python),'-u',str(entry),*sys.argv[1:]],env=env)
    raise SystemExit(result.returncode)
