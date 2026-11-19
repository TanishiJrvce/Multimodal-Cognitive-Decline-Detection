"""Wrapper: runs 04_extract_features.py and writes output to logs/feat.log.
Uses PYTHONUNBUFFERED to force unbuffered output.
"""
import subprocess, sys, pathlib, os

pathlib.Path("logs").mkdir(exist_ok=True)
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
with open("logs/feat.log", "w", buffering=1) as fh:
    proc = subprocess.run(
        [sys.executable, "-u", "scripts/04_extract_features.py"],
        stdout=fh, stderr=subprocess.STDOUT, text=True, env=env
    )
print("Exit code:", proc.returncode)
