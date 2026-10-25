"""Launch an owned MPI job and preserve diagnostics on timeout."""

import os
import signal
import subprocess


def run(command, *, env=None, timeout=120):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    expired = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        expired = True
        # Signal only the process group created for this job, never other users' jobs.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        stderr += "\nbenchmark timeout; owned launcher process group terminated\n"
    return subprocess.CompletedProcess(
        command, 124 if expired else process.returncode, stdout, stderr
    )
