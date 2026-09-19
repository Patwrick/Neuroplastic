"""Execute one bounded local command and retain its real status and output."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    a.directory.mkdir(parents=True, exist_ok=True)
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    start = time.perf_counter()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path = a.directory / (a.name + '.log')
    if path.exists():
        raise FileExistsError(path)
    with path.open('wb') as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
            code = result.returncode
        except subprocess.TimeoutExpired:
            log.write(b'\nABORTED: external wall-clock cap exceeded.\n')
            code = 124
    record = dict(command=subprocess.list2cmdline(command), exit_code=code,
                  duration_seconds=time.perf_counter()-start, log_path=str(path),
                  started_at_utc=started)
    (a.directory / (a.name + '.command.json')).write_text(json.dumps(record, indent=2)+'\n')
    print(path.read_text(errors='replace'))
    print(json.dumps(record))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
