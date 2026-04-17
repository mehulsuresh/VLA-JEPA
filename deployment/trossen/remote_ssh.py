#!/usr/bin/env python3

import argparse
import sys

import pexpect


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--user", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--copy", nargs=2, metavar=("SRC", "DST"))
    parser.add_argument("--pull", nargs=2, metavar=("SRC", "DST"))
    parser.add_argument("command", nargs="?", default="")
    args = parser.parse_args()

    target = args.host if args.user is None else f"{args.user}@{args.host}"
    ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

    if args.copy:
        src, dst = args.copy
        cmd = f"scp -r {ssh_opts} {src} {target}:{dst}"
    elif args.pull:
        src, dst = args.pull
        cmd = f"scp -r {ssh_opts} {target}:{src} {dst}"
    elif args.command:
        cmd = f"ssh {ssh_opts} {target} {args.command!r}"
    else:
        cmd = f"ssh {ssh_opts} {target}"

    child = pexpect.spawn(cmd, encoding="utf-8", timeout=args.timeout)
    child.logfile_read = sys.stdout

    while True:
        index = child.expect(
            [
                r"Are you sure you want to continue connecting \(yes/no(/\[fingerprint\])?\)\?",
                r"[Pp]assword:",
                pexpect.EOF,
                pexpect.TIMEOUT,
            ]
        )
        if index == 0:
            child.sendline("yes")
            continue
        if index == 1:
            child.sendline(args.password)
            continue
        if index == 2:
            return child.exitstatus or 0
        if index == 3:
            print("\nTimed out waiting for SSH/scp to complete.", file=sys.stderr)
            child.close(force=True)
            return 124

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
