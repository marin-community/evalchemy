"""Trusted test-side shell call transport."""

import base64
import os
import socket
import sys

from lm_eval.verifyit_function_worker import read, write


def main():
    merged = os.fstat(1) == os.fstat(2)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(sys.argv[1])
        stream = connection.makefile("rwb")
        max_bytes = int(sys.argv[3])
        write(stream, {"args": sys.argv[4:], "merge_streams": merged}, max_bytes)
        result = read(stream, max_bytes)
    sys.stdout.buffer.write(base64.b64decode(result["stdout"], validate=True))
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(base64.b64decode(result["stderr"], validate=True))
    sys.stderr.buffer.flush()
    with open(sys.argv[2], "ab") as completed:
        completed.write(b".")
    code = result["returncode"]
    sys.exit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
