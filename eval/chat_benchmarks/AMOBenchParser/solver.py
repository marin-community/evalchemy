"""Hard process boundary for AMO-Bench's model-controlled SymPy solves."""

import multiprocessing
from typing import Any, List


def _solve_many_worker(expressions: List[Any], connection) -> None:
    try:
        from sympy import solve

        connection.send(("ok", [solve(expression) for expression in expressions]))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _stop_process(process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join()


def solve_many_with_timeout(expressions: List[Any], timeout_seconds: float = 30) -> List[Any]:
    """Solve a variable row's expressions with a real wall-clock timeout."""
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(target=_solve_many_worker, args=(expressions, send_connection))
    process.start()
    send_connection.close()

    try:
        if not receive_connection.poll(timeout_seconds):
            raise TimeoutError(f"SymPy solve exceeded {timeout_seconds} seconds")
        try:
            status, payload = receive_connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"SymPy solve process exited with code {process.exitcode}") from exc
        if status != "ok":
            raise RuntimeError(f"SymPy solve failed: {payload}")
        return payload
    finally:
        receive_connection.close()
        process.join(timeout=1)
        _stop_process(process)
