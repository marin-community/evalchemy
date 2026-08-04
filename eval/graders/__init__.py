"""Self-contained graders that match lm-eval-harness verdicts without the harness.

Each grader exposes ``grade(problem, solution, reference_answer) -> float`` and
needs nothing else: no ``datasets`` doc, no task registry, no harness import.
"""
