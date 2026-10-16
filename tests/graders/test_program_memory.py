"""Canonical candidate children enforce their memory budget independently of safe mode."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
CHILD = """
import json
import resource
import sys
from eval.runtime.program_process import start_program_session

budget = 30 * 1024**3
if sys.argv[1] == 'lower':
    resource.setrlimit(resource.RLIMIT_AS, (256*1024**2, 512*1024**2))
    resource.setrlimit(resource.RLIMIT_DATA, (384*1024**2, 640*1024**2))
before = {name: resource.getrlimit(getattr(resource, name))
          for name in ('RLIMIT_AS', 'RLIMIT_DATA', 'RLIMIT_STACK')}
start_program_session()
after = {name: resource.getrlimit(getattr(resource, name)) for name in before}
small = bytearray(1024**2)
small[-1] = 1
memory_error = False
# An unfixed child must fail the regression without allocating tens of GiB.
if all(0 < after[name][0] <= budget for name in ('RLIMIT_AS', 'RLIMIT_DATA')):
    try:
        bytearray(budget + 1024**3)
    except MemoryError:
        memory_error = True
print(json.dumps({'before': before, 'after': after,
                  'small_allocation': small[-1], 'memory_error': memory_error}))
"""
BIGCODE = """
import json
import resource
import sys
from eval.chat_benchmarks.BigCodeBench.execution import check_correctness

code = '''import resource
import unittest
class TestCases(unittest.TestCase):
    def test_memory_budget(self):
        budget = 30 * 1024**3
        for kind in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            self.assertGreater(resource.getrlimit(kind)[0], 0)
            self.assertLessEqual(resource.getrlimit(kind)[0], budget)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_STACK), EXPECTED_STACK)
        small = bytearray(1024**2)
        self.assertEqual(len(small), 1024**2)
        with self.assertRaises(MemoryError):
            bytearray(budget + 1024**3)
'''
code = 'EXPECTED_STACK = ' + repr(resource.getrlimit(resource.RLIMIT_STACK)) + '\\n' + code
result = check_correctness('memory/control', {'test_code': code}, 'python',
                           timeout=3, tmp_dir=sys.argv[1], safe_mode=False)
print(json.dumps({'passed': result['passed'], 'result': result['result']}))
"""


@pytest.mark.parametrize("existing", ["default", "lower"])
def test_program_child_bounds_allocation_and_preserves_lower_limits(existing):
    child = subprocess.run([sys.executable, "-c", CHILD, existing], cwd=REPO,
                           capture_output=True, text=True, check=True, timeout=10)
    result = json.loads(child.stdout)
    assert result["small_allocation"] == 1
    assert result["memory_error"] is True
    assert result["after"]["RLIMIT_STACK"] == result["before"]["RLIMIT_STACK"]
    if existing == "lower":
        assert result["after"] == result["before"]


def test_native_bigcode_limits_memory_with_safe_mode_disabled(tmp_path):
    child = subprocess.run([sys.executable, "-c", BIGCODE, str(tmp_path)], cwd=REPO,
                           capture_output=True, text=True, check=True, timeout=10)
    assert json.loads(child.stdout) == {"passed": True, "result": "passed"}
