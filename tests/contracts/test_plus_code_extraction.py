"""Executable Python extracted from the Plus code benchmarks."""

import pytest

from eval.chat_benchmarks.HumanEvalPlus.utils.utils import extract_generation_code
from eval.chat_benchmarks.MBPPPlus.eval_instruct import MBPPPlusBenchmark


@pytest.mark.parametrize(
    "completion",
    [
        "```python\n    def add(a, b):\n        return a + b\n```",
        "    def add(a, b):\n        return a + b\n",
    ],
)
def test_mbppplus_indented_python_block_executes(completion):
    namespace = {}
    exec(MBPPPlusBenchmark().extract_code(completion), namespace)

    assert namespace["add"](2, 3) == 5


@pytest.mark.parametrize(
    "completion,function,args,expected",
    [
        (
            "```python\n    def similar_elements(list1, list2):\n        return set(list1) & set(list2)\n    ```",
            "similar_elements",
            ((3, 4, 5), (4, 5, 7)),
            {4, 5},
        ),
        (
            "```python\n   import math\n   def is_not_prime(n):\n       if n == 1:\n           return True\n"
            "       for i in range(2, int(math.sqrt(n))+1):\n           if n % i == 0:\n"
            "               return True\n       return False\n   ```",
            "is_not_prime",
            (35,),
            True,
        ),
    ],
)
def test_mbppplus_reported_indented_answers_execute(completion, function, args, expected):
    namespace = {}
    exec(MBPPPlusBenchmark().extract_code(completion), namespace)

    assert namespace[function](*args) == expected


@pytest.mark.parametrize(
    "completion,prompt",
    [
        (
            "```python\n    import math\n    def ceiling(x):\n        return math.ceil(x)\n```",
            "def ceiling(x):\n    \"\"\"Round x up.\"\"\"\n",
        ),
        (
            "```python\n    return math.ceil(x)\n```",
            "import math\n\ndef ceiling(x):\n    \"\"\"Round x up.\"\"\"\n",
        ),
        (
            "    def ceiling(x):\n        return math.ceil(x)\n",
            "import math\n\ndef ceiling(x):\n    \"\"\"Round x up.\"\"\"\n",
        ),
    ],
)
def test_humanevalplus_indented_python_solution_executes(completion, prompt):
    example = {
        "task_id": "HumanEval/test",
        "prompt": prompt,
        "output": completion,
    }

    namespace = {}
    exec(extract_generation_code(example, lang_code="python")["generation"], namespace)

    assert namespace["ceiling"](1.2) == 2
