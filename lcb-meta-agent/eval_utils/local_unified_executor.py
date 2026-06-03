"""Unified local code execution utilities shared by VERL-style and LiveCodeBench-style evaluation.

This module consolidates the overlapping mechanics:
- subprocess-isolated execution
- per-test timeout and global timeout protection
- basic reliability guard against destructive operations
- support for both call-based and stdio-based judging
"""

from __future__ import annotations

import ast
import faulthandler
import json
import multiprocessing
import os
import platform
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from io import StringIO
from types import ModuleType
from typing import Any
from unittest.mock import mock_open, patch


IMPORT_STRING = (
    "from string import *\n"
    "from re import *\n"
    "from datetime import *\n"
    "from collections import *\n"
    "from heapq import *\n"
    "from bisect import *\n"
    "from copy import *\n"
    "from math import *\n"
    "from random import *\n"
    "from statistics import *\n"
    "from itertools import *\n"
    "from functools import *\n"
    "from operator import *\n"
    "from io import *\n"
    "from sys import *\n"
    "from json import *\n"
    "from builtins import *\n"
    "from typing import *\n"
    "import string\n"
    "import re\n"
    "import datetime\n"
    "import collections\n"
    "import heapq\n"
    "import bisect\n"
    "import copy\n"
    "import math\n"
    "import random\n"
    "import statistics\n"
    "import itertools\n"
    "import functools\n"
    "import operator\n"
    "import io\n"
    "import sys\n"
    "import json\n"
    "sys.setrecursionlimit(500000)\n"
)


class TimeoutException(Exception):
    """Raised when signal alarm times out."""


class CodeType(Enum):
    CALL_BASED = "call_based"
    STDIO = "stdio"


@dataclass
class JudgeOutput:
    """Structured output for a single completion evaluation."""

    score: float
    passed: bool
    details: Any


class Capturing(list):
    """Capture stdout content as one string entry."""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda *args, **kwargs: None
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockBuffer:
    """Minimal bytes buffer compatible with common stdin.buffer calls."""

    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


class MockStdinWithBuffer:
    """String stdin with .buffer support."""

    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


def timeout_handler(signum, frame):
    raise TimeoutException("Timed out")


def truncate_text(value: Any, length: int = 300) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= length:
        return text
    return text[: length // 2] + "...(truncated)..." + text[-length // 2 :]


def extract_solution_code(completion: str) -> str:
    if "```" not in completion:
        return completion
    chunks = completion.split("```")
    if len(chunks) < 3:
        return completion
    for block in reversed(chunks[1::2]):
        content = block
        if "\n" in block:
            first, rest = block.split("\n", 1)
            if first.strip().lower() in {"python", "py"}:
                content = rest
        content = content.strip()
        if content:
            return content
    return completion


def parse_in_outs(sample_or_in_outs: dict | str) -> dict:
    if isinstance(sample_or_in_outs, str):
        sample_or_in_outs = json.loads(sample_or_in_outs)
    if "input_output" in sample_or_in_outs:
        return json.loads(sample_or_in_outs["input_output"])
    if "inputs" in sample_or_in_outs and "outputs" in sample_or_in_outs:
        return sample_or_in_outs
    raise ValueError("Input format must include either `input_output` or (`inputs`, `outputs`).")


def clean_if_name(code: str) -> str:
    try:
        parsed = ast.parse(code)
        if not parsed.body:
            return code
        last = parsed.body[-1]
        if isinstance(last, ast.If) and ast.unparse(last.test).strip() == "__name__ == '__main__'":
            return ast.unparse(parsed.body[:-1]) + "\n" + ast.unparse(last.body)
    except Exception:
        return code
    return code


def make_wrapped_stdio_function(code: str) -> str:
    try:
        parsed = ast.parse(code)
        import_stmts = []
        other_stmts = []
        for stmt in parsed.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        imports = ast.unparse(import_stmts) if import_stmts else ""
        return IMPORT_STRING + "\n" + imports + "\n" + ast.unparse(function_ast)
    except Exception:
        return code


def call_method(method, inputs: str | list[str]):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    inputs_line_iterator = iter(inputs.split("\n"))
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call(fn):
        try:
            return fn()
        except SystemExit:
            return None

    return _inner_call(method)


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        module = ModuleType("tmp_sol", "")
        exec(code, module.__dict__)
        if "class Solution" in code and hasattr(module, "Solution"):
            compiled = module.Solution()
        else:
            compiled = module
    finally:
        signal.alarm(0)
    return compiled


def get_function(compiled, fn_name: str):
    if not hasattr(compiled, fn_name):
        return None
    return getattr(compiled, fn_name)


def normalize_call_based_inputs(raw_inputs: Any) -> list[Any]:
    if isinstance(raw_inputs, list):
        return raw_inputs
    if isinstance(raw_inputs, str):
        lines = [line for line in raw_inputs.split("\n") if line != ""]
        return [json.loads(line) for line in lines]
    raise ValueError(f"Unsupported call-based input type: {type(raw_inputs)}")


def normalize_call_based_output(raw_output: Any) -> Any:
    if isinstance(raw_output, str):
        return json.loads(raw_output)
    return raw_output


def compare_call_based_output(prediction: Any, expected: Any) -> bool:
    if isinstance(prediction, tuple):
        prediction = list(prediction)
    return prediction == expected


def get_stripped_lines(value: str) -> list[str]:
    value = value.strip()
    return [line.strip() for line in value.split("\n")]


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimals = [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []
    return True, decimals


def compare_stdio_output(prediction: str, expected: str) -> tuple[bool, dict[str, Any] | None]:
    pred_lines = get_stripped_lines(prediction)
    exp_lines = get_stripped_lines(expected)
    metadata = {
        "output": truncate_text(prediction),
        "expected": truncate_text(expected),
        "error_code": -2,
    }

    if len(pred_lines) != len(exp_lines):
        metadata["error_message"] = "Wrong answer: mismatched output length"
        return False, metadata

    for idx, (pred_line, exp_line) in enumerate(zip(pred_lines, exp_lines)):
        if pred_line == exp_line:
            continue
        ok_pred, pred_dec = convert_line_to_decimals(pred_line)
        ok_exp, exp_dec = convert_line_to_decimals(exp_line)
        if ok_pred and ok_exp and pred_dec == exp_dec:
            continue
        metadata["error_message"] = (
            f"Wrong answer at line {idx}: {truncate_text(pred_line)} != {truncate_text(exp_line)}"
        )
        return False, metadata

    return True, None


def grade_call_based(code: str, in_outs: dict, timeout: int) -> tuple[list[Any], dict[str, Any]]:
    code = IMPORT_STRING + "\n" + code
    compiled = compile_code(code, timeout)
    method = get_function(compiled, in_outs["fn_name"])
    if method is None:
        return [-4], {
            "error_code": -4,
            "error_message": f"Function `{in_outs['fn_name']}` not found",
        }

    all_inputs = [normalize_call_based_inputs(item) for item in in_outs["inputs"]]
    all_outputs = [normalize_call_based_output(item) for item in in_outs["outputs"]]

    total_execution_time = 0.0
    results: list[Any] = []

    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            prediction = method(*gt_inp)
            total_execution_time += time.time() - start
            passed = compare_call_based_output(prediction, gt_out)
            results.append(True if passed else False)
            if not passed:
                return results, {
                    "output": truncate_text(prediction),
                    "inputs": truncate_text(gt_inp),
                    "expected": truncate_text(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as exc:
            if "timeoutexception" in repr(exc).lower() or isinstance(exc, TimeoutException):
                results.append(-3)
                return results, {
                    "error": repr(exc),
                    "inputs": truncate_text(gt_inp),
                    "expected": truncate_text(gt_out),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                }
            results.append(-4)
            return results, {
                "error": repr(exc),
                "inputs": truncate_text(gt_inp),
                "expected": truncate_text(gt_out),
                "error_code": -4,
                "error_message": "Runtime Error",
            }
        finally:
            signal.alarm(0)
            faulthandler.disable()

    return results, {"execution_time": total_execution_time}


def grade_stdio(code: str, in_outs: dict, timeout: int) -> tuple[list[Any], dict[str, Any]]:
    code = clean_if_name(code)
    code = make_wrapped_stdio_function(code)
    compiled = compile_code(code, timeout)
    method = get_function(compiled, "wrapped_function")
    if method is None:
        return [-4], {
            "error_code": -4,
            "error_message": "wrapped_function not found",
        }

    total_execution_time = 0.0
    results: list[Any] = []

    for gt_inp, gt_out in zip(in_outs["inputs"], in_outs["outputs"]):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            with Capturing() as captured_output:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
            prediction = captured_output[0]
            passed, failure_meta = compare_stdio_output(prediction, gt_out)
            results.append(True if passed else False)
            if not passed:
                assert failure_meta is not None
                failure_meta["inputs"] = truncate_text(gt_inp)
                return results, failure_meta
        except Exception as exc:
            if "timeoutexception" in repr(exc).lower() or isinstance(exc, TimeoutException):
                results.append(-3)
                return results, {
                    "error": repr(exc),
                    "inputs": truncate_text(gt_inp),
                    "expected": truncate_text(gt_out),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                }
            results.append(-4)
            return results, {
                "error": repr(exc),
                "inputs": truncate_text(gt_inp),
                "expected": truncate_text(gt_out),
                "error_code": -4,
                "error_message": "Runtime Error",
            }
        finally:
            signal.alarm(0)
            faulthandler.disable()

    return results, {"execution_time": total_execution_time}


def reliability_guard(maximum_memory_bytes: int | None = None):
    """Apply a best-effort guard. This is not a security sandbox."""

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None

    builtins.help = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def run_test(in_outs: dict, test: str, timeout: int = 6) -> tuple[list[Any], dict[str, Any]]:
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard()

    code_type = CodeType.CALL_BASED if in_outs.get("fn_name") else CodeType.STDIO
    if code_type == CodeType.CALL_BASED:
        return grade_call_based(test, in_outs, timeout)
    return grade_stdio(test, in_outs, timeout)


def _temp_run(in_outs, generation, result_conn, timeout, debug):
    try:
        if not debug:
            with open(os.devnull, "w") as devnull:
                sys.stdout = devnull
                sys.stderr = devnull
                res, metadata = run_test(in_outs, test=generation, timeout=timeout)
        else:
            res, metadata = run_test(in_outs, test=generation, timeout=timeout)
    except Exception:
        traceback.print_exc(10)
        res = [-1 for _ in range(len(in_outs.get("inputs", [])))]
        metadata = {"error_code": -5, "error_message": "TestRunnerError"}
    try:
        result_conn.send((res, metadata))
    finally:
        result_conn.close()


def _default_global_timeout(in_outs: dict, timeout: int) -> int:
    num_inputs = len(in_outs.get("inputs", []))
    num_inputs = max(1, num_inputs)
    return (timeout + 1) * num_inputs + 5


def check_correctness(
    in_outs: dict,
    generation: str,
    timeout: int = 6,
    debug: bool = False,
    global_timeout: int | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    try:
        mp_ctx = multiprocessing.get_context("fork")
    except ValueError:
        mp_ctx = multiprocessing.get_context()

    parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
    worker = mp_ctx.Process(
        target=_temp_run,
        args=(in_outs, generation, child_conn, timeout, debug),
    )

    worker.start()
    child_conn.close()
    worker.join(timeout=global_timeout or _default_global_timeout(in_outs, timeout))
    if worker.is_alive():
        worker.kill()
        worker.join()

    if parent_conn.poll():
        try:
            result_value, metadata_value = parent_conn.recv()
            parent_conn.close()
            return result_value, metadata_value
        except EOFError:
            parent_conn.close()
            result_value = [-1 for _ in range(len(in_outs.get("inputs", [])))]
            metadata_value = {
                "error_code": -5,
                "error_message": f"Child process exited unexpectedly (exit_code={worker.exitcode})",
            }
            return result_value, metadata_value

    parent_conn.close()
    result_value = [-1 for _ in range(len(in_outs.get("inputs", [])))]
    metadata_value = {"error_code": -3, "error_message": "Global timeout"}
    return result_value, metadata_value


def compute_score_local(
    completion: str,
    test_cases: dict | str,
    continuous: bool = False,
    timeout: int = 6,
    max_continuous_cases: int = 10,
) -> JudgeOutput:
    in_outs = parse_in_outs(test_cases)
    solution = extract_solution_code(completion)

    res, metadata = check_correctness(in_outs=in_outs, generation=solution, timeout=timeout, debug=False)
    passed = bool(res) and all(item is True for item in res)
    if passed:
        return JudgeOutput(score=1.0, passed=True, details=metadata)

    if not continuous:
        return JudgeOutput(score=0.0, passed=False, details=metadata)

    case_inputs = in_outs.get("inputs", [])
    case_outputs = in_outs.get("outputs", [])
    case_count = min(max_continuous_cases, len(case_inputs), len(case_outputs))
    if case_count == 0:
        return JudgeOutput(score=0.0, passed=False, details=[])

    passed_count = 0
    details = []
    fn_name = in_outs.get("fn_name")
    for idx in range(case_count):
        single_case = {
            "inputs": [case_inputs[idx]],
            "outputs": [case_outputs[idx]],
            "fn_name": fn_name,
        }
        single_res, single_meta = check_correctness(
            in_outs=single_case,
            generation=solution,
            timeout=timeout,
            debug=False,
        )
        current_pass = bool(single_res) and all(item is True for item in single_res)
        if current_pass:
            passed_count += 1
        details.append(
            {
                "test_case_id": idx,
                "input": truncate_text(case_inputs[idx]),
                "output": truncate_text(case_outputs[idx]),
                "result": single_res,
                "metadata": single_meta,
            }
        )

    score = passed_count / case_count
    return JudgeOutput(score=score, passed=score >= 1.0, details=details)


if __name__ == "__main__":
    print("Import this module and call compute_score_local(...) in your evaluator pipeline.")
