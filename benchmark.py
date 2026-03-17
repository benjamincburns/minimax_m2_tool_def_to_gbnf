#!/usr/bin/env python3
"""
Benchmark generate_minimax_tool_grammar with large/complex tool sets.

Uses timeit to measure conversion to GBNF with different options:
  allow_preamble: True/False
  strict: True/False (extra/additional properties)
  require_tool_call: True/False (at least one <invoke> required)

Run all benchmarks (default):
  python benchmark.py

Run a specific configuration only:
  python benchmark.py --default          # no preamble, non-strict (both require_tool_call T/F)
  python benchmark.py --preamble          # preamble, non-strict
  python benchmark.py --strict           # no preamble, strict
  python benchmark.py --preamble-strict   # preamble, strict
  python benchmark.py --require-tool-call # only require_tool_call=True (all preamble/strict combos)
  python benchmark.py --no-require-tool-call  # only require_tool_call=False
"""

from __future__ import annotations

import argparse
import timeit
from typing import Any

from minimax_m2_grammar import generate_minimax_tool_grammar

# ── Benchmark tool sets (large/complex) ────────────────────────────────────


def _make_many_tools(num_tools: int, params_per_tool: int) -> list[dict[str, Any]]:
    """Many tools, each with several string/enum/integer parameters."""
    tools = []
    for i in range(num_tools):
        props = {}
        required = []
        for j in range(params_per_tool):
            name = f"p{j}"
            required.append(name)
            if j % 3 == 0:
                props[name] = {"type": "string"}
            elif j % 3 == 1:
                props[name] = {"type": "integer"}
            else:
                props[name] = {"type": "string", "enum": [f"opt_{j}_a", f"opt_{j}_b", f"opt_{j}_c"]}
        tools.append(
            {
                "name": f"tool_{i}",
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            }
        )
    return tools


def _make_nested_and_refs() -> list[dict[str, Any]]:
    """Deep nesting, $defs, $ref, anyOf, arrays."""
    return [
        {
            "name": "complex",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {"$ref": "#/$defs/status"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "payload": {
                        "type": "object",
                        "properties": {
                            "level": {"type": "integer"},
                            "data": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "value": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "number"},
                                            {"type": "null"},
                                        ]
                                    },
                                    "meta": {
                                        "type": "object",
                                        "properties": {
                                            "source": {"$ref": "#/$defs/source"},
                                            "nested": {
                                                "type": "array",
                                                "items": {"$ref": "#/$defs/node"},
                                            },
                                        },
                                        "required": ["source"],
                                    },
                                },
                                "required": ["name", "meta"],
                            },
                        },
                        "required": ["level", "data"],
                    },
                },
                "required": ["id", "status", "payload"],
                "$defs": {
                    "status": {"type": "string", "enum": ["draft", "active", "archived"]},
                    "source": {"type": "string", "enum": ["api", "ui", "import"]},
                    "node": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                        },
                        "required": ["value"],
                    },
                },
            },
        },
    ]


def _make_recursive_root_ref() -> list[dict[str, Any]]:
    """Root $ref (#) recursion."""
    return [
        {
            "name": "tree",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#"}},
                },
                "required": ["value"],
            },
        },
    ]


def _make_additional_properties_variants() -> list[dict[str, Any]]:
    """Objects with additionalProperties True/False for strict vs non-strict."""
    return [
        {
            "name": "with_extra",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string"},
                            "b": {"type": "integer"},
                            "c": {"type": "string", "enum": ["x", "y"]},
                        },
                        "required": ["a"],
                        "additionalProperties": True,
                    },
                },
                "required": ["data"],
            },
        },
        {
            "name": "no_extra",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
                "required": ["data"],
            },
        },
    ]


def get_benchmark_tool_sets() -> dict[str, list[dict[str, Any]]]:
    """Return named tool sets for benchmarking."""
    return {
        "many_tools_50x10": _make_many_tools(50, 10),
        "many_tools_20x15": _make_many_tools(20, 15),
        "nested_and_refs": _make_nested_and_refs(),
        "recursive_root_ref": _make_recursive_root_ref(),
        "additional_properties": _make_additional_properties_variants(),
    }


# ── Benchmark runner ───────────────────────────────────────────────────────


def run_benchmark(
    tools: list[dict[str, Any]],
    allow_preamble: bool,
    strict: bool,
    require_tool_call: bool,
    number: int = 50,
) -> float:
    """Run timeit for generate_minimax_tool_grammar with given options. Returns seconds per call."""
    stmt = (
        "generate_minimax_tool_grammar(tools, allow_preamble=ap, strict=st, require_tool_call=rtc)"
    )
    globals_dict = {
        "generate_minimax_tool_grammar": generate_minimax_tool_grammar,
        "tools": tools,
        "ap": allow_preamble,
        "st": strict,
        "rtc": require_tool_call,
    }
    return timeit.timeit(stmt, number=number, globals=globals_dict) / number


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark generate_minimax_tool_grammar with different options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--number",
        type=int,
        default=50,
        help="Number of timeit iterations per run (default: 50)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--default", action="store_true", help="Run only allow_preamble=False, strict=False"
    )
    group.add_argument(
        "--preamble", action="store_true", help="Run only allow_preamble=True, strict=False"
    )
    group.add_argument(
        "--strict", action="store_true", help="Run only allow_preamble=False, strict=True"
    )
    group.add_argument(
        "--preamble-strict",
        action="store_true",
        dest="preamble_strict",
        help="Run only allow_preamble=True, strict=True",
    )
    group.add_argument(
        "--require-tool-call",
        action="store_true",
        dest="require_tool_call",
        help="Run only require_tool_call=True (all preamble/strict combos)",
    )
    group.add_argument(
        "--no-require-tool-call",
        action="store_true",
        dest="no_require_tool_call",
        help="Run only require_tool_call=False (all preamble/strict combos)",
    )
    args = parser.parse_args()

    # Build full 8-config matrix: (allow_preamble, strict, require_tool_call, label)
    def make_configs(require_tool_call_filter: bool | None) -> list[tuple[bool, bool, bool, str]]:
        rtc_values: list[bool] = (
            [True, False] if require_tool_call_filter is None else [require_tool_call_filter]
        )
        configs: list[tuple[bool, bool, bool, str]] = []
        for ap, st, label in [
            (False, False, "default (no preamble, non-strict)"),
            (True, False, "preamble (preamble, non-strict)"),
            (False, True, "strict (no preamble, strict)"),
            (True, True, "preamble-strict (preamble, strict)"),
        ]:
            for rtc in rtc_values:
                rtc_suffix = ", require_tool_call=True" if rtc else ", require_tool_call=False"
                configs.append((ap, st, rtc, label + rtc_suffix))
        return configs

    if args.default:
        configs = [
            (False, False, rtc, f"default, require_tool_call={rtc}") for rtc in (True, False)
        ]
    elif args.preamble:
        configs = [
            (True, False, rtc, f"preamble, require_tool_call={rtc}") for rtc in (True, False)
        ]
    elif args.strict:
        configs = [
            (False, True, rtc, f"strict, require_tool_call={rtc}") for rtc in (True, False)
        ]
    elif args.preamble_strict:
        configs = [
            (True, True, rtc, f"preamble-strict, require_tool_call={rtc}")
            for rtc in (True, False)
        ]
    elif args.require_tool_call:
        configs = make_configs(True)
    elif args.no_require_tool_call:
        configs = make_configs(False)
    else:
        configs = make_configs(None)

    tool_sets = get_benchmark_tool_sets()
    print("Benchmark: generate_minimax_tool_grammar")
    print("number =", args.number, "runs per configuration\n")

    for allow_preamble, strict, require_tool_call, label in configs:
        print(f"--- {label} ---")
        for name, tools in tool_sets.items():
            t = run_benchmark(
                tools,
                allow_preamble=allow_preamble,
                strict=strict,
                require_tool_call=require_tool_call,
                number=args.number,
            )
            print(f"  {name}: {t * 1000:.2f} ms/call")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
