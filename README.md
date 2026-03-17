# minimax-m2-tool-def-to-gbnf

[![CI](https://github.com/benjamincburns/minimax_m2_tool_def_to_gbnf/actions/workflows/ci.yml/badge.svg)](https://github.com/benjamincburns/minimax_m2_tool_def_to_gbnf/actions/workflows/ci.yml)

Look, I'm bad at naming things, okay?

Generate **GBNF grammars** for constrained decoding / guided generation with **MiniMax M2 series** models. Given OpenAI-style tool definitions, this library produces a GBNF grammar string that inference providers can use to ensure model output conforms to the MiniMax M2 XML tool-call format.

## Purpose

When using MiniMax M2 models for tool calling, the model must output XML in the `<minimax:tool_call>` / `<invoke>` format. This project generates GBNF (grammar for LLMs) rules from your tool schemas so that:

- Decoding is **constrained** to valid tool-call XML
- Only declared function names and parameters are allowed
- Parameter types (string, number, boolean, enum, object, array) and `$ref` / `$defs` are reflected in the grammar

You can feed the generated grammar into any inference stack that supports GBNF-based constrained decoding (e.g. llama.cpp, or other backends that accept GBNF).

## Installation

Lul this isn't on pypi yet. If you want it, drop an issue and I'll get it out there.

## Quick start

```python
from minimax_m2_tool_def_to_gbnf import generate_minimax_tool_grammar

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
]

grammar = generate_minimax_tool_grammar(tools)
# Use grammar with your inference provider's constrained-decoding API
```

The resulting string is a full GBNF grammar. Example of the kind of output it constrains to:

```xml
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Tokyo</parameter>
<parameter name="unit">celsius</parameter>
</invoke>
</minimax:tool_call>
```

## Tool definition format

Tool definitions follow the **OpenAI** style:

- **Nested**: `{"type": "function", "function": {"name": "...", "parameters": { ... }}}`
- **Flat**: `{"name": "...", "parameters": { ... }}`

`parameters` is a JSON Schema object (e.g. `type`, `properties`, `required`, `$defs`, `$ref`, `enum`, `anyOf`). The generator:

- Resolves `$ref` and `#/$defs/<name>` to separate GBNF rules (including recursive refs via `#`).
- Enforces valid function names, parameter names, required parameters in order, and value types (string, number, boolean, null, array, object, enum).

## API

```python
def generate_minimax_tool_grammar(
    tools: list[dict],
    *,
    allow_preamble: bool = False,
    require_tool_call: bool = True,
    strict: bool = False,
) -> str:
```

| Argument | Default | Description |
|----------|---------|-------------|
| `tools` | — | List of tool definitions (OpenAI-style). At least one required. |
| `allow_preamble` | `False` | If `True`, allow free text before the `<minimax:tool_call>` block. |
| `require_tool_call` | `True` | If `True`, at least one `<invoke>` is required; if `False`, empty tool call (no invocations) is allowed. |
| `strict` | `False` | If `True`, disallow `additionalProperties` on all objects (OpenAI strict mode). |

Returns a single string containing the full GBNF grammar.

## Limitations

- **`$ref`**: Only `"#"` and `"#/$defs/<name>"` are supported. External URIs or other JSON pointer paths raise `ValueError`.
- **allOf**: Not supported; falls back to a generic string/json-value rule.
- **Numeric/string bounds**: `minimum`, `maximum`, `minLength`, `maxLength`, `pattern`, `minItems`, `maxItems` are not enforced in the grammar (aligned with typical structured-output behavior).
- **`$defs` + string**: A `$defs` entry that is plain `{"type": "string"}` yields a quoted JSON string rule; MiniMax parsing still accepts it via `json.loads` fallback.

## Development

- **Tests**: `pytest` from the project root (tests in `tests/test_minimax_m2_tool_def_to_gbnf.py`).
- **Benchmark**: `python benchmark.py` to measure grammar generation time with large or complex tool sets; see `benchmark.py` for options (`--preamble`, `--strict`, `--require-tool-call`, etc.).
- **Lint/format**: Ruff, Pyright (see `pyproject.toml`).

## License

See repository for license information.
