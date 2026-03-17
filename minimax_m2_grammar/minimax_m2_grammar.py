"""
Generate GBNF grammars for MiniMax M2 XML tool call format.

Given a list of OpenAI-style tool definitions, produces a GBNF grammar string
that constrains generation to valid MiniMax M2 tool call XML.

The grammar enforces:
- At least one <invoke> block inside <minimax:tool_call>
- Only valid function names from the tool list
- Only valid parameter names per function, in schema-defined order
- All required parameters present
- Parameter values constrained to their declared types
- Enum values constrained to declared options
- $ref / $defs resolved to GBNF rule references (including recursive schemas)

Known limitations:
- Only $ref formats "#" and "#/$defs/<name>" are supported. External URIs
  and arbitrary JSON pointer paths (e.g. "#/properties/foo") raise ValueError.
- allOf is not supported (falls through to bare-string/json-value fallback).
- minimum/maximum, minLength/maxLength, pattern, minItems/maxItems are not
  enforced (consistent with OpenAI's own structured output engine).
- When strict=False and additionalProperties is allowed on an object where
  ALL declared properties are optional and none are emitted, additional
  properties may produce a leading-comma JSON formatting issue.
- $defs entries containing bare {"type": "string"} will produce json-string
  (quoted) rules even when referenced from a top-level XML parameter value.
  The MiniMax parser handles quoted strings correctly via json.loads fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Shared sub-grammars for JSON value types ────────────────────────────

_JSON_PRIMITIVES = r'''
json-string ::= "\"" json-string-inner "\""
json-string-inner ::= "" | json-string-char json-string-inner
json-string-char ::= [^"\\\x00-\x1f] | "\\" json-escape
json-escape ::= ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]

json-integer ::= "-"? ("0" | [1-9] [0-9]*)
json-number ::= json-integer ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
json-bool ::= "true" | "false"
json-null ::= "null"

json-value ::= json-string | json-number | json-bool | json-null | json-array | json-object

json-array ::= "[" ws json-array-body ws "]"
json-array-body ::= "" | json-value (ws "," ws json-value)*

json-object ::= "{" ws json-object-body ws "}"
json-object-body ::= "" | json-object-kv (ws "," ws json-object-kv)*
json-object-kv ::= json-string ws ":" ws json-value

ws ::= [ \t\n]*
'''.strip()

_RULE_NAME_RE = re.compile(r'[^a-zA-Z0-9_-]')


def _safe_rule_name(name: str) -> str:
    """Sanitize a string for use as a GBNF rule name."""
    return _RULE_NAME_RE.sub('-', name)


def _escape_gbnf_string(s: str) -> str:
    """Escape a string for use inside GBNF double-quoted literals."""
    return s.replace('\\', '\\\\').replace('"', '\\"')


def _extract_tool_info(tool: dict) -> tuple[str, dict]:
    """Extract (name, parameters_schema) from an OpenAI-style tool dict."""
    if "function" in tool:
        func = tool["function"]
        return func["name"], func.get("parameters", {})
    return tool["name"], tool.get("parameters", {})


# ── Grammar context ─────────────────────────────────────────────────────

@dataclass
class _GrammarContext:
    """Shared state threaded through grammar generation."""
    extra_rules: list[str] = field(default_factory=list)
    defs: dict[str, Any] = field(default_factory=dict)
    defs_prefix: str = ""
    root_schema: dict[str, Any] | None = None
    strict: bool = True
    _processed_defs: set[str] = field(default_factory=set)
    _root_ref_rule: str = ""
    _root_ref_generated: bool = False


# ── $ref resolution ─────────────────────────────────────────────────────

def _resolve_ref(ref: str, ctx: _GrammarContext) -> str:
    """Resolve a $ref string to a GBNF rule name.

    Supported formats:
    - "#/$defs/<name>" → maps to a named GBNF rule for the $defs entry
    - "#" → maps to a GBNF rule for the root parameter schema (recursive)

    Raises ValueError for unsupported $ref formats.
    """
    if ref == "#":
        if ctx.root_schema is None:
            raise ValueError("$ref '#' used but no root schema is available")
        if not ctx._root_ref_generated:
            ctx._root_ref_generated = True
            rule_name = ctx._root_ref_rule
            val_ref = _value_rule_for_schema(
                ctx.root_schema, rule_name, ctx, json_context=True
            )
            if val_ref != rule_name:
                ctx.extra_rules.append(f"{rule_name} ::= {val_ref}")
        return ctx._root_ref_rule

    if ref.startswith("#/$defs/"):
        def_name = ref[len("#/$defs/"):]
        if def_name not in ctx.defs:
            raise ValueError(
                f"$ref '#/$defs/{def_name}' references undefined definition"
            )
        return _ensure_def_processed(def_name, ctx)

    raise ValueError(
        f"Unsupported $ref: {ref!r}. "
        "Only '#/$defs/<name>' and '#' are supported."
    )


def _ensure_def_processed(def_name: str, ctx: _GrammarContext) -> str:
    """Ensure a $defs entry has a corresponding GBNF rule. Returns rule name.

    Uses lazy generation with cycle detection so that recursive $defs
    (A references B references A) produce valid recursive GBNF rules.
    """
    rule_name = f"{ctx.defs_prefix}defs-{_safe_rule_name(def_name)}"
    if def_name not in ctx._processed_defs:
        # Mark BEFORE processing to break infinite recursion
        ctx._processed_defs.add(def_name)
        schema = ctx.defs[def_name]
        val_ref = _value_rule_for_schema(
            schema, rule_name, ctx, json_context=True
        )
        if val_ref != rule_name:
            ctx.extra_rules.append(f"{rule_name} ::= {val_ref}")
    return rule_name


# ── Value type grammar generation ───────────────────────────────────────

def _value_rule_for_schema(
    schema: dict[str, Any],
    prefix: str,
    ctx: _GrammarContext,
    *,
    json_context: bool = False,
) -> str:
    """Return the GBNF rule reference for a value, given its JSON schema.

    Args:
        schema: JSON Schema dict for the value.
        prefix: Prefix for generated GBNF rule names (must be unique).
        ctx: Shared grammar context.
        json_context: If True, string values use json-string (quoted).
                      If False, string values use bare-string (unquoted).
    """
    # $ref resolution
    if "$ref" in schema:
        return _resolve_ref(schema["$ref"], ctx)

    # Enum constraint
    if "enum" in schema:
        alts = " | ".join(
            f'"{_escape_gbnf_string(str(v))}"' for v in schema["enum"]
        )
        rule_name = f"{prefix}-enum"
        ctx.extra_rules.append(f"{rule_name} ::= {alts}")
        return rule_name

    # Const constraint
    if "const" in schema:
        rule_name = f"{prefix}-const"
        ctx.extra_rules.append(
            f'{rule_name} ::= "{_escape_gbnf_string(str(schema["const"]))}"'
        )
        return rule_name

    typ = schema.get("type")

    # anyOf / oneOf
    if typ is None and ("anyOf" in schema or "oneOf" in schema):
        variants = schema.get("anyOf") or schema.get("oneOf", [])
        alt_refs = []
        for i, variant in enumerate(variants):
            ref = _value_rule_for_schema(
                variant, f"{prefix}-v{i}", ctx, json_context=json_context
            )
            alt_refs.append(ref)
        rule_name = f"{prefix}-union"
        ctx.extra_rules.append(f"{rule_name} ::= " + " | ".join(alt_refs))
        return rule_name

    # Type arrays: {"type": ["string", "null"]}
    if isinstance(typ, list):
        alt_refs = []
        for i, t in enumerate(typ):
            ref = _value_rule_for_schema(
                {"type": t}, f"{prefix}-t{i}", ctx, json_context=json_context
            )
            alt_refs.append(ref)
        rule_name = f"{prefix}-multi"
        ctx.extra_rules.append(f"{rule_name} ::= " + " | ".join(alt_refs))
        return rule_name

    # Scalar types
    if typ == "string":
        return "json-string" if json_context else "bare-string"

    if typ == "integer":
        return "json-integer"

    if typ == "number":
        return "json-number"

    if typ == "boolean":
        return "json-bool"

    if typ == "null":
        return "json-null"

    # Array
    if typ == "array":
        items_schema = schema.get("items")
        if items_schema:
            item_ref = _value_rule_for_schema(
                items_schema, f"{prefix}-item", ctx, json_context=True
            )
            arr_rule = f"{prefix}-array"
            body_rule = f"{arr_rule}-body"
            ctx.extra_rules.append(f'{arr_rule} ::= "[" ws {body_rule} ws "]"')
            ctx.extra_rules.append(
                f'{body_rule} ::= "" | {item_ref} (ws "," ws {item_ref})*'
            )
            return arr_rule
        return "json-array"

    # Object
    if typ == "object":
        properties = schema.get("properties")
        if properties:
            return _object_rule_for_schema(
                schema, prefix, ctx, json_context=json_context
            )
        return "json-object"

    # Fallback
    return "json-string" if json_context else "bare-string"


def _object_rule_for_schema(
    schema: dict[str, Any],
    prefix: str,
    ctx: _GrammarContext,
    *,
    json_context: bool = False,
) -> str:
    """Generate a GBNF rule for a JSON object with known properties."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    rule_name = f"{prefix}-obj"

    # Determine if additional properties are allowed
    additional = schema.get("additionalProperties", True)
    allow_additional = (not ctx.strict) and (additional is not False)

    # Generate per-property KV rules
    kv_parts: list[tuple[str, bool]] = []
    for pname, pschema in properties.items():
        safe = _safe_rule_name(pname)
        val_ref = _value_rule_for_schema(
            pschema, f"{prefix}-{safe}", ctx, json_context=True
        )
        kv_rule = f"{prefix}-kv-{safe}"
        ctx.extra_rules.append(
            f'{kv_rule} ::= "\\"{_escape_gbnf_string(pname)}\\"" ws ":" ws {val_ref}'
        )
        kv_parts.append((kv_rule, pname in required))

    if not kv_parts and not allow_additional:
        ctx.extra_rules.append(f'{rule_name} ::= "{{" ws "}}"')
        return rule_name

    if not kv_parts and allow_additional:
        # No declared properties, just arbitrary KV pairs
        ctx.extra_rules.append(
            f'{rule_name} ::= "{{" ws json-object-body ws "}}"'
        )
        return rule_name

    # Build body: required in order, optional with ?, additional at end
    inner_parts: list[str] = []
    for i, (kv_rule, is_required) in enumerate(kv_parts):
        needs_leading_comma = i > 0
        if is_required:
            if needs_leading_comma:
                inner_parts.append(f'ws "," ws {kv_rule}')
            else:
                inner_parts.append(kv_rule)
        else:
            if needs_leading_comma:
                inner_parts.append(f'(ws "," ws {kv_rule})?')
            else:
                inner_parts.append(f'({kv_rule})?')

    if allow_additional:
        inner_parts.append('(ws "," ws json-object-kv)*')

    body_expr = " ".join(inner_parts)
    ctx.extra_rules.append(f'{rule_name} ::= "{{" ws {body_expr} ws "}}"')
    return rule_name


# ── Top-level grammar generation ────────────────────────────────────────

def generate_minimax_tool_grammar(
    tools: list[dict],
    *,
    allow_preamble: bool = False,
    strict: bool = False,
) -> str:
    """Generate a GBNF grammar string from OpenAI-style tool definitions.

    Args:
        tools: List of tool definitions, either in OpenAI format
               ({"type": "function", "function": {...}}) or flat format
               ({"name": "...", "parameters": {...}}).
        allow_preamble: If True, allow free text before the tool call block.
        strict: If True, disallow additionalProperties on all objects
                (matching OpenAI strict mode). If False (default), respect
                the schema's additionalProperties setting.

    Returns:
        A GBNF grammar string suitable for StructuredOutputsParams(grammar=...).
    """
    if not tools:
        raise ValueError("At least one tool must be provided")

    rules: list[str] = []
    invocation_alts: list[str] = []

    # Root rule
    if allow_preamble:
        rules.append(
            'root ::= preamble "<minimax:tool_call>" "\\n" invocations '
            '"</minimax:tool_call>" "\\n"'
        )
        rules.append('preamble ::= [^<]*')
    else:
        rules.append(
            'root ::= "<minimax:tool_call>" "\\n" invocations '
            '"</minimax:tool_call>" "\\n"'
        )

    rules.append('invocations ::= invocation ("\\n" invocation)*')

    # Per-tool rules
    all_extra_rules: list[str] = []

    for tool in tools:
        name, params_schema = _extract_tool_info(tool)
        safe_name = _safe_rule_name(name)
        invoke_rule = f"invoke-{safe_name}"
        invocation_alts.append(invoke_rule)

        # Set up context for this tool
        defs = params_schema.get("$defs", {})
        ctx = _GrammarContext(
            defs_prefix=f"{safe_name}-",
            root_schema=params_schema,
            defs=defs,
            strict=strict,
            _root_ref_rule=f"{safe_name}-root-ref",
        )

        properties = params_schema.get("properties", {})
        required = set(params_schema.get("required", []))

        if not properties:
            rules.append(
                f'{invoke_rule} ::= "<invoke name=\\"{_escape_gbnf_string(name)}\\">'
                f'" "\\n" "</invoke>"'
            )
            all_extra_rules.extend(ctx.extra_rules)
            continue

        # Build per-parameter rules
        param_rules: list[tuple[str, bool]] = []
        for pname, pschema in properties.items():
            safe_pname = _safe_rule_name(pname)
            param_rule = f"{safe_name}-p-{safe_pname}"

            val_ref = _value_rule_for_schema(
                pschema, f"{safe_name}-{safe_pname}", ctx, json_context=False
            )

            rules.append(
                f'{param_rule} ::= "<parameter name=\\"{_escape_gbnf_string(pname)}\\">"'
                f" {val_ref} " f'"</parameter>"'
            )
            param_rules.append((param_rule, pname in required))

        # Parameter sequence: required in schema order, optional with ?
        param_parts: list[str] = []
        for param_rule, is_required in param_rules:
            if is_required:
                param_parts.append(f'{param_rule} "\\n"')
            else:
                param_parts.append(f'({param_rule} "\\n")?')

        params_body = " ".join(param_parts)
        rules.append(
            f'{invoke_rule} ::= "<invoke name=\\"{_escape_gbnf_string(name)}\\">"'
            f' "\\n" {params_body} "</invoke>"'
        )

        all_extra_rules.extend(ctx.extra_rules)

    # Invocation alternatives
    rules.append("invocation ::= " + " | ".join(invocation_alts))

    # Bare string for top-level string parameter values
    rules.append('bare-string ::= [^<]+')

    # Assemble
    all_rules = rules + all_extra_rules + [_JSON_PRIMITIVES]
    return "\n\n".join(all_rules) + "\n"