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
- $defs entries containing bare {"type": "string"} will produce json-string
  (quoted) rules even when referenced from a top-level XML parameter value.
  The MiniMax parser handles quoted strings correctly via json.loads fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ── Shared sub-grammars for JSON value types ────────────────────────────

_JSON_PRIMITIVES = r"""
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
""".strip()

_RULE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _safe_rule_name(name: str) -> str:
    """Sanitize a string for use as a GBNF rule name."""
    return _RULE_NAME_RE.sub("-", name)


def _escape_gbnf_string(s: str) -> str:
    """Escape a string for use inside GBNF double-quoted literals."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _schema_literal(value: Any, *, json_context: bool) -> str:
    """Serialize schema literals for enum/const in the target context."""
    if isinstance(value, str):
        # Top-level XML string params are bare, JSON contexts are quoted.
        return json.dumps(value) if json_context else value
    # JSON spellings for null/bool/number/object/array.
    return json.dumps(value, separators=(",", ":"))


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
    _processed_defs: set[tuple[str, bool]] = field(default_factory=set)
    _root_ref_rule: str = ""
    _root_ref_generated: bool = False


# ── $ref resolution ─────────────────────────────────────────────────────


def _resolve_ref(ref: str, ctx: _GrammarContext, *, json_context: bool) -> str:
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
            val_ref = _value_rule_for_schema(ctx.root_schema, rule_name, ctx, json_context=True)
            if val_ref != rule_name:
                ctx.extra_rules.append(f"{rule_name} ::= {val_ref}")
        return ctx._root_ref_rule

    if ref.startswith("#/$defs/"):
        def_name = ref[len("#/$defs/") :]
        if def_name not in ctx.defs:
            raise ValueError(f"$ref '#/$defs/{def_name}' references undefined definition")
        return _ensure_def_processed(def_name, ctx, json_context=json_context)

    raise ValueError(f"Unsupported $ref: {ref!r}. Only '#/$defs/<name>' and '#' are supported.")


def _ensure_def_processed(
    def_name: str,
    ctx: _GrammarContext,
    *,
    json_context: bool,
) -> str:
    """Ensure a $defs entry has a corresponding GBNF rule. Returns rule name.

    Uses lazy generation with cycle detection so that recursive $defs
    (A references B references A) produce valid recursive GBNF rules.
    """
    mode = "json" if json_context else "bare"
    rule_name = f"{ctx.defs_prefix}defs-{_safe_rule_name(def_name)}-{mode}"
    key = (def_name, json_context)
    if key not in ctx._processed_defs:
        # Mark BEFORE processing to break infinite recursion
        ctx._processed_defs.add(key)
        schema = ctx.defs[def_name]
        val_ref = _value_rule_for_schema(schema, rule_name, ctx, json_context=json_context)
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
        return _resolve_ref(schema["$ref"], ctx, json_context=json_context)

    # Enum constraint
    if "enum" in schema:
        alts = " | ".join(
            f'"{_escape_gbnf_string(_schema_literal(v, json_context=json_context))}"'
            for v in schema["enum"]
        )
        rule_name = f"{prefix}-enum"
        ctx.extra_rules.append(f"{rule_name} ::= {alts}")
        return rule_name

    # Const constraint
    if "const" in schema:
        rule_name = f"{prefix}-const"
        lit = _schema_literal(schema["const"], json_context=json_context)
        ctx.extra_rules.append(f'{rule_name} ::= "{_escape_gbnf_string(lit)}"')
        return rule_name

    typ = schema.get("type")

    # anyOf / oneOf
    if typ is None and ("anyOf" in schema or "oneOf" in schema):
        variants = schema.get("anyOf") or schema.get("oneOf", [])
        alt_refs = []
        for i, variant in enumerate(variants):
            ref = _value_rule_for_schema(variant, f"{prefix}-v{i}", ctx, json_context=json_context)
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
            ctx.extra_rules.append(f'{body_rule} ::= "" | {item_ref} (ws "," ws {item_ref})*')
            return arr_rule
        return "json-array"

    # Object
    if typ == "object":
        properties = schema.get("properties")
        if properties:
            return _object_rule_for_schema(schema, prefix, ctx, json_context=json_context)
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
    additional_kv_rule = "json-object-kv"
    if allow_additional and isinstance(additional, dict):
        additional_val_ref = _value_rule_for_schema(
            additional, f"{prefix}-additional", ctx, json_context=True
        )
        additional_kv_rule = f"{prefix}-kv-additional"
        ctx.extra_rules.append(
            f'{additional_kv_rule} ::= json-string ws ":" ws {additional_val_ref}'
        )

    # Generate per-property KV rules
    kv_parts: list[tuple[str, bool]] = []
    for pname, pschema in properties.items():
        safe = _safe_rule_name(pname)
        val_ref = _value_rule_for_schema(pschema, f"{prefix}-{safe}", ctx, json_context=True)
        kv_rule = f"{prefix}-kv-{safe}"
        ctx.extra_rules.append(
            f'{kv_rule} ::= "\\"{_escape_gbnf_string(pname)}\\"" ws ":" ws {val_ref}'
        )
        kv_parts.append((kv_rule, pname in required))

    # Build object body with two states: whether a previous KV was emitted.
    # This avoids leading-comma failures when initial optional fields are omitted.
    body_prefix = f"{prefix}-obj-body"
    count = len(kv_parts)
    for i in range(count, -1, -1):
        for has_prev in (False, True):
            state = f"{body_prefix}-{i}-{'p' if has_prev else 'n'}"
            if i == count:
                if not allow_additional:
                    expr = '""'
                elif has_prev:
                    expr = f'"" | ws "," ws {additional_kv_rule} (ws "," ws {additional_kv_rule})*'
                else:
                    expr = f'"" | {additional_kv_rule} (ws "," ws {additional_kv_rule})*'
            else:
                kv_rule, is_required = kv_parts[i]
                take_sep = 'ws "," ws ' if has_prev else ""
                take = f"{take_sep}{kv_rule} {body_prefix}-{i + 1}-p"
                if is_required:
                    expr = take
                else:
                    skip_state = f"{body_prefix}-{i + 1}-{'p' if has_prev else 'n'}"
                    expr = f"{skip_state} | {take}"
            ctx.extra_rules.append(f"{state} ::= {expr}")

    ctx.extra_rules.append(f'{rule_name} ::= "{{" ws {body_prefix}-0-n ws "}}"')
    return rule_name


# ── Preamble rules for excluding <minimax:tool_call> ─────────────────────

_TOOL_TAG = "<minimax:tool_call>"
_TOOL_TAG_SUFFIX = _TOOL_TAG[1:]  # "minimax:tool_call>"


def _preamble_rules() -> list[str]:
    """GBNF rules for preamble text that cannot consume '<minimax:tool_call>'.

    The ``pnt-after-lt`` rule matches ``<`` followed by characters that diverge
    from ``minimax:tool_call>`` at each position.  When the exact forbidden
    suffix follows ``<``, every alternative fails and ``pnt-char`` fails,
    so the preamble stops just before that ``<``.
    """
    alts: list[str] = []
    for i in range(len(_TOOL_TAG_SUFFIX)):
        prefix = _TOOL_TAG_SUFFIX[:i]
        reject = _TOOL_TAG_SUFFIX[i]
        if prefix:
            alts.append(f'"{_escape_gbnf_string(prefix)}" [^{reject}]')
        else:
            alts.append(f"[^{reject}]")
    return [
        "preamble ::= pnt-char*",
        "preamble-standalone ::= pnt-char+",
        'pnt-char ::= [^<] | "<" pnt-after-lt',
        "pnt-after-lt ::= " + " | ".join(alts),
    ]


def _standalone_root_alternatives() -> list[str]:
    """Root-rule alternatives for standalone preamble (no tool call block).

    Because ``pnt-char`` cannot consume ``<`` when it starts the forbidden
    tag ``<minimax:tool_call>``, a string ending with a *partial* prefix of
    that tag (e.g. ``hello<mini``) would leave the prefix unconsumed.
    We enumerate every proper prefix of the tag as an explicit suffix
    after ``preamble-standalone`` so those strings are still accepted.
    """
    alts = ["preamble-standalone"]
    for length in range(1, len(_TOOL_TAG)):
        prefix = _TOOL_TAG[:length]
        alts.append(f'preamble-standalone "{_escape_gbnf_string(prefix)}"')
    return alts


# ── Top-level grammar generation ────────────────────────────────────────


def generate_minimax_tool_grammar(
    tools: list[dict],
    *,
    allow_preamble: bool = True,
    require_tool_call: bool = False,
    strict: bool = False,
) -> str:
    """Generate a GBNF grammar string from OpenAI-style tool definitions.

    Args:
        tools: List of tool definitions, either in OpenAI format
               ({"type": "function", "function": {...}}) or flat format
               ({"name": "...", "parameters": {...}}).
        allow_preamble: If True (default), allow free text before the tool call
               block. If False, disallow free text before the tool call block.
        require_tool_call: If True, at least one <invoke> is required inside
               <minimax:tool_call>. If False (default), empty tool call (no
               invocations) is allowed.
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
        tool_call_alt = (
            'preamble "<minimax:tool_call>" "\\n" invocations "</minimax:tool_call>" "\\n"'
        )
        if require_tool_call:
            rules.append("root ::= " + tool_call_alt)
        else:
            standalone_alts = _standalone_root_alternatives()
            all_alts = standalone_alts + [tool_call_alt]
            rules.append("root ::= " + " | ".join(all_alts))
        rules.extend(_preamble_rules())
    else:
        rules.append(
            'root ::= "<minimax:tool_call>" "\\n" invocations "</minimax:tool_call>" "\\n"'
        )

    if require_tool_call:
        rules.append('invocations ::= invocation ("\\n" invocation)*')
    else:
        rules.append('invocations ::= "" | invocation ("\\n" invocation)*')

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
                f" {val_ref} "
                f'"</parameter>"'
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
    rules.append("bare-string ::= [^<]*")

    # Assemble
    all_rules = rules + all_extra_rules + [_JSON_PRIMITIVES]
    return "\n\n".join(all_rules) + "\n"
