"""
Tests for minimax_m2_grammar.py

Uses xgrammar to compile grammars and validate strings against them.

Requirements:
    pip install xgrammar
"""

import unittest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

from minimax_m2_grammar import generate_minimax_tool_grammar


def accepts(grammar_str: str, input_str: str) -> bool:
    return _is_grammar_accept_string(grammar_str, input_str)


def compiles(grammar_str: str) -> xgr.Grammar:
    return xgr.Grammar.from_ebnf(grammar_str)


# ── Helpers ─────────────────────────────────────────────────────────────

def tool_call(*invocations: str) -> str:
    body = "\n".join(invocations)
    return f"<minimax:tool_call>\n{body}</minimax:tool_call>\n"


def invoke(name: str, *params: str) -> str:
    body = "\n".join(params)
    return f'<invoke name="{name}">\n{body}\n</invoke>'


def param(name: str, value: str) -> str:
    return f'<parameter name="{name}">{value}</parameter>'


# ═══════════════════════════════════════════════════════════════════════
# Compilation tests
# ═══════════════════════════════════════════════════════════════════════


class TestCompilation(unittest.TestCase):

    def test_single_tool(self):
        tools = [{"name": "f", "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }}]
        compiles(generate_minimax_tool_grammar(tools))

    def test_multiple_tools(self):
        tools = [
            {"name": "a", "parameters": {"type": "object",
                "properties": {"x": {"type": "string"}}, "required": ["x"]}},
            {"name": "b", "parameters": {"type": "object",
                "properties": {"y": {"type": "integer"}}, "required": ["y"]}},
        ]
        compiles(generate_minimax_tool_grammar(tools))

    def test_no_params(self):
        tools = [{"name": "ping", "parameters": {"type": "object", "properties": {}}}]
        compiles(generate_minimax_tool_grammar(tools))

    def test_nested_object(self):
        tools = [{"name": "f", "parameters": {"type": "object", "properties": {
            "addr": {"type": "object", "properties": {
                "street": {"type": "string"}, "city": {"type": "string"}
            }, "required": ["street", "city"]}
        }, "required": ["addr"]}}]
        compiles(generate_minimax_tool_grammar(tools))

    def test_anyof(self):
        tools = [{"name": "f", "parameters": {"type": "object", "properties": {
            "v": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]}
        }, "required": ["v"]}}]
        compiles(generate_minimax_tool_grammar(tools))

    def test_type_array(self):
        tools = [{"name": "f", "parameters": {"type": "object", "properties": {
            "x": {"type": ["string", "null"]}
        }, "required": ["x"]}}]
        compiles(generate_minimax_tool_grammar(tools))

    def test_openai_format(self):
        tools = [{"type": "function", "function": {
            "name": "f", "parameters": {"type": "object",
                "properties": {"x": {"type": "string"}}, "required": ["x"]}
        }}]
        compiles(generate_minimax_tool_grammar(tools))

    def test_preamble(self):
        tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {"x": {"type": "string"}}, "required": ["x"]}}]
        compiles(generate_minimax_tool_grammar(tools, allow_preamble=True))

    def test_ref_defs(self):
        tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {"s": {"$ref": "#/$defs/status"}},
            "$defs": {"status": {"type": "string", "enum": ["on", "off"]}},
            "required": ["s"]}}]
        compiles(generate_minimax_tool_grammar(tools))

    def test_recursive_ref(self):
        tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {
                "value": {"type": "string"},
                "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
            },
            "$defs": {"node": {"type": "object", "properties": {
                "value": {"type": "string"},
                "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
            }, "required": ["value"]}},
            "required": ["value"]}}]
        compiles(generate_minimax_tool_grammar(tools))

    def test_root_ref(self):
        tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {
                "value": {"type": "string"},
                "children": {"type": "array", "items": {"$ref": "#"}},
            },
            "required": ["value"]}}]
        compiles(generate_minimax_tool_grammar(tools))

    def test_strict_mode(self):
        tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {"x": {"type": "string"}}, "required": ["x"]}}]
        compiles(generate_minimax_tool_grammar(tools, strict=True))

    def test_empty_tools_raises(self):
        with self.assertRaises(ValueError):
            generate_minimax_tool_grammar([])


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: basic tool calls
# ═══════════════════════════════════════════════════════════════════════


class TestSimpleAcceptance(unittest.TestCase):

    def setUp(self):
        self.tools = [{"name": "get_weather", "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        }}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_valid(self):
        s = tool_call(invoke("get_weather",
            param("location", "San Francisco"), param("unit", "celsius")))
        self.assertTrue(accepts(self.grammar, s))

    def test_other_enum(self):
        s = tool_call(invoke("get_weather",
            param("location", "Tokyo"), param("unit", "fahrenheit")))
        self.assertTrue(accepts(self.grammar, s))

    def test_rejects_invalid_enum(self):
        s = tool_call(invoke("get_weather",
            param("location", "Paris"), param("unit", "kelvin")))
        self.assertFalse(accepts(self.grammar, s))

    def test_rejects_missing_required(self):
        s = tool_call(invoke("get_weather", param("location", "London")))
        self.assertFalse(accepts(self.grammar, s))

    def test_rejects_wrong_function(self):
        s = tool_call(invoke("get_temp",
            param("location", "Berlin"), param("unit", "celsius")))
        self.assertFalse(accepts(self.grammar, s))

    def test_rejects_empty_block(self):
        self.assertFalse(accepts(self.grammar, "<minimax:tool_call>\n</minimax:tool_call>\n"))

    def test_rejects_bare_text(self):
        self.assertFalse(accepts(self.grammar, "Hello world"))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: optional parameters
# ═══════════════════════════════════════════════════════════════════════


class TestOptionalParams(unittest.TestCase):

    def setUp(self):
        self.tools = [{"name": "search", "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
                "lang": {"type": "string"},
            },
            "required": ["query"],
        }}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_required_only(self):
        s = tool_call(invoke("search", param("query", "vllm")))
        self.assertTrue(accepts(self.grammar, s))

    def test_required_plus_one(self):
        s = tool_call(invoke("search",
            param("query", "xgrammar"), param("max_results", "10")))
        self.assertTrue(accepts(self.grammar, s))

    def test_all_params(self):
        s = tool_call(invoke("search",
            param("query", "grammar"), param("max_results", "5"),
            param("lang", "en")))
        self.assertTrue(accepts(self.grammar, s))

    def test_rejects_wrong_order(self):
        s = tool_call(invoke("search",
            param("query", "test"), param("lang", "en"),
            param("max_results", "5")))
        self.assertFalse(accepts(self.grammar, s))

    def test_skip_first_optional(self):
        s = tool_call(invoke("search",
            param("query", "test"), param("lang", "en")))
        self.assertTrue(accepts(self.grammar, s))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: multiple tools & parallel calls
# ═══════════════════════════════════════════════════════════════════════


class TestMultipleTools(unittest.TestCase):

    def setUp(self):
        self.tools = [
            {"name": "get_weather", "parameters": {"type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]}},
            {"name": "search", "parameters": {"type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]}},
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_first_tool(self):
        self.assertTrue(accepts(self.grammar,
            tool_call(invoke("get_weather", param("location", "NYC")))))

    def test_second_tool(self):
        self.assertTrue(accepts(self.grammar,
            tool_call(invoke("search", param("query", "hello")))))

    def test_parallel(self):
        self.assertTrue(accepts(self.grammar, tool_call(
            invoke("get_weather", param("location", "NYC")),
            invoke("search", param("query", "weather nyc")))))

    def test_rejects_unknown(self):
        self.assertFalse(accepts(self.grammar,
            tool_call(invoke("delete", param("x", "y")))))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: value types
# ═══════════════════════════════════════════════════════════════════════


class TestValueTypes(unittest.TestCase):

    def setUp(self):
        self.tools = [{"name": "t", "parameters": {"type": "object",
            "properties": {
                "s": {"type": "string"}, "i": {"type": "integer"},
                "n": {"type": "number"}, "b": {"type": "boolean"},
            },
            "required": ["s", "i", "n", "b"]}}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_valid(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("t",
            param("s", "hello"), param("i", "42"),
            param("n", "3.14"), param("b", "true")))))

    def test_negative_int(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("t",
            param("s", "x"), param("i", "-7"),
            param("n", "0"), param("b", "false")))))

    def test_scientific(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("t",
            param("s", "x"), param("i", "1"),
            param("n", "1.5e10"), param("b", "true")))))

    def test_rejects_string_for_int(self):
        self.assertFalse(accepts(self.grammar, tool_call(invoke("t",
            param("s", "x"), param("i", "abc"),
            param("n", "1.0"), param("b", "true")))))

    def test_rejects_string_for_bool(self):
        self.assertFalse(accepts(self.grammar, tool_call(invoke("t",
            param("s", "x"), param("i", "1"),
            param("n", "1.0"), param("b", "yes")))))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: array parameters
# ═══════════════════════════════════════════════════════════════════════


class TestArrayParams(unittest.TestCase):

    def setUp(self):
        self.tools = [{"name": "batch", "parameters": {"type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "integer"}},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ids"]}}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_integer_array(self):
        self.assertTrue(accepts(self.grammar,
            tool_call(invoke("batch", param("ids", "[1, 2, 3]")))))

        self.assertFalse(accepts(self.grammar,
            tool_call(invoke("batch", param("ids", "[1, 2, 3.0]")))))

        self.assertFalse(accepts(self.grammar,
            tool_call(invoke("batch", param("ids", "[\"1\", \"2\", \"3\"]")))))

    def test_empty_array(self):
        self.assertTrue(accepts(self.grammar,
            tool_call(invoke("batch", param("ids", "[]")))))

        self.assertFalse(accepts(self.grammar,
            tool_call(invoke("batch", param("ids", "")))))

    def test_string_array_uses_json_string(self):
        """String items inside JSON arrays must be JSON-quoted."""
        self.assertTrue(accepts(self.grammar, tool_call(invoke("batch",
            param("ids", "[1]"),
            param("tags", '["foo", "bar"]')))))

    def test_rejects_unquoted_string_in_array(self):
        """Bare (unquoted) strings inside a JSON array should be rejected."""
        self.assertFalse(accepts(self.grammar, tool_call(invoke("batch",
            param("ids", "[1]"),
            param("tags", "[foo, bar]")))))

    def test_rejects_wrong_item_type(self):
        self.assertFalse(accepts(self.grammar,
            tool_call(invoke("batch", param("ids", '["not_int"]')))))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: nested objects
# ═══════════════════════════════════════════════════════════════════════


class TestNestedObject(unittest.TestCase):

    def setUp(self):
        self.tools = [{"name": "create_user", "parameters": {"type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"type": "object", "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                }, "required": ["street", "city"]},
            },
            "required": ["name", "address"]}}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_valid(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("create_user",
            param("name", "Alice"),
            param("address", '{"street": "123 Main", "city": "Springfield"}')))))

    def test_with_optional_field(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("create_user",
            param("name", "Bob"),
            param("address",
                  '{"street": "456 Oak", "city": "Portland", "zip": "97201"}')))))

    def test_rejects_missing_required_nested(self):
        self.assertFalse(accepts(self.grammar, tool_call(invoke("create_user",
            param("name", "Eve"),
            param("address", '{"street": "789 Pine"}')))))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: anyOf, const, type arrays
# ═══════════════════════════════════════════════════════════════════════


class TestAnyOf(unittest.TestCase):

    def setUp(self):
        self.tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {"value": {"anyOf": [
                {"type": "string"}, {"type": "integer"}, {"type": "null"},
            ]}},
            "required": ["value"]}}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_string(self):
        self.assertTrue(accepts(self.grammar,
            tool_call(invoke("f", param("value", "hello")))))

    def test_integer(self):
        self.assertTrue(accepts(self.grammar,
            tool_call(invoke("f", param("value", "42")))))

    def test_null(self):
        self.assertTrue(accepts(self.grammar,
            tool_call(invoke("f", param("value", "null")))))


class TestConstValue(unittest.TestCase):

    def setUp(self):
        self.tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {
                "action": {"const": "execute"},
                "target": {"type": "string"},
            },
            "required": ["action", "target"]}}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_correct(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("f",
            param("action", "execute"), param("target", "srv1")))))

    def test_rejects_wrong(self):
        self.assertFalse(accepts(self.grammar, tool_call(invoke("f",
            param("action", "delete"), param("target", "srv1")))))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: $ref / $defs (non-recursive)
# ═══════════════════════════════════════════════════════════════════════


class TestRefDefs(unittest.TestCase):
    """Non-recursive $ref to $defs entries."""

    def setUp(self):
        self.tools = [{"name": "order", "parameters": {
            "type": "object",
            "properties": {
                "status": {"$ref": "#/$defs/status_enum"},
                "item": {"$ref": "#/$defs/item_obj"},
            },
            "$defs": {
                "status_enum": {"type": "string", "enum": ["pending", "shipped", "delivered"]},
                "item_obj": {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "integer"},
                }, "required": ["name", "quantity"]},
            },
            "required": ["status", "item"],
        }}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_valid(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("order",
            param("status", "shipped"),
            param("item", '{"name": "Widget", "quantity": 5}')))))

    def test_rejects_invalid_enum_from_def(self):
        self.assertFalse(accepts(self.grammar, tool_call(invoke("order",
            param("status", "cancelled"),
            param("item", '{"name": "Widget", "quantity": 5}')))))

    def test_rejects_missing_required_in_def_obj(self):
        self.assertFalse(accepts(self.grammar, tool_call(invoke("order",
            param("status", "pending"),
            param("item", '{"name": "Widget"}')))))


class TestCrossRefDefs(unittest.TestCase):
    """$defs entries that reference other $defs entries."""

    def setUp(self):
        self.tools = [{"name": "f", "parameters": {
            "type": "object",
            "properties": {
                "data": {"$ref": "#/$defs/wrapper"},
            },
            "$defs": {
                "inner": {"type": "object", "properties": {
                    "value": {"type": "integer"},
                }, "required": ["value"]},
                "wrapper": {"type": "object", "properties": {
                    "payload": {"$ref": "#/$defs/inner"},
                    "label": {"type": "string"},
                }, "required": ["payload", "label"]},
            },
            "required": ["data"],
        }}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_valid(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("f",
            param("data", '{"payload": {"value": 42}, "label": "test"}')))))

    def test_rejects_wrong_inner_type(self):
        self.assertFalse(accepts(self.grammar, tool_call(invoke("f",
            param("data", '{"payload": {"value": "not_int"}, "label": "test"}')))))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: recursive $ref via $defs
# ═══════════════════════════════════════════════════════════════════════


class TestRecursiveRef(unittest.TestCase):
    """Recursive schemas via $defs (tree structures)."""

    def setUp(self):
        self.tools = [{"name": "process", "parameters": {
            "type": "object",
            "properties": {
                "tree": {"$ref": "#/$defs/node"},
            },
            "$defs": {
                "node": {"type": "object", "properties": {
                    "value": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                }, "required": ["value"]},
            },
            "required": ["tree"],
        }}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_leaf_node(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("process",
            param("tree", '{"value": "leaf"}')))))

    def test_one_level(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("process",
            param("tree",
                '{"value": "root", "children": [{"value": "child"}]}')))))

    def test_two_levels(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("process",
            param("tree",
                '{"value": "root", "children": ['
                '{"value": "a", "children": [{"value": "a1"}]}, '
                '{"value": "b"}'
                ']}')))))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: $ref "#" (root self-reference)
# ═══════════════════════════════════════════════════════════════════════


class TestRootRef(unittest.TestCase):
    """$ref "#" references the root parameters schema."""

    def setUp(self):
        self.tools = [{"name": "walk", "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "next": {"$ref": "#"},
            },
            "required": ["value"],
        }}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_no_recursion(self):
        """Just the base case, no 'next' field."""
        self.assertTrue(accepts(self.grammar, tool_call(invoke("walk",
            param("value", "start")))))

    def test_one_level(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("walk",
            param("value", "start"),
            param("next", '{"value": "end"}')))))

    def test_two_levels(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("walk",
            param("value", "a"),
            param("next", '{"value": "b", "next": {"value": "c"}}')))))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: json_context (strings in arrays use json-string)
# ═══════════════════════════════════════════════════════════════════════


class TestJsonContext(unittest.TestCase):
    """Verify that string handling differs between top-level and JSON contexts."""

    def setUp(self):
        self.tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {
                "bare": {"type": "string"},
                "arr": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["bare", "arr"]}}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_bare_string_unquoted(self):
        """Top-level string params are bare (unquoted)."""
        self.assertTrue(accepts(self.grammar, tool_call(invoke("f",
            param("bare", "hello world"),
            param("arr", "[]")))))

    def test_array_strings_quoted(self):
        """Strings inside JSON arrays must be quoted."""
        self.assertTrue(accepts(self.grammar, tool_call(invoke("f",
            param("bare", "hi"),
            param("arr", '["a", "b"]')))))

    def test_rejects_unquoted_in_array(self):
        """Unquoted strings inside a JSON array should fail."""
        self.assertFalse(accepts(self.grammar, tool_call(invoke("f",
            param("bare", "hi"),
            param("arr", "[a, b]")))))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: strict mode and additionalProperties
# ═══════════════════════════════════════════════════════════════════════


class TestStrictMode(unittest.TestCase):
    """strict=True disallows additional properties."""

    def setUp(self):
        self.tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {
                "data": {"type": "object", "properties": {
                    "name": {"type": "string"},
                }, "required": ["name"]},
            },
            "required": ["data"]}}]

    def test_strict_rejects_extra_key(self):
        grammar = generate_minimax_tool_grammar(self.tools, strict=True)
        s = tool_call(invoke("f",
            param("data", '{"name": "Alice", "age": 30}')))
        self.assertFalse(accepts(grammar, s))

    def test_strict_accepts_declared_only(self):
        grammar = generate_minimax_tool_grammar(self.tools, strict=True)
        s = tool_call(invoke("f", param("data", '{"name": "Alice"}')))
        self.assertTrue(accepts(grammar, s))


class TestAdditionalProperties(unittest.TestCase):
    """strict=False respects additionalProperties from schema."""

    def setUp(self):
        self.tools_allow = [{"name": "f", "parameters": {"type": "object",
            "properties": {
                "data": {"type": "object", "properties": {
                    "name": {"type": "string"},
                }, "required": ["name"],
                "additionalProperties": True},
            },
            "required": ["data"]}}]
        self.tools_deny = [{"name": "f", "parameters": {"type": "object",
            "properties": {
                "data": {"type": "object", "properties": {
                    "name": {"type": "string"},
                }, "required": ["name"],
                "additionalProperties": False},
            },
            "required": ["data"]}}]

    def test_allows_extra_when_permitted(self):
        grammar = generate_minimax_tool_grammar(self.tools_allow, strict=False)
        s = tool_call(invoke("f",
            param("data", '{"name": "Alice", "age": 30}')))
        self.assertTrue(accepts(grammar, s))

    def test_allows_no_extra(self):
        grammar = generate_minimax_tool_grammar(self.tools_allow, strict=False)
        s = tool_call(invoke("f", param("data", '{"name": "Alice"}')))
        self.assertTrue(accepts(grammar, s))

    def test_denies_extra_when_schema_says_false(self):
        grammar = generate_minimax_tool_grammar(self.tools_deny, strict=False)
        s = tool_call(invoke("f",
            param("data", '{"name": "Alice", "age": 30}')))
        self.assertFalse(accepts(grammar, s))

    def test_default_allows_extra(self):
        """When additionalProperties is absent, default is to allow (non-strict)."""
        tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {
                "data": {"type": "object", "properties": {
                    "name": {"type": "string"},
                }, "required": ["name"]},
            },
            "required": ["data"]}}]
        grammar = generate_minimax_tool_grammar(tools, strict=False)
        s = tool_call(invoke("f",
            param("data", '{"name": "Alice", "extra": "val"}')))
        self.assertTrue(accepts(grammar, s))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: preamble
# ═══════════════════════════════════════════════════════════════════════


class TestPreamble(unittest.TestCase):

    def setUp(self):
        self.tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {"x": {"type": "string"}}, "required": ["x"]}}]

    def test_no_preamble_rejects_leading_text(self):
        g = generate_minimax_tool_grammar(self.tools, allow_preamble=False)
        self.assertFalse(accepts(g,
            "Sure!\n" + tool_call(invoke("f", param("x", "hi")))))

    def test_preamble_accepts_leading_text(self):
        g = generate_minimax_tool_grammar(self.tools, allow_preamble=True)
        self.assertTrue(accepts(g,
            "Sure!" + tool_call(invoke("f", param("x", "hi")))))

    def test_preamble_accepts_empty(self):
        g = generate_minimax_tool_grammar(self.tools, allow_preamble=True)
        self.assertTrue(accepts(g, tool_call(invoke("f", param("x", "hi")))))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: no params
# ═══════════════════════════════════════════════════════════════════════


class TestNoParams(unittest.TestCase):

    def setUp(self):
        self.tools = [{"name": "ping", "parameters": {
            "type": "object", "properties": {}}}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_accepts(self):
        self.assertTrue(accepts(self.grammar,
            tool_call('<invoke name="ping">\n</invoke>')))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: complex real-world example
# ═══════════════════════════════════════════════════════════════════════


class TestComplexRealWorld(unittest.TestCase):

    def setUp(self):
        self.tools = [
            {"type": "function", "function": {
                "name": "create_github_issue",
                "parameters": {"type": "object", "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "priority": {"type": "string",
                        "enum": ["low", "medium", "high", "critical"]},
                }, "required": ["title", "body"]}}},
            {"type": "function", "function": {
                "name": "search_issues",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"},
                    "state": {"type": "string", "enum": ["open", "closed", "all"]},
                    "per_page": {"type": "integer"},
                }, "required": ["query"]}}},
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_create_minimal(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke(
            "create_github_issue",
            param("title", "Bug"), param("body", "Details")))))

    def test_create_with_optionals(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke(
            "create_github_issue",
            param("title", "Feature"),
            param("body", "Add XML grammar support"),
            param("labels", '["enhancement"]'),
            param("priority", "high")))))

    def test_search(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke(
            "search_issues",
            param("query", "minimax"), param("state", "open"),
            param("per_page", "20")))))

    def test_parallel(self):
        self.assertTrue(accepts(self.grammar, tool_call(
            invoke("create_github_issue",
                param("title", "Bug"), param("body", "Details")),
            invoke("search_issues", param("query", "related")))))

    def test_rejects_invalid_priority(self):
        self.assertFalse(accepts(self.grammar, tool_call(invoke(
            "create_github_issue",
            param("title", "Bug"), param("body", "Details"),
            param("priority", "urgent")))))

    def test_rejects_invalid_state(self):
        self.assertFalse(accepts(self.grammar, tool_call(invoke(
            "search_issues",
            param("query", "test"), param("state", "pending")))))


# ═══════════════════════════════════════════════════════════════════════
# Error handling: unsupported $ref formats
# ═══════════════════════════════════════════════════════════════════════


class TestUnsupportedRef(unittest.TestCase):

    def test_external_uri_raises(self):
        tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {"x": {"$ref": "https://example.com/schema.json"}},
            "required": ["x"]}}]
        with self.assertRaises(ValueError) as cm:
            generate_minimax_tool_grammar(tools)
        self.assertIn("Unsupported $ref", str(cm.exception))

    def test_relative_path_raises(self):
        tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {"x": {"$ref": "#/properties/y"}},
            "required": ["x"]}}]
        with self.assertRaises(ValueError) as cm:
            generate_minimax_tool_grammar(tools)
        self.assertIn("Unsupported $ref", str(cm.exception))

    def test_undefined_def_raises(self):
        tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {"x": {"$ref": "#/$defs/nonexistent"}},
            "$defs": {},
            "required": ["x"]}}]
        with self.assertRaises(ValueError) as cm:
            generate_minimax_tool_grammar(tools)
        self.assertIn("undefined", str(cm.exception))


# ═══════════════════════════════════════════════════════════════════════
# Known limitations (documented expected failures)
# ═══════════════════════════════════════════════════════════════════════


class TestKnownLimitations(unittest.TestCase):

    @unittest.expectedFailure
    def test_allof_not_supported(self):
        """allOf is not handled — falls through to bare-string fallback."""
        tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {
                "data": {"allOf": [
                    {"type": "object", "properties": {"a": {"type": "string"}},
                     "required": ["a"]},
                    {"type": "object", "properties": {"b": {"type": "integer"}},
                     "required": ["b"]},
                ]},
            },
            "required": ["data"]}}]
        grammar = generate_minimax_tool_grammar(tools)
        # allOf should produce {"a": "x", "b": 1} but grammar won't enforce
        # object structure — it falls back to bare-string, so this JSON obj
        # with < would be rejected but plain text accepted. This test expects
        # proper allOf support, which we don't have.
        s = tool_call(invoke("f", param("data", '{"a": "x", "b": 1}')))
        self.assertTrue(accepts(grammar, s))
        s = tool_call(invoke("f", param("data", "Hello, world!")))
        self.assertFalse(accepts(grammar, s))
        s = tool_call(invoke("f", param("data", "{}")))
        self.assertFalse(accepts(grammar, s))

    @unittest.expectedFailure
    def test_all_optional_object_props_with_additional_comma_issue(self):
        """When ALL declared props are optional, first absent + additional
        present produces a leading comma. This is a known grammar limitation."""
        tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {
                "data": {"type": "object", "properties": {
                    "opt_a": {"type": "string"},
                    "opt_b": {"type": "string"},
                }},  # no required, additionalProperties defaults True
            },
            "required": ["data"]}}]
        grammar = generate_minimax_tool_grammar(tools, strict=False)
        # With all optional props absent, additional property would need
        # no leading comma, but grammar always generates leading comma
        s = tool_call(invoke("f", param("data", '{"extra": "val"}')))
        self.assertTrue(accepts(grammar, s))


# ═══════════════════════════════════════════════════════════════════════
# Determinism
# ═══════════════════════════════════════════════════════════════════════


class TestDeterminism(unittest.TestCase):

    def test_same_output(self):
        tools = [{"name": "f", "parameters": {"type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"]}}]
        g1 = generate_minimax_tool_grammar(tools)
        g2 = generate_minimax_tool_grammar(tools)
        self.assertEqual(g1, g2)


if __name__ == "__main__":
    unittest.main(verbosity=2)