"""
Tests for minimax_m2_tool_def_to_gbnf (gbnf.py).

Uses xgrammar to compile grammars and validate strings against them.

Requirements:
    pip install xgrammar
"""

import unittest

import xgrammar as xgr
from minimax_m2_tool_def_to_gbnf import generate_minimax_tool_grammar
from xgrammar.testing import (
    _get_masked_tokens_from_bitmask,
    _get_matcher_from_grammar_and_tokenizer_info,
)

from .tokenizer import minimax_tokenizer

tokenizer_info = xgr.TokenizerInfo.from_huggingface(minimax_tokenizer)
compiler = xgr.GrammarCompiler(tokenizer_info)


def accepts(grammar_str: str, input_str: str) -> bool:
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar_str, tokenizer_info)
    tokens = minimax_tokenizer.encode(input_str)
    for token in tokens:
        if not matcher.accept_token(token):
            return False
    return matcher.accept_token(tokenizer_info.stop_token_ids[-1])


def compiles(grammar_str: str) -> xgr.CompiledGrammar:
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    return compiler.compile_grammar(grammar)


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
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            }
        ]
        compiles(generate_minimax_tool_grammar(tools))

    def test_multiple_tools(self):
        tools = [
            {
                "name": "a",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            },
            {
                "name": "b",
                "parameters": {
                    "type": "object",
                    "properties": {"y": {"type": "integer"}},
                    "required": ["y"],
                },
            },
        ]
        compiles(generate_minimax_tool_grammar(tools))

    def test_no_params(self):
        tools = [{"name": "ping", "parameters": {"type": "object", "properties": {}}}]
        compiles(generate_minimax_tool_grammar(tools))

    def test_nested_object(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "addr": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                            },
                            "required": ["street", "city"],
                        }
                    },
                    "required": ["addr"],
                },
            }
        ]
        compiles(generate_minimax_tool_grammar(tools))

    def test_anyof(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "v": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]}
                    },
                    "required": ["v"],
                },
            }
        ]
        compiles(generate_minimax_tool_grammar(tools))

    def test_type_array(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": ["string", "null"]}},
                    "required": ["x"],
                },
            }
        ]
        compiles(generate_minimax_tool_grammar(tools))

    def test_openai_format(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"],
                    },
                },
            }
        ]
        compiles(generate_minimax_tool_grammar(tools))

    def test_preamble(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            }
        ]
        compiles(generate_minimax_tool_grammar(tools, allow_preamble=True))

    def test_ref_defs(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"s": {"$ref": "#/$defs/status"}},
                    "$defs": {"status": {"type": "string", "enum": ["on", "off"]}},
                    "required": ["s"],
                },
            }
        ]
        compiles(generate_minimax_tool_grammar(tools))

    def test_recursive_ref(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                    },
                    "$defs": {
                        "node": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                            },
                            "required": ["value"],
                        }
                    },
                    "required": ["value"],
                },
            }
        ]
        compiles(generate_minimax_tool_grammar(tools))

    def test_root_ref(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "children": {"type": "array", "items": {"$ref": "#"}},
                    },
                    "required": ["value"],
                },
            }
        ]
        compiles(generate_minimax_tool_grammar(tools))

    def test_strict_mode(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            }
        ]
        compiles(generate_minimax_tool_grammar(tools, strict=True))

    def test_empty_tools_raises(self):
        with self.assertRaises(ValueError):
            generate_minimax_tool_grammar([])


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: basic tool calls
# ═══════════════════════════════════════════════════════════════════════


class TestSimpleAcceptance(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_valid(self):
        s = tool_call(
            invoke("get_weather", param("location", "San Francisco"), param("unit", "celsius"))
        )
        self.assertTrue(accepts(self.grammar, s))

    def test_other_enum(self):
        s = tool_call(
            invoke("get_weather", param("location", "Tokyo"), param("unit", "fahrenheit"))
        )
        self.assertTrue(accepts(self.grammar, s))

    def test_rejects_invalid_enum(self):
        s = tool_call(invoke("get_weather", param("location", "Paris"), param("unit", "kelvin")))
        self.assertFalse(accepts(self.grammar, s))

    def test_rejects_missing_required(self):
        s = tool_call(invoke("get_weather", param("location", "London")))
        self.assertFalse(accepts(self.grammar, s))

    def test_rejects_wrong_function(self):
        s = tool_call(invoke("get_temp", param("location", "Berlin"), param("unit", "celsius")))
        self.assertFalse(accepts(self.grammar, s))

    def test_accepts_empty_block(self):
        self.assertTrue(accepts(self.grammar, "<minimax:tool_call>\n</minimax:tool_call>\n"))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: optional parameters
# ═══════════════════════════════════════════════════════════════════════


class TestOptionalParams(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                        "lang": {"type": "string"},
                    },
                    "required": ["query"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_required_only(self):
        s = tool_call(invoke("search", param("query", "vllm")))
        self.assertTrue(accepts(self.grammar, s))

    def test_required_plus_one(self):
        s = tool_call(invoke("search", param("query", "xgrammar"), param("max_results", "10")))
        self.assertTrue(accepts(self.grammar, s))

    def test_all_params(self):
        s = tool_call(
            invoke(
                "search", param("query", "grammar"), param("max_results", "5"), param("lang", "en")
            )
        )
        self.assertTrue(accepts(self.grammar, s))

    def test_rejects_wrong_order(self):
        s = tool_call(
            invoke("search", param("query", "test"), param("lang", "en"), param("max_results", "5"))
        )
        self.assertFalse(accepts(self.grammar, s))

    def test_skip_first_optional(self):
        s = tool_call(invoke("search", param("query", "test"), param("lang", "en")))
        self.assertTrue(accepts(self.grammar, s))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: multiple tools & parallel calls
# ═══════════════════════════════════════════════════════════════════════


class TestMultipleTools(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
            {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "request.get",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
            {
                "name": "request.post",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}, "body": {"type": "string"}},
                    "required": ["url", "body"],
                },
            },
            {
                "name": "another_tool",
                "parameters": {
                    "type": "object",
                    "properties": {"property.name": {"type": "string"}},
                    "required": ["property.name"],
                },
            },
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_first_tool(self):
        self.assertTrue(
            accepts(self.grammar, tool_call(invoke("get_weather", param("location", "NYC"))))
        )

    def test_second_tool(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("search", param("query", "hello")))))

    def test_tool_with_special_char_in_name(self):
        self.assertTrue(
            accepts(
                self.grammar, tool_call(invoke("request.get", param("url", "https://example.com")))
            )
        )
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "request.post",
                        param("url", "https://example.com"),
                        param("body", "Hello, world!"),
                    )
                ),
            )
        )

    def test_tool_with_special_char_in_property_name(self):
        self.assertTrue(
            accepts(
                self.grammar, tool_call(invoke("another_tool", param("property.name", "value")))
            )
        )

    def test_parallel(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke("get_weather", param("location", "NYC")),
                    invoke("search", param("query", "weather nyc")),
                ),
            )
        )

        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke("get_weather", param("location", "NYC")),
                    invoke("delete", param("query", "x")),
                ),
            )
        )

    def test_rejects_unknown(self):
        self.assertFalse(accepts(self.grammar, tool_call(invoke("delete", param("x", "y")))))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: multiple invocations (same tool_call) — positive & negative
# ═══════════════════════════════════════════════════════════════════════


class TestMultipleInvocations(unittest.TestCase):
    """Tests for multiple <invoke> blocks in a single <minimax:tool_call>.

    Positive: multiple valid invocations (same or different tools).
    Negative: params valid for tool A used when invoking tool B, wrong param
    names per tool, missing required params in second invocation, etc.
    """

    def setUp(self):
        self.tools = [
            {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
            {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "request.post",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["url", "body"],
                },
            },
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    # ── Positive: multiple invocations ─────────────────────────────────

    def test_two_invocations_different_tools_valid(self):
        """Two invocations, each with correct tool and params."""
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke("get_weather", param("location", "NYC")),
                    invoke("search", param("query", "weather")),
                ),
            )
        )

    def test_three_invocations_mixed_tools_valid(self):
        """Three invocations with correct params for each tool."""
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke("get_weather", param("location", "London")),
                    invoke("search", param("query", "news")),
                    invoke(
                        "request.post",
                        param("url", "https://api.example.com"),
                        param("body", "{}"),
                    ),
                ),
            )
        )

    def test_two_invocations_same_tool_valid(self):
        """Multiple invocations of the same tool, each with valid params."""
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke("search", param("query", "first")),
                    invoke("search", param("query", "second")),
                ),
            )
        )

    def test_single_invocation_still_valid(self):
        """Single invocation in tool_call still accepted."""
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(invoke("get_weather", param("location", "Paris"))),
            )
        )

    # ── Negative: params for tool A used with tool B ────────────────────

    def test_rejects_search_with_location_param(self):
        """search expects 'query'; using get_weather's 'location' is invalid."""
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke("get_weather", param("location", "NYC")),
                    invoke("search", param("location", "NYC")),
                ),
            )
        )

    def test_rejects_get_weather_with_query_param(self):
        """get_weather expects 'location'; using search's 'query' is invalid."""
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke("search", param("query", "hello")),
                    invoke("get_weather", param("query", "NYC")),
                ),
            )
        )

    def test_rejects_request_post_with_location_and_body(self):
        """request.post expects 'url' and 'body'; using 'location' (get_weather) is invalid."""
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke("request.post", param("location", "NYC"), param("body", "{}")),
                ),
            )
        )

    def test_rejects_get_weather_with_url_and_body_params(self):
        """get_weather expects 'location'; using request.post's 'url'/'body' is invalid."""
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke("get_weather", param("url", "x"), param("body", "y")),
                ),
            )
        )

    def test_rejects_second_invocation_wrong_params(self):
        """First invoke correct; second invoke same tool but wrong param name."""
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke("get_weather", param("location", "NYC")),
                    invoke("get_weather", param("query", "Boston")),
                ),
            )
        )

    def test_rejects_mixed_params_on_single_invoke(self):
        """One invoke with param from tool A and param from tool B."""
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "search",
                        param("query", "test"),
                        param("location", "NYC"),
                    ),
                ),
            )
        )

    def test_rejects_request_post_missing_required_param(self):
        """request.post requires both url and body; only one is invalid."""
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke("get_weather", param("location", "NYC")),
                    invoke("request.post", param("url", "https://x.com")),
                ),
            )
        )


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: value types
# ═══════════════════════════════════════════════════════════════════════


class TestValueTypes(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "t",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "s": {"type": "string"},
                        "i": {"type": "integer"},
                        "n": {"type": "number"},
                        "b": {"type": "boolean"},
                    },
                    "required": ["s", "i", "n", "b"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_valid(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "t",
                        param("s", "hello"),
                        param("i", "42"),
                        param("n", "3.14"),
                        param("b", "true"),
                    )
                ),
            )
        )

        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "t",
                        param("s", "hello"),
                        param("i", "42"),
                        param("n", "3.14"),
                        param("b", "yes"),
                    )
                ),
            )
        )
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "t",
                        param("s", "hello"),
                        param("i", "3.14"),
                        param("n", "3.14"),
                        param("b", "true"),
                    )
                ),
            )
        )

    def test_negative_int(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "t", param("s", "x"), param("i", "-7"), param("n", "0"), param("b", "false")
                    )
                ),
            )
        )

        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "t",
                        param("s", "x"),
                        param("i", "-7.0"),
                        param("n", "0"),
                        param("b", "false"),
                    )
                ),
            )
        )
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke("t", param("s", "x"), param("i", "-7"), param("n", "0"), param("b", "0"))
                ),
            )
        )

    def test_scientific(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "t",
                        param("s", "x"),
                        param("i", "1"),
                        param("n", "1.5e10"),
                        param("b", "true"),
                    )
                ),
            )
        )

        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "t",
                        param("s", "x"),
                        param("i", "1"),
                        param("n", "not_a_number"),
                        param("b", "true"),
                    )
                ),
            )
        )
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "t",
                        param("s", "x"),
                        param("i", "1e2"),
                        param("n", "1.5e10"),
                        param("b", "true"),
                    )
                ),
            )
        )

    def test_rejects_string_for_int(self):
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "t",
                        param("s", "x"),
                        param("i", "abc"),
                        param("n", "1.0"),
                        param("b", "true"),
                    )
                ),
            )
        )

    def test_rejects_string_for_bool(self):
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "t", param("s", "x"), param("i", "1"), param("n", "1.0"), param("b", "yes")
                    )
                ),
            )
        )


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: array parameters
# ═══════════════════════════════════════════════════════════════════════


class TestArrayParams(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "batch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ids": {"type": "array", "items": {"type": "integer"}},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["ids"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_integer_array(self):
        self.assertTrue(
            accepts(self.grammar, tool_call(invoke("batch", param("ids", "[1, 2, 3]"))))
        )

        self.assertFalse(
            accepts(self.grammar, tool_call(invoke("batch", param("ids", "[1, 2, 3.0]"))))
        )

        self.assertFalse(
            accepts(self.grammar, tool_call(invoke("batch", param("ids", '["1", "2", "3"]'))))
        )

    def test_empty_array(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("batch", param("ids", "[]")))))

        self.assertFalse(accepts(self.grammar, tool_call(invoke("batch", param("ids", "")))))

    def test_string_array_uses_json_string(self):
        """String items inside JSON arrays must be JSON-quoted."""
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(invoke("batch", param("ids", "[1]"), param("tags", '["foo", "bar"]'))),
            )
        )

        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(invoke("batch", param("ids", "[1]"), param("tags", "[foo, bar]"))),
            )
        )

    def test_rejects_unquoted_string_in_array(self):
        """Bare (unquoted) strings inside a JSON array should be rejected."""
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(invoke("batch", param("ids", "[1]"), param("tags", "[foo, bar]"))),
            )
        )

    def test_rejects_wrong_item_type(self):
        self.assertFalse(
            accepts(self.grammar, tool_call(invoke("batch", param("ids", '["not_int"]'))))
        )


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: nested objects
# ═══════════════════════════════════════════════════════════════════════


class TestNestedObject(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "create_user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                                "zip": {"type": "string"},
                            },
                            "required": ["street", "city"],
                        },
                    },
                    "required": ["name", "address"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_valid(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "create_user",
                        param("name", "Alice"),
                        param("address", '{"street": "123 Main", "city": "Springfield"}'),
                    )
                ),
            )
        )

        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "create_user",
                        param("name", "Alice"),
                        param("address", '{"street": 123, "city": "Springfield"}'),
                    )
                ),
            )
        )

    def test_with_optional_field(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "create_user",
                        param("name", "Bob"),
                        param(
                            "address", '{"street": "456 Oak", "city": "Portland", "zip": "97201"}'
                        ),
                    )
                ),
            )
        )

        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "create_user",
                        param("name", "Bob"),
                        param("address", '{"street": "456 Oak", "city": 97201, "zip": "97201"}'),
                    )
                ),
            )
        )

    def test_rejects_missing_required_nested(self):
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "create_user",
                        param("name", "Eve"),
                        param("address", '{"street": "789 Pine"}'),
                    )
                ),
            )
        )


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: anyOf, const, type arrays
# ═══════════════════════════════════════════════════════════════════════


class TestAnyOf(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "integer"},
                                {"type": "null"},
                            ]
                        }
                    },
                    "required": ["value"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_string(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("f", param("value", "hello")))))

    def test_integer(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("f", param("value", "42")))))

    def test_null(self):
        self.assertTrue(accepts(self.grammar, tool_call(invoke("f", param("value", "null")))))


class TestConstValue(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"const": "execute"},
                        "target": {"type": "string"},
                    },
                    "required": ["action", "target"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_correct(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(invoke("f", param("action", "execute"), param("target", "srv1"))),
            )
        )

        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(invoke("f", param("action", "run"), param("target", "srv1"))),
            )
        )

        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(invoke("f", param("action", "exec"), param("target", "srv1"))),
            )
        )

        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(invoke("f", param("action", "executes"), param("target", "srv1"))),
            )
        )

    def test_rejects_wrong(self):
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(invoke("f", param("action", "delete"), param("target", "srv1"))),
            )
        )


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: $ref / $defs (non-recursive)
# ═══════════════════════════════════════════════════════════════════════


class TestRefDefs(unittest.TestCase):
    """Non-recursive $ref to $defs entries."""

    def setUp(self):
        self.tools = [
            {
                "name": "order",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"$ref": "#/$defs/status_enum"},
                        "item": {"$ref": "#/$defs/item_obj"},
                    },
                    "$defs": {
                        "status_enum": {
                            "type": "string",
                            "enum": ["pending", "shipped", "delivered"],
                        },
                        "item_obj": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "quantity": {"type": "integer"},
                            },
                            "required": ["name", "quantity"],
                        },
                    },
                    "required": ["status", "item"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_valid(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "order",
                        param("status", "shipped"),
                        param("item", '{"name": "Widget", "quantity": 5}'),
                    )
                ),
            )
        )

    def test_rejects_invalid_enum_from_def(self):
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "order",
                        param("status", "cancelled"),
                        param("item", '{"name": "Widget", "quantity": 5}'),
                    )
                ),
            )
        )

    def test_rejects_missing_required_in_def_obj(self):
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke("order", param("status", "pending"), param("item", '{"name": "Widget"}'))
                ),
            )
        )


class TestCrossRefDefs(unittest.TestCase):
    """$defs entries that reference other $defs entries."""

    def setUp(self):
        self.tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"$ref": "#/$defs/wrapper"},
                    },
                    "$defs": {
                        "inner": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "integer"},
                            },
                            "required": ["value"],
                        },
                        "wrapper": {
                            "type": "object",
                            "properties": {
                                "payload": {"$ref": "#/$defs/inner"},
                                "label": {"type": "string"},
                            },
                            "required": ["payload", "label"],
                        },
                    },
                    "required": ["data"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_valid(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke("f", param("data", '{"payload": {"value": 42}, "label": "test"}'))
                ),
            )
        )

    def test_rejects_wrong_inner_type(self):
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke("f", param("data", '{"payload": {"value": "not_int"}, "label": "test"}'))
                ),
            )
        )


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: recursive $ref via $defs
# ═══════════════════════════════════════════════════════════════════════


class TestRecursiveRef(unittest.TestCase):
    """Recursive schemas via $defs (tree structures)."""

    def setUp(self):
        self.tools = [
            {
                "name": "process",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tree": {"$ref": "#/$defs/node"},
                    },
                    "$defs": {
                        "node": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                            },
                            "required": ["value"],
                        },
                    },
                    "required": ["tree"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_single_node(self):
        self.assertTrue(
            accepts(self.grammar, tool_call(invoke("process", param("tree", '{"value": "leaf"}'))))
        )

    def test_one_level(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "process",
                        param("tree", '{"value": "root", "children": [{"value": "child"}]}'),
                    )
                ),
            )
        )

    def test_two_levels(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "process",
                        param(
                            "tree",
                            '{"value": "root", "children": ['
                            '{"value": "a", "children": [{"value": "a1"}]}, '
                            '{"value": "b"}'
                            "]}",
                        ),
                    )
                ),
            )
        )


class TestRecursiveUnionDefs(unittest.TestCase):
    """Recursive schema using oneOf/anyOf across multiple $defs."""

    def setUp(self):
        self.tools = [
            {
                "name": "analyze_graph",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "root": {"$ref": "#/$defs/node"},
                    },
                    "$defs": {
                        "node": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "meta": {"$ref": "#/$defs/meta"},
                                "next": {
                                    "oneOf": [
                                        {"$ref": "#/$defs/edge"},
                                        {"type": "null"},
                                    ]
                                },
                            },
                            "required": ["id", "next"],
                        },
                        "edge": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "targets": {
                                    "type": "array",
                                    "items": {
                                        "anyOf": [
                                            {"$ref": "#/$defs/node"},
                                            {"type": "integer"},
                                        ]
                                    },
                                },
                            },
                            "required": ["label", "targets"],
                        },
                        "meta": {
                            "type": "object",
                            "properties": {
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "score": {"type": "number"},
                            },
                            "required": ["tags"],
                        },
                    },
                    "required": ["root"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_recursive_defs_with_oneof_and_anyof(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "analyze_graph",
                        param(
                            "root",
                            '{"id":"n0","meta":{"tags":["entry"],"score":1.5},'
                            '"next":{"label":"fanout","targets":['
                            '{"id":"n1","next":null},'
                            "7,"
                            '{"id":"n2","next":{"label":"tail","targets":[11]}}'
                            "]}}",
                        ),
                    )
                ),
            )
        )

        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "analyze_graph",
                        param("root", '{"id":"bad","next":{"label":"oops","targets":[true]}}'),
                    )
                ),
            )
        )


class TestRecursiveUnionDefsEnumRegression(unittest.TestCase):
    """Recursive oneOf/anyOf with enum strings nested in JSON object context."""

    def setUp(self):
        self.tools = [
            {
                "name": "analyze_graph",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "root": {"$ref": "#/$defs/node"},
                    },
                    "$defs": {
                        "node": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "next": {
                                    "oneOf": [
                                        {"$ref": "#/$defs/node"},
                                        {"$ref": "#/$defs/branch"},
                                        {"type": "null"},
                                    ]
                                },
                            },
                            "required": ["id", "next"],
                        },
                        "branch": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "enum": ["fork"]},
                                "options": {
                                    "type": "array",
                                    "items": {
                                        "anyOf": [
                                            {"$ref": "#/$defs/node"},
                                            {"$ref": "#/$defs/leaf"},
                                        ]
                                    },
                                },
                            },
                            "required": ["kind", "options"],
                        },
                        "leaf": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "integer"},
                            },
                            "required": ["value"],
                        },
                    },
                    "required": ["root"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_should_accept_enum_string_in_json_object_context(self):
        # This payload is valid per the schema and should be accepted.
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "analyze_graph",
                        param(
                            "root",
                            '{"id":"n0","next":{"kind":"fork","options":['
                            '{"id":"n1","next":null},'
                            '{"value":7}'
                            "]}}",
                        ),
                    )
                ),
            )
        )


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: $ref "#" (root self-reference)
# ═══════════════════════════════════════════════════════════════════════


class TestRootRef(unittest.TestCase):
    """$ref "#" references the root parameters schema."""

    def setUp(self):
        self.tools = [
            {
                "name": "walk",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "next": {"$ref": "#"},
                    },
                    "required": ["value"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_no_recursion(self):
        """Just the base case, no 'next' field."""
        self.assertTrue(accepts(self.grammar, tool_call(invoke("walk", param("value", "start")))))

    def test_one_level(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke("walk", param("value", "start"), param("next", '{"value": "end"}'))
                ),
            )
        )

    def test_two_levels(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "walk",
                        param("value", "a"),
                        param("next", '{"value": "b", "next": {"value": "c"}}'),
                    )
                ),
            )
        )


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: json_context (strings in arrays use json-string)
# ═══════════════════════════════════════════════════════════════════════


class TestJsonContext(unittest.TestCase):
    """Verify that string handling differs between top-level and JSON contexts."""

    def setUp(self):
        self.tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bare": {"type": "string"},
                        "arr": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["bare", "arr"],
                },
            }
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_bare_string_unquoted(self):
        """Top-level string params are bare (unquoted)."""
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(invoke("f", param("bare", "hello world"), param("arr", "[]"))),
            )
        )

    def test_array_strings_quoted(self):
        """Strings inside JSON arrays must be quoted."""
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(invoke("f", param("bare", "hi"), param("arr", '["a", "b"]'))),
            )
        )

    def test_empty_string(self):
        """Empty string is allowed."""
        self.assertTrue(
            accepts(self.grammar, tool_call(invoke("f", param("bare", ""), param("arr", "[]"))))
        )

    def test_array_strings_escaped(self):
        """Strings inside JSON arrays must be escaped."""
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "f",
                        param("bare", "hello world"),
                        param("arr", '["Well this is \\"fun\\""]'),
                    )
                ),
            )
        )

    def test_rejects_unquoted_in_array(self):
        """Unquoted strings inside a JSON array should fail."""
        self.assertFalse(
            accepts(
                self.grammar, tool_call(invoke("f", param("bare", "hi"), param("arr", "[a, b]")))
            )
        )


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: strict mode and additionalProperties
# ═══════════════════════════════════════════════════════════════════════


class TestStrictMode(unittest.TestCase):
    """strict=True disallows additional properties."""

    def setUp(self):
        self.tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                    },
                    "required": ["data"],
                },
            }
        ]

    def test_strict_rejects_extra_key(self):
        grammar = generate_minimax_tool_grammar(self.tools, strict=True)
        s = tool_call(invoke("f", param("data", '{"name": "Alice", "age": 30}')))
        self.assertFalse(accepts(grammar, s))

    def test_strict_accepts_declared_only(self):
        grammar = generate_minimax_tool_grammar(self.tools, strict=True)
        s = tool_call(invoke("f", param("data", '{"name": "Alice"}')))
        self.assertTrue(accepts(grammar, s))

    def test_strict_ignores_additional_properties_true(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                            "required": ["name"],
                            "additionalProperties": True,
                        },
                    },
                    "required": ["data"],
                },
            }
        ]
        grammar = generate_minimax_tool_grammar(tools, strict=True)
        s = tool_call(invoke("f", param("data", '{"name": "Alice", "age": 30}')))
        self.assertFalse(accepts(grammar, s))


class TestAdditionalProperties(unittest.TestCase):
    """strict=False respects additionalProperties from schema."""

    def setUp(self):
        self.tools_allow = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "shoe_size": {"type": "integer"},
                            },
                            "required": ["name"],
                            "additionalProperties": True,
                        },
                    },
                    "required": ["data"],
                },
            }
        ]
        self.tools_deny = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["data"],
                },
            }
        ]

    def test_allows_extra_when_permitted(self):
        grammar = generate_minimax_tool_grammar(self.tools_allow, strict=False)
        s = tool_call(invoke("f", param("data", '{"name": "Alice", "age": 30}')))
        self.assertTrue(accepts(grammar, s))

    def test_allows_no_extra(self):
        grammar = generate_minimax_tool_grammar(self.tools_allow, strict=False)
        s = tool_call(invoke("f", param("data", '{"name": "Alice"}')))
        self.assertTrue(accepts(grammar, s))

    def test_denies_additional_property_with_invalid_syntax(self):
        grammar = generate_minimax_tool_grammar(self.tools_allow, strict=False)
        s = tool_call(invoke("f", param("data", '{"name": "Alice", "age": 30, extra: val}')))
        self.assertFalse(accepts(grammar, s))

        s = tool_call(invoke("f", param("data", '{"name": "Alice", "age": 30, "extra"}')))
        self.assertFalse(accepts(grammar, s))

        s = tool_call(invoke("f", param("data", '{"name": "Alice", "age": 30, "extra":}')))
        self.assertFalse(accepts(grammar, s))

    def test_denies_extra_when_schema_says_false(self):
        grammar = generate_minimax_tool_grammar(self.tools_deny, strict=False)
        s = tool_call(invoke("f", param("data", '{"name": "Alice", "age": 30}')))
        self.assertFalse(accepts(grammar, s))

    def test_default_allows_extra(self):
        """When additionalProperties is absent, default is to allow (non-strict)."""
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                    },
                    "required": ["data"],
                },
            }
        ]
        grammar = generate_minimax_tool_grammar(tools, strict=False)
        s = tool_call(invoke("f", param("data", '{"name": "Alice", "extra": "val"}')))
        self.assertTrue(accepts(grammar, s))

    def test_additional_properties_schema_is_enforced(self):
        """When additionalProperties is a schema, extras should match it."""
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                            "required": ["name"],
                            "additionalProperties": {"type": "integer"},
                        },
                    },
                    "required": ["data"],
                },
            }
        ]
        grammar = generate_minimax_tool_grammar(tools, strict=False)

        self.assertTrue(
            accepts(grammar, tool_call(invoke("f", param("data", '{"name": "Alice", "age": 30}'))))
        )
        self.assertFalse(
            accepts(
                grammar, tool_call(invoke("f", param("data", '{"name": "Alice", "age": "30"}')))
            )
        )

    def test_all_optional_props_can_be_skipped_with_additional(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "opt_a": {"type": "string"},
                                "opt_b": {"type": "string"},
                            },
                        },  # no required, additionalProperties defaults True
                    },
                    "required": ["data"],
                },
            }
        ]
        grammar = generate_minimax_tool_grammar(tools, strict=False)
        s = tool_call(invoke("f", param("data", '{"extra": "val"}')))
        self.assertTrue(accepts(grammar, s))

    def test_skip_optional_then_required_then_additional(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "opt_a": {"type": "string"},
                                "req_b": {"type": "integer"},
                            },
                            "required": ["req_b"],
                        },
                    },
                    "required": ["data"],
                },
            }
        ]
        grammar = generate_minimax_tool_grammar(tools, strict=False)

        # opt_a omitted, required req_b present, then additional key.
        self.assertTrue(
            accepts(grammar, tool_call(invoke("f", param("data", '{"req_b": 7, "extra": "ok"}'))))
        )


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: preamble
# ═══════════════════════════════════════════════════════════════════════


class TestPreamble(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            }
        ]

    def test_no_preamble_rejects_leading_text(self):
        g = generate_minimax_tool_grammar(self.tools, allow_preamble=False)
        self.assertFalse(accepts(g, "Sure!\n" + tool_call(invoke("f", param("x", "hi")))))

    def test_preamble_accepts_leading_text(self):
        g = generate_minimax_tool_grammar(self.tools, allow_preamble=True)
        self.assertTrue(accepts(g, "Sure!" + tool_call(invoke("f", param("x", "hi")))))

    def test_preamble_accepts_leading_text_with_tag(self):
        g = generate_minimax_tool_grammar(self.tools, allow_preamble=True, require_tool_call=True)
        self.assertTrue(
            accepts(g, "Sure! This is <i>awesome</i>!" + tool_call(invoke("f", param("x", "hi"))))
        )

        g_standalone = generate_minimax_tool_grammar(
            self.tools, allow_preamble=True, require_tool_call=False
        )
        self.assertTrue(accepts(g_standalone, "Sure! This is <i>awesome</i>!"))
        self.assertTrue(
            accepts(
                g_standalone,
                "Sure! This is <i>awesome</i>!" + tool_call(invoke("f", param("x", "hi"))),
            )
        )

    def test_preamble_accepts_leading_text_with_partial_tool_call_tag_up_to_closing_bracket(self):
        g = generate_minimax_tool_grammar(self.tools, allow_preamble=True, require_tool_call=True)
        g_standalone = generate_minimax_tool_grammar(
            self.tools, allow_preamble=True, require_tool_call=False
        )

        acceptable_part_tag = "<minimax:tool_call"

        # test preambles that include every prefix of the acceptable_part_tag with trailing chars
        # that are not `>`:
        for i in range(len(acceptable_part_tag)):
            prefix = acceptable_part_tag[:i]
            self.assertTrue(
                accepts(
                    g_standalone,
                    f"some text here {prefix} some more text here><"
                    + tool_call(invoke("f", param("x", "hi"))),
                )
            )

            self.assertTrue(accepts(g_standalone, f"some text here {prefix} some more text here><"))
            self.assertTrue(accepts(g_standalone, f"some text here {prefix}"))
            self.assertTrue(
                accepts(
                    g_standalone,
                    f"some text here {prefix}" + tool_call(invoke("f", param("x", "hi"))),
                )
            )
            self.assertTrue(
                accepts(
                    g,
                    f"some text here {prefix} some more text here><\n"
                    + tool_call(invoke("f", param("x", "hi"))),
                )
            )
            self.assertTrue(
                accepts(
                    g,
                    f"some text here {prefix}\n" + tool_call(invoke("f", param("x", "hi"))),
                )
            )
            self.assertTrue(
                accepts(
                    g,
                    f"some text here {prefix} some more text here>< "
                    + tool_call(invoke("f", param("x", "hi"))),
                )
            )
            self.assertTrue(
                accepts(
                    g,
                    f"some text here {prefix} " + tool_call(invoke("f", param("x", "hi"))),
                )
            )
            self.assertTrue(
                accepts(
                    g,
                    f"some text here {prefix} some more text here>"
                    + tool_call(invoke("f", param("x", "hi"))),
                )
            )
            self.assertTrue(
                accepts(
                    g,
                    f"some text here {prefix}xx" + tool_call(invoke("f", param("x", "hi"))),
                )
            )

    def test_preamble_rejects_leading_text_with_tool_call_start_tag_and_invalid_tool_call_content(
        self,
    ):
        g = generate_minimax_tool_grammar(self.tools, allow_preamble=True)
        self.assertFalse(
            accepts(
                g,
                "some text here <minimax:tool_call><totally invalid content></minimax:tool_call>",
            )
        )
        self.assertFalse(
            accepts(
                g,
                "\n".join(
                    [
                        "some text here",
                        "<minimax:tool_call>",
                        "<totally invalid content>",
                        "</minimax:tool_call>",
                    ]
                ),
            )
        )
        self.assertFalse(
            accepts(
                g,
                "\n".join(
                    [
                        "some text here",
                        "<minimax:tool_call>",
                        tool_call(invoke("f", param("x", "hi"))),
                    ]
                ),
            )
        )
        self.assertFalse(
            accepts(
                g,
                "some text here\n<minimax:tool_call>" + tool_call(invoke("f", param("x", "hi"))),
            )
        )
        self.assertFalse(
            accepts(
                g,
                "some text here <minimax:tool_call>" + tool_call(invoke("f", param("x", "hi"))),
            )
        )

    def test_preamble_accepts_empty(self):
        g = generate_minimax_tool_grammar(self.tools, allow_preamble=True)
        self.assertTrue(accepts(g, tool_call(invoke("f", param("x", "hi")))))

    def test_default_allows_bare_text(self):
        g = generate_minimax_tool_grammar(self.tools)
        self.assertTrue(accepts(g, "Hello world"))

    def test_default_allows_leading_text(self):
        g = generate_minimax_tool_grammar(self.tools)
        self.assertTrue(accepts(g, "Sure!\n" + tool_call(invoke("f", param("x", "hi")))))

    def test_preamble_accepts_newlines(self):
        g = generate_minimax_tool_grammar(self.tools, allow_preamble=True)
        self.assertTrue(
            accepts(g, "Sure!\n\nHere you go:\n" + tool_call(invoke("f", param("x", "hi"))))
        )


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: require_tool_call
# ═══════════════════════════════════════════════════════════════════════


class TestRequireToolCall(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            }
        ]

    def test_require_tool_call_true_rejects_empty(self):
        g = generate_minimax_tool_grammar(self.tools, require_tool_call=True)
        empty = "<minimax:tool_call>\n</minimax:tool_call>\n"
        self.assertFalse(accepts(g, empty))

    def test_require_tool_call_false_accepts_empty(self):
        g = generate_minimax_tool_grammar(self.tools, require_tool_call=False)
        empty = "<minimax:tool_call>\n</minimax:tool_call>\n"
        self.assertTrue(accepts(g, empty))

    def test_require_tool_call_true_accepts_one_invocation(self):
        g = generate_minimax_tool_grammar(self.tools, require_tool_call=True)
        self.assertTrue(accepts(g, tool_call(invoke("f", param("x", "hi")))))

    def test_require_tool_call_false_accepts_one_invocation(self):
        g = generate_minimax_tool_grammar(self.tools, require_tool_call=False)
        self.assertTrue(accepts(g, tool_call(invoke("f", param("x", "hi")))))

    def test_default_does_not_require_tool_call(self):
        g = generate_minimax_tool_grammar(self.tools)
        empty = "Hello world!"
        self.assertTrue(accepts(g, empty))

    def test_require_tool_call_true_allow_preamble_true_rejects_preamble_plus_empty(self):
        g = generate_minimax_tool_grammar(self.tools, require_tool_call=True, allow_preamble=True)
        empty_with_preamble = "Sure, here you go.\n" + "<minimax:tool_call>\n</minimax:tool_call>\n"
        self.assertFalse(accepts(g, empty_with_preamble))

    def test_require_tool_call_true_allow_preamble_true_rejects_preamble_without_tool_call(self):
        g = generate_minimax_tool_grammar(self.tools, require_tool_call=True, allow_preamble=True)
        preamble_no_tool_call = "Sure, here you go."
        self.assertFalse(accepts(g, preamble_no_tool_call))

    def test_require_tool_call_true_allow_preamble_true_rejects_empty_string(self):
        g = generate_minimax_tool_grammar(self.tools, require_tool_call=True, allow_preamble=True)
        empty = ""
        self.assertFalse(accepts(g, empty))

    def test_require_tool_call_true_allow_preamble_true_allows_tool_call_without_preamble(self):
        g = generate_minimax_tool_grammar(self.tools, require_tool_call=True, allow_preamble=True)
        tool_call_without_preamble = tool_call(invoke("f", param("x", "hi")))
        self.assertTrue(accepts(g, tool_call_without_preamble))

    def test_require_tool_call_true_allow_preamble_true_accepts_preamble_plus_invocation(self):
        g = generate_minimax_tool_grammar(self.tools, require_tool_call=True, allow_preamble=True)
        self.assertTrue(accepts(g, "Sure!\n" + tool_call(invoke("f", param("x", "hi")))))

    def test_require_tool_call_false_allow_preamble_true_accepts_preamble_plus_empty(self):
        g = generate_minimax_tool_grammar(self.tools, require_tool_call=False, allow_preamble=True)
        empty_with_preamble = "No tools needed.\n" + "<minimax:tool_call>\n</minimax:tool_call>\n"
        self.assertTrue(accepts(g, empty_with_preamble))

    def test_require_tool_call_false_allow_preamble_true_accepts_preamble_plus_invocation(self):
        g = generate_minimax_tool_grammar(self.tools, require_tool_call=False, allow_preamble=True)
        self.assertTrue(
            accepts(g, "Here is the result.\n" + tool_call(invoke("f", param("x", "hi"))))
        )


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: no params
# ═══════════════════════════════════════════════════════════════════════


class TestNoParams(unittest.TestCase):
    def setUp(self):
        self.tools = [{"name": "ping", "parameters": {"type": "object", "properties": {}}}]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_accepts(self):
        self.assertTrue(accepts(self.grammar, tool_call('<invoke name="ping">\n</invoke>')))

        self.assertFalse(accepts(self.grammar, tool_call(invoke("ping", param("x", "y")))))


# ═══════════════════════════════════════════════════════════════════════
# Acceptance: complex real-world example
# ═══════════════════════════════════════════════════════════════════════


class TestComplexRealWorld(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_github_issue",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "body": {"type": "string"},
                            "labels": {"type": "array", "items": {"type": "string"}},
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                            },
                        },
                        "required": ["title", "body"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_issues",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "state": {"type": "string", "enum": ["open", "closed", "all"]},
                            "per_page": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
        self.grammar = generate_minimax_tool_grammar(self.tools)

    def test_create_minimal(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke("create_github_issue", param("title", "Bug"), param("body", "Details"))
                ),
            )
        )

    def test_create_with_optionals(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "create_github_issue",
                        param("title", "Feature"),
                        param("body", "Add XML grammar support"),
                        param("labels", '["enhancement"]'),
                        param("priority", "high"),
                    )
                ),
            )
        )

    def test_search(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "search_issues",
                        param("query", "minimax"),
                        param("state", "open"),
                        param("per_page", "20"),
                    )
                ),
            )
        )

    def test_parallel(self):
        self.assertTrue(
            accepts(
                self.grammar,
                tool_call(
                    invoke("create_github_issue", param("title", "Bug"), param("body", "Details")),
                    invoke("search_issues", param("query", "related")),
                ),
            )
        )

    def test_rejects_invalid_priority(self):
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke(
                        "create_github_issue",
                        param("title", "Bug"),
                        param("body", "Details"),
                        param("priority", "urgent"),
                    )
                ),
            )
        )

    def test_rejects_invalid_state(self):
        self.assertFalse(
            accepts(
                self.grammar,
                tool_call(
                    invoke("search_issues", param("query", "test"), param("state", "pending"))
                ),
            )
        )


# ═══════════════════════════════════════════════════════════════════════
# Error handling: unsupported $ref formats
# ═══════════════════════════════════════════════════════════════════════


class TestUnsupportedRef(unittest.TestCase):
    def test_external_uri_raises(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"$ref": "https://example.com/schema.json"}},
                    "required": ["x"],
                },
            }
        ]
        with self.assertRaises(ValueError) as cm:
            generate_minimax_tool_grammar(tools)
        self.assertIn("Unsupported $ref", str(cm.exception))

    def test_relative_path_raises(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"$ref": "#/properties/y"}},
                    "required": ["x"],
                },
            }
        ]
        with self.assertRaises(ValueError) as cm:
            generate_minimax_tool_grammar(tools)
        self.assertIn("Unsupported $ref", str(cm.exception))

    def test_undefined_def_raises(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"$ref": "#/$defs/nonexistent"}},
                    "$defs": {},
                    "required": ["x"],
                },
            }
        ]
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
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "allOf": [
                                {
                                    "type": "object",
                                    "properties": {"a": {"type": "string"}},
                                    "required": ["a"],
                                },
                                {
                                    "type": "object",
                                    "properties": {"b": {"type": "integer"}},
                                    "required": ["b"],
                                },
                            ]
                        },
                    },
                    "required": ["data"],
                },
            }
        ]
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


class TestTokenMatching(unittest.TestCase):
    def setUp(self):
        tools = [
            {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                    "required": ["a"],
                },
            }
        ]

        self.g = generate_minimax_tool_grammar(tools)
        self.g_no_preamble = generate_minimax_tool_grammar(tools, allow_preamble=False)
        self.g_no_preamble_tool_call_required = generate_minimax_tool_grammar(
            tools, allow_preamble=False, require_tool_call=True
        )
        self.c = compiles(self.g)
        start_tool_call_token_seq = minimax_tokenizer.encode("<minimax:tool_call>")
        end_tool_call_token_seq = minimax_tokenizer.encode("</minimax:tool_call>")

        self.assertTrue(len(start_tool_call_token_seq) == len(end_tool_call_token_seq) == 1)

        self.start_tool_call_token = start_tool_call_token_seq[0]
        self.end_tool_call_token = end_tool_call_token_seq[0]
        self.assertTrue(self.start_tool_call_token != self.end_tool_call_token)
        self.assertTrue(self.start_tool_call_token == 200052)
        self.assertTrue(self.end_tool_call_token == 200053)

    def test_accepts_correct_tool_call_tokens_with_preamble(self):
        matcher = _get_matcher_from_grammar_and_tokenizer_info(self.g, tokenizer_info)

        message = "Hello, world!"
        tokens = minimax_tokenizer.encode(message)
        for token in tokens:
            self.assertTrue(matcher.accept_token(token))
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

        needs_apply = matcher.fill_next_token_bitmask(bitmask)
        self.assertTrue(needs_apply)
        masked_tokens = _get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)

        # make sure that start tool call token is allowed
        self.assertFalse(self.start_tool_call_token in masked_tokens)

        # start the toolcall
        self.assertTrue(matcher.accept_token(self.start_tool_call_token))

        needs_apply = matcher.fill_next_token_bitmask(bitmask)
        self.assertTrue(needs_apply)
        masked_tokens = _get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)

        # make sure that start tool call token is NOT allowed
        self.assertTrue(self.start_tool_call_token in masked_tokens)

        # now apply the invoke
        invoke_str = invoke("f", param("a", "b"))
        tokens = minimax_tokenizer.encode(f"\n{invoke_str}")
        for token in tokens:
            self.assertTrue(matcher.accept_token(token))

        needs_apply = matcher.fill_next_token_bitmask(bitmask)
        self.assertTrue(needs_apply)
        masked_tokens = _get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)

        # make sure that start tool call token is still NOT allowed
        self.assertTrue(self.start_tool_call_token in masked_tokens)

        # make sure that end tool call token is allowed
        self.assertTrue(matcher.accept_token(self.end_tool_call_token))

    def test_accepts_correct_tool_call_tokens_without_preamble(self):
        matcher = _get_matcher_from_grammar_and_tokenizer_info(self.g_no_preamble, tokenizer_info)

        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

        needs_apply = matcher.fill_next_token_bitmask(bitmask)
        self.assertTrue(needs_apply)
        masked_tokens = _get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)

        # make sure that start tool call token is allowed
        self.assertFalse(self.start_tool_call_token in masked_tokens)

        # start the toolcall
        self.assertTrue(matcher.accept_token(self.start_tool_call_token))

        needs_apply = matcher.fill_next_token_bitmask(bitmask)
        self.assertTrue(needs_apply)
        masked_tokens = _get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)

        # make sure that start tool call token is NOT allowed
        self.assertTrue(self.start_tool_call_token in masked_tokens)

        # now apply the invoke
        invoke_str = invoke("f", param("a", "b"))
        tokens = minimax_tokenizer.encode(f"\n{invoke_str}")
        for token in tokens:
            self.assertTrue(matcher.accept_token(token))

        needs_apply = matcher.fill_next_token_bitmask(bitmask)
        self.assertTrue(needs_apply)
        masked_tokens = _get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)

        # make sure that start tool call token is still NOT allowed
        self.assertTrue(self.start_tool_call_token in masked_tokens)

        # make sure that end tool call token is allowed
        self.assertTrue(matcher.accept_token(self.end_tool_call_token))

    def test_requires_tool_call_token_without_preamble(self):
        matcher = _get_matcher_from_grammar_and_tokenizer_info(self.g_no_preamble, tokenizer_info)

        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        needs_apply = matcher.fill_next_token_bitmask(bitmask)
        self.assertTrue(needs_apply)

        masked_tokens = _get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)

        # make sure that start tool call token is allowed
        self.assertFalse(self.start_tool_call_token in masked_tokens)

        # only "<", "<m", and "<minimax:tool_call>" should be allowed
        self.assertTrue(len(masked_tokens) == tokenizer_info.vocab_size - 3)

        # start the toolcall
        self.assertTrue(matcher.accept_token(self.start_tool_call_token))

        needs_apply = matcher.fill_next_token_bitmask(bitmask)
        self.assertTrue(needs_apply)
        masked_tokens = _get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)

        # make sure that start tool call token is NOT allowed
        self.assertTrue(self.start_tool_call_token in masked_tokens)

        # now apply the invoke
        invoke_str = invoke("f", param("a", "b"))
        tokens = minimax_tokenizer.encode(f"\n{invoke_str}")
        for token in tokens:
            self.assertTrue(matcher.accept_token(token))

        needs_apply = matcher.fill_next_token_bitmask(bitmask)
        self.assertTrue(needs_apply)
        masked_tokens = _get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)

        # make sure that start tool call token is still NOT allowed
        self.assertTrue(self.start_tool_call_token in masked_tokens)

        # make sure that end tool call token is allowed
        self.assertTrue(matcher.accept_token(self.end_tool_call_token))


if __name__ == "__main__":
    unittest.main(verbosity=2)
