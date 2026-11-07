"""Language-agnostic canonical node types for code graphs.

Parser node names are an implementation detail: the same construct may be an
``If``, ``if_statement`` or ``CONTROL_STRUCTURE_IF`` depending on whether the
graph came from Python's :mod:`ast`, tree-sitter, or Joern.  This module maps
those heterogeneous names to a small, stable vocabulary suitable for a shared
embedding table.

The mapper deliberately consumes *node types*, not source-code tokens.  It is
also conservative: ambiguous tags such as bare ``LITERAL`` and
``CONTROL_STRUCTURE`` map to their explicit generic categories, rather than
being guessed as a numeric literal or a particular branch construct.

IDs are stable, contiguous, and zero-based.  If a downstream model reserves
zero for padding, store ``get_canonical_id(raw_type) + 1`` and allocate
``NUM_CANONICAL_TYPES + 1`` embeddings.
"""

from __future__ import annotations


import re
import html
from collections.abc import Iterable
from typing import Final


# Do not reorder this tuple: its position is the serialized embedding ID.
CANONICAL_CATEGORIES: Final[tuple[str, ...]] = (
    "Control_If",
    "Control_Loop",
    "Control_Switch",
    "Control_Return",
    "Control_Break",
    "Var_Decl",
    "Func_Decl",
    "Class_Decl",
    "Param_Decl",
    "Assign_Op",
    "Binary_Op",
    "Unary_Op",
    "Call_Expr",
    "Access_Expr",
    "Literal_Num",
    "Literal_Str",
    "Literal_Bool",
    "Literal_Null",
    "Structural_Block",
    "Identifier_Context",
    "Literal_Generic",
    "Control_Generic",
    "Type_Meta",
    "Canonical_Unknown",
)

CANONICAL_TYPE_TO_ID: Final[dict[str, int]] = {
    name: index for index, name in enumerate(CANONICAL_CATEGORIES)
}
CANONICAL_ID_TO_TYPE: Final[tuple[str, ...]] = CANONICAL_CATEGORIES
NUM_CANONICAL_TYPES: Final[int] = len(CANONICAL_CATEGORIES)
CANONICAL_UNKNOWN_ID: Final[int] = CANONICAL_TYPE_TO_ID["Canonical_Unknown"]


_CAMEL_WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ACRONYM_WORD_BOUNDARY = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize(raw_type: object) -> str:
    """Normalize parser spelling without reading source-level token content."""
    if not isinstance(raw_type, str):
        return ""
    text = html.unescape(raw_type).strip()
    if not text:
        return ""
    text = _ACRONYM_WORD_BOUNDARY.sub("_", text)
    text = _CAMEL_WORD_BOUNDARY.sub("_", text)
    return _NON_ALNUM.sub("_", text.lower()).strip("_")


# Exact aliases cover the common output of Joern, Python ast, JavaParser,
# srcML, clang/tree-sitter C, and tree-sitter Python/Java.  Keys are normalized
# once at import time, so lookup remains inexpensive inside a DataLoader.
_ALIASES_BY_CATEGORY: Final[dict[str, tuple[str, ...]]] = {
    "Control_If": (
        "operator_conditional",
        "if",
        "if_stmt",
        "if_statement",
        "if_expression",
        "if_exp",
        "if_then_statement",
        "if_then_else_statement",
        "conditional_expression",
        "ternary_expression",
        "control_structure_if",
    ),
    "Control_Loop": (
        "for",
        "for_stmt",
        "for_statement",
        "for_each",
        "for_each_statement",
        "foreach_statement",
        "enhanced_for_statement",
        "async_for",
        "while",
        "while_stmt",
        "while_statement",
        "do",
        "do_stmt",
        "do_statement",
        "do_while_statement",
        "loop_statement",
        "list_comp",
        "set_comp",
        "comprehension",
        "dict_comp",
        "generator_exp",
        "control_structure_for",
        "control_structure_while",
        "control_structure_do",
    ),
    "Control_Switch": (
        "switch",
        "switch_stmt",
        "switch_statement",
        "switch_expression",
        "switch_entry",
        "case",
        "case_statement",
        "case_label",
        "match",
        "match_statement",
        "match_case",
        "control_structure_switch",
    ),
    "Control_Return": (
        "return",
        "return_stmt",
        "return_statement",
        "yield",
        "yield_expr",
        "yield_expression",
        "control_structure_return",
    ),
    "Control_Break": (
        "break",
        "break_stmt",
        "break_statement",
        "continue",
        "continue_stmt",
        "continue_statement",
        "goto",
        "goto_statement",
        "control_structure_break",
        "control_structure_continue",
    ),
    "Var_Decl": (
        "local",
        "member",
        "field_declaration",
        "variable_declaration",
        "variable_declarator",
        "local_variable_declaration",
        "local_declaration_statement",
        "declaration",
        "declaration_statement",
        "init_declarator",
        "var_decl",
        "let_declaration",
        "const_declaration",
        "global",
        "nonlocal",
    ),
    "Func_Decl": (
        "method",
        "method_declaration",
        "method_definition",
        "function_declaration",
        "function_definition",
        "function_def",
        "async_function_def",
        "constructor_declaration",
        "constructor_definition",
        "lambda",
        "lambda_expression",
        "func_decl",
    ),
    "Class_Decl": (
        "type_decl",
        "class_decl",
        "class_declaration",
        "class_definition",
        "class_def",
        "interface_declaration",
        "annotation_type_declaration",
        "enum_declaration",
        "struct_specifier",
        "struct_declaration",
        "union_specifier",
        "record_declaration",
    ),
    "Param_Decl": (
        "keyword",
        "arg",
        "arguments",
        "parameter",
        "parameter_declaration",
        "formal_parameter",
        "receiver_parameter",
        "spread_parameter",
        "typed_parameter",
        "default_parameter",
        "method_parameter_in",
        "method_parameter_out",
        "param_decl",
    ),
    "Assign_Op": (
        "assign",
        "ann_assign",
        "aug_assign",
        "named_expr",
        "assignment",
        "assignment_expression",
        "assignment_operator",
        "operator_assignment",
        "operator_assignment_plus",
        "operator_assignment_minus",
        "operator_assignment_multiplication",
        "operator_assignment_division",
        "operator_assignment_modulo",
        "operator_assignment_and",
        "operator_assignment_or",
        "operator_assignment_xor",
        "operator_assignment_shift_left",
        "operator_assignment_arithmetic_shift_right",
        "operator_assignment_logical_shift_right",
    ),
    "Binary_Op": (
        "bin_op",
        "bool_op",
        "compare",
        "binary_expression",
        "binary_operator",
        "infix_expression",
        "add",
        "sub",
        "mult",
        "mat_mult",
        "div",
        "floor_div",
        "mod",
        "pow",
        "l_shift",
        "r_shift",
        "bit_or",
        "bit_xor",
        "bit_and",
        "and",
        "or",
        "eq",
        "not_eq",
        "lt",
        "lt_e",
        "gt",
        "gt_e",
        "is",
        "is_not",
        "in",
        "not_in",
        "operator_addition",
        "operator_subtraction",
        "operator_multiplication",
        "operator_division",
        "operator_modulo",
        "operator_exponentiation",
        "operator_shift_left",
        "operator_arithmetic_shift_right",
        "operator_logical_shift_right",
        "operator_and",
        "operator_or",
        "operator_xor",
        "operator_logical_and",
        "operator_logical_or",
        "operator_equals",
        "operator_not_equals",
        "operator_less_than",
        "operator_less_equals_than",
        "operator_greater_than",
        "operator_greater_equals_than",
    ),
    "Unary_Op": (
        "operator_size_of",
        "operator_cast",
        "unary_op",
        "unary_expression",
        "prefix_expression",
        "postfix_expression",
        "u_add",
        "u_sub",
        "not",
        "invert",
        "operator_not",
        "operator_logical_not",
        "operator_minus",
        "operator_plus",
        "operator_pre_increment",
        "operator_post_increment",
        "operator_pre_decrement",
        "operator_post_decrement",
        "operator_address_of",
        "operator_indirection",
    ),
    "Call_Expr": (
        "init",
        "operator_alloc",
        "operator_new",
        "operator_constructor",
        "call",
        "call_expr",
        "call_expression",
        "method_call",
        "method_call_expression",
        "method_invocation",
        "function_call",
        "function_call_expression",
        "object_creation_expression",
        "class_instance_creation_expression",
        "new_expression",
    ),
    "Access_Expr": (
        "operator_index_access",
        "slice",
        "identifier",
        "field_identifier",
        "name",
        "qualified_name",
        "attribute",
        "attribute_access",
        "member_access",
        "member_expression",
        "field_access",
        "field_expression",
        "array_access",
        "array_access_expression",
        "subscript",
        "index_expression",
        "pointer_expression",
        "this",
        "super",
        "external",
    ),
    "Literal_Num": (
        "num",
        "number",
        "number_literal",
        "numeric_literal",
        "integer_literal",
        "decimal_integer_literal",
        "hex_integer_literal",
        "octal_integer_literal",
        "binary_integer_literal",
        "floating_point_literal",
        "float_literal",
        "double_literal",
        "imaginary_literal",
        "literal_int",
        "literal_integer",
        "literal_float",
        "literal_double",
        "constant_int",
        "constant_integer",
        "constant_float",
        "constant_double",
        "constant_complex",
    ),
    "Literal_Str": (
        "str",
        "joined_str",
        "formatted_value",
        "string",
        "string_literal",
        "character_literal",
        "char_literal",
        "concatenated_string",
        "template_string",
        "literal_str",
        "literal_string",
        "literal_char",
        "constant_str",
        "constant_string",
        "constant_bytes",
    ),
    "Literal_Bool": (
        "bool",
        "boolean",
        "boolean_literal",
        "true",
        "false",
        "true_literal",
        "false_literal",
        "literal_bool",
        "literal_boolean",
        "constant_bool",
        "constant_boolean",
    ),
    "Literal_Null": (
        "null",
        "none",
        "nil",
        "nullptr",
        "null_literal",
        "none_literal",
        "nil_literal",
        "literal_null",
        "literal_none",
        "constant_null",
        "constant_none",
    ),
    "Structural_Block": (
        "expr",
        "pass",
        "block",
        "module",
        "compilation_unit",
        "statement_list",
        "expression_list",
        "suite",
    ),
    "Identifier_Context": (
        "load",
        "store",
        "del",
        "starred",
    ),
    "Literal_Generic": (
        "tuple",
        "list",
        "dict",
        "set",
        "operator_array_initializer",
        "literal",
        "constant",
    ),
    "Control_Generic": (
        "try",
        "except_handler",
        "with",
        "withitem",
        "assert",
        "raise",
        "delete",
        "operator_throw",
        "control_structure",
    ),
    "Type_Meta": (
        "alias",
        "method_return",
        "modifier",
        "type",
        "type_ref",
        "annotation",
        "import",
        "import_from",
        "package",
        "namespace_block",
    ),
    "Canonical_Unknown": (
        "canonical_unknown",
        "unknown",
        "unknown_node",
    ),
}


def _build_alias_table() -> dict[str, int]:
    aliases: dict[str, int] = {}
    for category, raw_aliases in _ALIASES_BY_CATEGORY.items():
        category_id = CANONICAL_TYPE_TO_ID[category]
        for raw_alias in (category, *raw_aliases):
            alias = _normalize(raw_alias)
            previous = aliases.setdefault(alias, category_id)
            if previous != category_id:
                raise RuntimeError(f"Conflicting canonical alias: {raw_alias!r}")
    return aliases


RAW_TYPE_TO_CANONICAL_ID: Final[dict[str, int]] = _build_alias_table()


# Namespace-qualified parser classes are common (for example,
# ``com.github.javaparser.ast.stmt.IfStmt``).  Exact aliases cover normal
# inputs; these ordered patterns provide a conservative fallback for qualified
# names and parser-specific prefixes/suffixes.
_PATTERN_RULES: Final[tuple[tuple[int, re.Pattern[str]], ...]] = tuple(
    (CANONICAL_TYPE_TO_ID[category], re.compile(pattern))
    for category, pattern in (
        (
            "Control_If",
            r"(?:^|_)(?:if)(?:_stmt|_statement|_expression|_exp)?(?:_|$)",
        ),
        (
            "Control_Loop",
            r"(?:^|_)(?:for|for_each|foreach|enhanced_for|while|do_while|loop)(?:_stmt|_statement|_expression)?(?:_|$)",
        ),
        (
            "Control_Switch",
            r"(?:^|_)(?:switch|case|match)(?:_stmt|_statement|_expression|_entry|_label)?(?:_|$)",
        ),
        ("Control_Return", r"(?:^|_)(?:return|yield)(?:_stmt|_statement|_expr|_expression)?(?:_|$)"),
        ("Control_Break", r"(?:^|_)(?:break|continue|goto)(?:_stmt|_statement)?(?:_|$)"),
        (
            "Param_Decl",
            r"(?:^|_)(?:formal_|method_)?(?:parameter|param|argument)(?:_in|_out|_declaration|_decl)?(?:_|$)",
        ),
        (
            "Func_Decl",
            r"(?:^|_)(?:function|method|constructor)(?:_declaration|_definition|_def|_decl)(?:_|$)",
        ),
        (
            "Class_Decl",
            r"(?:^|_)(?:class|interface|enum|struct|union|record|type)(?:_declaration|_definition|_specifier|_decl|_def)(?:_|$)",
        ),
        (
            "Var_Decl",
            r"(?:^|_)(?:variable|local_variable|field|member|var|let|const)(?:_declaration|_declarator|_statement|_decl)?(?:_|$)",
        ),
        (
            "Assign_Op",
            r"(?:^|_)(?:assign|assignment|ann_assign|aug_assign|named_expr)(?:_operator|_expression)?(?:_|$)",
        ),
        (
            "Unary_Op",
            r"(?:^|_)(?:unary|prefix|postfix)(?:_op|_operator|_expression)?(?:_|$)",
        ),
        (
            "Binary_Op",
            r"(?:^|_)(?:binary|infix|bin_op|bool_op|compare)(?:_op|_operator|_expression)?(?:_|$)",
        ),
        (
            "Call_Expr",
            r"(?:^|_)(?:call|invocation|method_call|function_call|object_creation|class_instance_creation)(?:_expr|_expression)?(?:_|$)",
        ),
        (
            "Access_Expr",
            r"(?:^|_)(?:identifier|qualified_name|attribute|member_access|field_access|field_expression|array_access|subscript|index_expression)(?:_|$)",
        ),
        (
            "Literal_Bool",
            r"(?:^|_)(?:bool|boolean|true|false)(?:_literal)?(?:_|$)",
        ),
        (
            "Literal_Null",
            r"(?:^|_)(?:null|none|nil|nullptr)(?:_literal)?(?:_|$)",
        ),
        (
            "Literal_Str",
            r"(?:^|_)(?:str|string|char|character|bytes)(?:_literal)?(?:_|$)",
        ),
        (
            "Literal_Num",
            r"(?:^|_)(?:num|number|numeric|integer|int|float|double|decimal|hex|octal|binary_integer)(?:_literal)?(?:_|$)",
        ),
    )
)


def get_canonical_id(raw_type_string: str, *, language: str | None = None) -> int:
    """Return a language-agnostic canonical ID for one parser/Joern node.

    ``language`` is optional to preserve the public mapping API.  It is used
    only for a Joern export quirk: some Java CALL nodes are stored as their
    callee text (for example ``println`` or ``nextLong``) instead of ``CALL``.
    Those labels are safely treated as ``Call_Expr`` after all exact and
    parser-generic rules have failed; Java identifiers themselves are already
    emitted as ``IDENTIFIER`` and are handled before this fallback.
    """
    normalized = _normalize(raw_type_string)
    if not normalized:
        return CANONICAL_UNKNOWN_ID

    exact = RAW_TYPE_TO_CANONICAL_ID.get(normalized)
    if exact is not None:
        return exact

    for category_id, pattern in _PATTERN_RULES:
        if pattern.search(normalized):
            return category_id

    if str(language or "").strip().lower() == "java" and not normalized.startswith("operator_"):
        return CANONICAL_TYPE_TO_ID["Call_Expr"]
    return CANONICAL_UNKNOWN_ID


def get_canonical_name(raw_type_string: str) -> str:
    """Return the canonical category name for a raw node type."""
    return CANONICAL_ID_TO_TYPE[get_canonical_id(raw_type_string)]


def canonicalize_node_types(raw_types: Iterable[str]) -> list[int]:
    """Map an iterable of raw node types while preserving input order."""
    return [get_canonical_id(raw_type) for raw_type in raw_types]


__all__ = [
    "CANONICAL_CATEGORIES",
    "CANONICAL_ID_TO_TYPE",
    "CANONICAL_TYPE_TO_ID",
    "CANONICAL_UNKNOWN_ID",
    "NUM_CANONICAL_TYPES",
    "RAW_TYPE_TO_CANONICAL_ID",
    "canonicalize_node_types",
    "get_canonical_id",
    "get_canonical_name",
]
