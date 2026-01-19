pub const ERR_UNSUPPORTED_EXPRESSION: &str = "unsupported expression in IR lowering";
pub const ERR_UNSUPPORTED_EXPRESSION_ATOM: &str = "unsupported expression atom in IR lowering";
pub const ERR_UNSUPPORTED_PATTERN: &str =
    "unsupported pattern in IR lowering; expected binding, wildcard, or tuple";
pub const ERR_INVALID_MATCH_ARM: &str = "invalid match arm syntax";
pub const ERR_INVALID_MATCH_ARM_GUARD: &str = "invalid match arm guard syntax";
pub const ERR_MATCH_ARM_NEEDS_VALUE: &str = "match expects a value and at least one arm";
pub const ERR_UNRESOLVED_NAME: &str = "Unresolved name";
pub const ERR_MACRO_NEEDS_BODY: &str = "Macro definition requires a body";
pub const ERR_MACRO_PARAM_IDENT: &str = "Macro parameters must be identifiers";
pub const ERR_MACRO_SIGNATURE: &str = "Macro signature must be in parentheses";
pub const ERR_UNSUPPORTED_DEFINITION: &str = "Unsupported definition";
pub const ERR_EXPECTED_MACRO_NAME: &str = "Expected single identifier for macro name";
