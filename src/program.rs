use crate::macros::Macro;
use crate::parsing::{Expr, LExpr, Loc, Located, Parser, Token};
use std::collections::HashMap;
use thiserror::Error;

pub type CResult<T> = Result<T, CompileError>;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum CompileError {
    #[error("Expected single identifier for macro name")]
    InvalidMacroName { loc: Loc },

    #[error("Macro definition requires a body")]
    MissingMacroBody { loc: Loc },

    #[error("Macro signature must be in parentheses")]
    InvalidMacroSignature { loc: Loc },

    #[error("Macro parameters must be identifiers")]
    InvalidMacroParam { loc: Loc },

    #[error("Unsupported definition")]
    UnsupportedDefinition { loc: Loc },

    #[error("Expected expression")]
    ExpectedExpr { loc: Loc },

    #[error("Macro expansion failed: {message}")]
    MacroApply { message: String, loc: Loc },

    #[error(transparent)]
    Parse(#[from] crate::parsing::ParseError),
}

impl CompileError {
    pub fn loc(&self) -> Option<&Loc> {
        match self {
            CompileError::InvalidMacroName { loc }
            | CompileError::MissingMacroBody { loc }
            | CompileError::InvalidMacroSignature { loc }
            | CompileError::InvalidMacroParam { loc }
            | CompileError::UnsupportedDefinition { loc }
            | CompileError::ExpectedExpr { loc }
            | CompileError::MacroApply { loc, .. } => Some(loc),
            CompileError::Parse(_) => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct Program {
    pub macros: HashMap<String, Macro>,
    pub functions: Vec<LExpr>,
    pub structs: Vec<LExpr>,
    pub enums: Vec<LExpr>,
    pub unions: Vec<LExpr>,
}

impl Program {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_macro(&mut self, name: String, macro_def: Macro) {
        self.macros.insert(name, macro_def);
    }

    pub fn get_macro(&self, name: &str) -> Option<&Macro> {
        self.macros.get(name)
    }
}

pub struct ProgramParser<'a> {
    parser: Parser<'a>,
}

impl<'a> ProgramParser<'a> {
    pub fn new(src: &'a str, file: usize) -> Self {
        Self {
            parser: Parser::new(src, file),
        }
    }

    pub fn is_empty(&mut self) -> bool {
        self.parser.is_empty()
    }

    pub fn consume_expr(&mut self, program: &mut Program) -> CResult<Option<LExpr>> {
        let Some(expr) = self.parser.parse_stmt()? else {
            return Ok(None);
        };

        self.handle_definition(expr, program)
    }

    fn handle_definition(&mut self, expr: LExpr, program: &mut Program) -> CResult<Option<LExpr>> {
        match &expr.value {
            Expr::Postfix(op, items) if op.value.as_str() == ";" => {
                if let Some(inner) = items.get(0) {
                    let result = self.handle_definition(inner.clone(), program)?;
                    if let Some(expanded) = result {
                        return Ok(Some(Located {
                            loc: expr.loc.clone(),
                            value: Expr::Postfix(op.clone(), vec![expanded]),
                        }));
                    }
                    return Ok(None);
                }
                Ok(None)
            }
            Expr::Prefix(open, items) if open.value.as_str() == "{" => {
                let mut new_items = Vec::new();
                for item in items {
                    if let Some(expanded) = self.handle_definition(item.clone(), program)? {
                        new_items.push(expanded);
                    }
                }
                Ok(Some(Located {
                    loc: expr.loc.clone(),
                    value: Expr::Prefix(open.clone(), new_items),
                }))
            }
            Expr::Bin(eq, box_pair) if eq.value.as_str() == "=" => {
                let (lhs, rhs) = &**box_pair;
                self.handle_assignment(&expr, lhs, rhs, program)
            }
            _ => {
                let expanded = self.expand_macros_recursive(expr, program)?;
                match &expanded.value {
                    Expr::Bin(eq, _) if eq.value.as_str() == "=" => {
                        self.handle_definition(expanded, program)
                    }
                    Expr::Prefix(open, _) if open.value.as_str() == "{" => {
                        self.handle_definition(expanded, program)
                    }
                    Expr::Postfix(op, _) if op.value.as_str() == ";" => {
                        self.handle_definition(expanded, program)
                    }
                    _ => Ok(Some(expanded)),
                }
            }
        }
    }

    fn expand_macros_recursive(&mut self, expr: LExpr, program: &mut Program) -> CResult<LExpr> {
        match &expr.value {
            Expr::Postfix(open, args) => {
                if open.value.as_str() == "(" {
                    if let Some((callee, rest)) = args.split_first() {
                        if let Expr::Atom(Token::Ident(name)) = &callee.value {
                            if let Some(macro_def) = program.get_macro(name) {
                                let expanded = macro_def.apply(rest, &expr.loc).map_err(|e| {
                                    CompileError::MacroApply {
                                        message: e.message,
                                        loc: expr.loc.clone(),
                                    }
                                })?;
                                return self.expand_macros_recursive(expanded, program);
                            }
                        }
                    }
                }

                let mut new_args = Vec::with_capacity(args.len());
                for arg in args {
                    new_args.push(self.expand_macros_recursive(arg.clone(), program)?);
                }
                Ok(Located {
                    loc: expr.loc.clone(),
                    value: Expr::Postfix(open.clone(), new_args),
                })
            }
            Expr::Prefix(op, args) => {
                let mut new_args = Vec::with_capacity(args.len());
                for arg in args {
                    new_args.push(self.expand_macros_recursive(arg.clone(), program)?);
                }
                Ok(Located {
                    loc: expr.loc.clone(),
                    value: Expr::Prefix(op.clone(), new_args),
                })
            }
            Expr::Bin(op, box_pair) => {
                let (left_expr, right_expr) = &**box_pair;
                let left = self.expand_macros_recursive(left_expr.clone(), program)?;
                let right = self.expand_macros_recursive(right_expr.clone(), program)?;
                Ok(Located {
                    loc: expr.loc.clone(),
                    value: Expr::Bin(op.clone(), Box::new((left, right))),
                })
            }
            Expr::Atom(_) => Ok(expr),
        }
    }

    fn handle_assignment(
        &mut self,
        expr: &LExpr,
        lhs: &LExpr,
        rhs: &LExpr,
        program: &mut Program,
    ) -> CResult<Option<LExpr>> {
        match &rhs.value {
            Expr::Prefix(macro_kw, args) if macro_kw.value.as_str() == "macro" => {
                let name = get_single_ident(lhs)?;
                let macro_def = Macro::new(args, rhs.loc.clone())?;
                program.add_macro(name, macro_def);
                Ok(Some(expr.clone()))
            }
            Expr::Prefix(fn_kw, _)
                if fn_kw.value.as_str() == "fn" || fn_kw.value.as_str() == "cfn" =>
            {
                program.functions.push(rhs.clone());
                Ok(None)
            }
            Expr::Prefix(struct_kw, _) if struct_kw.value.as_str() == "struct" => {
                program.structs.push(rhs.clone());
                Ok(None)
            }
            Expr::Prefix(enum_kw, _) if enum_kw.value.as_str() == "enum" => {
                program.enums.push(rhs.clone());
                Ok(None)
            }
            Expr::Prefix(union_kw, _) if union_kw.value.as_str() == "union" => {
                program.unions.push(rhs.clone());
                Ok(None)
            }
            _ => Err(CompileError::UnsupportedDefinition {
                loc: lhs.loc.clone(),
            }),
        }
    }
}

pub fn get_single_ident(expr: &LExpr) -> CResult<String> {
    match &expr.value {
        Expr::Atom(Token::Ident(name)) => Ok(name.clone()),
        _ => Err(CompileError::InvalidMacroName {
            loc: expr.loc.clone(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_recursive_and_nested_macros() {
        let src = "\
            m = macro(name, var) { name = macro() { var } }\
            id = macro(x) { x }\
            m(foo, 7)\
            id(foo())\
        ";

        let mut program = Program::new();
        let mut parser = ProgramParser::new(src, 0);
        while !parser.is_empty() {
            let _ = parser.consume_expr(&mut program).unwrap();
        }

        assert!(program.get_macro("m").is_some());
        assert!(program.get_macro("id").is_some());
        assert!(program.get_macro("foo").is_some());
    }

    #[test]
    fn expands_macros_inside_arguments() {
        let src = "m = macro(x) { x(x) } m(m(f))";
        let mut program = Program::new();
        let mut parser = ProgramParser::new(src, 0);
        let mut last_expr = None;

        while !parser.is_empty() {
            if let Some(expr) = parser.consume_expr(&mut program).unwrap() {
                last_expr = Some(expr);
            }
        }

        let expr = last_expr.expect("expected expanded expression");
        match expr.value {
            Expr::Prefix(open, items) => {
                assert_eq!(open.as_str(), "{");
                assert_eq!(items.len(), 1);
                assert_double_blocked_ff_call(&items[0]);
            }
            _ => panic!("expected block expression"),
        }
    }

    fn assert_double_blocked_ff_call(expr: &LExpr) {
        match &expr.value {
            Expr::Postfix(open, args) => {
                assert_eq!(open.as_str(), "(");
                assert_eq!(args.len(), 2);
                for arg in args {
                    assert_blocked_ff_call(arg);
                }
            }
            _ => panic!("expected (f(f))(f(f)) expression"),
        }
    }

    fn assert_blocked_ff_call(expr: &LExpr) {
        match &expr.value {
            Expr::Prefix(open, items) => {
                assert_eq!(open.as_str(), "{");
                assert_eq!(items.len(), 1);
                assert_ff_call(&items[0]);
            }
            _ => panic!("expected block with f(f) expression"),
        }
    }

    fn assert_ff_call(expr: &LExpr) {
        match &expr.value {
            Expr::Postfix(open, args) => {
                assert_eq!(open.as_str(), "(");
                assert_eq!(args.len(), 2);
                assert!(matches!(&args[0].value, Expr::Atom(Token::Ident(name)) if name == "f"));
                assert!(matches!(&args[1].value, Expr::Atom(Token::Ident(name)) if name == "f"));
            }
            _ => panic!("expected f(f) expression"),
        }
    }
}
