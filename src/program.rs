use crate::macros::Macro;
use crate::parsing::{Expr, LExpr, Loc, Located, Parser, Token};
use std::collections::HashMap;
use thiserror::Error;

pub type CResult<T> = Result<T, CompileError>;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum CompileError {
    #[error("{s}")]
    SimpleError { loc: Loc, s: &'static str },

    #[error("Macro expansion failed: {message}")]
    MacroApply { message: String, loc: Loc },

    #[error(transparent)]
    Parse(#[from] crate::parsing::ParseError),
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

fn expand_macros_recursive(expr: LExpr, program: &mut Program) -> CResult<LExpr> {
    let Located { loc, value } = expr;
    match value {
        Expr::Postfix(open, args) => {
            if open.value.as_str() == "("
                && let Some((callee, rest)) = args.split_first()
                && let Expr::Atom(Token::Ident(name)) = &callee.value
                && let Some(macro_def) = program.get_macro(name)
            {
                let expanded = match macro_def.apply(rest, &loc) {
                    Ok(expanded) => expanded,
                    Err(e) => {
                        return Err(CompileError::MacroApply {
                            message: e.message,
                            loc,
                        });
                    }
                };
                return expand_macros_recursive(expanded, program);
            }

            let mut new_args = Vec::with_capacity(args.len());
            for arg in args {
                new_args.push(expand_macros_recursive(arg, program)?);
            }
            Ok(Located {
                loc,
                value: Expr::Postfix(open, new_args),
            })
        }
        Expr::Prefix(op, args) => {
            let mut new_args = Vec::with_capacity(args.len());
            for arg in args {
                new_args.push(expand_macros_recursive(arg, program)?);
            }
            Ok(Located {
                loc,
                value: Expr::Prefix(op, new_args),
            })
        }
        Expr::Bin(op, box_pair) => {
            let (left_expr, right_expr) = *box_pair;
            let left = expand_macros_recursive(left_expr, program)?;
            let right = expand_macros_recursive(right_expr, program)?;
            Ok(Located {
                loc,
                value: Expr::Bin(op, Box::new((left, right))),
            })
        }
        Expr::Atom(_) => Ok(Located { loc, value }),
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

    pub fn consume_expr(
        &mut self,
        program: &mut Program,
        on_expr: &mut dyn FnMut(&LExpr),
    ) -> CResult<bool> {
        let Some(expr) = self.parser.parse_stmt()? else {
            return Ok(false);
        };
        let expanded = expand_macros_recursive(expr, program)?;
        on_expr(&expanded);

        self.handle_definition(expanded, program)?;
        Ok(true)
    }

    fn handle_definition(&mut self, expr: LExpr, program: &mut Program) -> CResult<()> {
        let Located { loc: _, value } = expr;
        match value {
            Expr::Postfix(op, mut items) if op.value.as_str() == ";" => {
                self.handle_definition(items.pop().expect("bad structure"), program)
            }
            Expr::Prefix(open, items) if open.value.as_str() == "{" => {
                for item in items {
                    self.handle_definition(item, program)?;
                }
                Ok(())
            }
            Expr::Bin(eq, box_pair) if eq.value.as_str() == "=" => {
                let (lhs, rhs) = *box_pair;
                self.handle_assignment(lhs, rhs, program)?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn handle_assignment(&mut self, lhs: LExpr, rhs: LExpr, program: &mut Program) -> CResult<()> {
        let Located {
            loc: rhs_loc,
            value: rhs_value,
        } = rhs;

        match rhs_value {
            Expr::Prefix(macro_kw, args) if macro_kw.value.as_str() == "macro" => {
                let name = get_single_ident(lhs)?;
                let macro_def = Macro::new(args, rhs_loc)?;
                program.add_macro(name, macro_def);
                Ok(())
            }
            Expr::Prefix(fn_kw, args)
                if fn_kw.value.as_str() == "fn" || fn_kw.value.as_str() == "cfn" =>
            {
                program.functions.push(Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(fn_kw, args),
                });
                Ok(())
            }
            Expr::Prefix(struct_kw, args) if struct_kw.value.as_str() == "struct" => {
                program.structs.push(Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(struct_kw, args),
                });
                Ok(())
            }
            Expr::Prefix(enum_kw, args) if enum_kw.value.as_str() == "enum" => {
                program.enums.push(Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(enum_kw, args),
                });
                Ok(())
            }
            Expr::Prefix(union_kw, args) if union_kw.value.as_str() == "union" => {
                program.unions.push(Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(union_kw, args),
                });
                Ok(())
            }
            _ => Err(CompileError::SimpleError {
                loc: lhs.loc,
                s: "Unsupported definition",
            }),
        }
    }
}

pub fn get_single_ident(expr: LExpr) -> CResult<String> {
    match expr.value {
        Expr::Atom(Token::Ident(name)) => Ok(name),
        _ => Err(CompileError::SimpleError {
            loc: expr.loc,
            s: "Expected single identifier for macro name",
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
            let _ = parser.consume_expr(&mut program, &mut |_| {}).unwrap();
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
            let mut handler = |expr: &LExpr| {
                last_expr = Some(expr.clone());
            };
            let _ = parser.consume_expr(&mut program, &mut handler).unwrap();
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

    #[test]
    fn macro_requires_body() {
        let src = "m = macro(x)";
        let mut program = Program::new();
        let mut parser = ProgramParser::new(src, 0);
        let err = parser
            .consume_expr(&mut program, &mut |_| {})
            .expect_err("expected missing body error");

        assert!(matches!(
            err,
            CompileError::SimpleError {
                s: "Macro definition requires a body",
                ..
            }
        ));
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
