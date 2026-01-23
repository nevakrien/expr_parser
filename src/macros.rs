use crate::error_messages::{ERR_MACRO_NEEDS_BODY, ERR_MACRO_PARAM_IDENT, ERR_MACRO_SIGNATURE};
use crate::parsing::{LExpr, Loc, Located};
use crate::program::{CResult, CompileError, Program};
use crate::Expr;
use crate::Token;

#[derive(Debug)]
pub struct Macro {
    vars: Vec<String>,
    body: Expr,
}

impl Macro {
    pub fn new(args: Vec<LExpr>, loc: Loc) -> CResult<Self> {
        if args.len() < 2 {
            return Err(CompileError::SimpleError {
                loc,
                s: ERR_MACRO_NEEDS_BODY,
            });
        }

        let mut args = args.into_iter();
        let params_expr = args.next().expect("checked length");
        let body_expr = args.next().expect("checked length");

        let params = match &params_expr.value {
            Expr::Prefix(open, param_exprs) if open.value == "(" => {
                let mut param_names = Vec::new();
                for param_expr in param_exprs {
                    match &param_expr.value {
                        Expr::Atom(Token::Ident(name)) => param_names.push(name.clone()),
                        _ => {
                            return Err(CompileError::SimpleError {
                                loc: param_expr.loc.clone(),
                                s: ERR_MACRO_PARAM_IDENT,
                            });
                        }
                    }
                }
                param_names
            }
            _ => {
                return Err(CompileError::SimpleError {
                    loc: params_expr.loc.clone(),
                    s: ERR_MACRO_SIGNATURE,
                });
            }
        };

        Ok(Self {
            vars: params,
            body: body_expr.value,
        })
    }

    fn apply(&self, vars: &[LExpr], call_site: &Loc) -> CResult<LExpr> {
        if vars.len() != self.vars.len() {
            return Err(CompileError::Arity {
                loc: call_site.clone(),
                call_name: "Macro expansion",
                expected: self.vars.len(),
                got: vars.len(),
            });
        }

        Ok(self.substitute_expr(&self.body, vars, call_site))
    }

    fn substitute_expr(&self, expr: &Expr, args: &[LExpr], call_site: &Loc) -> LExpr {
        match expr {
            Expr::Atom(Token::Ident(s)) => {
                if let Some(idx) = self.vars.iter().position(|v| v == s) {
                    args[idx].clone()
                } else {
                    Located {
                        loc: call_site.clone(),
                        value: Expr::Atom(Token::Ident(s.clone())),
                    }
                }
            }
            Expr::Atom(token) => Located {
                loc: call_site.clone(),
                value: Expr::Atom(token.clone()),
            },
            Expr::Bin(op, box_exprs) => {
                let left = self.substitute_expr(&box_exprs.0.value, args, call_site);
                let right = self.substitute_expr(&box_exprs.1.value, args, call_site);
                let op_loc = Located {
                    loc: call_site.clone(),
                    value: op.value,
                };
                Located {
                    loc: call_site.clone(),
                    value: Expr::Bin(op_loc, Box::new((left, right))),
                }
            }
            Expr::Prefix(op, exprs) => {
                let new_exprs: Vec<_> = exprs
                    .iter()
                    .map(|e| self.substitute_expr(&e.value, args, call_site))
                    .collect();
                let op_loc = Located {
                    loc: call_site.clone(),
                    value: op.value,
                };
                Located {
                    loc: call_site.clone(),
                    value: Expr::Prefix(op_loc, new_exprs),
                }
            }
            Expr::Postfix(op, exprs) => {
                let new_exprs: Vec<_> = exprs
                    .iter()
                    .map(|e| self.substitute_expr(&e.value, args, call_site))
                    .collect();
                let op_loc = Located {
                    loc: call_site.clone(),
                    value: op.value,
                };
                Located {
                    loc: call_site.clone(),
                    value: Expr::Postfix(op_loc, new_exprs),
                }
            }
        }
    }
}

pub fn expand_macros_recursive(expr: &mut LExpr, program: &Program) -> CResult<()> {
    loop {
        let expansion = match &expr.value {
            Expr::Postfix(open, args) if open.value == "(" => {
                if let Some((callee, rest)) = args.split_first() {
                    if let Expr::Atom(Token::Ident(name)) = &callee.value {
                        program
                            .get_macro(name)
                            .map(|macro_def| macro_def.apply(rest, &expr.loc))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        match expansion {
            Some(expanded) => {
                *expr = expanded?;
            }
            None => break,
        }
    }

    match &mut expr.value {
        Expr::Postfix(_, args) | Expr::Prefix(_, args) => {
            for arg in args {
                expand_macros_recursive(arg, program)?;
            }
        }
        Expr::Bin(_, box_pair) => {
            let (left, right) = box_pair.as_mut();
            expand_macros_recursive(left, program)?;
            expand_macros_recursive(right, program)?;
        }
        Expr::Atom(_) => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error_messages::ERR_MACRO_NEEDS_BODY;
    use crate::parsing::Expr;
    use crate::program::{CompileError, Program};
    use crate::Parser;

    #[test]
    fn expands_recursive_and_nested_macros() {
        let src = "\
            m = macro(name, v) { name = macro() { v } }\
            id = macro(x) { x }\
            m(foo, 7)\
            id(foo())\
        ";

        let mut program = Program::new();
        let mut parser = Parser::new(src, 0);
        program.compile_all(&mut parser).unwrap();

        assert!(program.get_macro("m").is_some());
        assert!(program.get_macro("id").is_some());
        assert!(program.get_macro("foo").is_some());
    }

    #[test]
    fn expands_macros_inside_arguments() {
        let src = "m = macro(x) { x(x) } let f = 2; m(m(f))";
        let mut program = Program::new();
        let mut parser = Parser::new(src, 0);
        let mut last_expr = None;

        let mut pending = Vec::new();
        while !parser.is_empty() {
            let expr = parser.parse_with_macros(&program).unwrap().unwrap();
            last_expr = Some(expr.clone());
            program.gather_definition(expr, &mut pending).unwrap();
        }

        program
            .compile_pending_definitions(pending)
            .unwrap();

        let expr = last_expr.expect("expected expanded expression");
        match expr.value {
            Expr::Prefix(open, items) => {
                assert_eq!(open.value, "{");
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
        let mut parser = Parser::new(src, 0);
        let err = (|| {
            if let Some(exp) = parser.parse_with_macros(&program)? {
                let mut pending = Vec::new();
                program.gather_definition(exp, &mut pending)?;
            }

            Ok(())
        })()
        .expect_err("expected missing body error");

        assert!(matches!(
            err,
            CompileError::SimpleError {
                s: ERR_MACRO_NEEDS_BODY,
                ..
            }
        ));
    }

    fn assert_double_blocked_ff_call(expr: &LExpr) {
        match &expr.value {
            Expr::Postfix(open, args) => {
                assert_eq!(open.value, "(");
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
                assert_eq!(open.value, "{");
                assert_eq!(items.len(), 1);
                assert_ff_call(&items[0]);
            }
            _ => panic!("expected block with f(f) expression"),
        }
    }

    fn assert_ff_call(expr: &LExpr) {
        match &expr.value {
            Expr::Postfix(open, args) => {
                assert_eq!(open.value, "(");
                assert_eq!(args.len(), 2);
                assert!(matches!(&args[0].value, Expr::Atom(Token::Ident(name)) if name == "f"));
                assert!(matches!(&args[1].value, Expr::Atom(Token::Ident(name)) if name == "f"));
            }
            _ => panic!("expected f(f) expression"),
        }
    }
}
