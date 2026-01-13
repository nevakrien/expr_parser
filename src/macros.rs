use crate::Expr;
use crate::Token;
use crate::parsing::{LExpr, Loc, Located};
use crate::program::{CResult, CompileError};

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
                s: "Macro definition requires a body",
            });
        }

        let mut args = args.into_iter();
        let params_expr = args.next().expect("checked length");
        let body_expr = args.next().expect("checked length");

        let params = match &params_expr.value {
            Expr::Prefix(open, param_exprs) if open.value.as_str() == "(" => {
                let mut param_names = Vec::new();
                for param_expr in param_exprs {
                    match &param_expr.value {
                        Expr::Atom(Token::Ident(name)) => param_names.push(name.clone()),
                        _ => {
                            return Err(CompileError::SimpleError {
                                loc: param_expr.loc.clone(),
                                s: "Macro parameters must be identifiers",
                            });
                        }
                    }
                }
                param_names
            }
            _ => {
                return Err(CompileError::SimpleError {
                    loc: params_expr.loc.clone(),
                    s: "Macro signature must be in parentheses",
                });
            }
        };

        Ok(Self {
            vars: params,
            body: body_expr.value,
        })
    }

    pub fn apply(&self, vars: &[LExpr], call_site: &Loc) -> CResult<LExpr> {
        if vars.len() != self.vars.len() {
            return Err(CompileError::Arity {
                loc: call_site.clone(),
                call_name: "Macro expansion",
                expected: self.vars.len(),
                got: vars.len(),
            });
        }

        self.substitute_expr(&self.body, vars, call_site)
    }

    fn substitute_expr(&self, expr: &Expr, args: &[LExpr], call_site: &Loc) -> CResult<LExpr> {
        match expr {
            Expr::Atom(Token::Ident(s)) => {
                if let Some(idx) = self.vars.iter().position(|v| v == s) {
                    Ok(args[idx].clone())
                } else {
                    Ok(Located {
                        loc: call_site.clone(),
                        value: Expr::Atom(Token::Ident(s.clone())),
                    })
                }
            }
            Expr::Atom(token) => Ok(Located {
                loc: call_site.clone(),
                value: Expr::Atom(token.clone()),
            }),
            Expr::Bin(op, box_exprs) => {
                let left = self.substitute_expr(&box_exprs.0.value, args, call_site)?;
                let right = self.substitute_expr(&box_exprs.1.value, args, call_site)?;
                let op_loc = Located {
                    loc: call_site.clone(),
                    value: op.value,
                };
                Ok(Located {
                    loc: call_site.clone(),
                    value: Expr::Bin(op_loc, Box::new((left, right))),
                })
            }
            Expr::Prefix(op, exprs) => {
                let new_exprs: Result<Vec<_>, _> = exprs
                    .iter()
                    .map(|e| self.substitute_expr(&e.value, args, call_site))
                    .collect();
                let op_loc = Located {
                    loc: call_site.clone(),
                    value: op.value,
                };
                Ok(Located {
                    loc: call_site.clone(),
                    value: Expr::Prefix(op_loc, new_exprs?),
                })
            }
            Expr::Postfix(op, exprs) => {
                let new_exprs: Result<Vec<_>, _> = exprs
                    .iter()
                    .map(|e| self.substitute_expr(&e.value, args, call_site))
                    .collect();
                let op_loc = Located {
                    loc: call_site.clone(),
                    value: op.value,
                };
                Ok(Located {
                    loc: call_site.clone(),
                    value: Expr::Postfix(op_loc, new_exprs?),
                })
            }
        }
    }
}
