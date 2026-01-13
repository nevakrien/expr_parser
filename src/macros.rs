use crate::Expr;
use crate::Token;
use crate::parsing::{LExpr, Loc, Located};
use crate::program::{CResult, CompileError};

#[derive(Debug)]
pub struct MacroError {
    pub message: String,
}

impl std::fmt::Display for MacroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Macro error: {}", self.message)
    }
}

impl std::error::Error for MacroError {}

#[derive(Debug)]
pub struct Macro {
    vars: Vec<String>,
    body: Expr,
}

impl Macro {
    pub fn new(args: &[LExpr], loc: Loc) -> CResult<Self> {
        if args.is_empty() {
            return Err(CompileError::MissingMacroBody { loc });
        }

        let (params_expr, body_expr) = if args.len() > 1 {
            (Some(&args[0]), &args[1])
        } else {
            (None, &args[0])
        };

        let params = if let Some(sig) = params_expr {
            match &sig.value {
                Expr::Prefix(open, param_exprs) if open.value.as_str() == "(" => {
                    let mut param_names = Vec::new();
                    for param_expr in param_exprs {
                        match &param_expr.value {
                            Expr::Atom(Token::Ident(name)) => param_names.push(name.clone()),
                            _ => {
                                return Err(CompileError::InvalidMacroParam {
                                    loc: param_expr.loc.clone(),
                                });
                            }
                        }
                    }
                    param_names
                }
                _ => {
                    return Err(CompileError::InvalidMacroSignature {
                        loc: sig.loc.clone(),
                    });
                }
            }
        } else {
            Vec::new()
        };

        Ok(Self {
            vars: params,
            body: body_expr.value.clone(),
        })
    }

    pub fn apply(&self, vars: &[LExpr], call_site: &Loc) -> Result<LExpr, MacroError> {
        if vars.len() != self.vars.len() {
            return Err(MacroError {
                message: format!("Expected {} arguments, got {}", self.vars.len(), vars.len()),
            });
        }

        self.substitute_expr(&self.body, vars, call_site)
    }

    fn substitute_expr(
        &self,
        expr: &Expr,
        args: &[LExpr],
        call_site: &Loc,
    ) -> Result<LExpr, MacroError> {
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
                    value: op.value.clone(),
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
                    value: op.value.clone(),
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
                    value: op.value.clone(),
                };
                Ok(Located {
                    loc: call_site.clone(),
                    value: Expr::Postfix(op_loc, new_exprs?),
                })
            }
        }
    }
}
