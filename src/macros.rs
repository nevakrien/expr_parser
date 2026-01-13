use crate::Expr;
use crate::Token;
use crate::parsing::{LExpr, Loc, Located};

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

pub struct Macro {
    vars: Vec<String>,
    body: Expr,
}

impl Macro {
    pub fn new(vars: Vec<String>, body: Expr) -> Self {
        Self { vars, body }
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
