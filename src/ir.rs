use crate::parsing::{Expr, LExpr, Loc, Located, Token};
use crate::program::{CResult, CompileError, Program};
use std::collections::HashMap;

// Type aliases for commonly used typed/located constructs
pub type LName = Located<NameId>; // Located name identifier
pub type TName = Typed<NameId>; // Typed name identifier
// pub type TDecl = Typed<Decl>; // Typed declaration
pub type TValue = Typed<Value>; // Typed value/expression
pub type TPattern = Typed<Pattern>; // Typed pattern

// Core type definitions for the IR
#[derive(Debug, Clone, PartialEq)]
pub struct TypeInfo {
    /// Locations where this type is used in the code
    pub uses: Vec<Loc>,
}

/// Wrapper that adds location and type information to any value
#[derive(Debug, Clone, PartialEq)]
pub struct Typed<T> {
    /// Source location of this construct
    pub loc: Loc,
    /// Type information (if available/known)
    pub ty: Option<TypeInfo>,
    /// The underlying value
    pub value: T,
}

/// Unique identifier for names in the IR
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NameId(pub usize);

// /// The intermediate representation (IR) of a complete program
// #[derive(Debug, Clone, PartialEq)]
// pub struct ProgramIr {
//     /// Global/top-level declarations
//     pub decls: Vec<TDecl>,
//     /// Main program body expression
//     pub body: TValue,
// }

/// Generic parameters for declarations
#[derive(Debug, Clone, PartialEq)]
pub struct Generics {
    /// Parameter patterns (typically type variables)
    pub params: Vec<TPattern>,
}

// /// Declaration of a type, constant, or macro in the global scope
// ///
// /// TODO: Eventually support local declarations within functions
// /// TODO: Macros should be handled here (explicitly or conceptually)
// #[derive(Debug, Clone, PartialEq)]
// pub enum Decl {
//     /// Runtime value declaration (constants, functions, etc.)
//     RuntimeValue {
//         name: LName,
//         generics: Option<Generics>,
//         value: TValue,
//     },
//     /// Struct definition with named fields
//     Struct {
//         name: LName,
//         generics: Option<Generics>,
//         fields: Vec<(LName, TPattern)>,
//     },
//     /// Union definition with named fields (one field active at a time)
//     Union {
//         name: LName,
//         generics: Option<Generics>,
//         fields: Vec<(LName, TPattern)>,
//     },
//     /// Enum definition with variants
//     Enum {
//         name: LName,
//         generics: Option<Generics>,
//         variants: Vec<EnumVariant>,
//     },
//     /// Type alias declaration
//     Alias {
//         name: LName,
//         generics: Option<Generics>,
//         ty: TPattern,
//     },
//     /// Macro definition
//     Macro {
//         name: LName,
//         generics: Option<Generics>,
//         params: Vec<LName>,
//         body: TValue,
//     },
// }

/// Single variant within an enum declaration
#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: LName,
    pub fields: Vec<TPattern>,
}

/// Runtime values including functions, closures, and control flow constructs
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Reference to a named variable or function
    NameRef(NameId),
    /// Literal value (number, string, etc.)
    Literal(Literal),
    /// Function/method call
    Call {
        callee: Box<TValue>,
        args: Vec<TValue>,
    },
    /// Indexing operation (may also represent generic specialization)
    Index {
        base: Box<TValue>,
        args: Vec<TValue>,
    },
    /// Block of statements with optional return value
    Block {
        statements: Vec<TValue>,
        return_value: Option<Box<TValue>>,
    },
    /// Variable binding declaration
    Let { pat: TPattern, value: Box<TValue> },
    /// Assignment operation
    Assign {
        target: Box<TValue>,
        value: Box<TValue>,
    },
    /// Conditional expression
    If {
        cond: Box<TValue>,
        then: Box<TValue>,
        els: Option<Box<TValue>>,
    },
    /// While loop
    While {
        cond: Box<TValue>,
        body: Box<TValue>,
    },
    /// Function literal/anonymous function
    Func {
        params: Vec<Param>,
        ret: Option<TPattern>,
        body: Box<TValue>,
    },
    /// Return from function (with optional value)
    Return(Option<Box<TValue>>),
    /// Break from loop
    Break,
    /// Continue to next loop iteration
    Continue,
    /// Explicit type cast
    Cast { value: Box<TValue>, ty: TPattern },
    /// Type annotation for expressions
    TypeAnnotation { value: Box<TValue>, ty: TPattern },
    /// Pattern matching expression
    Match {
        value: Box<TValue>,
        arms: Vec<MatchArm>,
    },
}

/// Literal values that can appear in the code
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Num(u64),
    Float(f64),
    Str(String),
}

/// Function parameter declaration
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub pat: TPattern,
    pub ty: Option<TPattern>,
}

/// Patterns used for:
/// - Pattern matching (match expressions, function parameters)
/// - Type annotations (e.g., Option[Result[T, E]])
/// - Assignment targets (e.g., id[T] = fn(x: T) { x })
///
/// TODO: figure out if this should have a field for Value
///        this would come up for *x[T] = 2; or similar
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// Bind a value to a name (variable binding)
    Bind(NameId),
    /// Wildcard pattern that matches anything (_)
    Wildcard,
    /// Tuple pattern with multiple sub-patterns
    Tuple(Vec<TPattern>),
    /// Struct/enum destructoring pattern
    Destructure {
        ctor: LName,
        fields: Vec<PatternField>,
    },
    /// Generic type specialization (e.g., Foo[T, U])
    GenericSpecialization {
        base: Box<TPattern>,
        args: Vec<TPattern>,
    },
    /// Literal value pattern
    Literal(Literal),
}

/// Named field in a destructoring pattern
#[derive(Debug, Clone, PartialEq)]
pub struct PatternField {
    pub name: LName,
    pub value: TPattern,
}

/// Single arm in a match expression
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pat: TPattern,
    pub guard: Option<TValue>,
    pub body: TValue,
}

// Implementations for IR types

impl<T> Typed<T> {
    /// Create a new Typed value with the same location and type but different inner value
    pub fn with<U>(&self, value: U) -> Typed<U> {
        Typed {
            loc: self.loc.clone(),
            ty: self.ty.clone(),
            value,
        }
    }

    /// Transform the inner value using a function while preserving location and type
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Typed<U> {
        Typed {
            loc: self.loc,
            ty: self.ty,
            value: f(self.value),
        }
    }

    /// Convert to a Located value (dropping type information)
    pub fn into_located(self) -> Located<T> {
        Located {
            loc: self.loc,
            value: self.value,
        }
    }
}

impl<T> From<Located<T>> for Typed<T> {
    fn from(value: Located<T>) -> Self {
        Typed {
            loc: value.loc,
            ty: None,
            value: value.value,
        }
    }
}

// impl ProgramIr {
//     /// Create an empty program IR with a block body
//     pub fn empty(loc: Loc) -> Self {
//         Self {
//             decls: Vec::new(),
//             body: Typed {
//                 loc,
//                 ty: None,
//                 value: Value::Block {
//                     statements: Vec::new(),
//                     return_value: None,
//                 },
//             },
//         }
//     }
// }

impl Program {
    pub fn lower_global(&mut self, expr: LExpr) -> CResult<()> {
        match expr.value {
            // Global definition sugar: <pattern> = <expr>
            Expr::Bin(op, pair) if op.value == "=" => {
                let (lhs_expr, rhs_expr) = *pair;

                let lhs = self.lower_pattern(lhs_expr)?;
                // TODO: generics from the pattern go into RHS lowering environment

                // For now, only allow `name = value` at global scope
                let _name = match lhs.value {
                    Pattern::Bind(id) => id,
                    _ => {
                        // If this is truly impossible given your parser rules, make it a debug_assert.
                        return Err(CompileError::SimpleError {
                            loc: lhs.loc,
                            s: "Global definitions must bind a name",
                        });
                    }
                };

                let _rhs = self.lower_value(rhs_expr)?;

                todo!("put value in global scope")
            }

            Expr::Prefix(open, _items) if open.value == "let" => {
                todo!()
            }

            _ => Err(CompileError::SimpleError {
                loc: expr.loc,
                s: "Unsupported expression in global scale",
            }),
        }
    }

    #[inline]
    fn with_scope<T>(&mut self, f: impl FnOnce(&mut Program) -> CResult<T>) -> CResult<T> {
        self.push_scope();
        let result = f(self);
        self.pop_scope();
        result
    }

    //TODO: 
    // 1. local macros are intetionaly not handeled and scoping on macros is broken on purpose to be like C
    // 2. some places parse a value where a value/pattern check needs to be done
    pub fn lower_value(&mut self, expr: LExpr) -> CResult<TValue> {
        match expr.value {
            Expr::Atom(token) => self.lower_atom(&expr.loc, token),

            // { ... }
            Expr::Prefix(open, mut items) if open.value == "{" => {
                let loc = expr.loc;

                self.with_scope(|this| {
                    let mut statements = Vec::new();
                    let mut return_value = None;

                    if let Some(last) = items.pop() {
                        for item in items {
                            statements.push(this.lower_value(item)?);
                        }

                        if !matches!(last.value, Expr::Atom(Token::Operator(";"))) {
                            return_value = Some(Box::new(this.lower_value(last)?));
                        }
                    }

                    Ok(this.typed_value(
                        loc,
                        Value::Block {
                            statements,
                            return_value,
                        },
                    ))
                })
            }

            // let <pat> = <value>
            Expr::Prefix(open, mut items) if open.value == "let" => {
                let loc = expr.loc;

                if items.len() < 2 {
                    return Err(CompileError::SimpleError {
                        loc,
                        s: "let expects a pattern and a value",
                    });
                }
                if items.len() > 3 {
                    return Err(CompileError::SimpleError {
                        loc,
                        s: "let has too many parts",
                    });
                }

                let value_expr = items.pop().unwrap();
                let pat_expr = items.pop().unwrap();

                let pat = self.lower_pattern(pat_expr)?;
                let value = Box::new(self.lower_value(value_expr)?);

                Ok(self.typed_value(loc, Value::Let { pat, value }))
            }

            // call: <callee>(args...)
            Expr::Postfix(open, items) if open.value == "(" => {
                let loc = expr.loc;
                debug_assert!(!items.is_empty(), "call expression missing callee");

                let mut it = items.into_iter();
                let callee = Box::new(self.lower_value(it.next().unwrap())?);

                let mut args = Vec::new();
                for arg in it {
                    args.push(self.lower_value(arg)?);
                }

                Ok(self.typed_value(loc, Value::Call { callee, args }))
            }

            // index: <base>[args...]
            Expr::Postfix(open, items) if open.value == "[" => {
                let loc = expr.loc;
                debug_assert!(!items.is_empty(), "index expression missing base");

                let mut it = items.into_iter();
                //TODO this can actually be a generic so a pattern
                let base = Box::new(self.lower_value(it.next().unwrap())?);

                let mut args = Vec::new();
                for arg in it {
                    args.push(self.lower_value(arg)?);
                }

                Ok(self.typed_value(loc, Value::Index { base, args }))
            }

            // match <value> { arms... }
            Expr::Prefix(open, items) if open.value == "match" => {
                let loc = expr.loc;

                if items.len() < 2 {
                    return Err(CompileError::SimpleError {
                        loc,
                        s: "match expects a value and at least one arm",
                    });
                }

                let mut it = items.into_iter();
                let value = Box::new(self.lower_value(it.next().unwrap())?);

                let mut arms = Vec::new();
                for arm in it {
                    arms.push(self.lower_match_arm(arm)?);
                }

                Ok(self.typed_value(loc, Value::Match { value, arms }))
            }

            // assignment
            Expr::Bin(op, pair) if op.value == "=" => {
                let loc = expr.loc;
                let (lhs, rhs) = *pair;

                //TODO: target might be a pattern in rare cases? not sure
                let target = Box::new(self.lower_value(lhs)?);
                let value = Box::new(self.lower_value(rhs)?);

                Ok(self.typed_value(loc, Value::Assign { target, value }))
            }

            // fn (sig) body
            Expr::Prefix(open, items) if open.value == "fn" => {
                let loc = expr.loc;

                debug_assert!(
                    (1..=2).contains(&items.len()),
                    "fn expects signature and optional body"
                );

                let mut items = items;

                let body_expr = if items.len() == 2 {
                    items.pop().unwrap()
                } else {
                    todo!()
                };

                let sig_expr = items.pop().expect("fn missing signature");

                let (params_expr, ret_expr) = match sig_expr.value {
                    Expr::Bin(arrow, pair) if arrow.value == "->" => {
                        let (lhs, rhs) = *pair;
                        (lhs, Some(rhs))
                    }
                    _ => (sig_expr, None),
                };

                let Expr::Prefix(p_open, param_items) = params_expr.value else {
                    debug_assert!(false, "fn signature does not start with parameter list");
                    unreachable!();
                };
                debug_assert!(p_open.value == "(", "fn parameter list must start with '('");

                let mut params = Vec::with_capacity(param_items.len());
                for param in param_items {
                    let pat = self.lower_pattern(param)?;
                    params.push(Param { pat, ty: None });
                }

                let ret = match ret_expr {
                    Some(e) => Some(self.lower_pattern(e)?),
                    None => None,
                };

                let body = Box::new(self.lower_value(body_expr)?);

                Ok(self.typed_value(loc, Value::Func { params, ret, body }))
            }

            _ => Err(CompileError::SimpleError {
                loc: expr.loc,
                s: "Unsupported expression in IR lowering",
            }),
        }
    }

    fn lower_atom(&mut self, loc: &Loc, token: Token) -> CResult<TValue> {
        let value = match token {
            Token::NumLit(n) => Value::Literal(Literal::Num(n)),
            Token::FloatLit(f) => Value::Literal(Literal::Float(f)),
            Token::StrLit(s) => Value::Literal(Literal::Str(s)),

            Token::Ident(name) => {
                let id = self.resolve_value(loc, &name)?;
                Value::NameRef(id)
            }

            Token::Operator(_) => {
                return Err(CompileError::SimpleError {
                    loc: loc.clone(),
                    s: "Unexpected operator atom in IR lowering",
                });
            }
        };

        Ok(self.typed_value(loc.clone(), value))
    }

    pub fn lower_pattern(&mut self, expr: LExpr) -> CResult<TPattern> {
        match expr.value {
            Expr::Atom(Token::Ident(name)) if name == "_" => {
                Ok(self.typed_pattern(expr.loc, Pattern::Wildcard))
            }

            Expr::Atom(Token::Ident(name)) => {
                let id = self.insert_value_in_current_scope(&name);
                Ok(self.typed_pattern(expr.loc, Pattern::Bind(id)))
            }

            Expr::Prefix(open, items) if open.value == "(" => {
                let mut parts = Vec::with_capacity(items.len());
                for item in items {
                    parts.push(self.lower_pattern(item)?);
                }
                Ok(self.typed_pattern(expr.loc, Pattern::Tuple(parts)))
            }

            _ => Err(CompileError::SimpleError {
                loc: expr.loc,
                s: "Unsupported pattern in IR lowering",
            }),
        }
    }

    fn resolve_value(&mut self, loc: &Loc, name: &str) -> CResult<NameId> {
        for (value_scope, _) in self.scopes.iter().rev() {
            if let Some(id) = value_scope.get(name) {
                return Ok(*id);
            }
        }
        Err(CompileError::SimpleError {
            loc: loc.clone(),
            s: "Unresolved name",
        })
    }

    fn lower_match_arm(&mut self, expr: LExpr) -> CResult<MatchArm> {
        match expr.value {
            Expr::Bin(op, pair) if op.value == "=>" => {
                let (pat_expr, body_expr) = *pair;
                let pat = self.lower_pattern(pat_expr)?;
                let body = self.lower_value(body_expr)?;
                Ok(MatchArm { pat, guard: None, body })
            }

            Expr::Bin(op, pair) if op.value == "if" => {
                let (left, guard_expr) = *pair;

                if let Expr::Bin(arrow, inner_pair) = left.value
                    && arrow.value == "=>"
                {
                    let (pat_expr, body_expr) = *inner_pair;
                    let pat = self.lower_pattern(pat_expr)?;
                    let guard = self.lower_value(guard_expr)?;
                    let body = self.lower_value(body_expr)?;
                    Ok(MatchArm { pat, guard: Some(guard), body })
                } else {
                    Err(CompileError::SimpleError {
                        loc: expr.loc,
                        s: "Invalid match arm guard syntax",
                    })
                }
            }

            _ => Err(CompileError::SimpleError {
                loc: expr.loc,
                s: "Invalid match arm syntax",
            }),
        }
    }

        /// Create a typed value with the given location
    fn typed_value(&self, loc: Loc, value: Value) -> TValue {
        Typed {
            loc,
            ty: None,
            value,
        }
    }

    /// Create a typed pattern with the given location
    fn typed_pattern(&self, loc: Loc, value: Pattern) -> TPattern {
        Typed {
            loc,
            ty: None,
            value,
        }
    }

    /// Push a new variable scope onto the stack
    fn push_scope(&mut self) {
        self.scopes.push((HashMap::new(), HashMap::new()));
    }

    /// Pop the current variable scope
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }


    /// Insert a new binding into the current (innermost) scope, always creating a fresh ID.
    fn insert_value_in_current_scope(&mut self, name: &str) -> NameId {
        let id = self.fresh_name_id();
        if let Some((value_scope, _)) = self.scopes.last_mut() {
            value_scope.insert(name.to_string(), id);
        } else {
            // If you ever lower without at least one scope pushed, that's a bug.
            debug_assert!(false, "no scope available when inserting binding");
        }
        id
    }

    /// Generate a fresh unique name ID
    fn fresh_name_id(&mut self) -> NameId {
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        id
    }

}


// Public API functions

#[cfg(test)]
mod var_scope_test {
    use super::*;
    use crate::parsing::Parser;
    use crate::program::Program;
    #[test]
    fn var_scope_is_respected() {
        // { let a = 1; { let a = 2; a } a }
        let src = "{ let a = 1; { let a = 2; a; } a; }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().expect("failed to parse expr");
        let ir = program.lower_value(expr).expect("lowering failed");
        // top-level should be a block
        let top_block = match ir.value {
            Value::Block { ref statements, .. } => statements,
            _ => panic!("expected top-level block"),
        };
        // expect three statements: let a, inner block, final a
        assert_eq!(top_block.len(), 3);
        // Grab outer let bind id
        let outer_pat = match &top_block[0].value {
            Value::Let { pat, .. } => pat,
            _ => panic!("expected outer let"),
        };
        let outer_id = match &outer_pat.value {
            Pattern::Bind(id) => *id,
            _ => panic!("expected bind pattern"),
        };
        // Final name ref refers to outer
        let final_ref = match &top_block[2].value {
            Value::NameRef(id) => *id,
            _ => panic!("expected final name reference"),
        };
        assert_eq!(outer_id, final_ref);
        // Inner block
        let inner_block = match &top_block[1].value {
            Value::Block { statements, .. } => statements,
            _ => panic!("expected inner block"),
        };
        assert_eq!(inner_block.len(), 2);
        let inner_pat = match &inner_block[0].value {
            Value::Let { pat, .. } => pat,
            _ => panic!("expected inner let"),
        };
        let inner_id = match &inner_pat.value {
            Pattern::Bind(id) => *id,
            _ => panic!("expected bind pattern"),
        };
        let inner_ref = match &inner_block[1].value {
            Value::NameRef(id) => *id,
            _ => panic!("expected inner name reference"),
        };
        assert_eq!(inner_id, inner_ref);
        // Ensure inner and outer ids differ
        assert_ne!(
            outer_id, inner_id,
            "inner and outer bindings must not collide"
        );
    }
}

#[cfg(test)]
mod lowering_tests {
    use super::*;
    use crate::parsing::Parser;
    use crate::program::{CompileError, Program};

    fn lower_block(src: &str) -> TValue {
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().unwrap();
        program.lower_value(expr).unwrap()
    }

    fn bound_id(stmt: &TValue) -> NameId {
        match &stmt.value {
            Value::Let { pat, .. } => match pat.value {
                Pattern::Bind(id) => id,
                _ => panic!("expected bind pattern"),
            },
            _ => panic!("expected let statement"),
        }
    }

    #[test]
    fn lowers_call_and_index_with_bound_names() {
        let src = "{ let f = 1; let a = 2; f(a, 3)[a, 4]; }";
        let ir = lower_block(src);

        let statements = match ir.value {
            Value::Block { statements, .. } => statements,
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 3);

        let f_id = bound_id(&statements[0]);
        let a_id = bound_id(&statements[1]);

        let (base, index_args) = match &statements[2].value {
            Value::Index { base, args } => (base, args),
            _ => panic!("expected index expression"),
        };

        let (callee, call_args) = match &base.value {
            Value::Call { callee, args } => (callee, args),
            _ => panic!("expected call base"),
        };

        match callee.value {
            Value::NameRef(id) => assert_eq!(id, f_id),
            _ => panic!("expected callee to be name"),
        }
        assert_eq!(call_args.len(), 2);
        match call_args[0].value {
            Value::NameRef(id) => assert_eq!(id, a_id),
            _ => panic!("expected first call arg to be name"),
        }
        match call_args[1].value {
            Value::Literal(Literal::Num(3)) => {}
            _ => panic!("expected literal call arg"),
        }

        assert_eq!(index_args.len(), 2);
        match index_args[0].value {
            Value::NameRef(id) => assert_eq!(id, a_id),
            _ => panic!("expected first index arg to be name"),
        }
        match index_args[1].value {
            Value::Literal(Literal::Num(4)) => {}
            _ => panic!("expected literal index arg"),
        }
    }

    #[test]
    fn lowers_match_with_wildcard_arm() {
        let src = "{ let x = 1; match x { _ => x; }; }";
        let ir = lower_block(src);

        let statements = match ir.value {
            Value::Block { statements, .. } => statements,
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 2);

        let x_id = bound_id(&statements[0]);

        let (scrutinee, arms) = match &statements[1].value {
            Value::Match { value, arms } => (value, arms),
            _ => panic!("expected match"),
        };

        match scrutinee.value {
            Value::NameRef(id) => assert_eq!(id, x_id),
            _ => panic!("expected scrutinee name"),
        }

        assert_eq!(arms.len(), 1);
        match arms[0].pat.value {
            Pattern::Wildcard => {}
            _ => panic!("expected wildcard pattern"),
        }
        match arms[0].body.value {
            Value::NameRef(id) => assert_eq!(id, x_id),
            _ => panic!("expected arm body to reference x"),
        }
    }

    #[test]
    fn lowers_match_as_block_return_value() {
        let src = "{ let x = 1; match x { _ => x; } }";
        let ir = lower_block(src);

        let (statements, return_value) = match ir.value {
            Value::Block {
                statements,
                return_value,
            } => (statements, return_value),
            _ => panic!("expected block"),
        };

        assert_eq!(statements.len(), 1);
        assert!(return_value.is_some());

        let x_id = bound_id(&statements[0]);
        let match_expr = return_value.unwrap();
        let (scrutinee, arms) = match match_expr.value {
            Value::Match { value, arms } => (value, arms),
            _ => panic!("expected match"),
        };

        match scrutinee.value {
            Value::NameRef(id) => assert_eq!(id, x_id),
            _ => panic!("expected scrutinee name"),
        }
        assert_eq!(arms.len(), 1);
    }

    #[test]
    fn unsupported_expression_reports_error() {
        let src = "if 1 2";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().unwrap();
        let err = program.lower_value(expr).unwrap_err();
        match err {
            CompileError::SimpleError { s, .. } => {
                assert_eq!(s, "Unsupported expression in IR lowering");
            }
            _ => panic!("expected simple error"),
        }
    }
}
