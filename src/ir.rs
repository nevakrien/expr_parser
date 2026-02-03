/**
 * TODO: Convert IR from tree-shaped to flat list with ids.
 *    all in the outer level function.
 *
 * This will:
 * - Simplify type inference
 * - Avoid solver needing to rediscover operands
 * - Allow linear passes over IR
 */
use crate::error_messages::{
    ERR_ACCESS_EXPECTS_NAME, ERR_EXPECTED_GEN_NAME, ERR_INVALID_MATCH_ARM,
    ERR_INVALID_MATCH_ARM_GUARD, ERR_MATCH_ARM_NEEDS_VALUE, ERR_UNSUPPORTED_EXPRESSION,
    ERR_UNSUPPORTED_EXPRESSION_ATOM, ERR_UNSUPPORTED_PATTERN,
};
use crate::parsing::{Expr, LExpr, Loc, Located, Token};
use crate::program::{CResult, CompileError, Program};
use crate::string_intern::StrId;

//this file needs to move Value and Pattern into a dense array
//note that currently the only major diffrence between Value and Pattern is Bind
//the one place which actually reads them would become simpler if we merge the 2.
//would actually remove a lot of semi duplicate code from type infrence

// Type aliases for commonly used typed/located constructs
pub type LName = Located<NameId>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ValId(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PatId(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ArmId(pub usize);

///comment is now for explaining to LLM later LLM should rewrite it to actual docs explaining usage
///
///this type would be used to store &[Value] in our dynamic array
///things that used to push/collect a vec on the fly should be converted to:
///1. push some sentinal value say Value::Void ahead of time so we have the span of size N ready.
///2. compile sub expressions and overwrite the sentinal value with the correct thing
///
///its generally fine if errors leave sentinal values as errors imply we are gona not read the thing anyway
///if cleanup would seem needed we add it AFTER this change batch
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ValueSpan {
    _start: ValId,
    _count: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PatternSpan {
    _start: PatId,
    _count: usize,
}




impl ValueSpan {
    #[inline]
    pub fn new(start: ValId, count: usize) -> Self {
        Self {
            _start: start,
            _count: count,
        }
    }

    #[inline]
    pub fn start(&self) -> ValId {
        self._start
    }

    #[inline]
    pub fn len(&self) -> usize {
        self._count
    }

    #[inline]
    pub fn at(&self, index: usize) -> ValId {
        debug_assert!(index < self._count, "ValueSpan index out of bounds");
        ValId(self._start.0 + index)
    }

    #[inline]
    pub fn ids(&self) -> impl DoubleEndedIterator<Item = ValId> + '_ {
        (self._start.0..self._start.0 + self._count).map(ValId)
    }
}

impl PatternSpan {
    #[inline]
    pub fn new(start: PatId, count: usize) -> Self {
        Self {
            _start: start,
            _count: count,
        }
    }

    #[inline]
    pub fn start(&self) -> PatId {
        self._start
    }

    #[inline]
    pub fn len(&self) -> usize {
        self._count
    }

    #[inline]
    pub fn at(&self, index: usize) -> PatId {
        debug_assert!(index < self._count, "PatternSpan index out of bounds");
        PatId(self._start.0 + index)
    }

    #[inline]
    pub fn ids(&self) -> impl DoubleEndedIterator<Item = PatId> + '_ {
        (self._start.0..self._start.0 + self._count).map(PatId)
    }
}



/// Unique identifier for names in the IR
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NameId(pub usize);

/// Literal values that can appear in the code
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Literal {
    Num(u64),
    Float(f64),
    Str(StrId),
    Void,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AccessKind {
    Dot,
    Type,
    Ptr,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AccessName {
    pub name: StrId,
}

// /// Function parameter declaration
// #[derive(Debug, Clone, PartialEq)]
// pub struct Param {
//     pub pat: IPattern,
//     pub ty: Option<IValue>,
// }

/// Single variant within an enum declaration
#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: LName,
    pub fields: PatternSpan,
}
/// Pure binary operations.
///
/// Invariant:
/// - No control flow
/// - No short-circuiting
/// - Both operands are evaluated exactly once
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinOp {
    // arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    // bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,

    // comparisons
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Pure unary operations.
///
/// Invariant:
/// - No control flow
/// - Operand is evaluated exactly once
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UnOp {
    Neg,    // -x
    Not,    // !x
    BitNot, // ~x
    Deref,  // *x
    AddrOf, // &x
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Dir {
    Inc,
    Dec,
}

/// Assignment operator, where `None` means plain `=`.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum AssignOp {
    Nothing(ValId),
    Bin(BinOp, ValId),
    Pre(Dir),
    Post(Dir),
}

/// Short-circuiting logical operations.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
}

/// Runtime IR values.
///
/// This IR is *expression-oriented* but *effect-explicit*:
/// - Mutation is represented explicitly (`Assign`, `Let`)
/// - Control flow is explicit (`If`, `While`, `Match`)
/// - `BinOp` / `UnOp` are guaranteed to be pure
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Value {
    /// Reference to a resolved name
    NameRef(NameId),

    /// Literal constant
    Literal(Literal),

    /// Wildcard pattern that matches anything (_)
    Wildcard,

    Tuple(ValueSpan),

    // Enum(StructLike),
    // Struct(StructLike),
    // Union(StructLike),


    /// Pure binary operation
    BinOp {
        op: BinOp,
        values: (ValId, ValId),
    },

    /// Pure unary operation
    UnOp {
        op: UnOp,
        value: ValId,
    },

    //===== TYPES =====
    /// Explicit type cast
    Cast {
        value: ValId,
        ty: ValId,
    },

    /// Type annotation
    TypeAnnotation {
        value: ValId,
        ty: ValId,
    },

    //==== MUTATION GATES =====
    /// Function or callable invocation
    Call {
        callee: ValId,
        args: ValueSpan,
    },

    /// Assignment with explicit sequencing.
    ///
    /// Not a `BinOp` because:
    /// - LHS is evaluated first
    /// - Mutation occurs
    Assign {
        op: AssignOp,
        target: ValId,
    },

    /// Indexing or specialization
    Index {
        base: ValId,
        args: ValueSpan,
    },

    /// Field/type access with deferred name resolution
    Access {
        base: ValId,
        name: AccessName,
        kind: AccessKind,
    },

    // ===== SCOPE =====
    /// Immutable binding
    Let {
        pat: PatId,
        value: ValId,
        else_part: Option<ValId>,
    },

    /// Lexical block
    Block {
        statements: ValueSpan,
        return_value: Option<ValId>,
    },

    //==== CONTROL FLOW =====
    /// Short-circuiting logical operations.
    LogicOp {
        op: LogicOp,
        values: (ValId, ValId),
    },

    /// Conditional expression
    If {
        cond: ValId,
        then: ValId,
        els: Option<ValId>,
    },

    /// Loop
    While {
        cond: ValId,
        body: ValId,
    },

    /// Function literal
    Func {
        generics: PatternSpan,
        params: PatternSpan,
        output_type: Option<ValId>,
        body: ValId,
    },

    /// Early return
    Return(Option<ValId>),

    Break,
    Continue,

    /// Pattern match
    Match {
        value: ValId,
        arms: ValueSpan,
    },
    MatchArm(MatchArm)
}

/// Patterns used for:
/// - Pattern matching (match expressions, function parameters)
/// - Type annotations (e.g., Option[Result[T, E]])
/// - Assignment targets (e.g., id[T] = fn(x: T) { x })
///
/// TODO: figure out if this should have a field for Value
///        this would come up for *x[T] = 2; or similar
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Pattern {
    /// Bind a value to a name (variable binding)
    Bind(NameId),
    /// Wildcard pattern that matches anything (_)
    Wildcard,
    /// Tuple pattern with multiple sub-patterns
    Tuple(PatternSpan),
    /// Literal value pattern
    Literal(Literal),
    /// Type annotation pattern (x:T)
    TypeAnnotation { pat: PatId, ty: ValId },
    //==== TODOS: ========

    // /// Struct/enum destructoring pattern
    // Destructure {
    //     ctor: LName,
    //     fields: Vec<PatternField>,
    // },
    // /// Generic type specialization (e.g., Foo[T, U])
    // GenericSpecialization {
    //     base: Box<IPattern>,
    //     args: Vec<IPattern>,
    // },
}

/// Single arm in a match expression
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MatchArm {
    pub pat: PatId,
    pub body: ValId,
}

/// Single arm in a match expression
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct StructLike {
    pub generics: PatternSpan,
    pub fields: PatternSpan,
}

impl Program {
    //TODO:
    // 1. local macros are intetionaly not handeled and scoping on macros is broken on purpose to be like C
    // 2. some places parse a value where a value/pattern check needs to be done
    pub fn lower_value(&mut self, expr: LExpr) -> CResult<ValId> {
        let loc = expr.loc.clone();
        let value = self.lower_value_inner(expr)?;
        Ok(self.id_value(loc, value))
    }

    fn lower_value_into(&mut self, target: ValId, expr: LExpr) -> CResult<ValId> {
        let loc = expr.loc.clone();
        let value = self.lower_value_inner(expr)?;
        self.set_value(target, loc, value);
        Ok(target)
    }

    fn lower_value_inner(&mut self, expr: LExpr) -> CResult<Value> {
        match expr.value {
            Expr::Atom(token) => self.lower_atom(&expr.loc, token),

            // { ... }
            Expr::Prefix(open, items) if open.value == "{" => {
                self.lower_block_expr(expr.loc, items)
            }

            // let <pat> = <value>
            Expr::Prefix(open, items) if open.value == "let" => {
                self.lower_let_expr(expr.loc, items)
            }

            // call: <callee>(args...)
            Expr::Postfix(open, items) if open.value == "(" => {
                self.lower_call_expr(expr.loc, items)
            }

            // index: <base>[args...]
            Expr::Postfix(open, items) if open.value == "[" => {
                self.lower_index_expr(expr.loc, items)
            }

            // match <value> { arms... }
            Expr::Prefix(open, items) if open.value == "match" => {
                self.lower_match_expr(expr.loc, items)
            }

            // if <cond> <then> [else <else>]
            Expr::Prefix(open, items) if open.value == "if" => self.lower_if_expr(expr.loc, items),
            Expr::Prefix(open, items) if open.value == "while" => {
                self.lower_while_expr(expr.loc, items)
            }

            // assignment
            Expr::Bin(op, pair) if op.value == "=" => self.lower_assign_expr(expr.loc, *pair),

            // fn (sig) body
            Expr::Prefix(open, items) if open.value == "fn" => self.lower_fn_expr(expr.loc, items),

            Expr::Bin(op, pair) if (op.value == "as" || op.value == ":") => {
                self.lower_cast_expr(expr.loc, op, *pair)
            }

            Expr::Bin(op, pair) if matches!(op.value, "." | "::" | "->") => {
                let (lhs, rhs) = *pair;
                self.lower_access_expr(expr.loc, op, lhs, rhs)
            }

            //fallbacks
            Expr::Prefix(open, items) => self.lower_prefix_op(expr.loc, open, items),
            Expr::Postfix(open, items) => self.lower_postfix_op(expr.loc, open, items),
            Expr::Bin(op, pair) => {
                let (lhs, rhs) = *pair;
                self.lower_binary_op(expr.loc, op, lhs, rhs)
            }
        }
    }

    #[inline(always)]
    fn lower_atom(&mut self, loc: &Loc, token: Token) -> CResult<Value> {
        let value = match token {
            Token::NumLit(n) => Value::Literal(Literal::Num(n)),
            Token::FloatLit(f) => Value::Literal(Literal::Float(f)),
            Token::StrLit(s) => Value::Literal(Literal::Str(self.str_intern.intern(&s))),
            Token::Operator("(") => Value::Literal(Literal::Void),

            Token::Ident(name) if name == "_" => Value::Wildcard,
            Token::Ident(name) => {
                let id = self.resolve_name(loc, &name)?;
                Value::NameRef(id)
            }

            Token::Operator(op) => {
                return Err(CompileError::UnsupportedForm {
                    loc: loc.clone(),
                    op_loc: Some(loc.clone()),
                    op: Some(op),
                    message: ERR_UNSUPPORTED_EXPRESSION_ATOM,
                });
            }
        };

        Ok(value)
    }

    #[inline(always)]
    fn lower_block_expr(&mut self, _loc: Loc, mut items: Vec<LExpr>) -> CResult<Value> {
        self.with_scope(|this| {
            let (statements, return_value) = if let Some(last) = items.pop() {
                let span = this.reserve_value_span(items.len());
                for (index, item) in items.into_iter().enumerate() {
                    let target = span.at(index);
                    this.lower_value_into(target, item)?;
                }

                let return_value = if !matches!(last.value, Expr::Atom(Token::Operator(";"))) {
                    Some(this.lower_value(last)?)
                } else {
                    None
                };

                (span, return_value)
            } else {
                (this.reserve_value_span(0), None)
            };

            Ok(Value::Block {
                statements,
                return_value,
            })
        })
    }

    #[inline(always)]
    fn lower_let_expr(&mut self, loc: Loc, mut items: Vec<LExpr>) -> CResult<Value> {
        debug_assert!(2 <= items.len() && items.len() <= 3);

        let else_exp = if items.len() == 3 { items.pop() } else { None };

        let value_expr = items.pop().unwrap();
        let pat_expr = items.pop().unwrap();

        let value = self.lower_value(value_expr)?;
        let pat = self.lower_pattern(pat_expr)?;

        let else_part = if let Some(exp) = else_exp {
            let v = self.with_scope(|prog| prog.lower_value(exp))?;
            Some(v)
        } else {
            None
        };

        let _ = loc;

        Ok(Value::Let {
            pat,
            value,
            else_part,
        })
    }

    #[inline(always)]
    fn lower_call_expr(&mut self, _loc: Loc, items: Vec<LExpr>) -> CResult<Value> {
        debug_assert!(!items.is_empty(), "call expression missing callee");

        let args_span = self.reserve_value_span(items.len()-1);
        let mut items = items.into_iter();


        let callee = self.lower_value(items.next().unwrap())?;
        for (index, arg) in items.enumerate() {
            let target = args_span.at(index);
            self.lower_value_into(target, arg)?;
        }

        Ok(Value::Call {
            callee,
            args: args_span,
        })
    }

    #[inline(always)]
    fn lower_index_expr(&mut self, _loc: Loc, items: Vec<LExpr>) -> CResult<Value> {
        debug_assert!(!items.is_empty(), "index expression missing base");

        let mut items = items.into_iter();
        //TODO this can actually be a generic so a pattern
        let base = self.lower_value(items.next().unwrap())?;

        let args_span = self.reserve_value_span(items.len());
        for (index, arg) in items.enumerate() {
            let target = args_span.at(index);
            self.lower_value_into(target, arg)?;
        }

        Ok(Value::Index {
            base,
            args: args_span,
        })
    }

    #[inline(always)]
    fn lower_match_expr(&mut self, loc: Loc, items: Vec<LExpr>) -> CResult<Value> {
        if items.len() < 2 {
            return Err(CompileError::SimpleError {
                loc,
                s: ERR_MATCH_ARM_NEEDS_VALUE,
            });
        }

        let mut items = items.into_iter();
        let value = self.lower_value(items.next().unwrap())?;

        let arms = self.reserve_value_span(items.len());
        for (id,arm_expr) in arms.ids().zip(items){
            let loc = arm_expr.loc.clone();
            let arm = self.lower_match_arm(arm_expr)?;
            self.set_value(id,loc,Value::MatchArm(arm));
        }
        

        Ok(Value::Match { value, arms })
    }

    #[inline(always)]
    fn lower_if_expr(&mut self, loc: Loc, items: Vec<LExpr>) -> CResult<Value> {
        if items.len() < 2 || items.len() > 3 {
            return Err(CompileError::SimpleError {
                loc,
                s: "if expression requires condition and then branch, optional else branch",
            });
        }

        let mut items = items.into_iter();
        let cond_expr = items.next().unwrap();
        let then_expr = items.next().unwrap();
        let else_expr = if items.len() == 1 {
            Some(items.next().unwrap())
        } else {
            None
        };

        let cond = self.lower_value(cond_expr)?;
        let then = self.lower_value(then_expr)?;
        let els = if let Some(else_expr) = else_expr {
            Some(self.lower_value(else_expr)?)
        } else {
            None
        };
        let _ = loc;
        Ok(Value::If { cond, then, els })
    }

    #[inline(always)]
    fn lower_while_expr(&mut self, loc: Loc, items: Vec<LExpr>) -> CResult<Value> {
        if items.len() != 2 {
            return Err(CompileError::SimpleError {
                loc,
                s: "while expression requires condition",
            });
        }

        let mut items = items.into_iter();
        let cond_expr = items.next().unwrap();
        let then_expr = items.next().unwrap();

        let cond = self.lower_value(cond_expr)?;
        let body = self.lower_value(then_expr)?;
        let _ = loc;
        Ok(Value::While { cond, body })
    }

    #[inline(always)]
    fn lower_assign_expr(&mut self, loc: Loc, pair: (LExpr, LExpr)) -> CResult<Value> {
        let (lhs, rhs) = pair;

        //TODO: target might be a pattern in rare cases? not sure
        let target = self.lower_value(lhs)?;
        let value = self.lower_value(rhs)?;

        let _ = loc;
        Ok(Value::Assign {
            op: AssignOp::Nothing(value),
            target,
        })
    }

    #[inline(always)]
    fn lower_fn_expr(&mut self, loc: Loc, items: Vec<LExpr>) -> CResult<Value> {
        debug_assert!(
            (1..=3).contains(&items.len()),
            "fn expects optional generics, signature, and optional body"
        );

        let mut items = items.into_iter().peekable();

        let mut generics_expr = None;
        if matches!(&items.peek().unwrap().value, Expr::Prefix(open, _) if open.value == "[") {
            generics_expr = Some(items.next().unwrap());
        }

        let sig_expr = items.next().expect("fn missing signature");
        let body_expr = items.next();

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

        self.with_scope(|p| {
            let generics = match generics_expr {
                Some(gen_expr)=>{
                    let Expr::Prefix(open, items) = gen_expr.value else {
                    debug_assert!(false, "fn generics must use brackets");
                    unreachable!();
                    };
                    debug_assert!(open.value == "[", "fn generics must use brackets");

                    let ans = p.reserve_pattern_span(items.len());
                    for (index,expr) in items.into_iter().enumerate() {
                        let target = ans.at(index);
                        p.lower_pattern_into(target, expr)?;

                    }
                    ans
                },
                None=>{
                    p.reserve_pattern_span(0)
                }
            };

            let params_span = p.reserve_pattern_span(param_items.len());
            for (index, param) in param_items.into_iter().enumerate() {
                //TODO support type anotation
                let target = params_span.at(index);
                p.lower_pattern_into(target, param)?;
            }

            let output_type = match ret_expr {
                Some(e) => Some(p.lower_value(e)?),
                None => None,
            };

            let body_expr = match body_expr {
                Some(expr) => expr,
                None => todo!(),
            };
            let body = p.lower_value(body_expr)?;

            let _ = loc;
            Ok(Value::Func {
                generics,
                params: params_span,
                output_type,
                body,
            })
        })
    }

    #[inline(always)]
    fn lower_cast_expr(
        &mut self,
        loc: Loc,
        op: Located<&'static str>,
        pair: (LExpr, LExpr),
    ) -> CResult<Value> {
        let (value_expr, ty_expr) = pair;
        let value = self.lower_value(value_expr)?;
        let ty = self.lower_value(ty_expr)?;
        let v = match op.value {
            "as" => Value::Cast { value, ty },
            ":" => Value::TypeAnnotation { value, ty },
            _ => panic!("unsupported cast operator `{}`", op.value),
        };
        let _ = loc;
        Ok(v)
    }

    #[inline(always)]
    fn lower_access_expr(
        &mut self,
        loc: Loc,
        op: Located<&'static str>,
        lhs: LExpr,
        rhs: LExpr,
    ) -> CResult<Value> {
        let base = self.lower_value(lhs)?;
        let name = match rhs.value {
            Expr::Atom(Token::Ident(name)) => {
                let name = self.str_intern.intern(&name);
                AccessName { name }
            }
            _ => {
                return Err(CompileError::SimpleError {
                    loc: rhs.loc,
                    s: ERR_ACCESS_EXPECTS_NAME,
                });
            }
        };

        let kind = match op.value {
            "." => AccessKind::Dot,
            "::" => AccessKind::Type,
            "->" => AccessKind::Ptr,
            _ => panic!("unsupported access operator `{}`", op.value),
        };

        let _ = loc;
        Ok(Value::Access { base, name, kind })
    }

    pub fn lower_pattern(&mut self, expr: LExpr) -> CResult<PatId> {
        let loc = expr.loc.clone();
        let pattern = self.lower_pattern_inner(expr)?;
        Ok(self.id_pattern(loc, pattern))
    }

    fn lower_pattern_into(&mut self, target: PatId, expr: LExpr) -> CResult<PatId> {
        let loc = expr.loc.clone();
        let pattern = self.lower_pattern_inner(expr)?;
        self.set_pattern(target, loc, pattern);
        Ok(target)
    }

    fn lower_pattern_inner(&mut self, expr: LExpr) -> CResult<Pattern> {
        let loc = expr.loc.clone();
        match expr.value {
            Expr::Atom(Token::Ident(name)) if name == "_" => Ok(Pattern::Wildcard),

            Expr::Atom(Token::Ident(name)) => {
                let name = self.str_intern.intern(&name);
                let id = self.insert_value_in_current_scope(name);
                Ok(Pattern::Bind(id))
            }

            Expr::Prefix(open, items) if open.value == "(" => {
                let span = self.reserve_pattern_span(items.len());
                for (index, item) in items.into_iter().enumerate() {
                    let target = span.at(index);
                    self.lower_pattern_into(target, item)?;
                }
                Ok(Pattern::Tuple(span))
            }

            // Pattern with type annotation: x:T
            Expr::Bin(op, pair) if op.value == ":" => {
                let (pat_expr, ty_expr) = *pair;
                let pat = self.lower_pattern(pat_expr)?;
                let ty = self.lower_value(ty_expr)?;

                // Create a type annotation pattern
                let _ = loc;
                Ok(Pattern::TypeAnnotation { pat, ty })
            }

            Expr::Bin(op, _) => Err(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_PATTERN,
            }),

            Expr::Prefix(op, _) | Expr::Postfix(op, _) => Err(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_PATTERN,
            }),

            _ => Err(CompileError::UnsupportedForm {
                loc,
                op_loc: None,
                op: None,
                message: ERR_UNSUPPORTED_PATTERN,
            }),
        }
    }

    #[inline(always)]
    fn lower_match_arm(&mut self, expr: LExpr) -> CResult<MatchArm> {
        match expr.value {
            Expr::Bin(op, pair) if op.value == "=>" => {
                let (pat_expr, body_expr) = *pair;
                let pat = self.lower_pattern(pat_expr)?;
                let body = self.lower_value(body_expr)?;
                Ok(MatchArm { pat, body })
            }

            Expr::Bin(op, _) => Err(CompileError::UnsupportedForm {
                loc: expr.loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_INVALID_MATCH_ARM,
            }),

            Expr::Prefix(op, _) | Expr::Postfix(op, _) => Err(CompileError::UnsupportedForm {
                loc: expr.loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_INVALID_MATCH_ARM,
            }),

            _ => Err(CompileError::UnsupportedForm {
                loc: expr.loc,
                op_loc: None,
                op: None,
                message: ERR_INVALID_MATCH_ARM,
            }),
        }
    }

    // ===============================
    // Operator lowering helpers
    // ===============================

    #[inline(always)]
    fn lower_prefix_op(
        &mut self,
        loc: Loc,
        op: Located<&'static str>,
        items: Vec<LExpr>,
    ) -> CResult<Value> {
        let unop = match op.value {
            "-" => UnOp::Neg,
            "!" => UnOp::Not,
            "~" => UnOp::BitNot,
            "*" => UnOp::Deref,
            "&" => UnOp::AddrOf,

            "++" => return self.lower_inc_dec_prefix(op.map(|_| Dir::Inc), items),
            "--" => return self.lower_inc_dec_prefix(op.map(|_| Dir::Dec), items),

            _ => {
                return Err(CompileError::UnsupportedForm {
                    loc,
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: ERR_UNSUPPORTED_EXPRESSION,
                });
            }
        };

        if items.len() != 1 {
            panic!("prefix operator `{}` with {} operands", op, items.len());
        }

        let rhs = self.lower_value(items.into_iter().next().unwrap())?;

        let _ = loc;
        Ok(Value::UnOp {
            op: unop,
            value: rhs,
        })
    }

    #[inline(always)]
    fn lower_postfix_op(
        &mut self,
        _loc: Loc,
        op: Located<&'static str>,
        items: Vec<LExpr>,
    ) -> CResult<Value> {
        match op.value {
            // these are handled earlier and must never reach here
            "(" | "[" => unreachable!("call/index should be handled before postfix ops"),

            "++" => self.lower_inc_dec_postfix(op.map(|_| Dir::Inc), items),
            "--" => self.lower_inc_dec_postfix(op.map(|_| Dir::Dec), items),

            _ => Err(CompileError::UnsupportedForm {
                loc: _loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_EXPRESSION,
            }),
        }
    }

    #[inline(always)]
    fn lower_inc_dec_prefix(&mut self, op: Located<Dir>, mut items: Vec<LExpr>) -> CResult<Value> {
        if items.len() != 1 {
            panic!("prefix operator with {} operands", items.len());
        }

        let target = self.lower_value(items.pop().unwrap())?;

        let _ = op.loc;
        Ok(Value::Assign {
            op: AssignOp::Pre(op.value),
            target,
        })
    }

    #[inline(always)]
    fn lower_inc_dec_postfix(&mut self, op: Located<Dir>, mut items: Vec<LExpr>) -> CResult<Value> {
        if items.len() != 1 {
            panic!("postfix operator with {} operands", items.len());
        }

        let target = self.lower_value(items.pop().unwrap())?;

        let _ = op.loc.clone();
        Ok(Value::Assign {
            op: AssignOp::Post(op.value),
            target,
        })
    }

    #[inline(always)]
    fn lower_binary_op(
        &mut self,
        loc: Loc,
        op: Located<&'static str>,
        lhs: LExpr,
        rhs: LExpr,
    ) -> CResult<Value> {
        if let Some(assign_op) = match op.value {
            "=" => Some(None),
            "+=" => Some(Some(BinOp::Add)),
            "-=" => Some(Some(BinOp::Sub)),
            "*=" => Some(Some(BinOp::Mul)),
            "/=" => Some(Some(BinOp::Div)),
            "%=" => Some(Some(BinOp::Mod)),
            "&=" => Some(Some(BinOp::BitAnd)),
            "|=" => Some(Some(BinOp::BitOr)),
            "^=" => Some(Some(BinOp::BitXor)),
            "<<=" => Some(Some(BinOp::Shl)),
            ">>=" => Some(Some(BinOp::Shr)),
            _ => None,
        } {
            let target = self.lower_value(lhs)?;
            let value = self.lower_value(rhs)?;

            let _ = loc;
            return Ok(Value::Assign {
                target,
                op: if let Some(o) = assign_op {
                    AssignOp::Bin(o, value)
                } else {
                    AssignOp::Nothing(value)
                },
            });
        }

        if let Some(logic_op) = match op.value {
            "&&" => Some(LogicOp::And),
            "||" => Some(LogicOp::Or),
            _ => None,
        } {
            let left = self.lower_value(lhs)?;
            let right = self.lower_value(rhs)?;
            let _ = loc;
            return Ok(Value::LogicOp {
                op: logic_op,
                values: (left, right),
            });
        }

        let binop = match op.value {
            "+" => BinOp::Add,
            "-" => BinOp::Sub,
            "*" => BinOp::Mul,
            "/" => BinOp::Div,
            "%" => BinOp::Mod,

            "&" => BinOp::BitAnd,
            "|" => BinOp::BitOr,
            "^" => BinOp::BitXor,
            "<<" => BinOp::Shl,
            ">>" => BinOp::Shr,

            "==" => BinOp::Eq,
            "!=" => BinOp::Ne,
            "<" => BinOp::Lt,
            "<=" => BinOp::Le,
            ">" => BinOp::Gt,
            ">=" => BinOp::Ge,

            _ => {
                return Err(CompileError::UnsupportedForm {
                    loc,
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: ERR_UNSUPPORTED_EXPRESSION,
                });
            }
        };

        let left = self.lower_value(lhs)?;
        let right = self.lower_value(rhs)?;

        let _ = loc;
        Ok(Value::BinOp {
            op: binop,
            values: (left, right),
        })
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
        let top_block = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected top-level block"),
        };
        // expect three statements: let a, inner block, final a
        assert_eq!(top_block.len(), 3);
        // Grab outer let bind id
        let outer_pat = match program.value(top_block[0]) {
            Value::Let { pat, .. } => pat,
            _ => panic!("expected outer let"),
        };
        let outer_id = match program.pattern(outer_pat) {
            Pattern::Bind(id) => id,
            _ => panic!("expected bind pattern"),
        };
        // Final name ref refers to outer
        let final_ref = match program.value(top_block[2]) {
            Value::NameRef(id) => id,
            _ => panic!("expected final name reference"),
        };
        assert_eq!(outer_id, final_ref);
        // Inner block
        let inner_block = match program.value(top_block[1]) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected inner block"),
        };
        assert_eq!(inner_block.len(), 2);
        let inner_pat = match program.value(inner_block[0]) {
            Value::Let { pat, .. } => pat,
            _ => panic!("expected inner let"),
        };
        let inner_id = match program.pattern(inner_pat) {
            Pattern::Bind(id) => id,
            _ => panic!("expected bind pattern"),
        };
        let inner_ref = match program.value(inner_block[1]) {
            Value::NameRef(id) => id,
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
    use crate::error_messages::ERR_ACCESS_EXPECTS_NAME;
    use crate::parsing::Parser;
    use crate::program::{CompileError, Defined, Program};

    fn lower_block(src: &str) -> (Program, ValId) {
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().expect("failed to parse expr");
        let ir = program.lower_value(expr).expect("lowering failed");
        (program, ir)
    }

    fn bound_id(program: &Program, stmt: ValId) -> NameId {
        match program.value(stmt) {
            Value::Let { pat, .. } => match program.pattern(pat) {
                Pattern::Bind(id) => id,
                _ => panic!("expected bind pattern"),
            },
            _ => panic!("expected let statement"),
        }
    }

    #[test]
    fn lowers_call_and_index_with_bound_names() {
        let src = "{ let f = 1; let a = 2; f(a, 3)[a, 4]; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 3);

        let f_id = bound_id(&program, statements[0]);
        let a_id = bound_id(&program, statements[1]);

        let (base, index_args) = match program.value(statements[2]) {
            Value::Index { base, args } => (base, args),
            _ => panic!("expected index expression"),
        };

        let (callee, call_args) = match program.value(base) {
            Value::Call { callee, args } => (callee, args),
            _ => panic!("expected call base"),
        };

        match program.value(callee) {
            Value::NameRef(id) => assert_eq!(id, f_id),
            _ => panic!("expected callee to be name"),
        }
        let call_args = call_args.ids().collect::<Vec<_>>();
        assert_eq!(call_args.len(), 2);
        match program.value(call_args[0]) {
            Value::NameRef(id) => assert_eq!(id, a_id),
            _ => panic!("expected first call arg to be name"),
        }
        match program.value(call_args[1]) {
            Value::Literal(Literal::Num(3)) => {}
            _ => panic!("expected literal call arg"),
        }

        let index_args = index_args.ids().collect::<Vec<_>>();
        assert_eq!(index_args.len(), 2);
        match program.value(index_args[0]) {
            Value::NameRef(id) => assert_eq!(id, a_id),
            _ => panic!("expected first index arg to be name"),
        }
        match program.value(index_args[1]) {
            Value::Literal(Literal::Num(4)) => {}
            _ => panic!("expected literal index arg"),
        }
    }

    #[test]
    fn lowers_match_with_wildcard_arm() {
        let src = "{ let x = 1; match x { _ => x; }; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 2);

        let x_id = bound_id(&program, statements[0]);

        let (scrutinee, arms) = match program.value(statements[1]) {
            Value::Match { value, arms } => (value, arms),
            _ => panic!("expected match"),
        };

        match program.value(scrutinee) {
            Value::NameRef(id) => assert_eq!(id, x_id),
            _ => panic!("expected scrutinee name"),
        }

        assert_eq!(arms.len(), 1);
        let Value::MatchArm(arm) = program.value(arms.at(0)) else{
            panic!("expected match arm")
        };
        match program.pattern(arm.pat) {
            Pattern::Wildcard => {}
            _ => panic!("expected wildcard pattern"),
        }
        match program.value(arm.body) {
            Value::NameRef(id) => assert_eq!(id, x_id),
            _ => panic!("expected arm body to reference x"),
        }
    }

    #[test]
    fn lowers_match_as_block_return_value() {
        let src = "{ let x = 1; match x { _ => x; } }";
        let (program, ir) = lower_block(src);

        let (statements, return_value) = match program.value(ir) {
            Value::Block {
                statements,
                return_value,
            } => (statements, return_value),
            _ => panic!("expected block"),
        };

        let statements = statements.ids().collect::<Vec<_>>();
        assert_eq!(statements.len(), 1);
        assert!(return_value.is_some());

        let x_id = bound_id(&program, statements[0]);
        let match_expr = return_value.unwrap();
        let (scrutinee, arms) = match program.value(match_expr) {
            Value::Match { value, arms } => (value, arms),
            _ => panic!("expected match"),
        };

        match program.value(scrutinee) {
            Value::NameRef(id) => assert_eq!(id, x_id),
            _ => panic!("expected scrutinee name"),
        }
        assert_eq!(arms.len(), 1);
    }

    #[test]
    fn lowers_access_for_dot_and_paths() {
        let src = "{ let a = 1; let t = 2; a.b; t::c; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 4);

        let a_id = bound_id(&program, statements[0]);
        let t_id = bound_id(&program, statements[1]);

        match program.value(statements[2]) {
            Value::Access { base, name, kind } => {
                assert_eq!(kind, AccessKind::Dot);
                match program.value(base) {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected dot base name"),
                }
                assert_eq!(program.str_intern.resolve(name.name), "b");
            }
            _ => panic!("expected dot access"),
        }

        match program.value(statements[3]) {
            Value::Access { base, name, kind } => {
                assert_eq!(kind, AccessKind::Type);
                match program.value(base) {
                    Value::NameRef(id) => assert_eq!(id, t_id),
                    _ => panic!("expected type base name"),
                }
                assert_eq!(program.str_intern.resolve(name.name), "c");
            }
            _ => panic!("expected type access"),
        }
    }

    #[test]
    fn lowers_fn_generics_in_scope() {
        let src = "f = fn[T](x:T){ let y:T = x; y }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        program.lower_all(&mut parser).unwrap();

        let f_name = program.str_intern.intern("f");
        let f_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.get(&f_name))
            .expect("missing f binding");
        let defined = program.definitions.get(&f_id).expect("missing definition");

        match defined {
            Defined::Value(value) => match program.value(*value) {
                Value::Func { generics, .. } => assert_eq!(generics.len(), 1),
                _ => panic!("expected function value"),
            },
            _ => panic!("expected value definition"),
        }
    }

    #[test]
    fn lowers_mutual_function_references() {
        let src = "f = fn(){ g() } g = fn(){ f() }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        program.lower_all(&mut parser).unwrap();

        let f_name = program.str_intern.intern("f");
        let f_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.get(&f_name))
            .expect("missing f binding");
        let g_name = program.str_intern.intern("g");
        let g_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.get(&g_name))
            .expect("missing g binding");

        let f_def = program
            .definitions
            .get(&f_id)
            .expect("missing f definition");

        let g_def = program
            .definitions
            .get(&g_id)
            .expect("missing g definition");

        let f_body = match f_def {
            Defined::Value(value) => match program.value(*value) {
                Value::Func { body, .. } => body,
                _ => panic!("expected f to be a function"),
            },
            _ => panic!("expected f to lower to a value"),
        };

        let g_body = match g_def {
            Defined::Value(value) => match program.value(*value) {
                Value::Func { body, .. } => body,
                _ => panic!("expected g to be a function"),
            },
            _ => panic!("expected g to lower to a value"),
        };

        let f_call = match program.value(f_body) {
            Value::Call { callee, .. } => callee,
            Value::Block { return_value, .. } => return_value
                .as_ref()
                .map(|value| match program.value(*value) {
                    Value::Call { callee, .. } => callee,
                    _ => panic!("expected f return to be a call"),
                })
                .expect("expected f to return a call"),
            _ => panic!("expected f body to be a call or block"),
        };
        let g_call = match program.value(g_body) {
            Value::Call { callee, .. } => callee,
            Value::Block { return_value, .. } => return_value
                .as_ref()
                .map(|value| match program.value(*value) {
                    Value::Call { callee, .. } => callee,
                    _ => panic!("expected g return to be a call"),
                })
                .expect("expected g to return a call"),
            _ => panic!("expected g body to be a call or block"),
        };

        match program.value(f_call) {
            Value::NameRef(id) => assert_eq!(id, g_id),
            _ => panic!("expected f to call g"),
        }
        match program.value(g_call) {
            Value::NameRef(id) => assert_eq!(id, f_id),
            _ => panic!("expected g to call f"),
        }
    }

    #[test]
    fn access_requires_identifier_rhs() {
        let src = "{ let a = 1; a->++a; }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().unwrap();
        let err = program.lower_value(expr).unwrap_err();
        match err {
            CompileError::SimpleError { s, .. } => {
                assert_eq!(s, ERR_ACCESS_EXPECTS_NAME);
            }
            _ => panic!("expected simple error"),
        }
    }

    #[test]
    fn lowers_compound_assign_inc_and_logic_ops() {
        let src = "{ let a = 1; a += 2; ++a; a++; a && a; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 5);

        let a_id = bound_id(&program, statements[0]);

        match program.value(statements[1]) {
            Value::Assign { op, target } => {
                match op {
                    AssignOp::Bin(bin_op, value) => {
                        assert_eq!(bin_op, BinOp::Add);
                        match program.value(value) {
                            Value::Literal(Literal::Num(2)) => {}
                            _ => panic!("expected assign literal value"),
                        }
                    }
                    _ => panic!("expected compound assignment op"),
                }
                match program.value(target) {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected assign target name"),
                }
            }
            _ => panic!("expected compound assignment"),
        }

        match program.value(statements[2]) {
            Value::Assign { op, target } => {
                assert!(matches!(op, AssignOp::Pre(Dir::Inc)));
                match program.value(target) {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected inc target name"),
                }
            }
            _ => panic!("expected prefix inc assignment"),
        }

        match program.value(statements[3]) {
            Value::Assign { op, target } => {
                assert!(matches!(op, AssignOp::Post(Dir::Inc)));
                match program.value(target) {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected inc target name"),
                }
            }
            _ => panic!("expected postfix inc assignment"),
        }

        match program.value(statements[4]) {
            Value::LogicOp { op, values } => {
                assert_eq!(op, LogicOp::And);
                let (left, right) = values;
                match program.value(left) {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected logic left name"),
                }
                match program.value(right) {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected logic right name"),
                }
            }
            _ => panic!("expected logic op"),
        }
    }

    #[test]
    fn lowers_cast_expression() {
        let src = "{ let a = 1; a as int; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 2);

        let a_id = bound_id(&program, statements[0]);
        let cast_expr = statements[1];
        match program.value(cast_expr) {
            Value::Cast { value, ty } => {
                match program.value(value) {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected cast value to be name"),
                }
                match program.value(ty) {
                    Value::NameRef(id) => assert_ne!(id, a_id),
                    _ => panic!("expected cast type pattern"),
                }
            }
            _ => panic!("expected cast expression"),
        }
    }

    #[test]
    fn lowers_if_expression() {
        let src = "if 1 { 2 }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().unwrap();
        let ir = program.lower_value(expr).unwrap();

        match program.value(ir) {
            Value::If { cond, then, els } => {
                match program.value(cond) {
                    Value::Literal(Literal::Num(1)) => {}
                    _ => panic!("expected condition to be literal 1"),
                }
                match program.value(then) {
                    Value::Block {
                        statements,
                        return_value,
                    } => {
                        assert_eq!(statements.len(), 0);
                        assert!(return_value.is_some());
                        match program.value(return_value.unwrap()) {
                            Value::Literal(Literal::Num(2)) => {}
                            _ => panic!("expected then branch to return literal 2"),
                        }
                    }
                    _ => panic!("expected then branch to be a block"),
                }
                assert!(els.is_none(), "expected no else branch");
            }
            _ => panic!("expected if expression"),
        }
    }

    #[test]
    fn lowers_if_else_expression() {
        let src = "if 1 { 2 } else { 3 }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().unwrap();
        let ir = program.lower_value(expr).unwrap();

        match program.value(ir) {
            Value::If { cond, then, els } => {
                match program.value(cond) {
                    Value::Literal(Literal::Num(1)) => {}
                    _ => panic!("expected condition to be literal 1"),
                }
                match program.value(then) {
                    Value::Block {
                        statements,
                        return_value,
                    } => {
                        assert_eq!(statements.len(), 0);
                        assert!(return_value.is_some());
                        match program.value(return_value.unwrap()) {
                            Value::Literal(Literal::Num(2)) => {}
                            _ => panic!("expected then branch to return literal 2"),
                        }
                    }
                    _ => panic!("expected then branch to be a block"),
                }
                assert!(els.is_some(), "expected else branch");
                match program.value(els.unwrap()) {
                    Value::Block {
                        statements,
                        return_value,
                    } => {
                        assert_eq!(statements.len(), 0);
                        assert!(return_value.is_some());
                        match program.value(return_value.unwrap()) {
                            Value::Literal(Literal::Num(3)) => {}
                            _ => panic!("expected else branch to return literal 3"),
                        }
                    }
                    _ => panic!("expected else branch to be a block"),
                }
            }
            _ => panic!("expected if expression"),
        }
    }

    #[test]
    fn lowers_pattern_with_type_annotation() {
        let src = "{ let x:int = 1; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 1);

        let let_stmt = statements[0];
        match program.value(let_stmt) {
            Value::Let {
                pat,
                value,
                else_part,
            } => {
                assert!(else_part.is_none(), "expected no else part");
                match program.pattern(pat) {
                    Pattern::TypeAnnotation { pat: inner_pat, ty } => {
                        // The inner pattern should bind a new name 'x'
                        match program.pattern(inner_pat) {
                            Pattern::Bind(_x_id) => {
                                // Verify the value is the expected literal
                                match program.value(value) {
                                    Value::Literal(Literal::Num(1)) => {}
                                    _ => panic!("expected literal value"),
                                }
                            }
                            _ => panic!("expected bind pattern for variable name"),
                        }
                        // The type should resolve to the predefined 'int' name
                        match program.value(ty) {
                            Value::NameRef(_int_id) => {} // Type should be a name reference to 'int'
                            _ => panic!("expected type to be name reference to predefined type"),
                        }
                    }
                    _ => panic!("expected type annotation pattern"),
                }
            }
            _ => panic!("expected let statement"),
        }
    }
}
