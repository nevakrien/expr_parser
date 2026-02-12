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
    ERR_ACCESS_EXPECTS_NAME, ERR_GOTO_OUTSIDE_FUNCTION, ERR_INVALID_MATCH_ARM,
    ERR_LABEL_NAME_REQUIRED, ERR_LABEL_OUTSIDE_FUNCTION, ERR_MATCH_ARM_NEEDS_VALUE,
    ERR_PIPE_REQUIRES_CALL, ERR_POS_ARG_AFTER_NAMED, ERR_UNSUPPORTED_EXPRESSION,
    ERR_UNSUPPORTED_EXPRESSION_ATOM, ERR_UNSUPPORTED_PATTERN, ERR_UNSUPPORTED_TYPE_EXPR,
};
use crate::parsing::{Expr, LExpr, LFixed, Loc, Located, Token};
use crate::program::{CResult, CompileError, Program};
use crate::string_intern::StrId;

//this file needs to move Value and Pattern into a dense array
//note that currently the only major diffrence between Value and Pattern is Bind
//the one place which actually reads them would become simpler if we merge the 2.
//would actually remove a lot of semi duplicate code from type infrence

/// Unique identifier for names in the IR
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NameId(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct LabelId(pub usize);

impl LabelId {
    pub const PENDING: Self = Self(usize::MAX);
}

// Type aliases for commonly used typed/located constructs
pub type LName = Located<NameId>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ValId(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PatId(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TExpId(pub usize);

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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeExprSpan {
    _start: TExpId,
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
    pub fn is_empty(&self) -> bool {
        self._count == 0
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
    pub fn is_empty(&self) -> bool {
        self._count == 0
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

impl TypeExprSpan {
    #[inline]
    pub fn new(start: TExpId, count: usize) -> Self {
        Self {
            _start: start,
            _count: count,
        }
    }

    #[inline]
    pub fn start(&self) -> TExpId {
        self._start
    }

    #[inline]
    pub fn len(&self) -> usize {
        self._count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self._count == 0
    }

    #[inline]
    pub fn at(&self, index: usize) -> TExpId {
        debug_assert!(index < self._count, "PatternSpan index out of bounds");
        TExpId(self._start.0 + index)
    }

    #[inline]
    pub fn ids(&self) -> impl DoubleEndedIterator<Item = TExpId> + '_ {
        (self._start.0..self._start.0 + self._count).map(TExpId)
    }
}

/// Literal values that can appear in the code
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Literal {
    Num(u64),
    Float(f64),
    Bool(bool),
    Str(StrId),
    Void,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AccessKind {
    Dot,
    Static,
    Ptr,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CallingConvention {
    C,
    Hot,
    Unknown,
}

impl CallingConvention {
    #[inline(always)]
    pub fn from_fn_keyword(keyword: &str) -> Option<Self> {
        match keyword {
            "fn" => Some(Self::Hot),
            "cfn" => Some(Self::C),
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum StructLayoutSpec {
    C,
    Hot,
}

// /// Function parameter declaration
// #[derive(Debug, Clone, PartialEq)]
// pub struct Param {
//     pub pat: IPattern,
//     pub ty: Option<IValue>,
// }

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
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum UnOp {
    Neg,    // -x
    Not,    // !x
    BitNot, // ~x
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

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Call {
    pub base: ValId,
    pub args: ValueSpan,
    pub named_args_start: usize,
}

impl Call {
    pub fn pos_args(&self) -> ValueSpan {
        ValueSpan {
            _start: self.args._start,
            _count: self.named_args_start,
        }
    }

    pub fn named_args(&self) -> ValueSpan {
        ValueSpan {
            _start: ValId(self.args._start.0 + self.named_args_start),
            _count: self.args._count - self.named_args_start,
        }
    }
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

    /// things like struct arguments and function name arguments
    /// can be refered to by name as name=x
    Labeled {
        name: StrId,
        value: ValId,
    },

    /// Literal constant
    Literal(Literal),

    /// Wildcard pattern that matches anything (_)
    Wildcard,

    LabelDecl(LabelId),

    Tuple(ValueSpan),
    Array(ValueSpan),

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
    Deref(ValId),                   // *x
    AddrOf(ValId, Option<VarKind>), // &x

    Construct(Call),

    //===== TYPES =====
    /// Explicit type cast
    Cast {
        value: ValId,
        ty: TExpId,
    },

    /// Type annotation
    TypeAnnotation {
        value: ValId,
        ty: TExpId,
    },

    TypeDef {
        pat: PatId,
        ty: TExpId,
    },

    //==== MUTATION GATES =====
    /// Function or callable invocation
    Call(Call),

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
    Index(Call),

    /// Field/type access with deferred name resolution
    Access {
        base: ValId,
        name: StrId,
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
        calling_convention: CallingConvention,
        generics: PatternSpan,
        params: PatternSpan,
        output_type: Option<TExpId>,
        body: Option<ValId>,
    },

    /// Early return
    Return(Option<ValId>),

    Goto(LabelId),

    Break,
    Continue,

    /// Pattern match
    Match {
        value: ValId,
        arms: ValueSpan,
    },
    MatchArm(MatchArm),
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum VarKind {
    Mut,
    Const,
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
    Bind(NameId, VarKind),
    /// Wildcard pattern that matches anything (_)
    Wildcard(VarKind),
    /// Tuple pattern with multiple sub-patterns
    Tuple(PatternSpan),
    /// Literal value pattern
    Literal(Literal),
    /// Type annotation pattern (x:T)
    TypeAnnotation { pat: PatId, ty: TExpId },
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

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TypeExpr {
    /// Bind a value to a name (variable binding)
    NameRef(NameId),
    /// Wildcard pattern that matches anything (_)
    Wildcard,
    /// Tuple pattern with multiple sub-patterns
    Tuple(TypeExprSpan),

    /// specialization
    Index {
        base: TExpId,
        args: TypeExprSpan,
    },

    Ptr {
        base: TExpId,
        raw: bool,
        mutable: bool,
    },

    Enum(StructLike),
    Struct(StructLike),
    Union(StructLike),
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
    pub layout: StructLayoutSpec,
    pub generics: PatternSpan,
    pub fields: PatternSpan,
}

impl Program {
    //TODO:
    // 1. local macros are intetionaly not handeled and scoping on macros is broken on purpose to be like C
    // 2. some places parse a value where a value/pattern check needs to be done
    pub fn lower_value(&mut self, expr: LExpr) -> CResult<ValId> {
        let target = self.id_value(expr.loc.clone(), Value::Literal(Literal::Void));
        self.lower_value_into(target, expr)
    }

    #[inline]
    fn lower_value_into(&mut self, target: ValId, expr: LExpr) -> CResult<ValId> {
        let loc = expr.loc.clone();
        if let Expr::Prefix(op, items) = expr.value {
            if op.value == "goto" {
                return self.lower_goto_into(target, loc, op, items);
            }

            let value = self.lower_value_inner(loc.clone().with(Expr::Prefix(op, items)))?;
            self.set_value(target, loc, value);
            return Ok(target);
        }

        let value = self.lower_value_inner(expr)?;
        self.set_value(target, loc, value);
        Ok(target)
    }

    #[inline]
    fn lower_value_into_labeled(&mut self, target: ValId, expr: LExpr) -> CResult<bool> {
        let loc = expr.loc.clone();
        match expr.value {
            Expr::Bin(op, pair) if op.value == "=" => {
                if let Expr::Atom(Token::Ident(n)) = pair.0.value {
                    let value = self.lower_value(pair.1)?;
                    let name = self.str_intern.intern(&n);
                    self.set_value(target, loc, Value::Labeled { name, value });
                    Ok(true)
                } else {
                    self.lower_value_into(target, loc.with(Expr::Bin(op, pair)))?;
                    Ok(false)
                }
            }
            _ => {
                self.lower_value_into(target, expr)?;
                Ok(false)
            }
        }
    }

    fn lower_value_inner(&mut self, expr: LExpr) -> CResult<Value> {
        match expr.value {
            Expr::Atom(token) => self.lower_atom(&expr.loc, token),

            // { ... }
            Expr::Prefix(open, items) if open.value == "{" => {
                self.lower_block_expr(expr.loc, items)
            }

            Expr::Prefix(open, items) if open.value == "(" || open.value == "[" => {
                self.lower_tuple_expr(expr.loc, items, open.value)
            }

            // call: <base>(args...)
            Expr::Postfix(open, items) if matches!(open.value, "(" | "[" | "{") => {
                let call = self.lower_call_like_expr(expr.loc, items)?;
                Ok(match open.value {
                    "(" => Value::Call(call),
                    "[" => Value::Index(call),
                    "{" => Value::Construct(call),
                    _ => unreachable!(),
                })
            }

            // assignment
            Expr::Bin(op, pair) if op.value == "=" => self.lower_assign_expr(expr.loc, *pair),
            Expr::Bin(op, pair) if (op.value == "as" || op.value == ":") => {
                self.lower_cast_expr(expr.loc, op, *pair)
            }
            Expr::Bin(op, pair) if matches!(op.value, "." | "::" | "->") => {
                let (lhs, rhs) = *pair;
                self.lower_access_expr(expr.loc, op, lhs, rhs)
            }

            // let <pat> = <value>
            Expr::Prefix(open, items) if open.value == "let" => {
                self.lower_let_expr(expr.loc, items, VarKind::Const)
            }
            Expr::Prefix(open, items) if open.value == "var" => {
                self.lower_let_expr(expr.loc, items, VarKind::Mut)
            }

            Expr::Prefix(open, items) if open.value == "type" => {
                self.lower_typedef_expr(expr.loc, items)
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
            Expr::Prefix(open, items) if open.value == "break" => {
                self.lower_break_expr(expr.loc, open, items)
            }
            Expr::Prefix(open, items) if open.value == "continue" => {
                self.lower_continue_expr(expr.loc, open, items)
            }
            Expr::Prefix(open, items) if open.value == "return" => {
                self.lower_return_expr(expr.loc, open, items)
            }
            // fn/cfn (sig) [body]
            Expr::Prefix(open, items) if open.value == "fn" || open.value == "cfn" => {
                self.lower_fn_expr(expr.loc, open, items)
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
            Token::Operator("true") => Value::Literal(Literal::Bool(true)),
            Token::Operator("false") => Value::Literal(Literal::Bool(false)),

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
                    if let Some(name) = Self::extract_label_name(&item) {
                        if !this.in_function_body() {
                            return Err(CompileError::SimpleError {
                                loc: item.loc,
                                s: ERR_LABEL_OUTSIDE_FUNCTION,
                            });
                        }

                        let id = this.define_label_name(&item.loc, name)?;
                        this.set_value(target, item.loc, Value::LabelDecl(id));
                    } else {
                        this.lower_value_into(target, item)?;
                    }
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

    fn extract_label_name(expr: &LExpr) -> Option<&str> {
        let Expr::Prefix(op, args) = &expr.value else {
            return None;
        };
        if op.value != "`" || args.len() != 1 {
            return None;
        }

        match &args[0].value {
            Expr::Atom(Token::Ident(name)) => Some(name),
            _ => None,
        }
    }

    #[inline(always)]
    fn lower_typedef_expr(&mut self, loc: Loc, mut items: Vec<LExpr>) -> CResult<Value> {
        debug_assert!(2 == items.len());

        let value_expr = items.pop().unwrap();
        let pat_expr = items.pop().unwrap();

        let pat = self.lower_pattern(pat_expr, VarKind::Const)?;
        let ty = self.lower_type_expr(value_expr)?;

        let _ = loc;

        Ok(Value::TypeDef { pat, ty })
    }

    #[inline(always)]
    fn lower_let_expr(&mut self, loc: Loc, mut items: Vec<LExpr>, m: VarKind) -> CResult<Value> {
        debug_assert!(2 <= items.len() && items.len() <= 3);

        let else_exp = if items.len() == 3 { items.pop() } else { None };

        let value_expr = items.pop().unwrap();
        let pat_expr = items.pop().unwrap();

        let value = self.lower_value(value_expr)?;
        let pat = self.lower_pattern(pat_expr, m)?;

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
    fn lower_tuple_expr(
        &mut self,
        _loc: Loc,
        items: Vec<LExpr>,
        open: &'static str,
    ) -> CResult<Value> {
        let parts = self.reserve_value_span(items.len());

        for (index, arg) in items.into_iter().enumerate() {
            let target = parts.at(index);
            self.lower_value_into(target, arg)?;
        }

        Ok(match open {
            "(" => Value::Tuple(parts),
            "[" => Value::Array(parts),
            _ => unreachable!(),
        })
    }

    #[inline(always)]
    fn lower_call_like_expr(&mut self, _loc: Loc, items: Vec<LExpr>) -> CResult<Call> {
        debug_assert!(!items.is_empty(), "call expression missing base");

        let mut items = items.into_iter();
        let base = self.lower_value(items.next().unwrap())?;

        let args = self.reserve_value_span(items.len());
        let mut named_args_start = args.len();
        for (index, arg) in items.enumerate() {
            let target = args.at(index);
            let arg_loc = arg.loc.clone();
            if self.lower_value_into_labeled(target, arg)? {
                if named_args_start == args.len() {
                    named_args_start = index;
                }
            } else {
                if named_args_start != args.len() {
                    // Positional arguments after named ones break the contiguous split.
                    return Err(CompileError::SimpleError {
                        loc: arg_loc,
                        s: ERR_POS_ARG_AFTER_NAMED,
                    });
                }
            }
        }

        Ok(Call {
            base,
            args,
            named_args_start,
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
        for (id, arm_expr) in arms.ids().zip(items) {
            self.with_scope(|p|{
                let loc = arm_expr.loc.clone();
                let arm = p.lower_match_arm(arm_expr)?;
                p.set_value(id, loc, Value::MatchArm(arm));
                Ok(())
            })?;
            
        };

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
    fn lower_break_expr(&mut self, loc: Loc, op: LFixed, items: Vec<LExpr>) -> CResult<Value> {
        if !items.is_empty() {
            return Err(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_EXPRESSION,
            });
        }
        Ok(Value::Break)
    }

    #[inline(always)]
    fn lower_continue_expr(&mut self, loc: Loc, op: LFixed, items: Vec<LExpr>) -> CResult<Value> {
        if !items.is_empty() {
            return Err(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_EXPRESSION,
            });
        }
        Ok(Value::Continue)
    }

    #[inline(always)]
    fn lower_return_expr(&mut self, loc: Loc, op: LFixed, mut items: Vec<LExpr>) -> CResult<Value> {
        if items.len() > 1 {
            return Err(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_EXPRESSION,
            });
        }

        let value = if let Some(value) = items.pop() {
            Some(self.lower_value(value)?)
        } else {
            None
        };
        Ok(Value::Return(value))
    }

    #[inline(always)]
    fn lower_goto_into(
        &mut self,
        target: ValId,
        loc: Loc,
        op: LFixed,
        mut items: Vec<LExpr>,
    ) -> CResult<ValId> {
        if !self.in_function_body() {
            return Err(CompileError::SimpleError {
                loc,
                s: ERR_GOTO_OUTSIDE_FUNCTION,
            });
        }
        if items.len() != 1 {
            return Err(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_EXPRESSION,
            });
        }

        let label_expr = items.pop().unwrap();
        let Some(label_name) = Self::extract_label_name(&label_expr) else {
            return Err(CompileError::SimpleError {
                loc: label_expr.loc,
                s: ERR_LABEL_NAME_REQUIRED,
            });
        };

        let label = self.use_label_name_for_goto(target, &label_expr.loc, label_name)?;
        self.set_value(target, loc, Value::Goto(label));
        Ok(target)
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
    fn lower_fn_expr(&mut self, loc: Loc, fn_kw: LFixed, items: Vec<LExpr>) -> CResult<Value> {
        debug_assert!(
            (1..=3).contains(&items.len()),
            "fn expects optional generics, signature, and optional body"
        );

        let mut items = items.into_iter().peekable();

        let mut generics_expr = None;
        if let Some(peek) = items.peek()
            && matches!(&peek.value, Expr::Prefix(open, _) if open.value == "[")
        {
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

        let calling_convention = CallingConvention::from_fn_keyword(fn_kw.value).unwrap();

        self.with_function_labels(|this| {
            this.with_scope(|p| {
                let generics = match generics_expr {
                    Some(gen_expr) => {
                        let Expr::Prefix(open, items) = gen_expr.value else {
                            debug_assert!(false, "fn generics must use brackets");
                            unreachable!();
                        };
                        debug_assert!(open.value == "[", "fn generics must use brackets");

                        let ans = p.reserve_pattern_span(items.len());
                        for (index, expr) in items.into_iter().enumerate() {
                            let target = ans.at(index);
                            p.lower_pattern_into(target, expr, VarKind::Const)?;
                        }
                        ans
                    }
                    None => p.reserve_pattern_span(0),
                };

                let params_span = p.reserve_pattern_span(param_items.len());
                for (index, param) in param_items.into_iter().enumerate() {
                    //TODO support type anotation
                    let target = params_span.at(index);
                    p.lower_pattern_into(target, param, VarKind::Const)?;
                }

                let output_type = match ret_expr {
                    Some(e) => Some(p.lower_type_expr(e)?),
                    None => None,
                };

                let body = if let Some(body_expr) = body_expr {
                    Some(p.lower_value(body_expr)?)
                } else {
                    None
                };

                let _ = loc;
                Ok(Value::Func {
                    calling_convention,
                    generics,
                    params: params_span,
                    output_type,
                    body,
                })
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
        let ty = self.lower_type_expr(ty_expr)?;
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
            Expr::Atom(Token::Ident(name)) => self.str_intern.intern(&name),
            _ => {
                return Err(CompileError::SimpleError {
                    loc: rhs.loc,
                    s: ERR_ACCESS_EXPECTS_NAME,
                });
            }
        };

        let kind = match op.value {
            "." => AccessKind::Dot,
            "::" => AccessKind::Static,
            "->" => AccessKind::Ptr,
            _ => panic!("unsupported access operator `{}`", op.value),
        };

        let _ = loc;
        Ok(Value::Access { base, name, kind })
    }

    pub fn lower_pattern(&mut self, expr: LExpr, m: VarKind) -> CResult<PatId> {
        let loc = expr.loc.clone();
        let pattern = self.lower_pattern_inner(expr, m)?;
        Ok(self.id_pattern(loc, pattern))
    }

    fn lower_pattern_into(&mut self, target: PatId, expr: LExpr, m: VarKind) -> CResult<PatId> {
        let loc = expr.loc.clone();
        let pattern = self.lower_pattern_inner(expr, m)?;
        self.set_pattern(target, loc, pattern);
        Ok(target)
    }

    fn lower_pattern_inner(&mut self, expr: LExpr, m: VarKind) -> CResult<Pattern> {
        let loc = expr.loc.clone();
        match expr.value {
            Expr::Atom(Token::Ident(name)) if name == "_" => Ok(Pattern::Wildcard(m)),

            Expr::Atom(Token::Ident(name)) => {
                let name = self.str_intern.intern(&name);
                let id = self.insert_value_in_current_scope(name);
                Ok(Pattern::Bind(id, m))
            }

            // Pattern with type annotation: x:T
            Expr::Bin(op, pair) if op.value == ":" => {
                let (pat_expr, ty_expr) = *pair;
                let pat = self.lower_pattern(pat_expr, m)?;
                let ty = self.lower_type_expr(ty_expr)?;

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

            Expr::Prefix(open, mut items) if open.value == "mut" => {
                self.lower_pattern_inner(items.pop().unwrap(), VarKind::Mut)
            }
            Expr::Prefix(open, mut items) if open.value == "const" => {
                self.lower_pattern_inner(items.pop().unwrap(), VarKind::Const)
            }

            Expr::Prefix(open, items) if open.value == "(" => {
                let span = self.reserve_pattern_span(items.len());
                for (index, item) in items.into_iter().enumerate() {
                    let target = span.at(index);
                    self.lower_pattern_into(target, item, m)?;
                }
                Ok(Pattern::Tuple(span))
            }

            Expr::Prefix(op, _) | Expr::Postfix(op, _) => Err(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_PATTERN,
            }),

            Expr::Atom(_) => Err(CompileError::UnsupportedForm {
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
                let pat = self.lower_pattern(pat_expr, VarKind::Const)?;
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
        if items.len() != 1 {
            todo!("this should not be a hard error")
        }

        let mut rhs_expr = items.into_iter().next().unwrap();
        let unop = match op.value {
            "-" => UnOp::Neg,
            "!" => UnOp::Not,
            "~" => UnOp::BitNot,
            "*" => {
                let rhs = self.lower_value(rhs_expr)?;
                return Ok(Value::Deref(rhs));
            }
            "&" => {
                let mut kind = None;
                if let Expr::Prefix(ref inner_op, ref mut inner_items) = rhs_expr.value {
                    if matches!(inner_op.value, "mut" | "const") {
                        debug_assert_eq!(inner_items.len(), 1);
                        kind = Some(if inner_op.value == "mut" {
                            VarKind::Mut
                        } else {
                            VarKind::Const
                        });

                        let mut inner = inner_items.pop().unwrap();
                        std::mem::swap(&mut rhs_expr, &mut inner);
                    }
                }

                let rhs = self.lower_value(rhs_expr)?;
                return Ok(Value::AddrOf(rhs, kind));
            }

            "++" => return self.lower_inc_dec_prefix(op.map(|_| Dir::Inc), vec![rhs_expr]),
            "--" => return self.lower_inc_dec_prefix(op.map(|_| Dir::Dec), vec![rhs_expr]),

            _ => {
                return Err(CompileError::UnsupportedForm {
                    loc,
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: ERR_UNSUPPORTED_EXPRESSION,
                });
            }
        };

        let rhs = self.lower_value(rhs_expr)?;

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
        debug_assert_eq!(items.len(), 1);

        let target = self.lower_value(items.pop().unwrap())?;

        let _ = op.loc;
        Ok(Value::Assign {
            op: AssignOp::Pre(op.value),
            target,
        })
    }

    #[inline(always)]
    fn lower_inc_dec_postfix(&mut self, op: Located<Dir>, mut items: Vec<LExpr>) -> CResult<Value> {
        debug_assert_eq!(items.len(), 1);

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

            "|>" => {
                return self.lower_pipe_expr(loc, lhs, rhs);
            }

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

    #[inline(always)]
    fn lower_pipe_expr(&mut self, _loc: Loc, lhs: LExpr, rhs: LExpr) -> CResult<Value> {
        let Located { loc, value } = rhs;
        match value {
            Expr::Postfix(open, mut items) if open.value == "(" => {
                if items.is_empty() {
                    return Err(CompileError::SimpleError {
                        loc,
                        s: ERR_PIPE_REQUIRES_CALL,
                    });
                }

                items.insert(1, lhs);
                let call = self.lower_call_like_expr(loc.clone(), items)?;
                Ok(Value::Call(call))
            }
            _ => Err(CompileError::SimpleError {
                loc,
                s: ERR_PIPE_REQUIRES_CALL,
            }),
        }
    }

    pub fn lower_type_expr(&mut self, expr: LExpr) -> CResult<TExpId> {
        let loc = expr.loc.clone();
        let exp = self.lower_type_expr_inner(expr)?;
        Ok(self.id_type_expr(loc, exp))
    }

    fn lower_type_expr_into(&mut self, target: TExpId, expr: LExpr) -> CResult<TExpId> {
        let loc = expr.loc.clone();
        let exp = self.lower_type_expr_inner(expr)?;
        self.set_type_expr(target, loc, exp);
        Ok(target)
    }

    fn lower_type_expr_inner(&mut self, expr: LExpr) -> CResult<TypeExpr> {
        let loc = expr.loc.clone();
        match expr.value {
            Expr::Atom(Token::Ident(name)) if name == "_" => Ok(TypeExpr::Wildcard),

            Expr::Atom(Token::Ident(name)) => {
                let id = self.resolve_name(&loc, &name)?;
                Ok(TypeExpr::NameRef(id))
            }

            Expr::Atom(Token::Operator("(")) => {
                let span = self.reserve_type_expr_span(0);
                Ok(TypeExpr::Tuple(span))
            }

            Expr::Prefix(open, items) if open.value == "(" => {
                let span = self.reserve_type_expr_span(items.len());
                for (index, item) in items.into_iter().enumerate() {
                    let target = span.at(index);
                    self.lower_type_expr_into(target, item)?;
                }
                Ok(TypeExpr::Tuple(span))
            }

            Expr::Postfix(open, items) if open.value == "[" => {
                if items.is_empty() {
                    return Err(CompileError::UnsupportedForm {
                        loc,
                        op_loc: Some(open.loc),
                        op: Some(open.value),
                        message: ERR_UNSUPPORTED_TYPE_EXPR,
                    });
                }

                let mut items = items.into_iter();
                let base = self.lower_type_expr(items.next().unwrap())?;
                let args_span = self.reserve_type_expr_span(items.len());
                for (index, arg) in items.enumerate() {
                    let target = args_span.at(index);
                    self.lower_type_expr_into(target, arg)?;
                }

                Ok(TypeExpr::Index {
                    base,
                    args: args_span,
                })
            }

            Expr::Prefix(op, mut items) if matches!(op.value, "*" | "&") => {
                if items.len() != 1 {
                    return Err(CompileError::UnsupportedForm {
                        loc,
                        op_loc: Some(op.loc),
                        op: Some(op.value),
                        message: ERR_UNSUPPORTED_TYPE_EXPR,
                    });
                }

                let raw = op.value == "*";
                let mut mutable = raw;
                let mut inner = items.pop().unwrap();

                if let Expr::Prefix(ref inner_op, ref mut inner_items) = inner.value
                    && matches!(inner_op.value, "mut" | "const")
                {
                    if inner_items.len() != 1 {
                        return Err(CompileError::UnsupportedForm {
                            loc,
                            op_loc: Some(inner_op.loc.clone()),
                            op: Some(inner_op.value),
                            message: ERR_UNSUPPORTED_TYPE_EXPR,
                        });
                    }
                    mutable = inner_op.value == "mut";
                    inner = inner_items.pop().unwrap();
                }

                let base = self.lower_type_expr(inner)?;
                Ok(TypeExpr::Ptr { base, raw, mutable })
            }

            Expr::Prefix(open, items)
                if matches!(open.value, "struct" | "cstruct" | "enum" | "union") =>
            {
                self.lower_struct_like_type_expr(open, items)
            }

            Expr::Atom(Token::Operator(op)) => Err(CompileError::UnsupportedForm {
                loc: loc.clone(),
                op_loc: Some(loc),
                op: Some(op),
                message: ERR_UNSUPPORTED_TYPE_EXPR,
            }),

            Expr::Bin(op, _) => Err(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_TYPE_EXPR,
            }),

            Expr::Prefix(op, _) | Expr::Postfix(op, _) => Err(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_TYPE_EXPR,
            }),

            _ => Err(CompileError::UnsupportedForm {
                loc,
                op_loc: None,
                op: None,
                message: ERR_UNSUPPORTED_TYPE_EXPR,
            }),
        }
    }

    fn lower_struct_like_type_expr(&mut self, kw: LFixed, items: Vec<LExpr>) -> CResult<TypeExpr> {
        let mut items = items.into_iter().peekable();

        let mut generics_expr = None;
        if matches!(&items.peek().unwrap().value, Expr::Prefix(open, _) if open.value == "[") {
            generics_expr = Some(items.next().unwrap());
        }

        let fields_expr = match items.next() {
            Some(expr) => expr,
            None => {
                return Err(CompileError::UnsupportedForm {
                    loc: kw.loc.clone(),
                    op_loc: Some(kw.loc),
                    op: Some(kw.value),
                    message: ERR_UNSUPPORTED_TYPE_EXPR,
                });
            }
        };

        if items.next().is_some() {
            return Err(CompileError::UnsupportedForm {
                loc: kw.loc.clone(),
                op_loc: Some(kw.loc),
                op: Some(kw.value),
                message: ERR_UNSUPPORTED_TYPE_EXPR,
            });
        }

        let generics = match generics_expr {
            Some(gen_expr) => {
                let Expr::Prefix(open, items) = gen_expr.value else {
                    return Err(CompileError::UnsupportedForm {
                        loc: gen_expr.loc,
                        op_loc: Some(kw.loc),
                        op: Some(kw.value),
                        message: ERR_UNSUPPORTED_TYPE_EXPR,
                    });
                };
                debug_assert!(open.value == "[");

                let span = self.reserve_pattern_span(items.len());
                for (index, item) in items.into_iter().enumerate() {
                    let target = span.at(index);
                    self.lower_pattern_into(target, item, VarKind::Const)?;
                }
                span
            }
            None => self.reserve_pattern_span(0),
        };

        let fields = match fields_expr.value {
            Expr::Prefix(open, items) if open.value == "{" => {
                let span = self.reserve_pattern_span(items.len());
                for (index, item) in items.into_iter().enumerate() {
                    let target = span.at(index);
                    self.lower_pattern_into(target, item, VarKind::Mut)?;
                }
                span
            }
            _ => {
                return Err(CompileError::UnsupportedForm {
                    loc: fields_expr.loc,
                    op_loc: Some(kw.loc),
                    op: Some(kw.value),
                    message: ERR_UNSUPPORTED_TYPE_EXPR,
                });
            }
        };

        let def = StructLike {
            layout: match kw.value {
                "cstruct" => StructLayoutSpec::C,
                _ => StructLayoutSpec::Hot,
            },
            generics,
            fields,
        };
        Ok(match kw.value {
            "cstruct" => TypeExpr::Struct(def),
            "struct" => TypeExpr::Struct(def),
            "enum" => TypeExpr::Enum(def),
            "union" => TypeExpr::Union(def),
            _ => unreachable!(),
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
            Pattern::Bind(id, _) => id,
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
            Pattern::Bind(id, _) => id,
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
    use crate::error_messages::{
        ERR_ACCESS_EXPECTS_NAME, ERR_LABEL_ALREADY_DEFINED, ERR_LABEL_NAME_REQUIRED,
        ERR_MEMBER_METHOD_NAME_COLLISION,
    };
    use crate::parsing::Parser;
    use crate::program::{CompileError, Defined, Program};

    fn lower_block(src: &str) -> (Program, ValId) {
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().expect("failed to parse expr");
        let ir = program.lower_value(expr).expect("lowering failed");
        (program, ir)
    }

    fn assert_labeled_num(program: &Program, id: ValId, name: &str, value: u64) {
        match program.value(id) {
            Value::Labeled {
                name: labeled,
                value: val,
            } => {
                assert_eq!(program.str_intern.resolve(labeled), name);
                match program.value(val) {
                    Value::Literal(Literal::Num(num)) => assert_eq!(num, value),
                    other => panic!("expected labeled literal num, got {other:?}"),
                }
            }
            other => panic!("expected labeled arg, got {other:?}"),
        }
    }

    fn bound_id(program: &Program, stmt: ValId) -> NameId {
        match program.value(stmt) {
            Value::Let { pat, .. } => match program.pattern(pat) {
                Pattern::Bind(id, _) => id,
                _ => panic!("expected bind pattern"),
            },
            _ => panic!("expected let statement"),
        }
    }

    fn tuple_bind_kinds(program: &Program, pat: PatId) -> Vec<VarKind> {
        let Pattern::Tuple(span) = program.pattern(pat) else {
            panic!("expected tuple pattern")
        };
        span.ids()
            .map(|id| match program.pattern(id) {
                Pattern::Bind(_, kind) => kind,
                Pattern::Wildcard(kind) => kind,
                other => panic!("expected bind or wildcard pattern, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn tuple_pattern_mutability_is_per_binding_for_let_and_var() {
        let src = "{ let (mut a, b, const mut c) = 2; var (const x, y, const mut z) = 3; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 2);

        let let_pat = match program.value(statements[0]) {
            Value::Let { pat, .. } => pat,
            _ => panic!("expected let statement"),
        };
        let let_kinds = tuple_bind_kinds(&program, let_pat);
        assert_eq!(let_kinds, vec![VarKind::Mut, VarKind::Const, VarKind::Mut]);

        let var_pat = match program.value(statements[1]) {
            Value::Let { pat, .. } => pat,
            _ => panic!("expected var statement"),
        };
        let var_kinds = tuple_bind_kinds(&program, var_pat);
        assert_eq!(var_kinds, vec![VarKind::Const, VarKind::Mut, VarKind::Mut]);
    }

    #[test]
    fn member_method_duplicate_definition_errors_and_preserves_first() {
        let src = "type S = struct { a: int }; S.foo = fn(x){ x }; S.foo = fn(x, y){ x };";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();

        let expr = parser
            .parse_with_macros(&mut program)
            .expect("failed to parse type")
            .expect("missing type expr");
        program.gather_definition(expr).expect("type def failed");

        let expr = parser
            .parse_with_macros(&mut program)
            .expect("failed to parse first method")
            .expect("missing first method expr");
        program
            .gather_definition(expr)
            .expect("first method should lower");

        let expr = parser
            .parse_with_macros(&mut program)
            .expect("failed to parse second method")
            .expect("missing second method expr");
        let err = program.gather_definition(expr).unwrap_err();
        match err {
            CompileError::SimpleError { s, .. } => {
                assert_eq!(s, ERR_MEMBER_METHOD_NAME_COLLISION);
            }
            other => panic!("expected simple error, got {other:?}"),
        }

        let struct_name = program.str_intern.intern("S");
        let struct_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.get(&struct_name))
            .expect("missing struct name");
        let method_name = program.str_intern.intern("foo");
        let method_id = program
            .member_methods
            .get(&struct_id)
            .and_then(|methods| methods.get(&method_name))
            .copied()
            .expect("missing foo method");
        let Value::Func { params, .. } = program.value(method_id) else {
            panic!("expected foo to be a function");
        };
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn lowers_call_and_index_with_bound_names() {
        let src = "{ let f = 1; let a = 2; (a |> f(3))[a, 4]; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 3);

        let f_id = bound_id(&program, statements[0]);
        let a_id = bound_id(&program, statements[1]);

        let Value::Index(index_call) = program.value(statements[2]) else {
            panic!("expected index expression")
        };

        let Value::Call(call_call) = program.value(index_call.base) else {
            panic!("expected call base");
        };

        match program.value(call_call.base) {
            Value::NameRef(id) => assert_eq!(id, f_id),
            _ => panic!("expected callee to be name"),
        }
        assert_eq!(call_call.pos_args().len(), 2);
        assert!(call_call.named_args().is_empty());
        match program.value(call_call.args.at(0)) {
            Value::NameRef(id) => assert_eq!(id, a_id),
            _ => panic!("expected first call arg to be name"),
        }
        match program.value(call_call.args.at(1)) {
            Value::Literal(Literal::Num(3)) => {}
            _ => panic!("expected literal call arg"),
        }

        let index_args = index_call.pos_args().ids().collect::<Vec<_>>();
        assert_eq!(index_args.len(), 2);
        assert!(index_call.named_args().is_empty());
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
    fn lowers_labeled_args_for_call_index_and_construct() {
        let src = "{ let f = 1; let x = 2; let t = 3; f(x, a = 4); x[a = 5]; t{a = 6}; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 6);

        let Value::Call(call) = program.value(statements[3]) else {
            panic!("expected call expression");
        };

        let call_args = call.pos_args().ids().collect::<Vec<_>>();
        let call_named_args = call.named_args().ids().collect::<Vec<_>>();
        assert_eq!(call_args.len(), 1);
        assert_eq!(call_named_args.len(), 1);
        match program.value(call_args[0]) {
            Value::NameRef(_) => {}
            _ => panic!("expected first call arg to be name"),
        }
        assert_labeled_num(&program, call_named_args[0], "a", 4);

        let Value::Index(index) = program.value(statements[4]) else {
            panic!("expected index expression");
        };
        let index_args = index.pos_args().ids().collect::<Vec<_>>();
        let index_named_args = index.named_args().ids().collect::<Vec<_>>();
        assert_eq!(index_args.len(), 0);
        assert_eq!(index_named_args.len(), 1);
        assert_labeled_num(&program, index_named_args[0], "a", 5);

        let Value::Construct(construct) = program.value(statements[5]) else {
            panic!("expected construct expression");
        };
        let construct_args = construct.pos_args().ids().collect::<Vec<_>>();
        let construct_named_args = construct.named_args().ids().collect::<Vec<_>>();
        assert_eq!(construct_args.len(), 0);
        assert_eq!(construct_named_args.len(), 1);
        assert_labeled_num(&program, construct_named_args[0], "a", 6);
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
        let Value::MatchArm(arm) = program.value(arms.at(0)) else {
            panic!("expected match arm")
        };
        match program.pattern(arm.pat) {
            Pattern::Wildcard(_) => {}
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
                assert_eq!(program.str_intern.resolve(name), "b");
            }
            _ => panic!("expected dot access"),
        }

        match program.value(statements[3]) {
            Value::Access { base, name, kind } => {
                assert_eq!(kind, AccessKind::Static);
                match program.value(base) {
                    Value::NameRef(id) => assert_eq!(id, t_id),
                    _ => panic!("expected type base name"),
                }
                assert_eq!(program.str_intern.resolve(name), "c");
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
            Defined::Func(value) => match program.value(*value) {
                Value::Func { generics, .. } => assert_eq!(generics.len(), 1),
                _ => panic!("expected function value"),
            },
            _ => panic!("expected value definition"),
        }
    }

    #[test]
    fn lowers_cfn_without_body_as_external_declaration() {
        let src = "f = cfn(x:int)->int;";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        program.lower_all(&mut parser).unwrap();

        let f_name = program.str_intern.intern("f");
        let f_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.get(&f_name))
            .expect("missing f binding");

        let Defined::Func(value) = program
            .definitions
            .get(&f_id)
            .expect("missing f definition")
        else {
            panic!("expected function definition")
        };

        let Value::Func {
            calling_convention,
            params,
            output_type,
            body,
            ..
        } = program.value(*value)
        else {
            panic!("expected function value")
        };

        assert_eq!(calling_convention, CallingConvention::C);
        assert_eq!(params.len(), 1);
        assert!(output_type.is_some());
        assert!(body.is_none());
    }

    #[test]
    fn lowers_cstruct_definition_with_c_layout_marker() {
        let src = "type S = cstruct { x:int };";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        program.lower_all(&mut parser).unwrap();

        let s_name = program.str_intern.intern("S");
        let s_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.get(&s_name))
            .expect("missing S binding");

        let Defined::Type(texp) = program
            .definitions
            .get(&s_id)
            .expect("missing S definition")
        else {
            panic!("expected type definition")
        };

        let TypeExpr::Struct(def) = program.type_expr(*texp) else {
            panic!("expected struct type expression")
        };

        assert_eq!(def.layout, StructLayoutSpec::C);
        assert_eq!(def.fields.len(), 1);
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
            Defined::Func(value) => match program.value(*value) {
                Value::Func { body, .. } => body.expect("expected f to have a body"),
                _ => panic!("expected f to be a function"),
            },
            _ => panic!("expected f to lower to a value"),
        };

        let g_body = match g_def {
            Defined::Func(value) => match program.value(*value) {
                Value::Func { body, .. } => body.expect("expected g to have a body"),
                _ => panic!("expected g to be a function"),
            },
            _ => panic!("expected g to lower to a value"),
        };

        let f_call = match program.value(f_body) {
            Value::Call(Call { base, .. }) => base,
            Value::Block { return_value, .. } => return_value
                .as_ref()
                .map(|value| match program.value(*value) {
                    Value::Call(Call { base, .. }) => base,
                    _ => panic!("expected f return to be a call"),
                })
                .expect("expected f to return a call"),
            _ => panic!("expected f body to be a call or block"),
        };
        let g_call = match program.value(g_body) {
            Value::Call(Call { base, .. }) => base,
            Value::Block { return_value, .. } => return_value
                .as_ref()
                .map(|value| match program.value(*value) {
                    Value::Call(Call { base, .. }) => base,
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
                match program.type_expr(ty) {
                    TypeExpr::NameRef(id) => assert_ne!(id, a_id),
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
    fn lowers_break_and_continue() {
        let src = "{ break; continue; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 2);

        assert!(matches!(program.value(statements[0]), Value::Break));
        assert!(matches!(program.value(statements[1]), Value::Continue));
    }

    #[test]
    fn lowers_forward_goto_and_label_declaration_in_function() {
        let src = "f = fn(){ goto `err; `err; }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        program.lower_all(&mut parser).unwrap();

        let f_name = program.str_intern.intern("f");
        let f_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.get(&f_name))
            .expect("missing f binding");

        let Defined::Func(f_val) = program
            .definitions
            .get(&f_id)
            .expect("missing f definition")
        else {
            panic!("expected function definition")
        };

        let Value::Func { body, .. } = program.value(*f_val) else {
            panic!("expected function value")
        };
        let body = body.expect("expected function body");

        let statements = match program.value(body) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected function body block"),
        };
        assert_eq!(statements.len(), 2);

        let goto_label = match program.value(statements[0]) {
            Value::Goto(id) => id,
            _ => panic!("expected goto statement"),
        };

        let declared_label = match program.value(statements[1]) {
            Value::LabelDecl(id) => id,
            _ => panic!("expected label declaration statement"),
        };

        assert_eq!(goto_label, declared_label);
    }

    #[test]
    fn goto_label_must_be_defined_in_the_same_function() {
        let src = "f = fn(){ goto `missing; }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let err = program.lower_all(&mut parser).unwrap_err();

        match err {
            CompileError::UnresolvedLabel { name, .. } => {
                assert_eq!(name, "missing");
            }
            other => panic!("expected unresolved label error, got {other:?}"),
        }
    }

    #[test]
    fn goto_requires_direct_label_syntax() {
        let src = "f = fn(){ goto x; `x; }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let err = program.lower_all(&mut parser).unwrap_err();

        match err {
            CompileError::SimpleError { s, .. } => assert_eq!(s, ERR_LABEL_NAME_REQUIRED),
            other => panic!("expected label syntax error, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_label_definition_errors() {
        let src = "f = fn(){ `x; `x; }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let err = program.lower_all(&mut parser).unwrap_err();

        match err {
            CompileError::SimpleError { s, .. } => assert_eq!(s, ERR_LABEL_ALREADY_DEFINED),
            other => panic!("expected duplicate label error, got {other:?}"),
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
                            Pattern::Bind(_x_id, _) => {
                                // Verify the value is the expected literal
                                match program.value(value) {
                                    Value::Literal(Literal::Num(1)) => {}
                                    _ => panic!("expected literal value"),
                                }
                            }
                            _ => panic!("expected bind pattern for variable name"),
                        }
                        // The type should resolve to the predefined 'int' name
                        match program.type_expr(ty) {
                            TypeExpr::NameRef(_int_id) => {} // Type should be a name reference to 'int'
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
