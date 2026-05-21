use crate::data_structures::index::{Idx, IndexSpan};
use crate::data_structures::string_intern::StrId;
/**
 * TODO: Convert IR from tree-shaped to flat list with ids.
 *    all in the outer level function.
 *
 * This will:
 * - Simplify type inference
 * - Avoid solver needing to rediscover operands
 * - Allow linear passes over IR
 */
use crate::parsing::{Expr, LExpr, LFixed, Loc, Located, Token};
use crate::program::{CompileError, Program};

//this file needs to move Value and Pattern into a dense array
//note that currently the only major diffrence between Value and Pattern is Bind
//the one place which actually reads them would become simpler if we merge the 2.
//would actually remove a lot of semi duplicate code from type inference

/// Unique identifier for names in the IR
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NameId(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct LifeTimeId(pub usize);

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

macro_rules! impl_usize_idx {
    ($($id:ty),* $(,)?) => {
        $(
            impl Idx for $id {
                #[inline]
                fn new(idx: usize) -> Self {
                    Self(idx)
                }

                #[inline]
                fn index(self) -> usize {
                    self.0
                }
            }
        )*
    };
}

impl_usize_idx!(ValId, PatId, TExpId);

/// A contiguous range in the value arena.
pub type ValueSpan = IndexSpan<ValId>;
/// A contiguous range in the pattern arena.
pub type PatternSpan = IndexSpan<PatId>;
/// A contiguous range in the type-expression arena.
pub type TypeExprSpan = IndexSpan<TExpId>;

/// Literal values that can appear in the code
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Literal {
    Num(u64),
    Float(f64),
    Bool(bool),
    Str(StrId),
    Null,
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
        self.args.subslice(0, self.named_args_start)
    }

    pub fn named_args(&self) -> ValueSpan {
        self.args.subslice(
            self.named_args_start,
            self.args.len() - self.named_args_start,
        )
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct GenDec {
    pub parts: PatternSpan,
    ///exclusive
    pub lifetime_end: usize,
    pub where_clause: TypeExprSpan,
}

impl GenDec {
    pub fn lifetimes(&self) -> PatternSpan {
        self.parts.subslice(0, self.lifetime_end)
    }

    pub fn generics(&self) -> PatternSpan {
        self.parts
            .subslice(self.lifetime_end, self.parts.len() - self.lifetime_end)
    }

    pub fn where_clause(&self) -> TypeExprSpan {
        self.where_clause
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct GenIndex {
    pub base: TExpId,
    pub parts: TypeExprSpan,
    ///exclusive
    pub lifetime_end: usize,
}

impl GenIndex {
    pub fn lifetimes(&self) -> TypeExprSpan {
        self.parts.subslice(0, self.lifetime_end)
    }

    pub fn generics(&self) -> TypeExprSpan {
        self.parts
            .subslice(self.lifetime_end, self.parts.len() - self.lifetime_end)
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
    // LifeTime(LifeTimeId),
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

    IntAccess {
        base: ValId,
        id: usize,
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
        generics: GenDec,
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

    Poison,
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
    TypeAnnotation {
        pat: PatId,
        ty: TExpId,
    },

    LifeTime(LifeTimeId),
    AddrOf(PatId, VarKind),
    Poison,
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
        args: GenIndex,
    },

    Lt {
        lhs: TExpId,
        rhs: TExpId,
    },

    Ptr {
        base: TExpId,
        lifetime: Option<LifeTimeId>,
        raw: bool,
        mutable: bool,
    },

    LifeTime(LifeTimeId),

    Func {
        calling_convention: CallingConvention,
        params: TypeExprSpan,
        output_type: Option<TExpId>,
    },

    Array(TExpId, Option<usize>),

    Enum(StructLike),
    Struct(StructLike),
    Union(StructLike),

    Poison,
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
    pub generics: GenDec,
    pub fields: PatternSpan,
}

//these we look for in tests so...
pub(crate) const ACCESS_EXPECTS_NAME_MSG: &str =
    "access expressions require a simple identifier or integer literal after the operator";
pub(crate) const MEMBER_METHOD_COLLISION_MSG: &str =
    "member method name collides with an existing field or method";
pub(crate) const LABEL_NAME_REQUIRED_MSG: &str = "goto requires a direct label literal like `name`";
pub(crate) const LABEL_ALREADY_DEFINED_MSG: &str =
    "label already exists in this function; choose a new name";

impl Program {
    //TODO:
    // 1. local macros are intetionaly not handeled and scoping on macros is broken on purpose to be like C
    // 2. some places parse a value where a value/pattern check needs to be done
    pub fn lower_value(&mut self, expr: LExpr) -> ValId {
        let target = self.id_value(expr.loc.clone(), Value::Literal(Literal::Void));
        self.lower_value_into(target, expr)
    }

    #[inline]
    fn lower_value_into(&mut self, target: ValId, mut expr: LExpr) -> ValId {
        let loc = expr.loc.clone();

        //gotos have to know the valid
        if let Expr::Prefix(ref op, ref mut items) = expr.value
            && op.value == "goto"
        {
            return self.lower_goto_into(target, loc, op.clone(), items);
        }

        let value = self.lower_value_inner(expr);
        self.set_value(target, loc, value);
        target
    }

    #[inline]
    fn lower_value_into_labeled(&mut self, target: ValId, expr: LExpr) -> bool {
        let loc = expr.loc.clone();
        match expr.value {
            Expr::Bin(op, pair) if op.value == "=" => {
                if let Expr::Atom(Token::Ident(n)) = pair.0.value {
                    let value = self.lower_value(pair.1);
                    let name = self.str_intern.intern(&n);
                    self.set_value(target, loc, Value::Labeled { name, value });
                    true
                } else {
                    self.lower_value_into(target, loc.with(Expr::Bin(op, pair)));
                    false
                }
            }
            _ => {
                self.lower_value_into(target, expr);
                false
            }
        }
    }

    fn lower_value_inner(&mut self, expr: LExpr) -> Value {
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
                let call = self.lower_call_like_expr(expr.loc, items);
                match open.value {
                    "(" => Value::Call(call),
                    "[" => Value::Index(call),
                    "{" => Value::Construct(call),
                    _ => Value::Poison,
                }
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

            Expr::Prefix(open, items) if open.value == "&" => self.lower_addr_of(items),

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
    fn lower_atom(&mut self, loc: &Loc, token: Token) -> Value {
        match token {
            Token::NumLit(n) => Value::Literal(Literal::Num(n)),
            Token::FloatLit(f) => Value::Literal(Literal::Float(f)),
            Token::StrLit(s) => Value::Literal(Literal::Str(self.str_intern.intern(&s))),
            Token::Operator("(") => Value::Literal(Literal::Void),
            Token::Operator("true") => Value::Literal(Literal::Bool(true)),
            Token::Operator("false") => Value::Literal(Literal::Bool(false)),
            Token::Operator("null") | Token::Operator("nil") => Value::Literal(Literal::Null),

            Token::Ident(name) if name == "_" => Value::Wildcard,
            Token::Ident(name) => {
                let id = self.resolve_name(loc, &name);
                Value::NameRef(id)
            }

            Token::Operator(op) => {
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc: loc.clone(),
                    op_loc: Some(loc.clone()),
                    op: Some(op),
                    message: "operators cannot appear as standalone atoms; wrap them inside a full expression",
                });
                Value::Poison
            }
        }
    }

    #[inline(always)]
    fn lower_block_expr(&mut self, _loc: Loc, items: Vec<LExpr>) -> Value {
        self.with_scope_value(|this| this.lower_block_expr_inner(_loc, items))
    }

    fn lower_block_expr_inner(&mut self, _loc: Loc, mut items: Vec<LExpr>) -> Value {
        let (statements, return_value) = if let Some(last) = items.pop() {
            let span = self.reserve_value_span(items.len());
            for (index, item) in items.into_iter().enumerate() {
                let target = span.at(index);
                if let Some(name) = Self::extract_label_name(&item) {
                    if !self.in_function_body() {
                        self.push_lowering_error(CompileError::SimpleError {
                            loc: item.loc,
                            s: "labels can only be defined inside function bodies",
                        });
                    } else {
                        let id = self.define_label_name(&item.loc, name);
                        self.set_value(target, item.loc, Value::LabelDecl(id));
                    }
                } else {
                    self.lower_value_into(target, item);
                }
            }

            let return_value = if !matches!(last.value, Expr::Atom(Token::Operator(";"))) {
                Some(self.lower_value(last))
            } else {
                None
            };

            (span, return_value)
        } else {
            (self.reserve_value_span(0), None)
        };

        Value::Block {
            statements,
            return_value,
        }
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
    fn lower_typedef_expr(&mut self, loc: Loc, mut items: Vec<LExpr>) -> Value {
        debug_assert!(2 == items.len());

        let value_expr = items.pop().unwrap();
        let pat_expr = items.pop().unwrap();

        let pat = self.lower_pattern(pat_expr, VarKind::Const);
        let ty = self.lower_type_expr(value_expr);

        let _ = loc;

        Value::TypeDef { pat, ty }
    }

    #[inline(always)]
    fn lower_let_expr(&mut self, loc: Loc, mut items: Vec<LExpr>, m: VarKind) -> Value {
        debug_assert!(2 <= items.len() && items.len() <= 3);

        let else_exp = if items.len() == 3 { items.pop() } else { None };

        let value_expr = items.pop().unwrap();
        let pat_expr = items.pop().unwrap();

        let value = self.lower_value(value_expr);
        let pat = self.lower_pattern(pat_expr, m);

        let else_part = else_exp.map(|exp| self.with_scope_value(|p| p.lower_value(exp)));

        let _ = loc;

        Value::Let {
            pat,
            value,
            else_part,
        }
    }

    #[inline(always)]
    fn lower_tuple_expr(&mut self, _loc: Loc, items: Vec<LExpr>, open: &'static str) -> Value {
        let parts = self.reserve_value_span(items.len());

        for (index, arg) in items.into_iter().enumerate() {
            let target = parts.at(index);
            self.lower_value_into(target, arg);
        }

        match open {
            "(" => Value::Tuple(parts),
            "[" => Value::Array(parts),
            _ => Value::Poison,
        }
    }

    #[inline(always)]
    fn lower_call_like_expr(&mut self, _loc: Loc, items: Vec<LExpr>) -> Call {
        debug_assert!(!items.is_empty(), "call expression missing base");

        let mut items = items.into_iter();
        let base = self.lower_value(items.next().unwrap());

        let args = self.reserve_value_span(items.len());
        let mut named_args_start = args.len();
        for (index, arg) in items.enumerate() {
            let target = args.at(index);
            let arg_loc = arg.loc.clone();
            if self.lower_value_into_labeled(target, arg) {
                if named_args_start == args.len() {
                    named_args_start = index;
                }
            } else if named_args_start != args.len() {
                self.push_lowering_error(CompileError::SimpleError {
                    loc: arg_loc,
                    s: "positional arguments must come before named ones",
                });
            }
        }

        Call {
            base,
            args,
            named_args_start,
        }
    }

    #[inline(always)]
    fn lower_match_expr(&mut self, loc: Loc, items: Vec<LExpr>) -> Value {
        if items.len() < 2 {
            self.push_lowering_error(CompileError::SimpleError {
                loc,
                s: "match expressions require a value and at least one arm",
            });
            return Value::Poison;
        }

        let mut items = items.into_iter();
        let value = self.lower_value(items.next().unwrap());

        let arms = self.reserve_value_span(items.len());
        for (id, arm_expr) in arms.ids().zip(items) {
            self.with_scope_value(|p| {
                let loc = arm_expr.loc.clone();
                let arm = p.lower_match_arm(arm_expr);
                p.set_value(id, loc, Value::MatchArm(arm));
            });
        }

        Value::Match { value, arms }
    }

    #[inline(always)]
    fn lower_if_expr(&mut self, loc: Loc, items: Vec<LExpr>) -> Value {
        if items.len() < 2 || items.len() > 3 {
            self.push_lowering_error(CompileError::SimpleError {
                loc,
                s: "if expression requires condition and then branch, optional else branch",
            });
            return Value::Poison;
        }

        let mut items = items.into_iter();
        let cond_expr = items.next().unwrap();
        let then_expr = items.next().unwrap();
        let else_expr = if items.len() == 1 {
            Some(items.next().unwrap())
        } else {
            None
        };

        let cond = self.lower_value(cond_expr);
        let then = self.with_scope_value(|p| p.lower_value(then_expr));
        let els = else_expr.map(|else_expr| self.with_scope_value(|p| p.lower_value(else_expr)));
        let _ = loc;
        Value::If { cond, then, els }
    }

    #[inline(always)]
    fn lower_while_expr(&mut self, loc: Loc, items: Vec<LExpr>) -> Value {
        if items.len() != 2 {
            self.push_lowering_error(CompileError::SimpleError {
                loc,
                s: "while expression requires condition",
            });
            return Value::Poison;
        }

        let mut items = items.into_iter();
        let cond_expr = items.next().unwrap();
        let then_expr = items.next().unwrap();

        let (cond, body) = self.with_scope_value(|p| {
            let cond = p.lower_value(cond_expr);
            let body = p.lower_value(then_expr);
            (cond, body)
        });
        let _ = loc;
        Value::While { cond, body }
    }

    #[inline(always)]
    fn lower_break_expr(&mut self, loc: Loc, op: LFixed, items: Vec<LExpr>) -> Value {
        if items.len() > 1 {
            self.push_lowering_error(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: "break can only supply zero or one value",
            });
            return Value::Poison;
        }
        Value::Break
    }

    #[inline(always)]
    fn lower_continue_expr(&mut self, loc: Loc, op: LFixed, items: Vec<LExpr>) -> Value {
        if !items.is_empty() {
            self.push_lowering_error(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: "continue cannot take any values",
            });
            return Value::Poison;
        }
        Value::Continue
    }

    #[inline(always)]
    fn lower_return_expr(&mut self, loc: Loc, op: LFixed, mut items: Vec<LExpr>) -> Value {
        if items.len() > 1 {
            self.push_lowering_error(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: "return accepts at most one value",
            });
            return Value::Poison;
        }

        let value = items.pop().map(|value| self.lower_value(value));
        Value::Return(value)
    }

    #[inline(always)]
    fn lower_goto_into(
        &mut self,
        target: ValId,
        loc: Loc,
        op: LFixed,
        items: &mut Vec<LExpr>,
    ) -> ValId {
        if !self.in_function_body() {
            self.push_lowering_error(CompileError::SimpleError {
                loc,
                s: "goto statements must stay inside function bodies",
            });
            return target;
        }
        if items.len() != 1 {
            self.push_lowering_error(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: "goto requires a single label argument",
            });
            return target;
        }

        let label_expr = items.pop().unwrap();
        let Some(label_name) = Self::extract_label_name(&label_expr) else {
            self.push_lowering_error(CompileError::SimpleError {
                loc: label_expr.loc,
                s: LABEL_NAME_REQUIRED_MSG,
            });
            return target;
        };

        let label = self.use_label_name_for_goto(target, &label_expr.loc, label_name);
        self.set_value(target, loc, Value::Goto(label));
        target
    }

    #[inline(always)]
    fn lower_assign_expr(&mut self, loc: Loc, pair: (LExpr, LExpr)) -> Value {
        let (lhs, rhs) = pair;

        let target = self.lower_value(lhs);
        let value = self.lower_value(rhs);

        let _ = loc;
        Value::Assign {
            op: AssignOp::Nothing(value),
            target,
        }
    }

    #[inline(always)]
    fn lower_fn_expr(&mut self, loc: Loc, fn_kw: LFixed, items: Vec<LExpr>) -> Value {
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
            return Value::Poison;
        };
        debug_assert!(p_open.value == "(", "fn parameter list must start with '('");

        let _calling_convention = CallingConvention::from_fn_keyword(fn_kw.value).unwrap();

        match self.with_function_labels(|this| {
            Ok(this.lower_fn_expr_inner(
                loc,
                fn_kw,
                generics_expr,
                param_items,
                ret_expr,
                body_expr,
            ))
        }) {
            Ok(value) => value,
            Err(err) => {
                self.push_lowering_error(err);
                Value::Poison
            }
        }
    }

    fn lower_generic_dec(&mut self, generics_expr: Option<LExpr>) -> GenDec {
        let items_vec = match generics_expr {
            Some(gen_expr) => {
                let Expr::Prefix(open, items) = gen_expr.value else {
                    debug_assert!(false, "fn generics must use brackets");
                    return GenDec {
                        parts: self.reserve_pattern_span(0),
                        lifetime_end: 0,
                        where_clause: self.reserve_type_expr_span(0),
                    };
                };
                debug_assert!(open.value == "[", "fn generics must use brackets");

                Some(items)
            }
            None => None,
        };

        match items_vec {
            Some(items) => {
                let mut where_start = items.len();
                let mut where_prefix_count = 0;
                for (index, expr) in items.iter().enumerate() {
                    if let Expr::Prefix(op, nested_items) = &expr.value
                        && op.value == "where"
                    {
                        where_start = index;
                        where_prefix_count = nested_items.len();
                        break;
                    }
                }

                let parts = self.reserve_pattern_span(where_start);
                let where_count = if where_start == items.len() {
                    0
                } else {
                    where_prefix_count + (items.len() - where_start - 1)
                };
                let where_clause = self.reserve_type_expr_span(where_count);

                let mut where_index = 0;
                for (index, expr) in items.into_iter().enumerate() {
                    if index < where_start {
                        let target = parts.at(index);
                        self.lower_pattern_into(target, expr, VarKind::Const);
                        continue;
                    }

                    if index == where_start {
                        let Expr::Prefix(op, nested_items) = expr.value else {
                            unreachable!();
                        };
                        debug_assert!(op.value == "where");
                        for nested in nested_items {
                            let target = where_clause.at(where_index);
                            self.lower_type_expr_into(target, nested);
                            where_index += 1;
                        }
                        continue;
                    }

                    let target = where_clause.at(where_index);
                    self.lower_type_expr_into(target, expr);
                    where_index += 1;
                }

                debug_assert_eq!(where_index, where_clause.len());

                let lifetime_end = self.find_lifetime_end_in_pattern_span(parts);
                GenDec {
                    parts,
                    lifetime_end,
                    where_clause,
                }
            }
            None => GenDec {
                parts: self.reserve_pattern_span(0),
                lifetime_end: 0,
                where_clause: self.reserve_type_expr_span(0),
            },
        }
    }

    fn lower_fn_expr_inner(
        &mut self,
        loc: Loc,
        fn_kw: LFixed,
        generics_expr: Option<LExpr>,
        param_items: Vec<LExpr>,
        ret_expr: Option<LExpr>,
        body_expr: Option<LExpr>,
    ) -> Value {
        self.with_scope_value(|this| {
            let generics = this.lower_generic_dec(generics_expr);

            let params_span = this.reserve_pattern_span(param_items.len());
            for (index, param) in param_items.into_iter().enumerate() {
                let target = params_span.at(index);
                this.lower_pattern_into(target, param, VarKind::Const);
            }

            let output_type = ret_expr.map(|e| this.lower_type_expr(e));

            let body = body_expr.map(|body_expr| this.lower_value(body_expr));

            let _ = loc;
            Value::Func {
                calling_convention: CallingConvention::from_fn_keyword(fn_kw.value).unwrap(),
                generics,
                params: params_span,
                output_type,
                body,
            }
        })
    }

    #[inline(always)]
    fn lower_cast_expr(
        &mut self,
        loc: Loc,
        op: Located<&'static str>,
        pair: (LExpr, LExpr),
    ) -> Value {
        let (value_expr, ty_expr) = pair;
        let value = self.lower_value(value_expr);
        let ty = self.lower_type_expr(ty_expr);
        let v = match op.value {
            "as" => Value::Cast { value, ty },
            ":" => Value::TypeAnnotation { value, ty },
            _ => {
                self.push_lowering_error(CompileError::SimpleError {
                    loc: loc.clone(),
                    s: "unsupported cast operator",
                });
                Value::Poison
            }
        };
        let _ = loc;
        v
    }

    #[inline(always)]
    fn lower_access_expr(
        &mut self,
        loc: Loc,
        op: Located<&'static str>,
        lhs: LExpr,
        rhs: LExpr,
    ) -> Value {
        let base = self.lower_value(lhs);
        let kind = match op.value {
            "." => AccessKind::Dot,
            "::" => AccessKind::Static,
            "->" => AccessKind::Ptr,
            _ => {
                self.push_lowering_error(CompileError::SimpleError {
                    loc,
                    s: "unsupported access operator",
                });
                return Value::Poison;
            }
        };

        match rhs.value {
            Expr::Atom(Token::Ident(name)) => {
                let name = self.str_intern.intern(&name);
                Value::Access { base, name, kind }
            }
            Expr::Atom(Token::NumLit(id)) => Value::IntAccess {
                base,
                id: id as usize,
                kind,
            },
            _ => {
                self.push_lowering_error(CompileError::SimpleError {
                    loc: rhs.loc,
                    s: ACCESS_EXPECTS_NAME_MSG,
                });
                Value::Poison
            }
        }
    }

    pub fn lower_pattern(&mut self, expr: LExpr, m: VarKind) -> PatId {
        let loc = expr.loc.clone();
        let pattern = self.lower_pattern_inner(expr, m);
        self.id_pattern(loc, pattern)
    }

    fn lower_pattern_into(&mut self, target: PatId, expr: LExpr, m: VarKind) -> PatId {
        let loc = expr.loc.clone();
        let pattern = self.lower_pattern_inner(expr, m);
        self.set_pattern(target, loc, pattern);
        target
    }

    fn lower_pattern_inner(&mut self, expr: LExpr, m: VarKind) -> Pattern {
        let loc = expr.loc.clone();
        match expr.value {
            Expr::Atom(Token::Ident(name)) if name == "_" => Pattern::Wildcard(m),

            Expr::Atom(Token::Ident(name)) => {
                let name = self.str_intern.intern(&name);
                let id = self.insert_value_in_current_scope(name);
                Pattern::Bind(id, m)
            }

            Expr::Atom(_) => {
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc,
                    op_loc: None,
                    op: None,
                    message: "got a literal that isnt a name in a pattern",
                });
                Pattern::Poison
            }

            // Pattern with type annotation: x:T
            Expr::Bin(op, pair) if op.value == ":" => {
                let (pat_expr, ty_expr) = *pair;
                let pat = self.lower_pattern(pat_expr, m);
                let ty = self.lower_type_expr(ty_expr);

                Pattern::TypeAnnotation { pat, ty }
            }

            Expr::Bin(op, _) => {
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc,
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: "this pattern expression is not supported",
                });
                Pattern::Poison
            }

            Expr::Prefix(open, mut items) if open.value == "mut" => {
                self.lower_pattern_inner(items.pop().unwrap(), VarKind::Mut)
            }
            Expr::Prefix(open, mut items) if open.value == "const" => {
                self.lower_pattern_inner(items.pop().unwrap(), VarKind::Const)
            }

            Expr::Prefix(open, mut items) if open.value == "&" => {
                let mut rhs_expr = items.pop().unwrap();
                if let Some(_lifetime_expr) = items.pop() {
                    self.push_lowering_error(CompileError::SimpleError {
                        loc: rhs_expr.loc.clone(),
                        s: "lifetime specification not yet supported",
                    });
                    return Pattern::Poison;
                }

                let mut kind = VarKind::Const;
                if let Expr::Prefix(ref inner_op, ref mut inner_items) = rhs_expr.value
                    && matches!(inner_op.value, "mut" | "const")
                {
                    debug_assert_eq!(inner_items.len(), 1);
                    kind = if inner_op.value == "mut" {
                        VarKind::Mut
                    } else {
                        VarKind::Const
                    };

                    let mut inner = inner_items.pop().unwrap();
                    std::mem::swap(&mut rhs_expr, &mut inner);
                }

                let rhs = self.lower_pattern(rhs_expr, kind);
                Pattern::AddrOf(rhs, kind)
            }

            Expr::Prefix(open, items) if open.value == "(" => {
                let span = self.reserve_pattern_span(items.len());
                for (index, item) in items.into_iter().enumerate() {
                    let target = span.at(index);
                    self.lower_pattern_into(target, item, m);
                }
                Pattern::Tuple(span)
            }

            Expr::Prefix(open, items) if open.value == "'" => {
                let Some(Expr::Atom(Token::Ident(n))) = items.first().map(|x| &x.value) else {
                    self.push_lowering_error(CompileError::SimpleError {
                        loc: loc.clone(),
                        s: "invalid lifetime syntax",
                    });
                    return Pattern::Poison;
                };
                if items.len() > 1 {
                    self.push_lowering_error(CompileError::SimpleError {
                        loc: loc.clone(),
                        s: "invalid lifetime syntax",
                    });
                    return Pattern::Poison;
                }
                if matches!(n.as_str(), "_" | "static" | "raw") {
                    self.push_lowering_error(CompileError::SimpleError {
                        loc: loc.clone(),
                        s: "using a reserved name for a lifetime",
                    });
                    return Pattern::Poison;
                }
                let s = self.str_intern.intern(n);
                let life = self.insert_new_lifetiime(s);
                Pattern::LifeTime(life)
            }

            Expr::Prefix(op, _) | Expr::Postfix(op, _) => {
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc,
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: "this pattern expression is not supported",
                });
                Pattern::Poison
            }
        }
    }

    #[inline(always)]
    fn lower_match_arm(&mut self, expr: LExpr) -> MatchArm {
        match expr.value {
            Expr::Bin(op, pair) if op.value == "=>" => {
                let (pat_expr, body_expr) = *pair;
                let pat = self.lower_pattern(pat_expr, VarKind::Const);
                let body = self.lower_value(body_expr);
                MatchArm { pat, body }
            }

            Expr::Bin(op, _) => {
                let loc = expr.loc.clone();
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc: loc.clone(),
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: "match arms must be written as `<pattern> => <expr>`",
                });
                MatchArm {
                    pat: self.poison_pattern(loc.clone()),
                    body: self.poison_value(loc),
                }
            }

            Expr::Prefix(op, _) | Expr::Postfix(op, _) => {
                let loc = expr.loc.clone();
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc: loc.clone(),
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: "match arms must be written as `<pattern> => <expr>`",
                });
                MatchArm {
                    pat: self.poison_pattern(loc.clone()),
                    body: self.poison_value(loc),
                }
            }

            _ => {
                let loc = expr.loc.clone();
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc: loc.clone(),
                    op_loc: None,
                    op: None,
                    message: "match arms must be written as `<pattern> => <expr>`",
                });
                MatchArm {
                    pat: self.poison_pattern(loc.clone()),
                    body: self.poison_value(loc),
                }
            }
        }
    }

    #[inline(always)]
    fn lower_addr_of(&mut self, mut items: Vec<LExpr>) -> Value {
        let mut rhs_expr = items.pop().unwrap();
        if let Some(_lifetime_expr) = items.pop() {
            self.push_lowering_error(CompileError::SimpleError {
                loc: rhs_expr.loc.clone(),
                s: "lifetime specification not yet supported",
            });
            return Value::Poison;
        }

        let mut kind = None;
        if let Expr::Prefix(ref inner_op, ref mut inner_items) = rhs_expr.value
            && matches!(inner_op.value, "mut" | "const")
        {
            debug_assert_eq!(inner_items.len(), 1);
            kind = Some(if inner_op.value == "mut" {
                VarKind::Mut
            } else {
                VarKind::Const
            });

            let mut inner = inner_items.pop().unwrap();
            std::mem::swap(&mut rhs_expr, &mut inner);
        }

        let rhs = self.lower_value(rhs_expr);
        Value::AddrOf(rhs, kind)
    }

    // ===============================
    // Operator lowering helpers
    // ===============================

    #[inline(always)]
    fn lower_prefix_op(&mut self, loc: Loc, op: Located<&'static str>, items: Vec<LExpr>) -> Value {
        if items.len() != 1 {
            self.push_lowering_error(CompileError::SimpleError {
                loc: loc.clone(),
                s: "prefix operator requires exactly one operand",
            });
            return Value::Poison;
        }

        let rhs_expr = items.into_iter().next().unwrap();
        let unop = match op.value {
            "-" => UnOp::Neg,
            "!" => UnOp::Not,
            "~" => UnOp::BitNot,
            "*" => {
                let rhs = self.lower_value(rhs_expr);
                return Value::Deref(rhs);
            }
            "&" => {
                unreachable!()
            }

            "++" => return self.lower_inc_dec_prefix(op.map(|_| Dir::Inc), vec![rhs_expr]),
            "--" => return self.lower_inc_dec_prefix(op.map(|_| Dir::Dec), vec![rhs_expr]),

            _ => {
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc,
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: "this prefix operator is not supported as a value",
                });
                return Value::Poison;
            }
        };

        let rhs = self.lower_value(rhs_expr);

        let _ = loc;
        Value::UnOp {
            op: unop,
            value: rhs,
        }
    }

    #[inline(always)]
    fn lower_postfix_op(
        &mut self,
        _loc: Loc,
        op: Located<&'static str>,
        items: Vec<LExpr>,
    ) -> Value {
        match op.value {
            // these are handled earlier and must never reach here
            "(" | "[" => unreachable!("call/index should be handled before postfix ops"),

            "++" => self.lower_inc_dec_postfix(op.map(|_| Dir::Inc), items),
            "--" => self.lower_inc_dec_postfix(op.map(|_| Dir::Dec), items),

            _ => {
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc: _loc,
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: "this postfix operator is not supported by the IR lowering",
                });
                Value::Poison
            }
        }
    }

    #[inline(always)]
    fn lower_inc_dec_prefix(&mut self, op: Located<Dir>, mut items: Vec<LExpr>) -> Value {
        debug_assert_eq!(items.len(), 1);

        let target = self.lower_value(items.pop().unwrap());

        let _ = op.loc;
        Value::Assign {
            op: AssignOp::Pre(op.value),
            target,
        }
    }

    #[inline(always)]
    fn lower_inc_dec_postfix(&mut self, op: Located<Dir>, mut items: Vec<LExpr>) -> Value {
        debug_assert_eq!(items.len(), 1);

        let target = self.lower_value(items.pop().unwrap());

        let _ = op.loc.clone();
        Value::Assign {
            op: AssignOp::Post(op.value),
            target,
        }
    }

    #[inline(always)]
    fn lower_binary_op(
        &mut self,
        loc: Loc,
        op: Located<&'static str>,
        lhs: LExpr,
        rhs: LExpr,
    ) -> Value {
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
            let target = self.lower_value(lhs);
            let value = self.lower_value(rhs);

            let _ = loc;
            return Value::Assign {
                target,
                op: if let Some(o) = assign_op {
                    AssignOp::Bin(o, value)
                } else {
                    AssignOp::Nothing(value)
                },
            };
        }

        if let Some(logic_op) = match op.value {
            "&&" => Some(LogicOp::And),
            "||" => Some(LogicOp::Or),
            _ => None,
        } {
            let left = self.lower_value(lhs);
            let right = self.lower_value(rhs);
            let _ = loc;
            return Value::LogicOp {
                op: logic_op,
                values: (left, right),
            };
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
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc,
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: "this binary operator is not supported as a value",
                });
                return Value::Poison;
            }
        };

        let left = self.lower_value(lhs);
        let right = self.lower_value(rhs);

        let _ = loc;
        Value::BinOp {
            op: binop,
            values: (left, right),
        }
    }

    #[inline(always)]
    fn lower_pipe_expr(&mut self, _loc: Loc, lhs: LExpr, rhs: LExpr) -> Value {
        let Located { loc, value } = rhs;
        match value {
            Expr::Postfix(open, mut items) if open.value == "(" => {
                if items.is_empty() {
                    self.push_lowering_error(CompileError::SimpleError {
                        loc,
                        s: "pipe (`|>`) requires a call expression on the right-hand side",
                    });
                    return Value::Poison;
                }

                items.insert(1, lhs);
                let call = self.lower_call_like_expr(loc.clone(), items);
                Value::Call(call)
            }
            _ => {
                self.push_lowering_error(CompileError::SimpleError {
                    loc,
                    s: "pipe (`|>`) requires a call expression on the right-hand side",
                });
                Value::Poison
            }
        }
    }

    pub fn lower_type_expr(&mut self, expr: LExpr) -> TExpId {
        let loc = expr.loc.clone();
        let exp = self.lower_type_expr_inner(expr);
        self.id_type_expr(loc, exp)
    }

    fn lower_type_expr_into(&mut self, target: TExpId, expr: LExpr) -> TExpId {
        let loc = expr.loc.clone();
        let exp = self.lower_type_expr_inner(expr);
        self.set_type_expr(target, loc, exp);
        target
    }

    fn lower_type_expr_inner(&mut self, expr: LExpr) -> TypeExpr {
        let loc = expr.loc.clone();
        match expr.value {
            Expr::Atom(Token::Ident(name)) if name == "_" => TypeExpr::Wildcard,

            Expr::Atom(Token::Ident(name)) => {
                let id = self.resolve_name(&loc, &name);
                TypeExpr::NameRef(id)
            }

            Expr::Atom(Token::Operator("(")) => {
                let span = self.reserve_type_expr_span(0);
                TypeExpr::Tuple(span)
            }

            Expr::Prefix(open, items) if open.value == "(" => {
                let span = self.reserve_type_expr_span(items.len());
                for (index, item) in items.into_iter().enumerate() {
                    let target = span.at(index);
                    self.lower_type_expr_into(target, item);
                }
                TypeExpr::Tuple(span)
            }

            Expr::Prefix(open, mut items) if open.value == "[" => {
                if items.is_empty() || items.len() > 2 {
                    self.push_lowering_error(CompileError::UnsupportedForm {
                        loc,
                        op_loc: Some(open.loc),
                        op: Some(open.value),
                        message: "array type expressions must specify an element type and optional literal length",
                    });
                    return TypeExpr::Poison;
                }

                let element = self.lower_type_expr(items.remove(0));
                let len = if let Some(len_expr) = items.pop() {
                    match len_expr.value {
                        Expr::Atom(Token::NumLit(n)) => {
                            let Ok(n) = usize::try_from(n) else {
                                self.push_lowering_error(CompileError::UnsupportedForm {
                                    loc,
                                    op_loc: Some(open.loc),
                                    op: Some(open.value),
                                    message: "array length must be a non-negative integer literal",
                                });
                                return TypeExpr::Poison;
                            };
                            Some(n)
                        }
                        _ => {
                            self.push_lowering_error(CompileError::UnsupportedForm {
                                loc,
                                op_loc: Some(open.loc),
                                op: Some(open.value),
                                message: "array types with non-literal lengths are not supported",
                            });
                            return TypeExpr::Poison;
                        }
                    }
                } else {
                    None
                };

                TypeExpr::Array(element, len)
            }

            Expr::Postfix(open, items) if open.value == "[" => {
                if items.is_empty() {
                    self.push_lowering_error(CompileError::UnsupportedForm {
                        loc,
                        op_loc: Some(open.loc),
                        op: Some(open.value),
                        message: "type indexing with brackets is not supported yet",
                    });
                    return TypeExpr::Poison;
                }

                let items_len = items.len() - 1;
                let mut items = items.into_iter();
                let base = self.lower_type_expr(items.next().unwrap());

                let parts = self.reserve_type_expr_span(items_len);
                for (index, arg) in items.enumerate() {
                    let target = parts.at(index);
                    self.lower_type_expr_into(target, arg);
                }

                let mut lifetime_end = items_len;
                let mut seen_generic = false;
                for index in 0..items_len {
                    let arg = parts.at(index);
                    let is_lifetime = matches!(self.type_expr(arg), TypeExpr::LifeTime(_));
                    if is_lifetime {
                        if seen_generic {
                            self.push_lowering_error(CompileError::SimpleError {
                                loc: self.type_expr_loc(arg),
                                s: "lifetimes must come before generic parameters",
                            });
                        }
                    } else {
                        seen_generic = true;
                        if lifetime_end == items_len {
                            lifetime_end = index;
                        }
                    }
                }

                TypeExpr::Index {
                    base,
                    args: GenIndex {
                        base: parts.start(),
                        parts,
                        lifetime_end,
                    },
                }
            }

            Expr::Prefix(op, items) if op.value == "'" => {
                if items.len() != 1 {
                    self.push_lowering_error(CompileError::SimpleError {
                        loc,
                        s: "invalid lifetime syntax",
                    });
                    return TypeExpr::Poison;
                }

                let Expr::Atom(Token::Ident(n)) = &items[0].value else {
                    self.push_lowering_error(CompileError::SimpleError {
                        loc,
                        s: "invalid lifetime syntax",
                    });
                    return TypeExpr::Poison;
                };

                if n == "_" {
                    return TypeExpr::LifeTime(LifeTimeId::WILDCARD);
                }

                let sid = self.str_intern.intern(n);
                let Some(life) = self.try_get_lifetime(sid) else {
                    self.push_lowering_error(CompileError::SimpleError {
                        loc,
                        s: "unknown lifetime",
                    });
                    return TypeExpr::Poison;
                };

                TypeExpr::LifeTime(life)
            }

            Expr::Prefix(op, mut items) if matches!(op.value, "*" | "&") => {
                let raw = op.value == "*";
                let mut mutable = raw;
                let mut inner = items.pop().unwrap();

                let lifetime = match items.pop() {
                    None => None,
                    Some(lexp) => {
                        let Expr::Prefix(op, mut items2) = lexp.value else {
                            self.push_lowering_error(CompileError::SimpleError {
                                loc: loc.clone(),
                                s: "invalid lifetime syntax",
                            });
                            return TypeExpr::Poison;
                        };
                        if op.value != "'" {
                            self.push_lowering_error(CompileError::SimpleError {
                                loc: loc.clone(),
                                s: "invalid lifetime syntax",
                            });
                            return TypeExpr::Poison;
                        }
                        let subexp = items2.pop().unwrap();
                        let Expr::Atom(Token::Ident(n)) = subexp.value else {
                            self.push_lowering_error(CompileError::SimpleError {
                                loc: loc.clone(),
                                s: "invalid lifetime syntax",
                            });
                            return TypeExpr::Poison;
                        };
                        if n == "_" {
                            None
                        } else {
                            let s = self.str_intern.intern(&n);
                            let Some(life) = self.try_get_lifetime(s) else {
                                self.push_lowering_error(CompileError::SimpleError {
                                    loc: loc.clone(),
                                    s: "unknown lifetime",
                                });
                                return TypeExpr::Poison;
                            };

                            if life == LifeTimeId::RAW {
                                mutable = true;
                            }
                            Some(life)
                        }
                    }
                };

                if let Expr::Prefix(ref inner_op, ref mut inner_items) = inner.value
                    && matches!(inner_op.value, "mut" | "const")
                {
                    if inner_items.len() != 1 {
                        self.push_lowering_error(CompileError::UnsupportedForm {
                            loc,
                            op_loc: Some(inner_op.loc.clone()),
                            op: Some(inner_op.value),
                            message: "pointer qualifiers must wrap exactly one inner type",
                        });
                        return TypeExpr::Poison;
                    }
                    mutable = inner_op.value == "mut";
                    inner = inner_items.pop().unwrap();
                }

                let base = self.lower_type_expr(inner);
                TypeExpr::Ptr {
                    base,
                    raw,
                    mutable,
                    lifetime,
                }
            }

            Expr::Prefix(open, items)
                if matches!(open.value, "struct" | "cstruct" | "enum" | "union") =>
            {
                self.lower_struct_like_type_expr(open, items)
            }

            Expr::Prefix(open, items) if open.value == "fn" || open.value == "cfn" => {
                self.lower_fn_type_expr(expr.loc, open, items)
            }

            Expr::Atom(Token::Operator(op)) => {
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc: loc.clone(),
                    op_loc: Some(loc),
                    op: Some(op),
                    message: "operators cannot be used as standalone type expressions",
                });
                TypeExpr::Poison
            }

            Expr::Bin(op, pair) if op.value == "<" => {
                let (lhs, rhs) = *pair;
                TypeExpr::Lt {
                    lhs: self.lower_type_expr(lhs),
                    rhs: self.lower_type_expr(rhs),
                }
            }

            Expr::Bin(op, _) => {
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc,
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: "operators cannot be used as standalone type expressions",
                });
                TypeExpr::Poison
            }

            Expr::Prefix(op, _) | Expr::Postfix(op, _) => {
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc,
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: "operators cannot be used as standalone type expressions",
                });
                TypeExpr::Poison
            }

            _ => {
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc,
                    op_loc: None,
                    op: None,
                    message: "this type expression form is not supported yet",
                });
                TypeExpr::Poison
            }
        }
    }

    #[inline(always)]
    fn lower_fn_type_expr(&mut self, loc: Loc, fn_kw: LFixed, items: Vec<LExpr>) -> TypeExpr {
        debug_assert!(
            (1..=3).contains(&items.len()),
            "fn expects optional generics, signature, and optional body"
        );

        let mut items = items.into_iter().peekable();

        if let Some(peek) = items.peek()
            && matches!(&peek.value, Expr::Prefix(open, _) if open.value == "[")
        {
            let generics_expr = items.next().unwrap();
            self.push_lowering_error(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(generics_expr.loc),
                op: Some("["),
                message: "functions type expressions may not contain generics (may be added later for some subset)",
            });
            return TypeExpr::Poison;
        }

        let sig_expr = items.next().expect("fn missing signature");
        if let Some(body_expr) = items.next() {
            self.push_lowering_error(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(body_expr.loc),
                op: None,
                message: "functions type expressions dont have a body",
            });
            return TypeExpr::Poison;
        }

        let (params_expr, ret_expr) = match sig_expr.value {
            Expr::Bin(arrow, pair) if arrow.value == "->" => {
                let (lhs, rhs) = *pair;
                (lhs, Some(rhs))
            }
            _ => (sig_expr, None),
        };

        let Expr::Prefix(p_open, param_items) = params_expr.value else {
            debug_assert!(false, "fn signature does not start with parameter list");
            return TypeExpr::Poison;
        };
        debug_assert!(p_open.value == "(", "fn parameter list must start with '('");

        let calling_convention = CallingConvention::from_fn_keyword(fn_kw.value).unwrap();

        let params_span = self.reserve_type_expr_span(param_items.len());
        for (index, param) in param_items.into_iter().enumerate() {
            let target = params_span.at(index);
            self.lower_type_expr_into(target, param);
        }

        let output_type = ret_expr.map(|e| self.lower_type_expr(e));

        TypeExpr::Func {
            calling_convention,
            params: params_span,
            output_type,
        }
    }

    fn lower_struct_like_type_expr(&mut self, kw: LFixed, items: Vec<LExpr>) -> TypeExpr {
        let mut items = items.into_iter().peekable();

        let mut generics_expr = None;
        if matches!(&items.peek().unwrap().value, Expr::Prefix(open, _) if open.value == "[") {
            generics_expr = Some(items.next().unwrap());
        }

        let fields_expr = match items.next() {
            Some(expr) => expr,
            None => {
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc: kw.loc.clone(),
                    op_loc: Some(kw.loc),
                    op: Some(kw.value),
                    message: "type literals must provide exactly one field block",
                });
                return TypeExpr::Poison;
            }
        };

        if items.next().is_some() {
            self.push_lowering_error(CompileError::UnsupportedForm {
                loc: kw.loc.clone(),
                op_loc: Some(kw.loc),
                op: Some(kw.value),
                message: "type literals cannot have extra items beyond the field block",
            });
            return TypeExpr::Poison;
        }

        let generics = self.lower_generic_dec(generics_expr);

        let fields = match fields_expr.value {
            Expr::Prefix(open, items) if open.value == "{" => {
                let span = self.reserve_pattern_span(items.len());
                for (index, item) in items.into_iter().enumerate() {
                    let target = span.at(index);
                    self.lower_pattern_into(target, item, VarKind::Mut);
                }
                span
            }
            _ => {
                self.push_lowering_error(CompileError::UnsupportedForm {
                    loc: fields_expr.loc,
                    op_loc: Some(kw.loc),
                    op: Some(kw.value),
                    message: "type literals currently require a `{}` block of fields",
                });
                return TypeExpr::Poison;
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
        match kw.value {
            "cstruct" => TypeExpr::Struct(def),
            "struct" => TypeExpr::Struct(def),
            "enum" => TypeExpr::Enum(def),
            "union" => TypeExpr::Union(def),
            _ => TypeExpr::Poison,
        }
    }

    #[inline(always)]
    fn find_lifetime_end_in_pattern_span(&mut self, parts: PatternSpan) -> usize {
        let mut lifetime_end = parts.len();
        let mut seen_generic = false;
        for index in 0..parts.len() {
            let pat = parts.at(index);
            let is_lifetime = matches!(self.pattern(pat), Pattern::LifeTime(_));
            if is_lifetime {
                if seen_generic {
                    self.push_lowering_error(CompileError::SimpleError {
                        loc: self.pattern_loc(pat),
                        s: "lifetimes must come before generic parameters",
                    });
                }
            } else {
                seen_generic = true;
                if lifetime_end == parts.len() {
                    lifetime_end = index;
                }
            }
        }
        lifetime_end
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
        let ir = program.lower_value(expr);
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
    use crate::parsing::Parser;
    use crate::program::{CompileError, Defined, Program};

    fn lower_block(src: &str) -> (Program, ValId) {
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().expect("failed to parse expr");
        let ir = program.lower_value(expr);
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
    fn member_method_duplicate_definitions_form_a_function_set() {
        let src = "type S = struct { a: int }; S.foo = fn(x){ x }; S.foo = fn(x, y){ x };";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();

        let expr = parser
            .parse_with_macros(&mut program)
            .expect("failed to parse type")
            .expect("missing type expr");
        program.gather_definition(expr);

        let expr = parser
            .parse_with_macros(&mut program)
            .expect("failed to parse first method")
            .expect("missing first method expr");
        program.gather_definition(expr);

        let expr = parser
            .parse_with_macros(&mut program)
            .expect("failed to parse second method")
            .expect("missing second method expr");
        program.gather_definition(expr);
        assert!(program.lowering_errors.is_empty());

        let struct_name = program.str_intern.intern("S");
        let struct_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.0.get(&struct_name))
            .expect("missing struct name");
        let method_name = program.str_intern.intern("foo");
        let methods = program
            .member_methods
            .get(&struct_id)
            .and_then(|methods| methods.get(&method_name))
            .expect("missing foo method");
        assert!(methods.declarations.is_empty());
        assert_eq!(methods.implementations.len(), 2);

        let first = methods.implementations[0];
        let second = methods.implementations[1];

        let Value::Func { params, .. } = program.value(first) else {
            panic!("expected first foo declaration to be a function");
        };
        assert_eq!(params.len(), 1);

        let Value::Func { params, .. } = program.value(second) else {
            panic!("expected second foo declaration to be a function");
        };
        assert_eq!(params.len(), 2);
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
    fn lowers_int_access_for_dot_and_ptr() {
        let src = "{ let t = (1, 2); t.0; t->1; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 3);

        let t_id = bound_id(&program, statements[0]);

        match program.value(statements[1]) {
            Value::IntAccess { base, id, kind } => {
                assert_eq!(kind, AccessKind::Dot);
                match program.value(base) {
                    Value::NameRef(id) => assert_eq!(id, t_id),
                    _ => panic!("expected dot base name"),
                }
                assert_eq!(id, 0);
            }
            _ => panic!("expected dot int access"),
        }

        match program.value(statements[2]) {
            Value::IntAccess { base, id, kind } => {
                assert_eq!(kind, AccessKind::Ptr);
                match program.value(base) {
                    Value::NameRef(id) => assert_eq!(id, t_id),
                    _ => panic!("expected ptr base name"),
                }
                assert_eq!(id, 1);
            }
            _ => panic!("expected ptr int access"),
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
            .and_then(|scope| scope.0.get(&f_name))
            .expect("missing f binding");
        let defined = program.definitions.get(&f_id).expect("missing definition");

        match defined {
            Defined::Func(funcs) => {
                let value = funcs
                    .implementations
                    .first()
                    .copied()
                    .expect("expected function implementation");
                match program.value(value) {
                    Value::Func { generics, .. } => assert_eq!(generics.generics().len(), 1),
                    _ => panic!("expected function value"),
                }
            }
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
            .and_then(|scope| scope.0.get(&f_name))
            .expect("missing f binding");

        let Defined::Func(funcs) = program
            .definitions
            .get(&f_id)
            .expect("missing f definition")
        else {
            panic!("expected function definition")
        };

        let value = funcs
            .declarations
            .first()
            .copied()
            .expect("expected function declaration");

        let Value::Func {
            calling_convention,
            params,
            output_type,
            body,
            ..
        } = program.value(value)
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
            .and_then(|scope| scope.0.get(&s_name))
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
            .and_then(|scope| scope.0.get(&f_name))
            .expect("missing f binding");
        let g_name = program.str_intern.intern("g");
        let g_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.0.get(&g_name))
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
            Defined::Func(funcs) => {
                let value = funcs
                    .implementations
                    .first()
                    .copied()
                    .expect("expected function implementation");
                match program.value(value) {
                    Value::Func { body, .. } => body.expect("expected f to have a body"),
                    _ => panic!("expected f to be a function"),
                }
            }
            _ => panic!("expected f to lower to a value"),
        };

        let g_body = match g_def {
            Defined::Func(funcs) => {
                let value = funcs
                    .implementations
                    .first()
                    .copied()
                    .expect("expected function implementation");
                match program.value(value) {
                    Value::Func { body, .. } => body.expect("expected g to have a body"),
                    _ => panic!("expected g to be a function"),
                }
            }
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
        program.lower_value(expr);
        let mut errors = std::mem::take(&mut program.lowering_errors);
        assert_eq!(errors.len(), 1);
        let err = errors.pop().unwrap();
        match err {
            CompileError::SimpleError { s, .. } => {
                assert_eq!(s, ACCESS_EXPECTS_NAME_MSG);
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
        let ir = program.lower_value(expr);

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
        let ir = program.lower_value(expr);

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
            .and_then(|scope| scope.0.get(&f_name))
            .expect("missing f binding");

        let Defined::Func(funcs) = program
            .definitions
            .get(&f_id)
            .expect("missing f definition")
        else {
            panic!("expected function definition")
        };

        let f_val = funcs
            .implementations
            .first()
            .copied()
            .expect("expected function implementation");

        let Value::Func { body, .. } = program.value(f_val) else {
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
        let errors = program.lower_all(&mut parser).unwrap_err();

        let err = errors
            .into_iter()
            .find(|e| matches!(e, CompileError::UnresolvedLabel { .. }))
            .unwrap();
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
        let errors = program.lower_all(&mut parser).unwrap_err();

        let err = errors
            .into_iter()
            .find(|e| matches!(e, CompileError::SimpleError { .. }))
            .unwrap();
        match err {
            CompileError::SimpleError { s, .. } => assert_eq!(s, LABEL_NAME_REQUIRED_MSG),
            other => panic!("expected label syntax error, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_label_definition_errors() {
        let src = "f = fn(){ `x; `x; }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let errors = program.lower_all(&mut parser).unwrap_err();

        let err = errors
            .into_iter()
            .find(|e| matches!(e, CompileError::SimpleError { .. }))
            .unwrap();
        match err {
            CompileError::SimpleError { s, .. } => assert_eq!(s, LABEL_ALREADY_DEFINED_MSG),
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

    #[test]
    fn lowers_sized_array_type_annotation() {
        let src = "{ let x:[int;3] = [1:int, 2:int, 3:int]; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 1);

        let let_stmt = statements[0];
        match program.value(let_stmt) {
            Value::Let { pat, .. } => match program.pattern(pat) {
                Pattern::TypeAnnotation { ty, .. } => match program.type_expr(ty) {
                    TypeExpr::Array(element, Some(3)) => match program.type_expr(element) {
                        TypeExpr::NameRef(_) => {}
                        _ => panic!("expected array element type name"),
                    },
                    _ => panic!("expected sized array type expression"),
                },
                _ => panic!("expected type annotation pattern"),
            },
            _ => panic!("expected let statement"),
        }
    }

    #[test]
    fn lowers_unsized_array_type_annotation() {
        let src = "{ let x:[int] = [1:int]; }";
        let (program, ir) = lower_block(src);

        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 1);

        let let_stmt = statements[0];
        match program.value(let_stmt) {
            Value::Let { pat, .. } => match program.pattern(pat) {
                Pattern::TypeAnnotation { ty, .. } => match program.type_expr(ty) {
                    TypeExpr::Array(_, None) => {}
                    _ => panic!("expected unsized array type expression"),
                },
                _ => panic!("expected type annotation pattern"),
            },
            _ => panic!("expected let statement"),
        }
    }

    #[test]
    fn fn_generics_only() {
        let src = "f = fn[T](x: T) -> T { x }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        program.lower_all(&mut parser).unwrap();

        let f_name = program.str_intern.intern("f");
        let f_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.0.get(&f_name))
            .expect("missing f binding");
        let defined = program.definitions.get(&f_id).expect("missing definition");

        match defined {
            Defined::Func(funcs) => {
                let value = funcs
                    .implementations
                    .first()
                    .copied()
                    .expect("expected function implementation");
                match program.value(value) {
                    Value::Func { generics, .. } => {
                        assert!(generics.lifetimes().is_empty());
                        assert_eq!(generics.generics().len(), 1);
                    }
                    _ => panic!("expected function value"),
                }
            }
            _ => panic!("expected value definition"),
        }
    }

    #[test]
    fn fn_lifetimes_only() {
        let src = "f = fn['a](x: &'a int) -> &'a int { x }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        program.lower_all(&mut parser).unwrap();

        let f_name = program.str_intern.intern("f");
        let f_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.0.get(&f_name))
            .expect("missing f binding");
        let defined = program.definitions.get(&f_id).expect("missing definition");

        match defined {
            Defined::Func(funcs) => {
                let value = funcs
                    .implementations
                    .first()
                    .copied()
                    .expect("expected function implementation");
                match program.value(value) {
                    Value::Func { generics, .. } => {
                        assert_eq!(generics.lifetimes().len(), 1);
                        assert!(generics.generics().is_empty());
                    }
                    _ => panic!("expected function value"),
                }
            }
            _ => panic!("expected value definition"),
        }
    }

    #[test]
    fn fn_lifetimes_and_generics() {
        let src = "f = fn['a, T](x: &'a T) -> &'a T { x }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        program.lower_all(&mut parser).unwrap();

        let f_name = program.str_intern.intern("f");
        let f_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.0.get(&f_name))
            .expect("missing f binding");
        let defined = program.definitions.get(&f_id).expect("missing definition");

        match defined {
            Defined::Func(funcs) => {
                let value = funcs
                    .implementations
                    .first()
                    .copied()
                    .expect("expected function implementation");
                match program.value(value) {
                    Value::Func { generics, .. } => {
                        assert_eq!(generics.lifetimes().len(), 1);
                        assert_eq!(generics.generics().len(), 1);
                        assert!(generics.where_clause().is_empty());
                    }
                    _ => panic!("expected function value"),
                }
            }
            _ => panic!("expected value definition"),
        }
    }

    #[test]
    fn fn_generics_with_where_clause() {
        let src = "f = fn['a, 'b, T, where T<'a, T<'b](x: T) -> T { x }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        program.lower_all(&mut parser).unwrap();

        let f_name = program.str_intern.intern("f");
        let f_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.0.get(&f_name))
            .expect("missing f binding");
        let defined = program.definitions.get(&f_id).expect("missing definition");

        match defined {
            Defined::Func(funcs) => {
                let value = funcs
                    .implementations
                    .first()
                    .copied()
                    .expect("expected function implementation");
                match program.value(value) {
                    Value::Func { generics, .. } => {
                        assert_eq!(generics.lifetimes().len(), 2);
                        assert_eq!(generics.generics().len(), 1);
                        assert_eq!(generics.where_clause().len(), 2);

                        for constraint in generics.where_clause().ids() {
                            let TypeExpr::Lt { lhs, rhs } = program.type_expr(constraint) else {
                                panic!("expected `<` type expression in where clause");
                            };
                            assert!(matches!(program.type_expr(lhs), TypeExpr::NameRef(_)));
                            assert!(matches!(program.type_expr(rhs), TypeExpr::LifeTime(_)));
                        }
                    }
                    _ => panic!("expected function value"),
                }
            }
            _ => panic!("expected value definition"),
        }
    }

    #[test]
    fn type_index_generics_only() {
        let src = "{ let x: Vec[int] = Vec.new(); }";
        let (program, ir) = lower_block(src);
        let statements = match program.value(ir) {
            Value::Block { statements, .. } => statements.ids().collect::<Vec<_>>(),
            _ => panic!("expected block"),
        };
        let let_stmt = statements[0];
        let Value::Let { pat, .. } = program.value(let_stmt) else {
            panic!("expected let");
        };
        let Pattern::TypeAnnotation { ty, .. } = program.pattern(pat) else {
            panic!("expected type annotation");
        };
        let TypeExpr::Index { args, .. } = program.type_expr(ty) else {
            panic!("expected index");
        };
        assert!(args.lifetimes().is_empty());
        assert_eq!(args.generics().len(), 1);
    }

    #[test]
    fn type_index_lifetime_then_generic_lowers_lifetime_arg() {
        let src = "Box = struct['a, T]{inner:T}; f = fn['a](x: Box['a, int]) -> void {}";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        program.lower_all(&mut parser).unwrap();
        assert!(program.lowering_errors.is_empty());

        let f_name = program.str_intern.intern("f");
        let f_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.0.get(&f_name))
            .expect("missing f binding");
        let defined = program.definitions.get(&f_id).expect("missing definition");

        let value = match defined {
            Defined::Func(funcs) => funcs
                .implementations
                .first()
                .copied()
                .expect("expected function implementation"),
            _ => panic!("expected function definition"),
        };

        let Value::Func { params, .. } = program.value(value) else {
            panic!("expected function value");
        };
        let x_pat = params.at(0);
        let Pattern::TypeAnnotation { ty, .. } = program.pattern(x_pat) else {
            panic!("expected typed parameter");
        };
        let TypeExpr::Index { args, .. } = program.type_expr(ty) else {
            panic!("expected indexed type");
        };

        assert_eq!(args.lifetimes().len(), 1);
        assert_eq!(args.generics().len(), 1);

        let lifetime_arg = args
            .lifetimes()
            .ids()
            .next()
            .expect("expected one lifetime arg");
        match program.type_expr(lifetime_arg) {
            TypeExpr::LifeTime(_) => {}
            _ => panic!("expected lowered lifetime type arg"),
        }
    }

    #[test]
    fn type_index_lifetime_without_scope_errors() {
        let src = "{ let x: Box['a, int] = Box.new(); }";
        let (program, _) = lower_block(src);
        assert!(!program.lowering_errors.is_empty());
        let err_msg = program.lowering_errors[0].to_string();
        assert!(err_msg.contains("unknown lifetime"));
    }

    #[test]
    fn fn_mixed_lifetimes_generics_error() {
        let src = "f = fn[T, 'a](x: &'a T) -> T { x }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let errs = program.lower_all(&mut parser).unwrap_err();

        let err_msg = errs[0].to_string();
        assert!(err_msg.contains("lifetimes must come before generic parameters"));
    }

    #[test]
    fn type_index_mixed_lifetimes_generics_error() {
        let src = "{ let x: Box[int, 'raw] = Box.new(); }";
        let (program, _) = lower_block(src);
        assert!(!program.lowering_errors.is_empty());
        let err_msg = program.lowering_errors[0].to_string();
        assert!(err_msg.contains("lifetimes must come before generic parameters"));
    }
}
