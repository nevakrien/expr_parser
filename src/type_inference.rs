//! Type inference sketch (constraint-first, local, non-integrated)
//
// ================================================================
// DESIGN GOALS (SHOULD STAY TRUE)
// ================================================================
// 1) Constraints are RECORDS, not guesses.
// 2) Inference is a SEPARATE, mutable process.
// 3) Every value has exactly ONE producer, and 0..N consumers.
// 4) Errors come from constraints (produce + consume sites).
//
// ================================================================

use crate::parsing::Located;
use std::collections::HashMap;

use crate::{
    ir::{BinOp, Literal, NameId, Pattern, TPattern, TValue, UnOp, Value},
    parsing::Loc,
    program::{Defined, Program},
};

/* ================================================================
 * Core IDs (STABLE)
 * ================================================================ */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferId {
    Nothing,
    Concrete(TypeId),
    Infered(usize),
    GenericArg(usize),
}

/* ================================================================
 * Types & TypeStore (STABLE)
 * ================================================================ */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    Int,
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    F32,
    F64,
    Bool,
    Str,
    Void,
    Type,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeValue {
    Builtin(BuiltinType),
    Tuple(Vec<TypeId>),
    Func { params: Vec<TypeId>, ret: TypeId },
    Ptr(TypeId),
}

#[derive(Debug)]
pub struct TypeStore {
    values: Vec<TypeValue>,
    intern: HashMap<TypeValue, TypeId>,
}

impl Default for TypeStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeStore {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            intern: HashMap::new(),
        }
    }

    pub fn get(&self, id: TypeId) -> &TypeValue {
        &self.values[id.0]
    }

    pub fn intern(&mut self, ty: TypeValue) -> TypeId {
        if let Some(&id) = self.intern.get(&ty) {
            return id;
        }
        let id = TypeId(self.values.len());
        self.values.push(ty.clone());
        self.intern.insert(ty, id);
        id
    }
}

/* ================================================================
 * Typed wrapper (STABLE)
 * ================================================================ */

#[derive(Debug, Clone, PartialEq)]
pub struct Typed<T> {
    pub loc: Loc,
    pub ty: InferId,
    pub value: T,
}

impl Program {
    pub(crate) fn insert_builtin_types(&mut self) {
        use BuiltinType::*;

        // One place to update when adding builtin types.
        // Note: `"float"` is an alias for `f64` in this sketch.
        const BUILTINS: &[(&str, BuiltinType)] = &[
            ("int", Int),
            ("i8", I8),
            ("i16", I16),
            ("i32", I32),
            ("i64", I64),
            ("i128", I128),
            ("isize", Isize),
            ("u8", U8),
            ("u16", U16),
            ("u32", U32),
            ("u64", U64),
            ("u128", U128),
            ("usize", Usize),
            ("f32", F32),
            ("f64", F64),
            ("float", F64),
            ("bool", Bool),
            ("str", Str),
            ("void", Void),
            ("Type", Type),
        ];

        for &(name, builtin) in BUILTINS {
            let ty = self.type_store.intern(TypeValue::Builtin(builtin));
            let name = self.str_intern.intern(name);
            let id = self.insert_value_in_current_scope(name);
            self.definitions.insert(id, Defined::TypeRef(ty));
        }
    }

    /// Lowering helper: create a typed value (no inference slot allocation here).
    pub(crate) fn typed_value(&mut self, loc: Loc, value: Value) -> TValue {
        Typed {
            loc,
            ty: InferId::Nothing,
            value,
        }
    }

    /// Lowering helper: create a typed pattern.
    pub(crate) fn typed_pattern(&mut self, loc: Loc, value: Pattern) -> TPattern {
        Typed {
            loc,
            ty: InferId::Nothing,
            value,
        }
    }
}

/* ================================================================
 * Errors (STABLE SHAPE)
 * ================================================================ */

#[derive(Debug)]
pub enum TypeError {
    Unresolved {
        produced_loc: Loc,
        message: &'static str,
    },

    SimpleMismatch {
        required_loc: Loc,
        produced_loc: Loc,
        expected: TypeId,
        found: TypeId,
        note: &'static str,
    },

    Unsupported {
        loc: Loc,
        message: &'static str,
    },

    ExpectedType {
        loc: Loc,
        message: &'static str,
    },
}

/* ================================================================
 * Constraint model (CORE)
 * ================================================================ */

#[derive(Debug, Clone)]
pub enum ProducedBy {
    Literal {
        lit: Literal,
    },
    NameRef {
        name: NameId,
    },
    LetBind {
        name: NameId,
        annotated: Option<Located<InferId>>,
    },
    Cast {
        target: InferId,
    },
    TypeAnnotationExpr {
        ty: InferId,
    },
    BinOp {
        op: BinOp,
    },
    UnOp {
        op: UnOp,
    },
    Block,
    Call,
    Func,
    Other(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub enum ConsumedAs {
    SameAs(InferId),
    Numeric,
    IntNumeric,
    FloatNumeric,
    Explicit(InferId),
    Other(&'static str),
}

#[derive(Debug)]
pub struct TypeConstraints {
    pub produced_loc: Loc,
    pub produced: ProducedBy,
    pub consumed: Vec<(Loc, ConsumedAs)>,
}

impl TypeConstraints {
    fn new(loc: Loc, produced: ProducedBy) -> Self {
        Self {
            produced_loc: loc,
            produced,
            consumed: Vec::new(),
        }
    }
}

/* ================================================================
 * Constraint collection
 * ================================================================ */

struct CollectCtx {
    infer: InferState,

    /// Constraints indexed by InferId
    constraints: HashMap<InferId, TypeConstraints>,

    /// Map each NameId to its InferId
    name_infers: HashMap<NameId, InferId>,
}

impl CollectCtx {
    fn new() -> Self {
        Self {
            infer: InferState::new(),
            constraints: HashMap::new(),
            name_infers: HashMap::new(),
        }
    }

    fn ensure_expr_id(&mut self, v: &mut TValue) -> InferId {
        match v.ty {
            InferId::Nothing => {
                let id = self.infer.fresh();
                v.ty = id;
                id
            }
            id => id,
        }
    }

    fn infer_for_name(&mut self, name: NameId) -> InferId {
        *self
            .name_infers
            .entry(name)
            .or_insert_with(|| self.infer.fresh())
    }

    fn set_producer(&mut self, id: InferId, loc: Loc, produced: ProducedBy) {
        self.constraints
            .entry(id)
            .and_modify(|c| {
                c.produced_loc = loc.clone();
                c.produced = produced.clone();
            })
            .or_insert_with(|| TypeConstraints::new(loc, produced));
    }

    fn add_consume(&mut self, id: InferId, loc: Loc, c: ConsumedAs) {
        self.constraints
            .entry(id)
            .or_insert_with(|| {
                TypeConstraints::new(loc.clone(), ProducedBy::Other("producer not recorded yet"))
            })
            .consumed
            .push((loc, c));
    }
}

/* ================================================================
 * Constraint walk
 * ================================================================ */

fn collect_constraints(program: &mut Program, value: &mut TValue) -> CollectCtx {
    let mut ctx = CollectCtx::new();
    collect_value(program, &mut ctx, value);
    ctx
}
fn collect_value(program: &mut Program, ctx: &mut CollectCtx, v: &mut TValue) {
    let this = ctx.ensure_expr_id(v);

    //we assume no operator overloads so that we can do NumericInt on things like a&b regardless of their types.

    match &mut v.value {
        Value::Literal(lit) => {
            ctx.set_producer(
                this,
                v.loc.clone(),
                ProducedBy::Literal { lit: lit.clone() },
            );
        }

        Value::Block {
            statements,
            return_value,
        } => {
            for s in statements.iter_mut() {
                collect_value(program, ctx, s);
            }

            if let Some(ret) = return_value.as_mut() {
                collect_value(program, ctx, ret);

                // block expr == return expr
                ctx.add_consume(this, v.loc.clone(), ConsumedAs::SameAs(ret.ty));
                ctx.add_consume(ret.ty, ret.loc.clone(), ConsumedAs::SameAs(this));
            }

            ctx.set_producer(this, v.loc.clone(), ProducedBy::Block);
        }

        Value::NameRef(name) => {
            let nid = ctx.infer_for_name(*name);

            ctx.set_producer(this, v.loc.clone(), ProducedBy::NameRef { name: *name });

            // expression <-> name-value equivalence
            ctx.add_consume(this, v.loc.clone(), ConsumedAs::SameAs(nid));
            ctx.add_consume(nid, v.loc.clone(), ConsumedAs::SameAs(this));
        }

        Value::Cast { value, ty } => {
            collect_value(program, ctx, value);
            collect_value(program, ctx, ty);

            let target = resolve_type_expr(program, ty).unwrap_or(InferId::Nothing);

            ctx.set_producer(this, v.loc.clone(), ProducedBy::Cast { target });

            // Cast is an explicit requirement
            if target != InferId::Nothing {
                ctx.add_consume(this, v.loc.clone(), ConsumedAs::Explicit(target));
            }
        }

        Value::TypeAnnotation { value, ty } => {
            collect_value(program, ctx, value);
            collect_value(program, ctx, ty);

            let ann = resolve_type_expr(program, ty).unwrap_or(InferId::Nothing);

            ctx.set_producer(
                this,
                v.loc.clone(),
                ProducedBy::TypeAnnotationExpr { ty: ann },
            );

            // RHS must satisfy the annotation
            if ann != InferId::Nothing {
                ctx.add_consume(value.ty, value.loc.clone(), ConsumedAs::Explicit(ann));
            }
        }

        Value::BinOp { op, values } => {
            let (l, r) = values.as_mut();
            collect_value(program, ctx, l);
            collect_value(program, ctx, r);

            // This expression is produced by a binary operator
            ctx.set_producer(this, v.loc.clone(), ProducedBy::BinOp { op: *op });

            match op {
                // Bitwise ops: operands must be integer-like
                BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::BitXor
                | BinOp::Shl
                | BinOp::Shr
                | BinOp::Mod => {
                    ctx.add_consume(l.ty, v.loc.clone(), ConsumedAs::IntNumeric);
                    ctx.add_consume(r.ty, v.loc.clone(), ConsumedAs::IntNumeric);
                }

                // Arithmetic ops: operands must be numeric
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                    ctx.add_consume(l.ty, v.loc.clone(), ConsumedAs::Numeric);
                    ctx.add_consume(r.ty, v.loc.clone(), ConsumedAs::Numeric);
                }

                // Comparisons: operands numeric, result handled by producer later
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    ctx.add_consume(l.ty, v.loc.clone(), ConsumedAs::Numeric);
                    ctx.add_consume(r.ty, v.loc.clone(), ConsumedAs::Numeric);
                }
            }
        }

        Value::UnOp { op, value } => {
            collect_value(program, ctx, value);

            ctx.set_producer(this, v.loc.clone(), ProducedBy::UnOp { op: *op });

            match op {
                UnOp::BitNot => {
                    ctx.add_consume(value.ty, v.loc.clone(), ConsumedAs::IntNumeric);
                }

                UnOp::Neg => {
                    ctx.add_consume(value.ty, v.loc.clone(), ConsumedAs::Numeric);
                }

                UnOp::Not => {
                    // operand is bool-like, result decided by producer
                    // (you *could* add ConsumedAs::Explicit(bool) on operand later)
                }

                UnOp::Deref | UnOp::AddrOf => {
                    ctx.add_consume(this, v.loc.clone(), ConsumedAs::Other("ptr op (todo)"));
                }
            }
        }

        Value::Let { pat, value, .. } => {
            collect_pattern(program, ctx, pat);
            collect_value(program, ctx, value);

            ctx.set_producer(this, v.loc.clone(), ProducedBy::Other("let expression"));

            record_let_bind_link(program, ctx, pat, value, &v.loc);
        }

        _ => {
            ctx.set_producer(this, v.loc.clone(), ProducedBy::Other("unmodeled"));
        }
    }
}

fn collect_pattern(program: &mut Program, ctx: &mut CollectCtx, p: &mut TPattern) {
    // Patterns may contain:
    // - bindings (handled later by record_let_bind_link)
    // - type annotations
    // - nested patterns

    match &mut p.value {
        Pattern::Bind(_) => {
            // Do nothing here.
            // The binding is handled by record_let_bind_link so that
            // the RHS is available.
        }

        Pattern::TypeAnnotation { pat, ty } => {
            // Recurse into the inner pattern
            collect_pattern(program, ctx, pat);

            // Collect the *type expression* so it gets an InferId
            collect_value(program, ctx, ty);
        }

        Pattern::Tuple(items) => {
            for it in items.iter_mut() {
                collect_pattern(program, ctx, it);
            }
        }

        Pattern::Literal(_) | Pattern::Wildcard => {
            // No bindings, no annotations
        }
    }
}

fn record_let_bind_link(
    program: &mut Program,
    ctx: &mut CollectCtx,
    pat: &mut TPattern,
    rhs: &mut TValue,
    loc: &Loc,
) {
    match &mut pat.value {
        Pattern::Bind(name) => {
            let nid = ctx.infer_for_name(*name);

            ctx.set_producer(
                nid,
                loc.clone(),
                ProducedBy::LetBind {
                    name: *name,
                    annotated: None,
                },
            );

            ctx.add_consume(nid, loc.clone(), ConsumedAs::SameAs(rhs.ty));
            ctx.add_consume(rhs.ty, rhs.loc.clone(), ConsumedAs::SameAs(nid));
        }

        Pattern::TypeAnnotation { pat: inner, ty } => {
            collect_value(program, ctx, ty);

            record_let_bind_link(program, ctx, inner, rhs, loc);

            if let Pattern::Bind(name) = inner.value {
                let nid = ctx.infer_for_name(name);

                let ann = Located {
                    loc: ty.loc.clone(),
                    value: ty.ty,
                };

                ctx.set_producer(
                    nid,
                    loc.clone(),
                    ProducedBy::LetBind {
                        name,
                        annotated: Some(ann),
                    },
                );

                ctx.add_consume(nid, pat.loc.clone(), ConsumedAs::Explicit(ty.ty));
                ctx.add_consume(rhs.ty, rhs.loc.clone(), ConsumedAs::Explicit(ty.ty));
            }
        }

        Pattern::Tuple(items) => {
            for it in items.iter_mut() {
                record_let_bind_link(program, ctx, it, rhs, loc);
            }
        }

        Pattern::Literal(_) | Pattern::Wildcard => {}
    }
}

/* ================================================================
 * Type expressions
 * ================================================================ */

fn resolve_type_expr(program: &Program, v: &TValue) -> Result<InferId, TypeError> {
    match &v.value {
        Value::NameRef(name) => match program.definitions.get(name) {
            Some(Defined::TypeRef(t)) => Ok(InferId::Concrete(*t)),
            _ => Err(TypeError::ExpectedType {
                loc: v.loc.clone(),
                message: "expected type",
            }),
        },
        _ => Err(TypeError::ExpectedType {
            loc: v.loc.clone(),
            message: "unsupported type expr",
        }),
    }
}

fn builtin(program: &mut Program, b: BuiltinType) -> InferId {
    let ty = program.type_store.intern(TypeValue::Builtin(b));
    InferId::Concrete(ty)
}

/* ================================================================
 * InferState (UNION-FIND + CONCRETE ASSIGNMENT)
 * ================================================================ */

#[derive(Debug)]
struct InferState {
    parent: Vec<usize>,            // union-find parent
    concrete: Vec<Option<TypeId>>, // concrete type per root
}

impl InferState {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            concrete: Vec::new(),
        }
    }

    fn fresh(&mut self) -> InferId {
        let id = self.parent.len();
        self.parent.push(id);
        self.concrete.push(None);
        InferId::Infered(id)
    }

    fn find(&mut self, id: InferId) -> InferId {
        match id {
            InferId::Infered(i) => {
                let p = self.parent[i];
                if p != i {
                    let root = match self.find(InferId::Infered(p)) {
                        InferId::Infered(r) => r,
                        _ => unreachable!(),
                    };
                    self.parent[i] = root;
                }
                InferId::Infered(self.parent[i])
            }
            other => other,
        }
    }

    fn unify(&mut self, a: InferId, b: InferId) -> Result<bool, (TypeId, TypeId)> {
        let a = self.find(a);
        let b = self.find(b);

        if a == b {
            return Ok(false);
        }

        match (a, b) {
            (InferId::Concrete(ta), InferId::Concrete(tb)) => {
                if ta == tb {
                    Ok(false)
                } else {
                    Err((ta, tb))
                }
            }

            (InferId::Concrete(t), InferId::Infered(i))
            | (InferId::Infered(i), InferId::Concrete(t)) => {
                let root = self.find(InferId::Infered(i));
                if let InferId::Infered(r) = root {
                    match self.concrete[r] {
                        Some(existing) => {
                            if existing == t {
                                Ok(false)
                            } else {
                                Err((existing, t))
                            }
                        }
                        None => {
                            self.concrete[r] = Some(t);
                            Ok(true)
                        }
                    }
                } else {
                    Ok(false)
                }
            }

            (InferId::Infered(a), InferId::Infered(b)) => {
                let ra = self.find(InferId::Infered(a));
                let rb = self.find(InferId::Infered(b));

                if ra == rb {
                    return Ok(false);
                }

                let (ra, rb) = match (ra, rb) {
                    (InferId::Infered(x), InferId::Infered(y)) => (x, y),
                    _ => unreachable!(),
                };

                let ca = self.concrete[ra];
                let cb = self.concrete[rb];

                self.parent[rb] = ra;

                match (ca, cb) {
                    (Some(ta), Some(tb)) if ta != tb => Err((ta, tb)),
                    (None, Some(t)) => {
                        self.concrete[ra] = Some(t);
                        Ok(true)
                    }
                    _ => Ok(true),
                }
            }

            _ => Ok(false),
        }
    }

    fn get_concrete(&mut self, id: InferId) -> Option<TypeId> {
        match self.find(id) {
            InferId::Concrete(t) => Some(t),
            InferId::Infered(i) => self.concrete[i],
            _ => None,
        }
    }
}

/* ================================================================
 * BASIC SOLVER (FIXED-POINT, LOOP-SAFE)
 * ================================================================ */

fn numeric_kind(ty: BuiltinType) -> Option<NumericKind> {
    match ty {
        BuiltinType::Int
        | BuiltinType::I8
        | BuiltinType::I16
        | BuiltinType::I32
        | BuiltinType::I64
        | BuiltinType::I128
        | BuiltinType::Isize
        | BuiltinType::U8
        | BuiltinType::U16
        | BuiltinType::U32
        | BuiltinType::U64
        | BuiltinType::U128
        | BuiltinType::Usize => Some(NumericKind::Int),

        BuiltinType::F32 | BuiltinType::F64 => Some(NumericKind::Float),

        _ => None,
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum NumericKind {
    Int,
    Float,
}

fn seed_from_producers(
    program: &mut Program,
    infer: &mut InferState,
    constraints: &HashMap<InferId, TypeConstraints>,
) -> Result<bool, TypeError> {
    let mut changed = false;

    for (&id, c) in constraints.iter() {
        match &c.produced {
            ProducedBy::Literal { lit } => {
                let ty = match lit {
                    Literal::Num(_) => builtin(program, BuiltinType::Int),
                    Literal::Float(_) => builtin(program, BuiltinType::F64),
                    Literal::Str(_) => builtin(program, BuiltinType::Str),
                    Literal::Void => builtin(program, BuiltinType::Void),
                };

                changed |= infer
                    .unify(id, ty)
                    .map_err(|(e, f)| TypeError::SimpleMismatch {
                        required_loc: c.produced_loc.clone(),
                        produced_loc: c.produced_loc.clone(),
                        expected: e,
                        found: f,
                        note: "literal contradicts existing constraints",
                    })?;
            }

            ProducedBy::Cast { target } | ProducedBy::TypeAnnotationExpr { ty: target } => {
                if *target != InferId::Nothing {
                    changed |=
                        infer
                            .unify(id, *target)
                            .map_err(|(e, f)| TypeError::SimpleMismatch {
                                required_loc: c.produced_loc.clone(),
                                produced_loc: c.produced_loc.clone(),
                                expected: e,
                                found: f,
                                note: "explicit type contradicts existing constraints",
                            })?;
                }
            }

            ProducedBy::BinOp { op } => match op {
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    let bool_ty = builtin(program, BuiltinType::Bool);

                    changed |=
                        infer
                            .unify(id, bool_ty)
                            .map_err(|(e, f)| TypeError::SimpleMismatch {
                                required_loc: c.produced_loc.clone(),
                                produced_loc: c.produced_loc.clone(),
                                expected: e,
                                found: f,
                                note: "comparison operator result is always bool",
                            })?;
                }

                _ => {}
            },

            _ => {}
        }
    }

    Ok(changed)
}

fn apply_equivalence_constraints(
    infer: &mut InferState,
    constraints: &HashMap<InferId, TypeConstraints>,
) -> Result<bool, TypeError> {
    let mut changed = false;

    for (&id, c) in constraints.iter() {
        for (use_loc, cons) in &c.consumed {
            match *cons {
                ConsumedAs::SameAs(other) | ConsumedAs::Explicit(other) => {
                    changed |=
                        infer
                            .unify(id, other)
                            .map_err(|(e, f)| TypeError::SimpleMismatch {
                                required_loc: use_loc.clone(),
                                produced_loc: c.produced_loc.clone(),
                                expected: e,
                                found: f,
                                note: "equivalence constraint failed",
                            })?;
                }
                _ => {}
            }
        }
    }

    Ok(changed)
}

fn apply_operator_semantics(
    program: &mut Program,
    infer: &mut InferState,
    constraints: &HashMap<InferId, TypeConstraints>,
) -> Result<bool, TypeError> {
    let mut changed = false;

    for (&id, c) in constraints.iter() {
        let ProducedBy::BinOp { op } = c.produced else {
            continue;
        };

        // Collect operand InferIds from constraints
        let mut operands = Vec::new();
        for (_, cons) in &c.consumed {
            match *cons {
                ConsumedAs::Numeric | ConsumedAs::IntNumeric => {
                    operands.push(id);
                }
                _ => {}
            }
        }

        if operands.len() != 2 {
            continue;
        }

        let lhs = operands[0];
        let rhs = operands[1];

        let Some(lt) = infer.get_concrete(lhs) else {
            continue;
        };
        let Some(rt) = infer.get_concrete(rhs) else {
            continue;
        };

        let TypeValue::Builtin(lb) = program.type_store.get(lt) else {
            continue;
        };
        let TypeValue::Builtin(rb) = program.type_store.get(rt) else {
            continue;
        };

        let lk = numeric_kind(*lb);
        let rk = numeric_kind(*rb);

        match op {
            // ---------- arithmetic ----------
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => match (lk, rk) {
                (Some(NumericKind::Float), Some(NumericKind::Float)) => {
                    changed |= infer
                        .unify(id, builtin(program, BuiltinType::F64))
                        .map_err(|(e, f)| TypeError::SimpleMismatch {
                            required_loc: c.produced_loc.clone(),
                            produced_loc: c.produced_loc.clone(),
                            expected: e,
                            found: f,
                            note: "float arithmetic",
                        })?;
                }

                (Some(NumericKind::Int), Some(NumericKind::Int)) => {
                    changed |= infer
                        .unify(id, builtin(program, BuiltinType::Int))
                        .map_err(|(e, f)| TypeError::SimpleMismatch {
                            required_loc: c.produced_loc.clone(),
                            produced_loc: c.produced_loc.clone(),
                            expected: e,
                            found: f,
                            note: "integer arithmetic",
                        })?;
                }

                _ => {
                    return Err(TypeError::SimpleMismatch {
                        required_loc: c.produced_loc.clone(),
                        produced_loc: c.produced_loc.clone(),
                        expected: lt,
                        found: rt,
                        note: "invalid arithmetic operands",
                    });
                }
            },

            // ---------- bitwise ----------
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr | BinOp::Mod => {
                match (lk, rk) {
                    (Some(NumericKind::Int), Some(NumericKind::Int)) => {
                        changed |= infer
                            .unify(id, builtin(program, BuiltinType::Int))
                            .map_err(|(e, f)| TypeError::SimpleMismatch {
                                required_loc: c.produced_loc.clone(),
                                produced_loc: c.produced_loc.clone(),
                                expected: e,
                                found: f,
                                note: "bitwise op result",
                            })?;
                    }

                    _ => {
                        return Err(TypeError::SimpleMismatch {
                            required_loc: c.produced_loc.clone(),
                            produced_loc: c.produced_loc.clone(),
                            expected: lt,
                            found: rt,
                            note: "bitwise ops require integer operands",
                        });
                    }
                }
            }

            _ => {}
        }
    }

    Ok(changed)
}

fn solve_basic(
    program: &mut Program,
    infer: &mut InferState,
    constraints: &HashMap<InferId, TypeConstraints>,
) -> Result<(), TypeError> {
    loop {
        let mut changed = false;

        changed |= seed_from_producers(program, infer, constraints)?;
        changed |= apply_equivalence_constraints(infer, constraints)?;
        changed |= apply_operator_semantics(program, infer, constraints)?;

        if !changed {
            break;
        }
    }

    Ok(())
}

/* ================================================================
 * Finalization (ROOT ONLY, NO GLOBAL CHECKS)
 * ================================================================ */

fn finalize_root(
    value: &TValue,
    infer: &mut InferState,
    constraints: &HashMap<InferId, TypeConstraints>,
) -> Result<TypeId, TypeError> {
    let root = value.ty;

    match infer.find(root) {
        InferId::Concrete(t) => Ok(t),

        InferId::Infered(i) => {
            if let Some(t) = infer.concrete[i] {
                Ok(t)
            } else {
                let produced_loc = constraints
                    .get(&root)
                    .map(|c| c.produced_loc.clone())
                    .unwrap_or_else(|| value.loc.clone());

                Err(TypeError::Unresolved {
                    produced_loc,
                    message: "could not infer a concrete type (try adding an explicit annotation)",
                })
            }
        }

        InferId::Nothing => Err(TypeError::Unresolved {
            produced_loc: value.loc.clone(),
            message: "expression never received an InferId slot (internal bug)",
        }),

        InferId::GenericArg(_) => Err(TypeError::Unsupported {
            loc: value.loc.clone(),
            message: "generic arguments not supported in basic solver",
        }),
    }
}

/* ================================================================
 * Entry point
 * ================================================================ */

pub fn infer_value(program: &mut Program, value: &mut TValue) -> Result<TypeId, TypeError> {
    let mut collected = collect_constraints(program, value);

    // Phase 1: constraint propagation (no guessing)
    solve_basic(program, &mut collected.infer, &collected.constraints)?;

    // Phase 2: resolve ONLY the root expression
    finalize_root(value, &mut collected.infer, &collected.constraints)
}

#[cfg(test)]
mod type_infer_tests {
    use super::*;
    use crate::parsing::Parser;

    /// Parse + lower + gather definitions,
    /// but DO NOT run type inference.
    fn gather_program(src: &str) -> Program {
        let mut program = Program::new();
        program.insert_builtin_types();

        let mut parser = Parser::new(src, 0);

        while !parser.is_empty() {
            match parser.parse_with_macros(&mut program) {
                Ok(Some(expr)) => {
                    program
                        .gather_definition(expr)
                        .expect("gather_definition failed");
                }
                Ok(None) => break,
                Err(e) => panic!("parse error: {:?}", e),
            }
        }

        program
            .check_pending_names()
            .expect("pending name resolution failed");

        program
    }

    /// Extract the body of the *single* function in the program.
    fn extract_single_fn_body(program: &Program) -> TValue {
        let def = program
            .definitions
            .iter()
            .find_map(|(_, def)| match def {
                Defined::Value(v) => Some(v),
                _ => None,
            })
            .expect("expected a function definition");

        match &def.value {
            Value::Func { body, .. } => (**body).clone(),
            _ => panic!("expected function value"),
        }
    }

    /// Run inference on a single function body.
    fn infer_fn(src: &str) -> Result<(Program, TypeId), TypeError> {
        let mut program = gather_program(src);
        let mut body = extract_single_fn_body(&program);
        let ty = infer_value(&mut program, &mut body)?;
        Ok((program, ty))
    }

    macro_rules! assert_fn_type {
        ($src:expr, $builtin:expr) => {{
            let (program, ty) = infer_fn($src).expect("inference failed");
            match program.type_store.get(ty) {
                TypeValue::Builtin(b) => assert_eq!(*b, $builtin),
                other => panic!("expected builtin type, got {:?}", other),
            }
        }};
    }

    /* ------------------------------------------------------------
     * Positive cases
     * ------------------------------------------------------------ */

    #[test]
    fn infer_literal() {
        assert_fn_type!("f = fn(){ 1 }", BuiltinType::Int);
    }

    #[test]
    fn infer_cast() {
        assert_fn_type!("f = fn(){ 1 as int }", BuiltinType::Int);
    }

    #[test]
    fn infer_let_binding() {
        assert_fn_type!("f = fn(){ let x = 1; x }", BuiltinType::Int);
    }

    #[test]
    fn infer_let_with_annotation() {
        assert_fn_type!("f = fn(){ let x:int = 1; x }", BuiltinType::Int);
    }

    #[test]
    fn infer_block_return() {
        assert_fn_type!("f = fn(){ { let x = 1; x } }", BuiltinType::Int);
    }

    #[test]
    fn cast_allows_type_change() {
        assert_fn_type!("f = fn(){ let x:int = 1; x as bool }", BuiltinType::Bool);
    }

    // #[test]
    // fn arithmetic_on_float_is_allowed() {
    //     assert_fn_type!("f = fn(){ 1.0 + 2.0 }", BuiltinType::F64);
    // }

    /* ------------------------------------------------------------
     * Error cases
     * ------------------------------------------------------------ */

    #[test]
    fn unresolved_variable_errors() {
        let err = infer_fn("f = fn(y){ let x = y; x }").unwrap_err();
        match err {
            TypeError::Unresolved { .. } => {}
            other => panic!("expected Unresolved, got {:?}", other),
        }
    }

    /*    #[test]
    fn bitwise_on_float_errors() {
        let err = infer_fn("f = fn(){ 1.0 & 2 }").unwrap_err();
        match err {
            TypeError::SimpleMismatch { .. } => {}
            other => panic!("expected SimpleMismatch, got {:?}", other),
        }
    }

    #[test]
    fn annotated_float_bitwise_errors() {
        let err = infer_fn("f = fn(){ let x: f64 = 1; x & 3 }").unwrap_err();
        match err {
            TypeError::SimpleMismatch { .. } => {}
            other => panic!("expected SimpleMismatch, got {:?}", other),
        }
    }*/
}
