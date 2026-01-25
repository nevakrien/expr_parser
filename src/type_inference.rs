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

use crate::ir::NameId;
use crate::parsing::Located;
use std::collections::HashMap;

use crate::{
    ir::{BinOp, Literal, Pattern, IPattern, IValue, UnOp, Value,},
    parsing::Loc,
    program::{Defined, Program,ValId},
};


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

    InvalidOperator {
        loc: Loc,
        op: BinOp,
        lhs: TypeId,
        rhs: TypeId,
        note: &'static str,
    },

    InvalidLiteral {
        loc: Loc,
        loc_reqired:Loc,
        literal: Literal,
        target: TypeId,
        note: &'static str,
    },

}


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
 * Constraint model (CORE)
 * ================================================================ */

#[derive(Debug, Clone)]
pub enum ProducedBy {
    Literal {
        lit: Literal,
    },

    /// A name reference *after* name-resolution / lowering:
    /// refers directly to the value being referenced (its ValId).
    ///
    /// This removes NameId from inference: names are not type-nodes; values are.
    NameRef {
        target: NameId,
    },

    /// A `let` binding is also expressed in terms of the value-id that represents
    /// the bound slot (or the referenced value, depending on your lowering).
    ///
    /// The important part for inference is "this binding slot's type".
    LetBind {
        bind: NameId,
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

    /// Constraints indexed by InferId (unchanged for now)
    constraints: HashMap<InferId, TypeConstraints>,

    /// anoyingly some names dont have values,
    ///    for ones that do InferId points at those values
    ///    the ones that dont have a specific conrete type
    name_infers: HashMap<NameId, InferId>,
    value_infers: HashMap<ValId, InferId>,
}

impl CollectCtx {
    fn new() -> Self {
        Self {
            infer: InferState::new(),
            constraints: HashMap::new(),
            value_infers: HashMap::new(),
            name_infers:HashMap::new(),
        }
    }

    /// Get (or create) the InferId for this value.
    /// This replaces *all* uses of `v.ty`.
    fn ensure_expr_id(&mut self, v: &IValue) -> InferId {
        *self
            .value_infers
            .entry(v.id)
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

    /// Every name gets exactly one InferId slot.
    /// This is temporary until bindings are lowered to values.
    fn infer_for_name(&mut self, name: NameId) -> InferId {
        *self
            .name_infers
            .entry(name)
            .or_insert_with(|| self.infer.fresh())
    }
}


/* ================================================================
 * Constraint walk
 * ================================================================ */

fn collect_constraints(program: &mut Program, value: &IValue) -> CollectCtx {
    let mut ctx = CollectCtx::new();
    collect_value(program, &mut ctx, value);
    ctx
}

fn collect_value(program: &mut Program, ctx: &mut CollectCtx, v: &IValue) {
    let this = ctx.ensure_expr_id(v);

    //we assume no operator overloads so that we can do NumericInt on things like a&b regardless of their types.

    match &v.value {
        Value::Literal(lit) => {
            ctx.set_producer(
                this,
                program.get_loc(v.id),
                ProducedBy::Literal { lit: lit.clone() },
            );
        }

        Value::Block {
            statements,
            return_value,
        } => {
            for s in statements.iter() {
                collect_value(program, ctx, s);
            }

            if let Some(ret) = return_value.as_ref() {
                collect_value(program, ctx, ret);
                let ret_id = ctx.ensure_expr_id(ret);

                // block expr == return expr
                ctx.add_consume(this, program.get_loc(v.id), ConsumedAs::SameAs(ret_id));
                ctx.add_consume(ret_id, program.get_loc(ret.id), ConsumedAs::SameAs(this));
            }

            ctx.set_producer(this, program.get_loc(v.id), ProducedBy::Block);
        }

        Value::NameRef(target) => {
            let target_id = ctx.infer_for_name(*target);

            ctx.set_producer(
                this,
                program.get_loc(v.id),
                ProducedBy::NameRef { target: *target },
            );

            // expression <-> referenced value equivalence
            ctx.add_consume(this, program.get_loc(v.id), ConsumedAs::SameAs(target_id));
            ctx.add_consume(target_id, program.get_loc(v.id), ConsumedAs::SameAs(this));
        }

        Value::Cast { value, ty } => {
            collect_value(program, ctx, value);
            collect_value(program, ctx, ty);

            let target = resolve_type_expr(program, ty).unwrap_or(InferId::Nothing);

            ctx.set_producer(
                this,
                program.get_loc(v.id),
                ProducedBy::Cast { target },
            );

            // Cast is an explicit requirement
            if target != InferId::Nothing {
                ctx.add_consume(this, program.get_loc(v.id), ConsumedAs::Explicit(target));
            }
        }

        Value::TypeAnnotation { value, ty } => {
            collect_value(program, ctx, value);
            collect_value(program, ctx, ty);

            let value_id = ctx.ensure_expr_id(value);
            let ann = resolve_type_expr(program, ty).unwrap_or(InferId::Nothing);

            ctx.set_producer(
                this,
                program.get_loc(v.id),
                ProducedBy::TypeAnnotationExpr { ty: ann },
            );

            // RHS must satisfy the annotation
            if ann != InferId::Nothing {
                ctx.add_consume(value_id, program.get_loc(value.id), ConsumedAs::Explicit(ann));
            }
        }

        Value::BinOp { op, values } => {
            let (l, r) = values.as_ref();
            collect_value(program, ctx, l);
            collect_value(program, ctx, r);

            let l_id = ctx.ensure_expr_id(l);
            let r_id = ctx.ensure_expr_id(r);

            // This expression is produced by a binary operator
            ctx.set_producer(
                this,
                program.get_loc(v.id),
                ProducedBy::BinOp { op: *op },
            );

            match op {
                // Bitwise ops: operands must be integer-like
                BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::BitXor
                | BinOp::Shl
                | BinOp::Shr
                | BinOp::Mod => {
                    ctx.add_consume(l_id, program.get_loc(v.id), ConsumedAs::IntNumeric);
                    ctx.add_consume(r_id, program.get_loc(v.id), ConsumedAs::IntNumeric);
                }

                // Arithmetic ops: operands must be numeric
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                    ctx.add_consume(l_id, program.get_loc(v.id), ConsumedAs::Numeric);
                    ctx.add_consume(r_id, program.get_loc(v.id), ConsumedAs::Numeric);
                }

                // Comparisons: operands numeric, result handled by producer later
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    ctx.add_consume(l_id, program.get_loc(v.id), ConsumedAs::Numeric);
                    ctx.add_consume(r_id, program.get_loc(v.id), ConsumedAs::Numeric);
                }
            }
        }

        Value::UnOp { op, value } => {
            collect_value(program, ctx, value);
            let value_id = ctx.ensure_expr_id(value);

            ctx.set_producer(
                this,
                program.get_loc(v.id),
                ProducedBy::UnOp { op: *op },
            );

            match op {
                UnOp::BitNot => {
                    ctx.add_consume(value_id, program.get_loc(v.id), ConsumedAs::IntNumeric);
                }

                UnOp::Neg => {
                    ctx.add_consume(value_id, program.get_loc(v.id), ConsumedAs::Numeric);
                }

                UnOp::Not => {
                    // operand is bool-like, result decided by producer
                    // (you *could* add ConsumedAs::Explicit(bool) on operand later)
                }

                UnOp::Deref | UnOp::AddrOf => {
                    ctx.add_consume(
                        this,
                        program.get_loc(v.id),
                        ConsumedAs::Other("ptr op (todo)"),
                    );
                }
            }
        }

        Value::Let { pat, value, .. } => {
            collect_pattern(program, ctx, pat);
            collect_value(program, ctx, value);

            ctx.set_producer(
                this,
                program.get_loc(v.id),
                ProducedBy::Other("let expression"),
            );

            record_let_bind_link(program, ctx, pat, value, &program.get_loc(v.id));
        }

        _ => {
            ctx.set_producer(
                this,
                program.get_loc(v.id),
                ProducedBy::Other("unmodeled"),
            );
        }
    }
}

fn collect_pattern(program: &mut Program, ctx: &mut CollectCtx, p: &IPattern) {
    // Patterns may contain:
    // - bindings (handled later by record_let_bind_link)
    // - type annotations
    // - nested patterns

    match &p.value {
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
            for it in items.iter() {
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
    pat: &IPattern,
    rhs: &IValue,
    loc: &Loc,
) {
    match &pat.value {
        Pattern::Bind(name) => {
            let nid = ctx.infer_for_name(*name);
            let rhs_id = ctx.ensure_expr_id(rhs);

            ctx.set_producer(
                nid,
                loc.clone(),
                ProducedBy::LetBind {
                    bind: *name,
                    annotated: None,
                },
            );

            ctx.add_consume(nid, loc.clone(), ConsumedAs::SameAs(rhs_id));
            ctx.add_consume(rhs_id, program.get_loc(rhs.id), ConsumedAs::SameAs(nid));
        }

        Pattern::TypeAnnotation { pat: inner, ty } => {
            collect_value(program, ctx, ty);

            // Recurse so nested tuple patterns etc. still work
            record_let_bind_link(program, ctx, inner, rhs, loc);

            // Only if the inner pattern is actually a binder do we attach the annotation
            if let Pattern::Bind(name) = inner.value {
                let nid = ctx.infer_for_name(name);
                let rhs_id = ctx.ensure_expr_id(rhs);

                let ann = resolve_type_expr(program, ty).unwrap_or(InferId::Nothing);

                ctx.set_producer(
                    nid,
                    loc.clone(),
                    ProducedBy::LetBind {
                        bind:name,
                        annotated: Some(Located {
                            loc: program.get_loc(ty.id),
                            value: ann,
                        }),
                    },
                );

                if ann != InferId::Nothing {
                    ctx.add_consume(nid, loc.clone(), ConsumedAs::Explicit(ann));
                    ctx.add_consume(rhs_id, program.get_loc(rhs.id), ConsumedAs::Explicit(ann));
                }
            }
        }

        Pattern::Tuple(items) => {
            for it in items.iter() {
                record_let_bind_link(program, ctx, it, rhs, loc);
            }
        }

        Pattern::Literal(_) | Pattern::Wildcard => {}
    }
}


/* ================================================================
 * Type expressions
 * ================================================================ */

fn resolve_type_expr(program: &Program, v: &IValue) -> Result<InferId, TypeError> {
    match &v.value {
        Value::NameRef(name) => match program.definitions.get(name) {
            Some(Defined::TypeRef(t)) => Ok(InferId::Concrete(*t)),
            _ => Err(TypeError::ExpectedType {
                loc: program.get_loc(v.id),
                message: "expected type",
            }),
        },
        _ => Err(TypeError::ExpectedType {
            loc: program.get_loc(v.id),
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

    /// Require that `a` and `b` represent *the same concrete type*.
    ///
    /// IMPORTANT SEMANTIC RULES:
    /// -------------------------
    /// - This function enforces *equality*, nothing else.
    /// - It must NOT reason about operator overloading
    ///
    /// If this errors, it means:
    ///     "these two things cannot possibly be the same type"
    ///
    /// All other semantic failures MUST be reported at the call site
    /// where enough context exists to produce a meaningful error.
    fn require_same(&mut self, a: InferId, b: InferId) -> Result<bool, (TypeId, TypeId)> {
        let a = self.find(a);
        let b = self.find(b);

        if a == b {
            return Ok(false);
        }

        match (a, b) {
            // Concrete vs concrete: hard failure if unequal
            (InferId::Concrete(ta), InferId::Concrete(tb)) => {
                if ta == tb {
                    Ok(false)
                } else {
                    Err((ta, tb))
                }
            }

            // Concrete binds inferred
            (InferId::Concrete(t), InferId::Infered(i))
            | (InferId::Infered(i), InferId::Concrete(t)) => {
                let InferId::Infered(r) = self.find(InferId::Infered(i)) else {
                    return Ok(false);
                };

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
            }

            // Inferred vs inferred: union, propagate concrete if present
            (InferId::Infered(a), InferId::Infered(b)) => {
                let InferId::Infered(ra) = self.find(InferId::Infered(a)) else { unreachable!() };
                let InferId::Infered(rb) = self.find(InferId::Infered(b)) else { unreachable!() };

                if ra == rb {
                    return Ok(false);
                }

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
                    .require_same(id, ty)
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
                    changed |= infer
                        .require_same(id, *target)
                        .map_err(|(e, f)| TypeError::SimpleMismatch {
                            required_loc: c.produced_loc.clone(),
                            produced_loc: c.produced_loc.clone(),
                            expected: e,
                            found: f,
                            note: "explicit type contradicts existing constraints",
                        })?;
                }
            }

            ProducedBy::BinOp { op } => {
                // NOTE:
                // We do NOT validate operand compatibility here.
                // That is handled in `apply_operator_semantics`,
                // where we have access to *both* operands and the operator.
                if matches!(
                    op,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                ) {
                    let bool_ty = builtin(program, BuiltinType::Bool);
                    changed |= infer
                        .require_same(id, bool_ty)
                        .map_err(|(e, f)| TypeError::SimpleMismatch {
                            required_loc: c.produced_loc.clone(),
                            produced_loc: c.produced_loc.clone(),
                            expected: e,
                            found: f,
                            note: "comparison operator result is always bool",
                        })?;
                }
            }

            ProducedBy::NameRef { .. } => {
                // TODO(global name refs):
                // If a NameRef refers to a fully-typed, non-generic global,
                // we may seed here. Otherwise, defer to local constraints.
            }

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
                    changed |= infer
                        .require_same(id, other)
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

        let mut operands = Vec::new();
        for (_, cons) in &c.consumed {
            match *cons {
                ConsumedAs::Numeric | ConsumedAs::IntNumeric => operands.push(id),
                _ => {}
            }
        }

        if operands.len() != 2 {
            continue;
        }

        let lhs = operands[0];
        let rhs = operands[1];

        let (Some(lt), Some(rt)) = (infer.get_concrete(lhs), infer.get_concrete(rhs)) else {
            continue;
        };

        let (TypeValue::Builtin(lb), TypeValue::Builtin(rb)) =
            (program.type_store.get(lt), program.type_store.get(rt))
        else {
            continue;
        };

        let lk = numeric_kind(*lb);
        let rk = numeric_kind(*rb);

        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => match (lk, rk) {
                (Some(NumericKind::Float), Some(NumericKind::Float)) => {
                    changed |= infer
                        .require_same(id, builtin(program, BuiltinType::F64))
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
                        .require_same(id, builtin(program, BuiltinType::Int))
                        .map_err(|(e, f)| TypeError::SimpleMismatch {
                            required_loc: c.produced_loc.clone(),
                            produced_loc: c.produced_loc.clone(),
                            expected: e,
                            found: f,
                            note: "integer arithmetic",
                        })?;
                }
                _ => {
                    return Err(TypeError::InvalidOperator {
                        loc: c.produced_loc.clone(),
                        op,
                        lhs: lt,
                        rhs: rt,
                        note: "invalid arithmetic operands",
                    });
                }
            },

            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr | BinOp::Mod => {
                match (lk, rk) {
                    (Some(NumericKind::Int), Some(NumericKind::Int)) => {
                        changed |= infer
                            .require_same(id, builtin(program, BuiltinType::Int))
                            .map_err(|(e, f)| TypeError::SimpleMismatch {
                                required_loc: c.produced_loc.clone(),
                                produced_loc: c.produced_loc.clone(),
                                expected: e,
                                found: f,
                                note: "bitwise op result",
                            })?;
                    }
                    _ => {
                        return Err(TypeError::InvalidOperator {
                            loc: c.produced_loc.clone(),
                            op,
                            lhs: lt,
                            rhs: rt,
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
    program: &Program,
    value: &IValue,
    root: InferId,
    infer: &mut InferState,
    constraints: &HashMap<InferId, TypeConstraints>,
) -> Result<TypeId, TypeError> {
    match infer.find(root) {
        InferId::Concrete(t) => Ok(t),

        InferId::Infered(i) => {
            if let Some(t) = infer.concrete[i] {
                Ok(t)
            } else {
                let produced_loc = constraints
                    .get(&root)
                    .map(|c| c.produced_loc.clone())
                    .unwrap_or_else(|| program.get_loc(value.id));

                Err(TypeError::Unresolved {
                    produced_loc,
                    message: "could not infer a concrete type (try adding an explicit annotation)",
                })
            }
        }

        InferId::Nothing => Err(TypeError::Unresolved {
            produced_loc: program.get_loc(value.id),
            message: "expression never received an InferId slot (internal bug)",
        }),

        InferId::GenericArg(_) => Err(TypeError::Unsupported {
            loc: program.get_loc(value.id),
            message: "generic arguments not supported in basic solver",
        }),
    }
}


/* ================================================================
 * Entry point
 * ================================================================ */

pub fn infer_value(program: &mut Program, value: &IValue) -> Result<TypeId, TypeError> {
    let collected = collect_constraints(program, value);

    // Root InferId is now owned by CollectCtx, not stored in the AST.
    let root = *collected
        .value_infers
        .get(&value.id)
        .expect("root value did not get an InferId (internal bug)");

    // Phase 1: constraint propagation (no guessing)
    let mut infer = collected.infer;
    solve_basic(program, &mut infer, &collected.constraints)?;

    // Phase 2: resolve ONLY the root expression
    finalize_root(program, value, root, &mut infer, &collected.constraints)
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
    fn extract_single_fn_body(program: &Program) -> IValue {
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

    #[test]
    fn infer_let_with_num_literal() {
        assert_fn_type!("f = fn(){ let x:i32 = 1; x }", BuiltinType::I32);
    }

    #[test]
    fn arithmetic_on_float_is_allowed() {
        assert_fn_type!("f = fn(){ 1.0 + 2.0 }", BuiltinType::F64);
    }

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

    //  #[test]
    // fn bitwise_on_float_errors() {
    //     let err = infer_fn("f = fn(){ 1.0 & 2 }").unwrap_err();
    //     match err {
    //         TypeError::SimpleMismatch { .. } => {}
    //         other => panic!("expected SimpleMismatch, got {:?}", other),
    //     }
    // }

    // #[test]
    // fn annotated_float_bitwise_errors() {
    //     let err = infer_fn("f = fn(){ let x: f64 = 1.0; x & 3 }").unwrap_err();
    //     match err {
    //         TypeError::SimpleMismatch { .. } => {}
    //         other => panic!("expected SimpleMismatch, got {:?}", other),
    //     }
    // }
}
