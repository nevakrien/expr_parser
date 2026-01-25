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
    I8, I16, I32, I64, I128, Isize,
    U8, U16, U32, U64, U128, Usize,
    F32, F64,
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

impl TypeStore {
    pub fn new()->Self{
        Self{
            values:Vec::new(),
            intern:HashMap::new(),
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
            let id = self.insert_value_in_current_scope(name.to_string());
            self.definitions
                .insert(id, (name.to_string(), Defined::TypeRef(ty)));
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

#[derive(Debug,Clone)]
pub enum ProducedBy {
    Literal { lit: Literal },
    NameRef { name: NameId },
    LetBind {
        name: NameId,
        annotated: Option<Located<InferId>>,
    },
    Cast { target: InferId },
    TypeAnnotationExpr { ty: InferId },
    BinOp { op: BinOp },
    UnOp { op: UnOp },
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
        Self { produced_loc: loc, produced, consumed: Vec::new() }
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
        *self.name_infers.entry(name).or_insert_with(|| self.infer.fresh())
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
                TypeConstraints::new(
                    loc.clone(),
                    ProducedBy::Other("producer not recorded yet"),
                )
            })
            .consumed.push((loc, c));
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

        Value::NameRef(name) => {
            let nid = ctx.infer_for_name(*name);

            ctx.set_producer(
                this,
                v.loc.clone(),
                ProducedBy::NameRef { name: *name },
            );

            // expression <-> name-value equivalence
            ctx.add_consume(this, v.loc.clone(), ConsumedAs::SameAs(nid));
            ctx.add_consume(nid, v.loc.clone(), ConsumedAs::SameAs(this));
        }

        Value::Cast { value, ty } => {
            collect_value(program, ctx, value);
            collect_value(program, ctx, ty);

            let target = resolve_type_expr(program, ty).unwrap_or(InferId::Nothing);

            ctx.set_producer(
                this,
                v.loc.clone(),
                ProducedBy::Cast { target },
            );

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

            ctx.set_producer(
                this,
                v.loc.clone(),
                ProducedBy::BinOp { op: *op },
            );

            match op {
                BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::BitXor
                | BinOp::Shl
                | BinOp::Shr
                | BinOp::Mod => {
                    ctx.add_consume(l.ty, v.loc.clone(), ConsumedAs::IntNumeric);
                    ctx.add_consume(r.ty, v.loc.clone(), ConsumedAs::IntNumeric);
                }

                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                    ctx.add_consume(l.ty, v.loc.clone(), ConsumedAs::Numeric);
                    ctx.add_consume(r.ty, v.loc.clone(), ConsumedAs::Numeric);
                }

                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    ctx.add_consume(l.ty, v.loc.clone(), ConsumedAs::Numeric);
                    ctx.add_consume(r.ty, v.loc.clone(), ConsumedAs::Numeric);

                    let bool_ty = builtin(program, BuiltinType::Bool);
                    ctx.add_consume(this, v.loc.clone(), ConsumedAs::Explicit(bool_ty));
                }
            }
        }

        Value::UnOp { op, value } => {
            collect_value(program, ctx, value);

            ctx.set_producer(
                this,
                v.loc.clone(),
                ProducedBy::UnOp { op: *op },
            );

            match op {
                UnOp::BitNot => {
                    ctx.add_consume(value.ty, v.loc.clone(), ConsumedAs::IntNumeric);
                }
                UnOp::Neg => {
                    ctx.add_consume(value.ty, v.loc.clone(), ConsumedAs::Numeric);
                }
                UnOp::Not => {
                    let bool_ty = builtin(program, BuiltinType::Bool);
                    ctx.add_consume(this, v.loc.clone(), ConsumedAs::Explicit(bool_ty));
                }
                UnOp::Deref | UnOp::AddrOf => {
                    ctx.add_consume(this, v.loc.clone(), ConsumedAs::Other("ptr op (todo)"));
                }
            }
        }

        Value::Let { pat, value, .. } => {
            collect_pattern(program, ctx, pat);
            collect_value(program, ctx, value);

            ctx.set_producer(
                this,
                v.loc.clone(),
                ProducedBy::Other("let expression"),
            );

            record_let_bind_link(program, ctx, pat, value, &v.loc);
        }

        _ => {
            ctx.set_producer(
                this,
                v.loc.clone(),
                ProducedBy::Other("unmodeled"),
            );
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
            Some((_, Defined::TypeRef(t))) => Ok(InferId::Concrete(*t)),
            _ => Err(TypeError::ExpectedType { loc: v.loc.clone(), message: "expected type" }),
        },
        _ => Err(TypeError::ExpectedType { loc: v.loc.clone(), message: "unsupported type expr" }),
    }
}

fn builtin(program: &mut Program, b: BuiltinType) -> InferId {
    let ty = program.type_store.intern(TypeValue::Builtin(b));
    InferId::Concrete(ty)
}


/* ================================================================
 * Inference state (BASIC SOLVER: UNION-FIND + TYPE ASSIGNMENT)
 * ================================================================ */

#[derive(Debug)]
struct InferState {
    // union-find parent for Infered(i)
    parent: Vec<usize>,
    // optional concrete assignment for the root
    concrete: Vec<Option<TypeId>>,
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

    fn is_slot(&self, id: InferId) -> bool {
        matches!(id, InferId::Infered(_))
    }

    fn find_slot(&mut self, i: usize) -> usize {
        // path compression
        let p = self.parent[i];
        if p == i {
            i
        } else {
            let r = self.find_slot(p);
            self.parent[i] = r;
            r
        }
    }

    fn find(&mut self, id: InferId) -> InferId {
        match id {
            InferId::Infered(i) => InferId::Infered(self.find_slot(i)),
            other => other,
        }
    }

    fn get_concrete(&mut self, id: InferId) -> Option<TypeId> {
        match self.find(id) {
            InferId::Concrete(t) => Some(t),
            InferId::Infered(i) => self.concrete[i],
            _ => None,
        }
    }

    fn assign_concrete(&mut self, id: InferId, ty: TypeId) {
        match self.find(id) {
            InferId::Infered(i) => self.concrete[i] = Some(ty),
            InferId::Concrete(_) => {
                // nothing to do
            }
            _ => {}
        }
    }

    /// Merge two InferIds. If both sides already have concrete types, they must match.
    fn unify(
        &mut self,
        a: InferId,
        b: InferId,
    ) -> Result<(), (TypeId, TypeId)> {
        let a = self.find(a);
        let b = self.find(b);

        // If either side is Nothing/GenericArg we don't do anything in the basic solver.
        if matches!(a, InferId::Nothing | InferId::GenericArg(_))
            || matches!(b, InferId::Nothing | InferId::GenericArg(_))
        {
            return Ok(());
        }

        // concrete vs concrete
        if let (InferId::Concrete(ta), InferId::Concrete(tb)) = (a, b) {
            if ta == tb {
                return Ok(());
            } else {
                return Err((ta, tb));
            }
        }

        // concrete vs slot
        if let (InferId::Concrete(t), InferId::Infered(_)) = (a, b) {
            return self.unify_slot_with_concrete(b, t);
        }
        if let (InferId::Infered(_), InferId::Concrete(t)) = (a, b) {
            return self.unify_slot_with_concrete(a, t);
        }

        // slot vs slot
        if let (InferId::Infered(ai), InferId::Infered(bi)) = (a, b) {
            if ai == bi {
                return Ok(());
            }

            // union by "prefer the one that already has a concrete assigned" (very simple heuristic)
            let ac = self.concrete[ai];
            let bc = self.concrete[bi];

            match (ac, bc) {
                (Some(ta), Some(tb)) => {
                    if ta != tb {
                        return Err((ta, tb));
                    }
                    // same type; merge arbitrarily
                    self.parent[bi] = ai;
                    return Ok(());
                }
                (Some(_), None) => {
                    self.parent[bi] = ai;
                    return Ok(());
                }
                (None, Some(_)) => {
                    self.parent[ai] = bi;
                    return Ok(());
                }
                (None, None) => {
                    self.parent[bi] = ai;
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    fn unify_slot_with_concrete(
        &mut self,
        slot: InferId,
        t: TypeId,
    ) -> Result<(), (TypeId, TypeId)> {
        let slot = self.find(slot);
        match slot {
            InferId::Infered(i) => match self.concrete[i] {
                None => {
                    self.concrete[i] = Some(t);
                    Ok(())
                }
                Some(existing) => {
                    if existing == t {
                        Ok(())
                    } else {
                        Err((existing, t))
                    }
                }
            },
            InferId::Concrete(tc) => {
                if tc == t { Ok(()) } else { Err((tc, t)) }
            }
            _ => Ok(()),
        }
    }
}

/* ================================================================
 * Solver (BASIC: SAME-AS + EXPLICIT ONLY)
 * ================================================================ */

fn produced_loc_of(
    constraints: &HashMap<InferId, TypeConstraints>,
    id: InferId,
) -> Loc {
    constraints
        .get(&id)
        .expect("internal error: InferId has no producer recorded")
        .produced_loc
        .clone()
}

/// Runs a tiny fixed-point loop applying only the obvious constraints:
/// - SameAs(a,b)
/// - Explicit(a == b)   (where b is typically Concrete(type) or other resolved InferId)
///
/// Everything else is ignored for now.
fn solve_basic(
    infer: &mut InferState,
    constraints: &HashMap<InferId, TypeConstraints>,
) -> Result<(), TypeError> {
    let mut changed = true;

    while changed {
        changed = false;

        for (&id, c) in constraints.iter() {
            // NOTE: producer does not drive anything yet in this basic solver.
            // We only react to consumes.
            for (use_loc, cons) in c.consumed.iter() {
                match *cons {
                    ConsumedAs::SameAs(other) => {
                        let before_a = infer.find(id);
                        let before_b = infer.find(other);

                        match infer.unify(id, other) {
                            Ok(()) => {
                                let after_a = infer.find(id);
                                let after_b = infer.find(other);
                                if after_a != before_a || after_b != before_b {
                                    changed = true;
                                }
                            }
                            Err((expected, found)) => {
                                // Here: "required_loc" is the consume site,
                                // and "produced_loc" should point to the produced value that conflicts.
                                // We choose `id` as "the thing being constrained".
                                return Err(TypeError::SimpleMismatch {
                                    required_loc: use_loc.clone(),
                                    produced_loc: c.produced_loc.clone(),
                                    expected,
                                    found,
                                    note: "basic unify failed (SameAs)",
                                });
                            }
                        }
                    }

                    ConsumedAs::Explicit(expected_id) => {
                        let before = infer.find(id);

                        match infer.unify(id, expected_id) {
                            Ok(()) => {
                                let after = infer.find(id);
                                if after != before {
                                    changed = true;
                                }
                            }
                            Err((expected, found)) => {
                                // For Explicit, it's especially important that required_loc points
                                // at the explicit annotation/cast site.
                                return Err(TypeError::SimpleMismatch {
                                    required_loc: use_loc.clone(),
                                    produced_loc: c.produced_loc.clone(),
                                    expected,
                                    found,
                                    note: "basic unify failed (Explicit)",
                                });
                            }
                        }
                    }

                    // Ignored in the basic solver:
                    ConsumedAs::Numeric
                    | ConsumedAs::IntNumeric
                    | ConsumedAs::FloatNumeric
                    | ConsumedAs::Other(_) => {}
                }
            }
        }
    }

    Ok(())
}

/* ================================================================
 * Finalization (VERIFY EVERYTHING IS CONCRETE)
 * ================================================================ */

fn finalize_infer_id(
    infer: &mut InferState,
    constraints: &HashMap<InferId, TypeConstraints>,
    id: InferId,
) -> Result<TypeId, TypeError> {
    match infer.find(id) {
        InferId::Concrete(t) => Ok(t),
        InferId::Infered(i) => {
            if let Some(t) = infer.concrete[i] {
                Ok(t)
            } else {
                Err(TypeError::Unresolved {
                    produced_loc: produced_loc_of(constraints, id),
                    message: "could not infer a concrete type (try adding an explicit annotation)",
                })
            }
        }
        InferId::Nothing => Err(TypeError::Unresolved {
            produced_loc: produced_loc_of(constraints, id),
            message: "expression never received an InferId slot (bug: expected ensure_expr_id)",
        }),
        InferId::GenericArg(_) => Err(TypeError::Unsupported {
            loc: produced_loc_of(constraints, id),
            message: "generic args not supported in basic solver",
        }),
    }
}

/// TODO make this possible to use as a public api, 
/// it should use just external signatures
/// (Optional but useful) verify that *all* tracked InferIds ended up concrete.
/// You can call this later once you start caring about full-program completeness.
fn finalize_all(
    infer: &mut InferState,
    constraints: &HashMap<InferId, TypeConstraints>,
) -> Result<(), TypeError> {
    for (&id, _) in constraints.iter() {
        let _ = finalize_infer_id(infer, constraints, id)?;
    }
    Ok(())
}

/* ================================================================
 * Entry point (NOW WIRED TO BASIC SOLVER)
 * ================================================================ */

pub fn infer_value(program: &mut Program, value: &mut TValue) -> Result<TypeId, TypeError> {
    let mut collected = collect_constraints(program, value);

    // Basic solving pass: SameAs + Explicit only.
    solve_basic(&mut collected.infer, &collected.constraints)?;

    // If you want: enforce global completeness later.
    // finalize_all(&mut collected.infer, &collected.constraints)?;

    // Return the type of the root expression.
    finalize_infer_id(&mut collected.infer, &collected.constraints, value.ty)
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
            match parser.parse_with_macros(&program) {
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
            .find_map(|(_, (_name, def))| match def {
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

    /* ------------------------------------------------------------
     * Error cases
     * ------------------------------------------------------------ */

    #[test]
    fn unresolved_variable_errors() {
        let err = infer_fn("f = fn(){ let x = y; x }").unwrap_err();
        match err {
            TypeError::Unresolved { .. } => {}
            other => panic!("expected Unresolved, got {:?}", other),
        }
    }

    #[test]
    fn conflicting_annotation_errors() {
        let err = infer_fn("f = fn(){ let x:int = 1; x as bool }").unwrap_err();
        match err {
            TypeError::SimpleMismatch { .. } => {}
            other => panic!("expected SimpleMismatch, got {:?}", other),
        }
    }
}
