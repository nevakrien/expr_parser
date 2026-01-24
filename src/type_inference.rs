//! Type inference sketch (constraint-first, local, non-integrated)
//
// ================================================================
// DESIGN GOALS (SHOULD STAY TRUE)
// ================================================================
// 1) Constraints are RECORDS, not guesses.
//    - They describe how values are PRODUCED and CONSUMED.
//    - They are mostly immutable and only appended to.
//
// 2) Inference is a SEPARATE, mutable process.
//    - It tries to assign concrete types to inference variables.
//    - Inference variables can be merged.
//    - Constraints are NEVER merged.
//
// 3) Every value has exactly ONE producer, and 0..N consumers.
//
// 4) Error reporting should come from constraints:
//    - Mismatch must be able to cite:
//      * where the expectation came from (consume site)
//      * where the conflicting value was produced (produce site)
//
// This file is intentionally incomplete.
// Anything uncertain is left as `todo!()` with a comment.
//
// ================================================================

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

/// Inference-time identifier for an expression's type.
/// Stored inside `Typed<T>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferId {
    /// Lowering placeholder (before inference assigns a slot)
    Nothing,

    /// Fully resolved global type
    Concrete(TypeId),

    /// Local inference variable (slot index in an inference run)
    Infered(usize),

    /// Placeholder for generics (future)
    GenericArg(usize),
}

/* ================================================================
 * Types & TypeStore (STABLE)
 * ================================================================ */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    // C-like int
    Int,

    // signed
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,

    // unsigned
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,

    // floats
    F32,
    F64,

    // others
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
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            intern: HashMap::new(),
        }
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

    pub fn get(&self, id: TypeId) -> &TypeValue {
        &self.values[id.0]
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

/* ================================================================
 * Program helpers (STABLE)
 * ================================================================ */

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
 * Error model (IMPROVED SHAPE)
 * ================================================================ */

#[derive(Debug)]
pub enum TypeError {
    /// "We couldn't determine a concrete type"
    Unresolved {
        produced_loc: Loc,
        message: &'static str,
    },

    /// Concrete mismatch with BOTH sides:
    /// - where the expectation came from (consume site)
    /// - where the mismatching value was produced (produce site)
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

/// We need to talk about both expression-values (InferId) and name-values (NameId).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueKey {
    Expr(InferId),
    Name(NameId),
}

/// Each value has exactly one producer.
/// (Producer does not "merge"; it is a record.)
#[derive(Debug)]
pub enum ProducedBy {
    /// Expression is a literal.
    Literal { lit: Literal },

    /// Expression is a name reference (its type is "same as" the referenced name-value).
    NameRef { name: NameId },

    /// A name-value is created by a let-binding.
    LetBind { name: NameId },

    /// Expression is a cast to a known target type.
    Cast { target: TypeId },

    /// Expression is a type annotation expression.
    /// (Meaning: this node's type is the annotated type.)
    TypeAnnotationExpr { ty: TypeId },

    /// Expression is some operator node. (Details recorded in consumes.)
    BinOp { op: BinOp },
    UnOp { op: UnOp },

    /// Expression is a block.
    Block,

    /// Expression is a function call.
    Call,

    /// Expression is a function literal.
    Func,

    /// Placeholder for more producers later.
    Other(&'static str),
}

/// A single consumption constraint placed at some location.
/// (Constraints are appended; they are not merged.)
#[derive(Debug, Clone, Copy)]
pub enum ConsumedAs {
    /// Must have the same type as another value.
    SameAs(ValueKey),

    /// Must be numeric (int-like OR float-like).
    Numeric,

    /// Must be int-like numeric.
    IntNumeric,

    /// Must be float-like numeric.
    FloatNumeric,

    /// Must equal a concrete type.
    Concrete(TypeId),

    /// Placeholder for future structured constraints.
    Other(&'static str),
}

/// All constraints for a single value.
#[derive(Debug)]
pub struct TypeConstraints {
    pub produced_loc: Loc,
    pub produced: ProducedBy,
    pub consumed: Vec<(Loc, ConsumedAs)>,
}

impl TypeConstraints {
    fn new(produced_loc: Loc, produced: ProducedBy) -> Self {
        Self {
            produced_loc,
            produced,
            consumed: Vec::new(),
        }
    }

    fn add_consume(&mut self, loc: Loc, c: ConsumedAs) {
        self.consumed.push((loc, c));
    }
}

/* ================================================================
 * Inference state (MUTABLE, MERGEABLE — STILL A STUB)
 * ================================================================ */

#[derive(Debug)]
struct InferState {
    slots: Vec<InferSlot>,
}

#[derive(Debug)]
struct InferSlot {
    ty: Option<TypeId>,
}

impl InferState {
    fn new() -> Self {
        Self { slots: Vec::new() }
    }

    fn fresh(&mut self) -> InferId {
        let id = self.slots.len();
        self.slots.push(InferSlot { ty: None });
        InferId::Infered(id)
    }

    fn get(&self, id: InferId) -> Option<TypeId> {
        match id {
            InferId::Concrete(t) => Some(t),
            InferId::Infered(i) => self.slots.get(i)?.ty,
            _ => None,
        }
    }

    fn assign(&mut self, id: InferId, ty: TypeId) {
        if let InferId::Infered(i) = id {
            self.slots[i].ty = Some(ty);
        }
    }

    // TODO: union-find / "merge variables"
    // TODO: numeric-kind tracking
}

/* ================================================================
 * Constraint collection (MORE MEAT, STILL SAFE)
 * ================================================================ */

struct CollectCtx {
    infer: InferState,
    constraints: HashMap<ValueKey, TypeConstraints>,
}

impl CollectCtx {
    fn new() -> Self {
        Self {
            infer: InferState::new(),
            constraints: HashMap::new(),
        }
    }

    fn ensure_expr_id(&mut self, v: &mut TValue) -> InferId {
        match v.ty {
            InferId::Nothing => {
                let id = self.infer.fresh();
                v.ty = id;
                id
            }
            other => other,
        }
    }

    fn ensure_expr_entry(&mut self, key: ValueKey, produced_loc: Loc, produced: ProducedBy) {
        self.constraints
            .entry(key)
            .or_insert_with(|| TypeConstraints::new(produced_loc, produced));
    }

    fn add_consume(&mut self, key: ValueKey, loc: Loc, c: ConsumedAs) {
        // It is legal to add consumes before we’ve visited the producer
        // (e.g., forward edges). If producer isn't known yet, insert a placeholder.
        let entry = self.constraints.entry(key).or_insert_with(|| {
            TypeConstraints::new(
                loc.clone(),
                ProducedBy::Other("producer not recorded yet"),
            )
        });
        entry.add_consume(loc, c);
    }

    fn set_producer(&mut self, key: ValueKey, produced_loc: Loc, produced: ProducedBy) {
        match self.constraints.get_mut(&key) {
            Some(existing) => {
                // Producer is single-assignment.
                // If we already had a placeholder producer, overwrite it.
                existing.produced_loc = produced_loc;
                existing.produced = produced;
            }
            None => {
                self.constraints
                    .insert(key, TypeConstraints::new(produced_loc, produced));
            }
        }
    }
}

/// Collect constraints for a value tree.
/// This function MUTATES `value.ty` to allocate InferIds locally.
/// It DOES NOT solve types.
fn collect_constraints(program: &mut Program, value: &mut TValue) -> CollectCtx {
    let mut ctx = CollectCtx::new();
    collect_value(program, &mut ctx, value);
    ctx
}

fn collect_value(program: &mut Program, ctx: &mut CollectCtx, v: &mut TValue) {
    let id = ctx.ensure_expr_id(v);
    let key = ValueKey::Expr(id);

    match &mut v.value {
        Value::Literal(lit) => {
            ctx.set_producer(key, v.loc.clone(), ProducedBy::Literal { lit: lit.clone() });
            // Note: literal polymorphism is handled in inference, not constraints,
            // so we do not force "int" here.
        }

        Value::NameRef(name) => {
            ctx.set_producer(key, v.loc.clone(), ProducedBy::NameRef { name: *name });

            // The type of this expression must equal the type of the name-value.
            ctx.add_consume(key, v.loc.clone(), ConsumedAs::SameAs(ValueKey::Name(*name)));

            // Record that the name-value is consumed by this use site (useful for errors later).
            ctx.add_consume(
                ValueKey::Name(*name),
                v.loc.clone(),
                ConsumedAs::SameAs(key),
            );
        }

        Value::Cast { value: inner, ty } => {
            // Collect children first (so they have IDs).
            collect_value(program, ctx, inner);
            collect_value(program, ctx, ty);

            let target = match resolve_type_expr(program, ty) {
                Ok(t) => t,
                Err(_) => {
                    // Don't invent types here; record producer but leave inference to report.
                    ctx.set_producer(key, v.loc.clone(), ProducedBy::Other("cast (unresolved type expr)"));
                    return;
                }
            };

            ctx.set_producer(key, v.loc.clone(), ProducedBy::Cast { target });

            // The cast expression itself is known to be `target`.
            // Still, we keep it as a consume constraint, not "solved".
            ctx.add_consume(key, v.loc.clone(), ConsumedAs::Concrete(target));

            // The inner value is consumed as "numeric" only if target is numeric, etc.
            // That policy is inference-time, so: TODO later.
        }

        Value::TypeAnnotation { value: inner, ty } => {
            collect_value(program, ctx, inner);
            collect_value(program, ctx, ty);

            let ann = match resolve_type_expr(program, ty) {
                Ok(t) => t,
                Err(_) => {
                    ctx.set_producer(
                        key,
                        v.loc.clone(),
                        ProducedBy::Other("type annotation (unresolved type expr)"),
                    );
                    return;
                }
            };

            ctx.set_producer(key, v.loc.clone(), ProducedBy::TypeAnnotationExpr { ty: ann });

            // This expression node itself is "ann".
            ctx.add_consume(key, v.loc.clone(), ConsumedAs::Concrete(ann));

            // The inner expression must match the annotation.
            ctx.add_consume(ValueKey::Expr(inner.ty), inner.loc.clone(), ConsumedAs::Concrete(ann));
        }

        Value::BinOp { op, values } => {
            let (lhs, rhs) = values.as_mut();
            collect_value(program, ctx, lhs);
            collect_value(program, ctx, rhs);

            ctx.set_producer(key, v.loc.clone(), ProducedBy::BinOp { op: *op });

            // Record operand constraints.
            // We do NOT decide the result type here; inference does that.
            match op {
                // bitwise/shifts/mod are int-only
                BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr | BinOp::Mod => {
                    ctx.add_consume(ValueKey::Expr(lhs.ty), v.loc.clone(), ConsumedAs::IntNumeric);
                    ctx.add_consume(ValueKey::Expr(rhs.ty), v.loc.clone(), ConsumedAs::IntNumeric);
                    ctx.add_consume(key, v.loc.clone(), ConsumedAs::IntNumeric);
                }

                // arithmetic: number
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                    ctx.add_consume(ValueKey::Expr(lhs.ty), v.loc.clone(), ConsumedAs::Numeric);
                    ctx.add_consume(ValueKey::Expr(rhs.ty), v.loc.clone(), ConsumedAs::Numeric);
                    ctx.add_consume(key, v.loc.clone(), ConsumedAs::Numeric);
                }

                // comparisons produce bool, and operands are numeric for now
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    ctx.add_consume(ValueKey::Expr(lhs.ty), v.loc.clone(), ConsumedAs::Numeric);
                    ctx.add_consume(ValueKey::Expr(rhs.ty), v.loc.clone(), ConsumedAs::Numeric);

                    let bool_ty = builtin(program, BuiltinType::Bool);
                    ctx.add_consume(key, v.loc.clone(), ConsumedAs::Concrete(bool_ty));
                }
            }
        }

        Value::UnOp { op, value: inner } => {
            collect_value(program, ctx, inner);
            ctx.set_producer(key, v.loc.clone(), ProducedBy::UnOp { op: *op });

            match op {
                UnOp::BitNot => {
                    ctx.add_consume(ValueKey::Expr(inner.ty), v.loc.clone(), ConsumedAs::IntNumeric);
                    ctx.add_consume(key, v.loc.clone(), ConsumedAs::IntNumeric);
                }
                UnOp::Neg => {
                    ctx.add_consume(ValueKey::Expr(inner.ty), v.loc.clone(), ConsumedAs::Numeric);
                    ctx.add_consume(key, v.loc.clone(), ConsumedAs::Numeric);
                }
                UnOp::Not => {
                    // TODO: constrain inner to bool later
                    let bool_ty = builtin(program, BuiltinType::Bool);
                    ctx.add_consume(key, v.loc.clone(), ConsumedAs::Concrete(bool_ty));
                }

                UnOp::Deref | UnOp::AddrOf => {
                    // Leave as record only; inference later.
                    ctx.add_consume(key, v.loc.clone(), ConsumedAs::Other("ptr op (todo)"));
                }
            }
        }

        Value::Let { pat, value: rhs, else_part } => {
            collect_pattern(program, ctx, pat);
            collect_value(program, ctx, rhs);
            if let Some(e) = else_part {
                collect_value(program, ctx, e);
            }

            // Producer for the let expression itself.
            // We do NOT commit to what "let expression returns" here.
            ctx.set_producer(key, v.loc.clone(), ProducedBy::Other("let (todo: return semantics)"));

            // If pattern is Bind(name) or TypeAnnotation{pat:Bind(name), ty:...},
            // we record a name-value producer and link it to RHS.
            record_let_bind_link(program, ctx, pat, rhs, &v.loc);
        }

        Value::Block { statements, return_value } => {
            for s in statements.iter_mut() {
                collect_value(program, ctx, s);
            }
            if let Some(ret) = return_value.as_mut() {
                collect_value(program, ctx, ret);

                // Record: block type == return expression type (if present).
                ctx.add_consume(key, v.loc.clone(), ConsumedAs::SameAs(ValueKey::Expr(ret.ty)));
                ctx.add_consume(ValueKey::Expr(ret.ty), ret.loc.clone(), ConsumedAs::SameAs(key));
            }

            ctx.set_producer(key, v.loc.clone(), ProducedBy::Block);
        }

        Value::Call { callee, args } => {
            collect_value(program, ctx, callee);
            for a in args.iter_mut() {
                collect_value(program, ctx, a);
            }
            ctx.set_producer(key, v.loc.clone(), ProducedBy::Call);

            // TODO: add "callee must be callable" and arg constraints later.
            ctx.add_consume(key, v.loc.clone(), ConsumedAs::Other("call typing (todo)"));
        }

        Value::Func { .. } => {
            ctx.set_producer(key, v.loc.clone(), ProducedBy::Func);
            // TODO: collect body, params, ret patterns, generics
            ctx.add_consume(key, v.loc.clone(), ConsumedAs::Other("func typing (todo)"));
        }

        // Everything else: record producer but don’t invent constraints yet.
        _ => {
            ctx.set_producer(key, v.loc.clone(), ProducedBy::Other("producer not modeled yet"));
        }
    }
}

fn collect_pattern(program: &mut Program, ctx: &mut CollectCtx, p: &mut TPattern) {
    // NOTE: patterns also have InferId, but we’re not using it yet.
    // We'll still recurse to find binds + type annotations.
    match &mut p.value {
        Pattern::Bind(name) => {
            // no-op here; let-binding logic records producer
            let _ = name;
        }

        Pattern::TypeAnnotation { pat, ty } => {
            collect_pattern(program, ctx, pat);
            collect_value(program, ctx, ty);
        }

        Pattern::Tuple(items) => {
            for it in items.iter_mut() {
                collect_pattern(program, ctx, it);
            }
        }

        Pattern::Literal(_) | Pattern::Wildcard => {}
    }
}

/// If `pat` binds a name, create a `ValueKey::Name(name)` producer entry and link it to RHS.
/// Also apply pattern type annotation constraint if present.
fn record_let_bind_link(
    program: &Program,
    ctx: &mut CollectCtx,
    pat: &mut TPattern,
    rhs: &mut TValue,
    let_loc: &Loc,
) {
    match &mut pat.value {
        Pattern::Bind(name) => {
            let nk = ValueKey::Name(*name);
            ctx.set_producer(nk, let_loc.clone(), ProducedBy::LetBind { name: *name });

            // The name-value must match RHS type.
            ctx.add_consume(nk, let_loc.clone(), ConsumedAs::SameAs(ValueKey::Expr(rhs.ty)));
            ctx.add_consume(ValueKey::Expr(rhs.ty), rhs.loc.clone(), ConsumedAs::SameAs(nk));
        }

        Pattern::TypeAnnotation { pat: inner, ty } => {
            // First ensure we link binding.
            record_let_bind_link(program, ctx, inner, rhs, let_loc);

            // Then apply annotation: the bound name must be that type.
            let ann_ty = match resolve_type_expr(program, ty) {
                Ok(t) => t,
                Err(_) => return, // don't guess
            };

            // Apply to the *binding* if it exists and is a bind.
            if let Pattern::Bind(name) = inner.value {
                let nk = ValueKey::Name(name);
                ctx.add_consume(nk, pat.loc.clone(), ConsumedAs::Concrete(ann_ty));

                // Also constrain RHS to match (so mismatch can cite both sites later).
                ctx.add_consume(ValueKey::Expr(rhs.ty), rhs.loc.clone(), ConsumedAs::Concrete(ann_ty));
            }
        }

        _ => {}
    }
}

/* ================================================================
 * Type expression resolution (SAFE SUBSET)
 * ================================================================ */

fn resolve_type_expr(program: &Program, v: &TValue) -> Result<TypeId, TypeError> {
    match &v.value {
        Value::NameRef(name) => match program.definitions.get(name) {
            Some((_, Defined::TypeRef(ty))) => Ok(*ty),
            _ => Err(TypeError::ExpectedType {
                loc: v.loc.clone(),
                message: "expected a type name (Defined::TypeRef)",
            }),
        },

        // TODO: generic type application, qualified access, tuples-as-types etc.
        _ => Err(TypeError::ExpectedType {
            loc: v.loc.clone(),
            message: "type expression form not supported in sketch yet",
        }),
    }
}

fn builtin(program: &mut Program, b: BuiltinType) -> TypeId {
    // We intentionally use the interned TypeStore.
    // This is stable and doesn't guess.
    program
        .type_store
        .intern(TypeValue::Builtin(b))
}

/* ================================================================
 * Inference driver (STILL A STUB — but with real inputs)
 * ================================================================ */

/// Runs:
/// 1) constraint collection (mutates `value.ty` to allocate local InferIds)
/// 2) inference solve (TODO)
pub fn infer_value(program: &mut Program, value: &mut TValue) -> Result<TypeId, TypeError> {
    let collected = collect_constraints(program, value);

    // At this point we have:
    // - collected.constraints: records of produced/consumed relationships
    // - collected.infer: a slot arena whose indices match InferId::Infered(_)
    //
    // Next step is solving those constraints:
    // - seed from ProducedBy (where possible)
    // - apply consumes, merging inference vars
    // - detect contradictions and report with required_loc + produced_loc
    //
    // That solver is intentionally not written yet.

    let _constraints = collected.constraints;
    let _infer = collected.infer;

    todo!("solver not implemented yet (constraints are collected and ready)");
}
