//! Type inference sketch
//
// ================================================================
// DESIGN GOALS
// ================================================================
// 1) make simple infrence dead obvious and have good errors
// 2) get a working sketch
// 3) be still open to add overloads+lifetimes
//
// ================================================================

use crate::identity_hasher::IdHashMap;
use crate::ir::{NameId, PatId, ValId};
use std::collections::HashMap;
use std::ops::{Index, IndexMut};

use crate::{
    ir::{BinOp, Literal, Pattern, Value},
    program::{Defined, Program},
};

/* ================================================================
 * Core IDs (STABLE)
 * ================================================================ */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub usize);

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    Int = 0, //for now this enum MUST start at 0 and we also need values in order
    Uint,
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

impl From<BuiltinType> for TypeId {
    #[inline(always)]
    fn from(b: BuiltinType) -> Self {
        TypeId(b as usize)
    }
}
impl TryFrom<TypeId> for BuiltinType {
    type Error = ();

    #[inline(always)]
    fn try_from(id: TypeId) -> Result<Self, ()> {
        match id.0 as u8 {
            x if x == BuiltinType::Int as u8 => Ok(BuiltinType::Int),
            x if x == BuiltinType::Uint as u8 => Ok(BuiltinType::Uint),
            x if x == BuiltinType::I8 as u8 => Ok(BuiltinType::I8),
            x if x == BuiltinType::I16 as u8 => Ok(BuiltinType::I16),
            x if x == BuiltinType::I32 as u8 => Ok(BuiltinType::I32),
            x if x == BuiltinType::I64 as u8 => Ok(BuiltinType::I64),
            x if x == BuiltinType::I128 as u8 => Ok(BuiltinType::I128),
            x if x == BuiltinType::Isize as u8 => Ok(BuiltinType::Isize),
            x if x == BuiltinType::U8 as u8 => Ok(BuiltinType::U8),
            x if x == BuiltinType::U16 as u8 => Ok(BuiltinType::U16),
            x if x == BuiltinType::U32 as u8 => Ok(BuiltinType::U32),
            x if x == BuiltinType::U64 as u8 => Ok(BuiltinType::U64),
            x if x == BuiltinType::U128 as u8 => Ok(BuiltinType::U128),
            x if x == BuiltinType::Usize as u8 => Ok(BuiltinType::Usize),
            x if x == BuiltinType::F32 as u8 => Ok(BuiltinType::F32),
            x if x == BuiltinType::F64 as u8 => Ok(BuiltinType::F64),
            x if x == BuiltinType::Bool as u8 => Ok(BuiltinType::Bool),
            x if x == BuiltinType::Str as u8 => Ok(BuiltinType::Str),
            x if x == BuiltinType::Void as u8 => Ok(BuiltinType::Void),
            x if x == BuiltinType::Type as u8 => Ok(BuiltinType::Type),
            _ => Err(()),
        }
    }
}

/*const _: () = {
    if std::mem::size_of::<BuiltinType>() != 1 {
        panic!("BuiltinType must be 1 byte");
    }
};*/

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeValue {
    Builtin(BuiltinType),
    Tuple(Vec<TypeId>),
    Func { params: Vec<TypeId>, ret: TypeId },
    Ptr(TypeId),
    Type, //meta programing
}

impl Program {
    //TODO make it so we can store TypeId here directly
    //or perhaps move type expressions to use some sort of global type context
    #[inline(always)]
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
            let name = self.str_intern.intern(name);
            let id = self.insert_value_in_current_scope(name);
            self.definitions
                .insert(id, Defined::BuildinType(TypeValue::Builtin(builtin)));
        }
    }
}

#[derive(Debug)]
pub struct TypeStore {
    values: Vec<TypeValue>,
    intern: HashMap<TypeValue, TypeId>,
    global_types: IdHashMap<ValId, TypeId>,
}

impl Default for TypeStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeStore {
    pub fn new() -> Self {
        let mut ans = Self {
            values: Vec::new(),
            intern: HashMap::new(),
            global_types: IdHashMap::default(),
        };

        for i in 0.. {
            let Ok(builtin) = BuiltinType::try_from(TypeId(i)) else {
                break;
            };
            ans.intern(TypeValue::Builtin(builtin));
        }
        ans
    }

    #[inline(always)]
    pub fn get_global(&self, id: ValId) -> Option<TypeId> {
        self.global_types.get(&id).copied()
    }

    #[inline(always)]
    pub fn type_value(&self, id: TypeId) -> &TypeValue {
        &self.values[id.0]
    }

    #[inline]
    pub fn intern(&mut self, ty: TypeValue) -> TypeId {
        if let Some(&id) = self.intern.get(&ty) {
            return id;
        }
        let id = TypeId(self.values.len());
        self.values.push(ty.clone());
        self.intern.insert(ty, id);
        id
    }

    #[inline]
    pub fn as_builtin(&self, t: TypeId) -> Option<BuiltinType> {
        match self.type_value(t) {
            TypeValue::Builtin(b) => Some(*b),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn is_int_like(&self, t: TypeId) -> bool {
        use BuiltinType::*;
        matches!(
            self.as_builtin(t),
            Some(Int | I8 | I16 | I32 | I64 | I128 | Isize | U8 | U16 | U32 | U64 | U128 | Usize)
        )
    }

    #[inline(always)]
    pub fn is_float_like(&self, t: TypeId) -> bool {
        use BuiltinType::*;
        matches!(self.as_builtin(t), Some(F32 | F64))
    }
}

pub struct LocalTypes {
    val_types: IdHashMap<ValId, TypeId>,
    pat_types: IdHashMap<PatId, TypeId>,
}

impl Default for LocalTypes {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalTypes {
    pub fn new() -> Self {
        Self {
            pat_types: IdHashMap::default(),
            val_types: IdHashMap::default(),
        }
    }

    #[inline(always)]
    pub fn type_of(&self, id: ValId) -> Option<TypeId> {
        self.val_types.get(&id).copied()
    }
}

// ==============================
// Errors
// ==============================

#[derive(Debug)]
pub enum TypeError {
    /// Could not infer a concrete type for this value
    Unresolved { value: ValId, message: &'static str },
    UnresolvedPattern {
        pattern: PatId,
        message: &'static str,
    },

    /// Type expression (the RHS of `:` / `as`) wasn't a valid type
    ExpectedType {
        type_expr: ValId,
        message: &'static str,
    },

    /// `expr : T` or `pat : T` conflicts with what the value/pattern already implies.
    /// Carries BOTH the annotation node and the constrained node so diagnostics can point at both.
    AnnotationMismatch {
        /// The annotation node (Value::TypeAnnotation / Pattern::TypeAnnotation)
        annotation: ValId,
        /// The value/pattern being constrained (the `value` inside the annotation)
        constrained: ValId,
        expected: TypeId,
        found: TypeId,
        note: &'static str,
    },

    /// Pattern annotation mismatch
    PatternAnnotationMismatch {
        annotation: PatId,
        constrained: PatId,
        expected: TypeId,
        found: TypeId,
        note: &'static str,
    },

    /// Equality constraint failure at some site (let/match/etc).
    /// Carries a site ValId so you can point at the operator/let/match that demanded equality.
    IncompatibleTypes {
        site: ValId,
        left: TypeId,
        right: TypeId,
        note: &'static str,
    },

    /// Literal cluster resolved to an incompatible concrete type, or stayed unresolved.
    InvalidLiteral {
        literal: ValId,
        resolved: Option<TypeId>,
        message: &'static str,
    },

    /// (future) Operator rule failure.
    InvalidOperator {
        site: ValId,
        op: BinOp,
        lhs: TypeId,
        rhs: TypeId,
        note: &'static str,
    },
}

// ===================================
// Entry point
// ===================================

pub fn infer_value_internals(
    program: &Program,
    store: &mut TypeStore,
    value: ValId,
) -> Result<LocalTypes, TypeError> {
    let mut ctx = InferState::new(store, program);

    let _root = gather_constraints(&mut ctx, value)?;

    loop {
        let mut progress = false;
        progress |= resolve_func_types(&mut ctx)?;
        progress |= resolve_operator_types(&mut ctx)?;
        if !progress {
            break;
        }
    }

    // One linear normalization pass (no extra allocations).
    ctx.normalize_clusters();

    validate_literals(&ctx)?;
    finalize(&mut ctx)?;

    Ok(ctx.ans)
}

// ===================================
// Inference state + unify-find clusters
// ===================================
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CId(usize);

struct ClusterVec<T>(Vec<T>);
impl<T> ClusterVec<T> {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl<T> Index<CId> for ClusterVec<T> {
    type Output = T;
    fn index(&self, id: CId) -> &T {
        &self.0[id.0]
    }
}

impl<T> IndexMut<CId> for ClusterVec<T> {
    fn index_mut(&mut self, id: CId) -> &mut T {
        &mut self.0[id.0]
    }
}

struct InferState<'a> {
    store: &'a mut TypeStore,
    program: &'a Program,

    //ir -> cid
    val_cluster: IdHashMap<ValId, CId>,
    pat_cluster: IdHashMap<PatId, CId>,
    names: IdHashMap<NameId, CId>,

    // unify-find
    parent: ClusterVec<CId>,
    cluster: ClusterVec<Cluster>,

    // literal bookkeeping: keep ValId for error context
    int_lits: Vec<(ValId, CId)>,
    float_lits: Vec<(ValId, CId)>,

    //functions
    func_decs: Vec<(CId, CallSite)>,
    func_calls: IdHashMap<CId, CallSite>,

    //operators
    op_sites: Vec<OpSite>,

    ans: LocalTypes,
}

#[derive(Debug)]
struct Cluster {
    ty: Option<TypeId>,
    // call:Option<CallInfer>
}

#[derive(Debug)]
struct CallSite {
    loc: ValId,
    inputs: Vec<CId>,
    output: CId,
}

#[derive(Debug, Clone, Copy)]
struct OpSite {
    loc: ValId,
    op: BinOp,
    lhs: CId,
    rhs: CId,
    output: CId,
}

#[inline(always)]
fn find_root(parent: &mut ClusterVec<CId>, x: CId) -> CId {
    let p = parent[x];
    if p != x {
        let r = find_root(parent, p);
        parent[x] = r;
    }
    parent[x]
}

fn unify_clusters(
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    a: CId,
    b: CId,
) -> Result<CId, Clash> {
    let ra = find_root(parent, a);
    let rb = find_root(parent, b);
    if ra == rb {
        return Ok(ra);
    }

    let ta = cluster[ra].ty;
    let tb = cluster[rb].ty;
    if let (Some(a), Some(b)) = (ta, tb) {
        if a != b {
            return Err(Clash { a, b });
        }
    }

    // No rank: simplest correct UF (you can add rank later if you care)
    parent[rb] = ra;

    let other_ty = cluster[rb].ty; //.clone();
    let root_c = &mut cluster[ra];

    root_c.ty = root_c.ty.or(other_ty);
    // root_c.has_int_lit |= other_c.has_int_lit;
    // root_c.has_float_lit |= other_c.has_float_lit;

    Ok(ra)
}

impl<'a> InferState<'a> {
    fn new(store: &'a mut TypeStore, program: &'a Program) -> Self {
        Self {
            store,
            program,
            val_cluster: IdHashMap::default(),
            pat_cluster: IdHashMap::default(),
            names: IdHashMap::default(),
            parent: ClusterVec::new(),
            cluster: ClusterVec::new(),
            int_lits: Vec::new(),
            float_lits: Vec::new(),
            func_decs: Vec::new(),
            func_calls: IdHashMap::default(),
            op_sites: Vec::new(),
            ans: LocalTypes::new(),
        }
    }

    fn new_cluster(&mut self) -> CId {
        let id = CId(self.parent.len());
        self.parent.0.push(id);
        self.cluster.0.push(Cluster { ty: None });
        id
    }

    fn bind_val(&mut self, v: ValId, c: CId) {
        self.val_cluster.insert(v, c);
    }

    fn bind_pat(&mut self, p: PatId, c: CId) {
        self.pat_cluster.insert(p, c);
    }

    /// Default: values get their own cluster unless the semantics aliases them
    // fn cluster_of(&mut self, v: ValId) -> usize {
    //     if let Some(&c) = self.val_cluster.get(&v) {
    //         return c;
    //     }
    //     let c = self.new_cluster();
    //     self.bind_val(v, c);
    //     c
    // }

    #[inline(always)]
    fn find(&mut self, x: CId) -> CId {
        find_root(&mut self.parent, x)
    }

    /// Normalize everything once so later phases can use parent[c] without calling find().
    fn normalize_clusters(&mut self) {
        for i in 0..self.parent.len() {
            let i = CId(i);
            let r = self.find(i);
            self.parent[i] = r;
        }
    }

    //TODO: actually check call and have proper errors for when it fails
    fn unify(&mut self, a: CId, b: CId) -> Result<CId, Clash> {
        unify_clusters(&mut self.parent, &mut self.cluster, a, b)
    }

    // fn force_type(&mut self, c: CId, ty: TypeId) -> Result<(), Clash> {
    //     let r = self.find(c);
    //     match self.cluster[r].ty {
    //         None => {
    //             self.cluster[r].ty = Some(ty);
    //             Ok(())
    //         }
    //         Some(t) if t == ty => Ok(()),
    //         Some(t) => Err(Clash { a: t, b: ty }),
    //     }
    // }

    fn builtin(&mut self, b: BuiltinType) -> TypeId {
        // self.store.intern(TypeValue::Builtin(b))
        b.into()
    }
}

#[derive(Debug, Clone, Copy)]
struct Clash {
    a: TypeId,
    b: TypeId,
}

// ===================================
// Constraint gathering (alias where possible)
// ===================================

///this enum is just for initial gathering
///the main point is we can constant fold literals
///howver non literals are kinda tricky and should be avoided for agressive merges
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferStyle {
    ///can merge super agressively
    Literal,

    LocalVar,

    ///these are implictly generic in some ways which is trick
    LocalFunc,

    StructName,
}

impl InferStyle {
    fn delit(self) -> Self {
        match self {
            InferStyle::Literal => InferStyle::LocalVar,
            t => t,
        }
    }
}

fn gather_constraints(ctx: &mut InferState, v: ValId) -> Result<(CId, InferStyle), TypeError> {
    match ctx.program.value(v) {
        Value::Literal(Literal::Num(_)) => {
            let c = ctx.new_cluster();
            // ctx.cluster[c].has_int_lit = true;
            ctx.bind_val(v, c);
            ctx.int_lits.push((v, c));
            Ok((c, InferStyle::Literal))
        }

        Value::Literal(Literal::Float(_)) => {
            let c = ctx.new_cluster();
            // ctx.cluster[c].has_float_lit = true;
            ctx.bind_val(v, c);
            ctx.float_lits.push((v, c));
            Ok((c, InferStyle::Literal))
        }

        Value::Literal(Literal::Str(_)) => {
            let c = ctx.new_cluster();
            let t = ctx.builtin(BuiltinType::Str);
            ctx.cluster[c].ty = Some(t);
            ctx.bind_val(v, c);
            Ok((c, InferStyle::Literal))
        }

        Value::Literal(Literal::Void) => {
            let c = ctx.new_cluster();
            let t = ctx.builtin(BuiltinType::Void);
            ctx.cluster[c].ty = Some(t);
            ctx.bind_val(v, c);
            Ok((c, InferStyle::Literal))
        }

        Value::NameRef(n) => {
            if let Some(&c) = ctx.names.get(&n) {
                //names might refer to something that us generic in the local scope...
                //so this here is actually wrong for when users define local genric stuff
                ctx.bind_val(v, c);
                return Ok((c, InferStyle::LocalVar));
            }

            if ctx.program.definitions.contains_key(&n) {
                todo!("global name resolution / overload sets");
            }

            unreachable!("name used before binding");
        }

        Value::TypeAnnotation { value, ty } => {
            let (rhs_cluster, style) = gather_constraints(ctx, value)?;
            let ann_ty = compile_type_expr(ctx, ty)?;

            if let Err(Clash { a, b }) = ctx.unify(rhs_cluster, ann_ty) {
                return Err(TypeError::AnnotationMismatch {
                    annotation: v,
                    constrained: value,
                    expected: b,
                    found: a,
                    note: "type annotation does not match value",
                });
            }

            // Annotation does not introduce a new type identity: alias to the value
            ctx.bind_val(v, rhs_cluster);
            Ok((rhs_cluster, style))
        }

        Value::Cast { value, ty } => {
            let _ = gather_constraints(ctx, value)?;
            // Cast produces a new type identity: the target type
            let c = compile_type_expr(ctx, ty)?;
            ctx.bind_val(v, c);
            Ok((c, InferStyle::LocalVar))
        }

        Value::Let {
            pat,
            value,
            else_part,
        } => {
            let (rhs, _) = gather_constraints(ctx, value)?;
            let lhs = gather_pattern_constraints(ctx, pat)?;

            if let Err(Clash { a, b }) = ctx.unify(lhs, rhs) {
                return Err(TypeError::IncompatibleTypes {
                    site: v,
                    left: a,
                    right: b,
                    note: "let binding types do not match",
                });
            }

            if let Some(e) = else_part {
                let (ec, _) = gather_constraints(ctx, e)?;
                if let Err(Clash { a, b }) = ctx.unify(lhs, ec) {
                    return Err(TypeError::IncompatibleTypes {
                        site: e,
                        left: a,
                        right: b,
                        note: "let-else requires the else value to match the pattern type",
                    });
                }
            }

            // let-expr evaluates to the bound pattern value => alias
            ctx.bind_val(v, lhs);
            Ok((lhs, InferStyle::LocalVar))
        }

        Value::Block {
            statements,
            return_value,
        } => {
            for s in statements.ids() {
                gather_constraints(ctx, s)?;
            }

            // block aliases its return value cluster (or void)
            let (c, style) = match return_value {
                Some(r) => gather_constraints(ctx, r)?,
                None => {
                    let c = ctx.new_cluster();
                    let t = ctx.builtin(BuiltinType::Void);
                    ctx.cluster[c].ty = Some(t);
                    (c, InferStyle::LocalVar)
                }
            };

            ctx.bind_val(v, c);
            Ok((c, style))
        }

        Value::BinOp { op, values } => {
            let (lhs, rhs) = values;

            let (lc, ls) = gather_constraints(ctx, lhs)?;
            let (rc, rs) = gather_constraints(ctx, rhs)?;

            let (style, is_trivial) = match (ls, rs) {
                (InferStyle::Literal, InferStyle::Literal) => (InferStyle::Literal, true),
                _ => (InferStyle::LocalVar, false),
            };

            if !is_trivial {
                let output = match op {
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        let c = ctx.new_cluster();
                        let t = ctx.builtin(BuiltinType::Bool);
                        ctx.cluster[c].ty = Some(t);
                        c
                    }
                    BinOp::Add
                    | BinOp::Sub
                    | BinOp::Mul
                    | BinOp::Div
                    | BinOp::Mod
                    | BinOp::BitAnd
                    | BinOp::BitOr
                    | BinOp::BitXor
                    | BinOp::Shl
                    | BinOp::Shr => ctx.new_cluster(),
                };

                ctx.bind_val(v, output);
                ctx.op_sites.push(OpSite {
                    loc: v,
                    op,
                    lhs: lc,
                    rhs: rc,
                    output,
                });
                return Ok((output, style));
            }

            // Result cluster:
            // - comparisons always produce bool
            // - arithmetic / bitwise produce a value cluster
            match op {
                // ======================
                // Comparisons: bool
                // ======================
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    // operands must be comparable -> same cluster
                    if let Err(Clash { a, b }) = ctx.unify(lc, rc) {
                        return Err(TypeError::IncompatibleTypes {
                            site: v,
                            left: a,
                            right: b,
                            note: "comparison operands must have the same type",
                        });
                    }

                    let c = ctx.new_cluster();
                    let t = ctx.builtin(BuiltinType::Bool);
                    ctx.cluster[c].ty = Some(t);
                    ctx.bind_val(v, c);
                    Ok((c, InferStyle::Literal))
                }

                // ======================
                // Arithmetic / bitwise
                // ======================
                BinOp::Add
                | BinOp::Sub
                | BinOp::Mul
                | BinOp::Div
                | BinOp::Mod
                | BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::BitXor
                | BinOp::Shl
                | BinOp::Shr => {
                    // First, operands must have the same type
                    let root = match ctx.unify(lc, rc) {
                        Ok(r) => r,
                        Err(Clash { a, b }) => {
                            return Err(TypeError::IncompatibleTypes {
                                site: v,
                                left: a,
                                right: b,
                                note: "binary operator requires operands of the same type",
                            });
                        }
                    };

                    // Now: literal handling (currently a no op)
                    // when we add overloading we need to check here that we actualyl merge literals explictly
                    ctx.bind_val(v, root);
                    Ok((root, style))
                }
            }
        }
        Value::While { cond: _, body: _ } => {
            todo!()
        }
        Value::If {
            cond: _,
            then: _,
            els: _,
        } => {
            todo!()
        }
        Value::Func {
            generics: _,
            params,
            output_type,
            body,
        } => {
            let inputs = params
                .ids()
                .map(|pat| gather_pattern_constraints(ctx, pat))
                .collect::<Result<_, _>>()?;

            let output = if let Some(x) = output_type {
                compile_type_expr(ctx, x)?
            } else {
                let c = ctx.new_cluster();
                ctx.cluster[c].ty = Some(BuiltinType::Void.into());
                c
            };
            let f = ctx.new_cluster();
            ctx.bind_val(v, f);
            ctx.func_decs.push((
                f,
                CallSite {
                    inputs,
                    output,
                    loc: v,
                },
            ));

            let (body_cluster, _) = gather_constraints(ctx, body)?;

            if let Err(Clash { a, b }) = ctx.unify(body_cluster, output) {
                return Err(TypeError::AnnotationMismatch {
                    annotation: v,
                    constrained: body,
                    expected: b,
                    found: a,
                    note: "type annotation does not match function output",
                });
            }

            //TODO limit f on params and out somehow
            //this might need to be done ahead of time globaly for all funcs
            //so that we can have weird type recursions
            //if thats the case this part might be just compiling cluster,(params need to be gathered so we get them in as vars we can use)
            Ok((f, InferStyle::LocalFunc))
        }
        _ => panic!("more expressions {:?}", ctx.program.value(v)),
    }
}

fn gather_pattern_constraints(ctx: &mut InferState, p: PatId) -> Result<CId, TypeError> {
    match ctx.program.pattern(p) {
        Pattern::Bind(n) => {
            let c = ctx.new_cluster();
            ctx.names.insert(n, c);
            ctx.bind_pat(p, c);
            Ok(c)
        }

        Pattern::TypeAnnotation { pat, ty } => {
            let c = gather_pattern_constraints(ctx, pat)?;
            let t = compile_type_expr(ctx, ty)?;

            if let Err(Clash { a, b }) = ctx.unify(c, t) {
                return Err(TypeError::PatternAnnotationMismatch {
                    annotation: p,
                    constrained: pat,
                    expected: b,
                    found: a,
                    note: "pattern annotation does not match the value bound here",
                });
            }

            ctx.bind_pat(p, c);
            Ok(c)
        }

        _ => todo!(),
    }
}

fn compile_type_expr(ctx: &mut InferState, v: ValId) -> Result<CId, TypeError> {
    match ctx.program.value(v) {
        Value::NameRef(n) => {
            let t = match ctx.program.definitions.get(&n) {
                Some(Defined::BuildinType(b)) => ctx.store.intern(b.clone()),
                Some(Defined::Type { ty, .. }) => *ty,
                _ => {
                    return Err(TypeError::ExpectedType {
                        type_expr: v,
                        message: "expected type",
                    })
                }
            };

            let c = ctx.new_cluster();
            ctx.cluster[c].ty = Some(t);
            ctx.bind_val(v, c);
            Ok(c)
        }
        Value::Wildcard => {
            let c = ctx.new_cluster();
            ctx.bind_val(v, c);
            Ok(c)
        }
        _ => {
            return Err(TypeError::ExpectedType {
                type_expr: v,
                message: "unsupported type expression",
            })
        }
    }
}

// ===================================
// middle phase
// ===================================

fn resolve_func_types(ctx: &mut InferState) -> Result<bool, TypeError> {
    let mut progress = false;
    'outer: for (cid, dec) in ctx.func_decs.iter_mut() {
        let mut params = Vec::with_capacity(dec.inputs.len());
        for x in dec.inputs.iter_mut() {
            *x = find_root(&mut ctx.parent, *x);
            let Some(t) = ctx.cluster[*x].ty else {
                continue 'outer;
            };
            params.push(t)
        }
        dec.output = find_root(&mut ctx.parent, dec.output);
        let Some(ret) = ctx.cluster[dec.output].ty else {
            continue 'outer;
        };

        let tid = ctx.store.intern(TypeValue::Func { params, ret });
        *cid = find_root(&mut ctx.parent, *cid);
        let f = &mut ctx.cluster[*cid];
        if let Some(found) = f.ty {
            if found != tid {
                todo!()
            }
        } else {
            progress = true;
        }
        f.ty = Some(tid);
    }
    Ok(progress)
}

#[inline]
fn operator_allows_type(ctx: &TypeStore, op: BinOp, t: TypeId) -> bool {
    use BinOp::*;
    match op {
        Eq | Ne => true,

        Add | Sub | Mul | Div | Mod => ctx.is_int_like(t) || ctx.is_float_like(t),

        BitAnd | BitOr | BitXor | Shl | Shr => ctx.is_int_like(t),

        // not handled here yet
        Lt | Le | Gt | Ge => true,
    }
}

#[inline]
fn operator_requires_exact_match(op: BinOp) -> bool {
    matches!(op, BinOp::Eq | BinOp::Ne)
}

fn resolve_operator_types(ctx: &mut InferState) -> Result<bool, TypeError> {
    let mut progress = false;

    for site in ctx.op_sites.iter() {
        let lhs = find_root(&mut ctx.parent, site.lhs);
        let rhs = find_root(&mut ctx.parent, site.rhs);

        let lhs_ty = ctx.cluster[lhs].ty;
        let rhs_ty = ctx.cluster[rhs].ty;

        let op = site.op;

        match (lhs_ty, rhs_ty) {
            // ============================================================
            // Case A: both sides known
            // ============================================================
            (Some(lt), Some(rt)) => {
                // applicability check (int/float/bitwise)
                if !operator_allows_type(ctx.store, op, lt)
                    || !operator_allows_type(ctx.store, op, rt)
                {
                    return Err(TypeError::InvalidOperator {
                        site: site.loc,
                        op,
                        lhs: lt,
                        rhs: rt,
                        note: "operator not supported for this type",
                    });
                }

                // Eq / Ne require EXACT same type
                if operator_requires_exact_match(op) && lt != rt {
                    return Err(TypeError::InvalidOperator {
                        site: site.loc,
                        op,
                        lhs: lt,
                        rhs: rt,
                        note: "equality requires operands of the exact same type",
                    });
                }
            }

            // ============================================================
            // Case B: exactly one side known
            // ============================================================
            (Some(t), None) | (None, Some(t)) => {
                // early rejection (void excluded intentionally)
                if !operator_allows_type(ctx.store, op, t) {
                    return Err(TypeError::InvalidOperator {
                        site: site.loc,
                        op,
                        lhs: t,
                        rhs: t,
                        note: "operator cannot apply to this type",
                    });
                }

                // Eq / Ne force unification
                if operator_requires_exact_match(op) {
                    if let Err(Clash { a, b }) =
                        unify_clusters(&mut ctx.parent, &mut ctx.cluster, lhs, rhs)
                    {
                        return Err(TypeError::InvalidOperator {
                            site: site.loc,
                            op,
                            lhs: a,
                            rhs: b,
                            note: "equality requires operands of the same type",
                        });
                    }
                    progress = true;
                }
            }

            // ============================================================
            // Case C: nothing known yet
            // ============================================================
            (None, None) => {}
        }
    }

    Ok(progress)
}

// ===================================
// Late phases (normalized parent[] access)
// ===================================

fn validate_literals(ctx: &InferState) -> Result<(), TypeError> {
    for &(lit, c) in ctx.int_lits.iter() {
        let r = ctx.parent[c];
        match ctx.cluster[r].ty {
            Some(t) => {
                if !ctx.store.is_int_like(t) {
                    return Err(TypeError::InvalidLiteral {
                        literal: lit,
                        resolved: Some(t),
                        message: "integer literal used as non-integer type",
                    });
                }
            }
            None => {
                return Err(TypeError::InvalidLiteral {
                    literal: lit,
                    resolved: None,
                    message: "cannot infer type of integer literal",
                });
            }
        }
    }

    for &(lit, c) in ctx.float_lits.iter() {
        let r = ctx.parent[c];
        match ctx.cluster[r].ty {
            Some(t) => {
                if !ctx.store.is_float_like(t) {
                    return Err(TypeError::InvalidLiteral {
                        literal: lit,
                        resolved: Some(t),
                        message: "float literal used as non-float type",
                    });
                }
            }
            None => {
                return Err(TypeError::InvalidLiteral {
                    literal: lit,
                    resolved: None,
                    message: "cannot infer type of float literal",
                });
            }
        }
    }

    Ok(())
}

fn finalize(ctx: &mut InferState) -> Result<(), TypeError> {
    // ctx.parent[] already normalized
    for (&v, &c) in ctx.val_cluster.iter() {
        let r = ctx.parent[c];
        if let Some(t) = ctx.cluster[r].ty {
            ctx.ans.val_types.insert(v, t);
        } else {
            return Err(TypeError::Unresolved {
                value: v,
                message: "could not infer type",
            });
        }
    }
    for (&p, &c) in ctx.pat_cluster.iter() {
        let r = ctx.parent[c];
        if let Some(t) = ctx.cluster[r].ty {
            ctx.ans.pat_types.insert(p, t);
        } else {
            return Err(TypeError::UnresolvedPattern {
                pattern: p,
                message: "could not infer type",
            });
        }
    }
    Ok(())
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
    fn extract_single_fn(program: &Program) -> ValId {
        *program
            .definitions
            .iter()
            .find_map(|(_, def)| match def {
                Defined::Value(v) => Some(v),
                _ => None,
            })
            .expect("expected a function definition")
    }

    /// Run inference on a single function body.
    fn infer_fn(src: &str, store: &mut TypeStore) -> Result<TypeId, TypeError> {
        let program = gather_program(src);
        let f = extract_single_fn(&program);
        let body = match program.value(f) {
            Value::Func { body, .. } => body,
            _ => panic!("expected function value"),
        };

        let types = infer_value_internals(&program, store, f)?;
        Ok(types.type_of(body).unwrap())
    }

    //this is a hack for just testing
    fn infer_fn_body(src: &str, store: &mut TypeStore) -> Result<TypeId, TypeError> {
        let program = gather_program(src);
        let f = extract_single_fn(&program);
        let body = match program.value(f) {
            Value::Func { body, .. } => body,
            _ => panic!("expected function value"),
        };

        let types = infer_value_internals(&program, store, body)?;
        Ok(types.type_of(body).unwrap())
    }

    macro_rules! assert_fn_type {
        ($src:expr, $builtin:expr) => {{
            let mut store = TypeStore::new();
            let ty = infer_fn_body($src, &mut store).unwrap();
            match store.type_value(ty) {
                TypeValue::Builtin(b) => assert_eq!(*b, $builtin),
                other => panic!("expected builtin type, got {:?}", other),
            }
        }};
    }

    /* ------------------------------------------------------------
     * Positive cases
     * ------------------------------------------------------------ */

    #[test]
    fn infer_cast() {
        assert_fn_type!("f = fn(){ 1 : u32 as int }", BuiltinType::Int);
    }

    #[test]
    fn infer_let_with_annotation() {
        assert_fn_type!("f = fn(){ let x:int = 1; x }", BuiltinType::Int);
    }

    #[test]
    fn infer_block_return() {
        assert_fn_type!("f = fn(){ { let x : usize = 1; x } }", BuiltinType::Usize);
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
        assert_fn_type!("f = fn(){ (1.0 : f64) + 2.0 }", BuiltinType::F64);
    }

    #[test]
    fn resolves_eq() {
        assert_fn_type!(
            r#"
            f = fn() {
                let a = 1 + 2;
                let c = a == (2 + 1);
                let d: i64 = a;
            }
            "#,
            BuiltinType::Void
        )
    }

    #[test]
    fn large_lit_chains() {
        assert_fn_type!(
            r#"
            f = fn() {
                let a = 1 + 2;

                let b = 3.0 + 4.0;
                let z = b + 1.0:float;

                let c = a == (2 + 1);
                let d: i64 = a;
                let e = d + 5;
                let f = b as i64;
            }
            "#,
            BuiltinType::Void
        )
    }

    #[test]
    fn large_mixed_types_with_casts() {
        assert_fn_type!(
            r#"
            f = fn() {
                let a = 1 + 2;

                let b = 3.0 + 4.0;
                let z = b + 1.0:float;

                let c = a == (2 + 1);
                let d: i64 = a;
                let e = d + 5;
                let f = b as i64;

                let g = f == e;

                {
                    let h = g;
                    h
                }
            }
            "#,
            BuiltinType::Bool
        );
    }

    #[test]
    fn infer_empty_function() {
        let mut store = TypeStore::new();
        infer_fn("f=fn(){}", &mut store).unwrap();
    }

    /* ------------------------------------------------------------
     * Error cases
     * ------------------------------------------------------------ */

    // #[test]
    // fn unresolved_variable_errors() {
    //     let mut store = TypeStore::new();
    //     let err = infer_fn("f = fn(y){ let x = y; x }", &mut store).unwrap_err();
    //     match err {
    //         TypeError::Unresolved { .. } => {}
    //         other => panic!("expected Unresolved, got {:?}", other),
    //     }
    // }

    #[test]
    fn unresolved_int_errors() {
        let mut store = TypeStore::new();
        let err = infer_fn_body("f = fn(){ let x = 1; x }", &mut store).unwrap_err();
        match err {
            TypeError::Unresolved { .. } => {}
            TypeError::InvalidLiteral { .. } => {}
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
