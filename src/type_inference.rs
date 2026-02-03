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

///this should only be used for specifiying ERRORS
///it is a way of representing unknown types that we can intern
///TypeId should not point to this as a general rule
pub const UNKNOWN_TYPE: TypeId = TypeId(usize::MAX);
pub const UNKNOWN_INT_SIZE: TypeId = TypeId(usize::MAX - 1);
pub const UNKNOWN_FLOAT_SIZE: TypeId = TypeId(usize::MAX - 2);

///this type specifically has internals containing UNKNOWN_TYPE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BadTypeId(pub TypeId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenId(pub usize);

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
    Func {
        params: Vec<TypeId>,
        ret: TypeId,
    },
    Ptr(TypeId),
    Type, //meta programing
    WithGenerics {
        count: usize,
        ///note that the body can refer to external generics
        body: TypeId,
    },
    Generic(GenId),
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

    #[inline(always)]
    pub fn get_bad_type_string(&self, t: BadTypeId) -> String {
        self.get_type_string(t.0)
    }
    pub fn get_type_string(&self, t: TypeId) -> String {
        self.get_type_string_nested(t, 0)
    }
    pub fn get_type_string_nested(&self, t: TypeId, gen_count: usize) -> String {
        if t == UNKNOWN_TYPE {
            return "_".to_string();
        }
        if t == UNKNOWN_INT_SIZE {
            return "int?".to_string();
        }
        if t == UNKNOWN_FLOAT_SIZE {
            return "float?".to_string();
        }

        match self.type_value(t) {
            TypeValue::Builtin(b) => match b {
                BuiltinType::Int => "int".to_string(),
                BuiltinType::Uint => "uint".to_string(),
                BuiltinType::I8 => "i8".to_string(),
                BuiltinType::I16 => "i16".to_string(),
                BuiltinType::I32 => "i32".to_string(),
                BuiltinType::I64 => "i64".to_string(),
                BuiltinType::I128 => "i128".to_string(),
                BuiltinType::Isize => "isize".to_string(),
                BuiltinType::U8 => "u8".to_string(),
                BuiltinType::U16 => "u16".to_string(),
                BuiltinType::U32 => "u32".to_string(),
                BuiltinType::U64 => "u64".to_string(),
                BuiltinType::U128 => "u128".to_string(),
                BuiltinType::Usize => "usize".to_string(),
                BuiltinType::F32 => "f32".to_string(),
                BuiltinType::F64 => "f64".to_string(),
                BuiltinType::Bool => "bool".to_string(),
                BuiltinType::Str => "str".to_string(),
                BuiltinType::Void => "void".to_string(),
                BuiltinType::Type => "Type".to_string(),
            },
            TypeValue::Tuple(items) => {
                let inner = items
                    .iter()
                    .map(|id| self.get_type_string_nested(*id, gen_count))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({})", inner)
            }
            TypeValue::Func { params, ret } => {
                let params = params
                    .iter()
                    .map(|id| self.get_type_string_nested(*id, gen_count))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "fn({}) -> {}",
                    params,
                    self.get_type_string_nested(*ret, gen_count)
                )
            }
            TypeValue::Ptr(inner) => format!("*{}", self.get_type_string_nested(*inner, gen_count)),
            TypeValue::Type => "Type".to_string(),
            TypeValue::WithGenerics { count, body } => {
                let new_count = gen_count + count;
                let pars = (gen_count..new_count)
                    .map(|i| format!("T{i}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "for<{pars}> {}",
                    self.get_type_string_nested(*body, new_count)
                )
            }
            TypeValue::Generic(g) => format!("T{}", g.0),
        }
    }
}

pub struct LocalTypes {
    pub val_types: IdHashMap<ValId, TypeId>,
    pub pat_types: IdHashMap<PatId, TypeId>,
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
    Unresolved {
        value: ValId,
    },
    UnresolvedPattern {
        pattern: PatId,
    },

    /// Type expression (the RHS of `:` / `as`) wasn't a valid type
    ExpectedTypeExpr {
        type_expr: ValId,
    },

    /// 2 values are required to be the same type because of some value producing expression
    /// but they are found to not be
    ValuesContradict {
        expectation_reason: &'static str,
        ///call causing the requirment
        site: ValId,
        ///most representative value from the found bin
        ///usually just the outer most value in an expression
        ///however in something like (let y:*int=null; y=let x = 2) we will show 2 as the reason
        ///since in that case 2 represents a requirment that the value is int
        found: ValId,
        ///closest value we can anotate as the reason for the expected cluster
        expected_place: ValId,
        clash: TypeClash,
    },

    /// `expr : T` or `pat : T` conflicts with what the value/pattern already implies.
    /// Carries BOTH the annotation node and the constrained node so diagnostics can point at both.
    AnnotationMismatch {
        /// The annotation node (Value::TypeAnnotation / Pattern::TypeAnnotation)
        annotation: ValId,
        /// The value/pattern being constrained (the `value` inside the annotation)
        constrained: ValId,
        clash: TypeClash,
    },

    /// Pattern annotation mismatch
    PatternAnnotationMismatch {
        annotation: PatId,
        constrained: PatId,
        clash: TypeClash,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct TypeClash {
    pub found: Option<BadTypeId>,
    pub wanted: Option<BadTypeId>,
}

// ===================================
// Entry point
// ===================================

pub fn infer_value_internals(
    program: &Program,
    store: &mut TypeStore,
    value: ValId,
) -> Result<LocalTypes, Vec<TypeError>> {
    let mut ctx = InferState::new(store, program);

    let _root = gather_constraints(&mut ctx, value);

    loop {
        let mut progress = false;
        progress |= resolve_operator_types(&mut ctx);
        progress |= resolve_func_types(&mut ctx);
        if !progress {
            break;
        }
    }

    if !ctx.errors.is_empty() {
        return Err(ctx.errors);
    }

    let types = finalize(&mut ctx);
    if ctx.errors.is_empty() {
        Ok(types)
    } else {
        Err(ctx.errors)
    }
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
    fn swap(&mut self,a:CId,b:CId){
        self.0.swap(a.0,b.0)
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
    val_cluster: Vec<(ValId, CId)>,
    pat_cluster: Vec<(PatId, CId)>,
    names: IdHashMap<NameId, CId>,

    // unify-find
    parent: ClusterVec<CId>,
    cluster: ClusterVec<Cluster>,

    //operators
    op_sites: Vec<OpSite>,

    call_sites: Vec<CallSite>,

    errors: Vec<TypeError>,
}

#[derive(Debug)]
struct Cluster {
    state: ResolveKind,

}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CallSiteId(usize);

#[derive(Debug, Clone, Copy)]
enum ResolveKind {
    Solved(TypeId),
    Nothing,
    ///internal refs are going to always share the cluster with their source
    ExternRef(NameId),

    ///the val is the last entity easily considered a lit like (2+1+3) in (let y = let x = 2+1+3)
    ///these lits can be used for error reporting
    IntLike,
    ///same as intlike but for float
    FloatLike,
    ///not all functions are like this but if something is declared as a function its this
    Func(CallSiteId),
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
    lhs_val: ValId,
    rhs_val: ValId,
    lhs: CId,
    rhs: CId,
    output: CId,
    had_error: bool,
}

impl<'a> InferState<'a> {
    fn new(store: &'a mut TypeStore, program: &'a Program) -> Self {
        Self {
            store,
            program,
            val_cluster: Vec::default(),
            pat_cluster: Vec::default(),
            names: IdHashMap::default(),
            parent: ClusterVec::new(),
            cluster: ClusterVec::new(),
            op_sites: Vec::new(),
            call_sites: Vec::new(),
            errors: Vec::new(),
        }
    }

    fn new_cluster(&mut self) -> CId {
        let id = CId(self.parent.len());
        self.parent.0.push(id);
        self.cluster.0.push(Cluster {
            state: ResolveKind::Nothing,
        });
        id
    }

    fn new_solved(&mut self, t: TypeId) -> CId {
        let id = CId(self.parent.len());
        self.parent.0.push(id);
        self.cluster.0.push(Cluster {
            state: ResolveKind::Solved(t),
        });
        id
    }

    fn new_int_like(&mut self, v: ValId) -> CId {
        let id = self.new_cluster();
        self.cluster[id].state = ResolveKind::IntLike;
        id
    }

    fn new_float_like(&mut self, v: ValId) -> CId {
        let id = self.new_cluster();
        self.cluster[id].state = ResolveKind::FloatLike;
        id
    }

    fn new_func(&mut self, call: CallSite) -> CId {
        let call_id = CallSiteId(self.call_sites.len());
        self.call_sites.push(call);
        let id = self.new_cluster();
        self.cluster[id].state = ResolveKind::Func(call_id);
        id
    }

    fn bind_val(&mut self, v: ValId, c: CId) {
        self.val_cluster.push((v, c));
    }

    fn bind_pat(&mut self, p: PatId, c: CId) {
        self.pat_cluster.push((p, c));
    }

    fn push_error(&mut self, err: TypeError) {
        self.errors.push(err);
    }

    fn unify(&mut self, a: CId, b: CId) -> Result<CId, TypeClash> {
        unify_clusters(
            self.store,
            &mut self.parent,
            &mut self.cluster,
            &mut self.call_sites,
            a,
            b,
        )
    }

    fn force_type(&mut self, a: CId, t: TypeId) -> Result<(), TypeClash> {
        force_type(
            self.store,
            &mut self.parent,
            &mut self.cluster,
            &mut self.call_sites,
            a,
            t,
        )
    }
}

// =====================================================
// general union find + error resolution
// =====================================================

#[inline(always)]
fn find_root(parent: &mut ClusterVec<CId>, x: CId) -> CId {
    let p = parent[x];
    if p != x {
        let r = find_root(parent, p);
        parent[x] = r;
    }
    parent[x]
}

///tries to combine 2 clusters
///on fail produces a type_clash and keeps the 2 seprate
///infrence can continye with the 2 types seperated for the purpose of gathering more errors (obviously not Unresolved style errors)
fn unify_clusters(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    call_sites: &mut Vec<CallSite>,
    found: CId,
    wanted: CId,
) -> Result<CId, TypeClash> {
    let rf = find_root(parent, found);
    let rw = find_root(parent, wanted);
    if rf == rw {
        return Ok(rw);
    }

    // Try found <- wanted
    if try_absorb(store, parent, cluster, call_sites, rw, rf)? {
        if rf != parent[rf]{
            todo!()
        }

        parent[rf]=rw;
        return Ok(rw)
    }

    // Otherwise try wanted <- found
    if try_absorb(store, parent, cluster, call_sites, rf, rw)? {
        if rw != parent[rw]{
            todo!()
        }

        parent[rw]=rf;
        return Ok(rf)
    }

    // Neither direction worked → real contradiction
    Err(TypeClash {
        found: extract_bad_type(store, parent, cluster, call_sites, found),
        wanted: extract_bad_type(store, parent, cluster, call_sites, wanted),
    })
}

fn try_absorb(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    call_sites: &mut Vec<CallSite>,
    dst: CId,
    src: CId,
) -> Result<bool, TypeClash> {
    use ResolveKind::*;

    let dst_state = cluster[dst].state;
    let src_state = cluster[src].state;

    match (dst_state, src_state) {
        // =====================================================
        // this is a hack for making literals not apear in errors as much
        // =====================================================
        (Nothing, IntLike)|(Nothing, FloatLike) => {
            cluster[dst].state=cluster[src].state;
            Ok(true)
        },

        // =====================================================
        // src has no information → always safe to absorb
        // =====================================================
        (_, Nothing) => Ok(true),

        // =====================================================
        // Solved types
        // =====================================================
        (Solved(t1), Solved(t2)) => {
            if t1 == t2 {
                Ok(true)
            } else {
                Err(TypeClash {
                    found: Some(BadTypeId(t2)),
                    wanted: Some(BadTypeId(t1)),
                })
            }
        }

        // =====================================================
        // Solved absorbs literals if compatible
        // =====================================================
        (Solved(t), IntLike) => {
            if !store.is_int_like(t) {
                return Err(type_vs_literal_clash(t));
            }
            Ok(true)
        }

        (Solved(t), FloatLike) => {
            if !store.is_float_like(t) {
                return Err(type_vs_literal_clash(t));
            }
            Ok(true)
        }

        // =====================================================
        // Same-kind weak info: merge
        // =====================================================
        (IntLike, IntLike) | (FloatLike, FloatLike) => {
            Ok(true)
        }

        // =====================================================
        // Function placeholders
        // =====================================================
        (Func(dst_call), Func(src_call)) => {
            let (dst_len, src_len) = {
                let dst_call = &call_sites[dst_call.0];
                let src_call = &call_sites[src_call.0];
                (dst_call.inputs.len(), src_call.inputs.len())
            };
            if dst_len != src_len {
                return Err(func_call_clash(
                    store, parent, cluster, call_sites, dst_call, src_call,
                ));
            }

            for i in 0..dst_len {
                let (a, b) = {
                    let dst_call = &call_sites[dst_call.0];
                    let src_call = &call_sites[src_call.0];
                    (dst_call.inputs[i], src_call.inputs[i])
                };
                if unify_clusters(store, parent, cluster, call_sites, a, b).is_err() {
                    return Err(func_call_clash(
                        store, parent, cluster, call_sites, dst_call, src_call,
                    ));
                }
            }
            let (dst_out, src_out) = {
                let dst_call = &call_sites[dst_call.0];
                let src_call = &call_sites[src_call.0];
                (dst_call.output, src_call.output)
            };
            if unify_clusters(store, parent, cluster, call_sites, dst_out, src_out).is_err() {
                return Err(func_call_clash(
                    store, parent, cluster, call_sites, dst_call, src_call,
                ));
            }

            if let Some(t) = try_resolve_func_type(store, parent, cluster, call_sites, dst_call) {
                cluster[dst].state = Solved(t);
            }
            Ok(true)
        }

        (Solved(t), Func(call)) => {
            unify_func_with_type(store, parent, cluster, call_sites, call, t)?;
            Ok(true)
        }

        // =====================================================
        // ExternRef can be cast into things
        // =====================================================
        (_, ExternRef(_)) => {
            Ok(true)
        }

        // =====================================================
        // Everything else: do not guess
        // =====================================================
        _ => Ok(false),
    }
}

fn force_type(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    call_sites: &mut Vec<CallSite>,
    target: CId,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let root = find_root(parent, target);
    let state = cluster[root].state;
    match state {
        ResolveKind::Nothing => {
            cluster[root].state = ResolveKind::Solved(ty);
            Ok(())
        }
        ResolveKind::Solved(t) if t == ty => Ok(()),
        ResolveKind::Solved(t) => Err(simple_type_clash(t, ty)),
        ResolveKind::IntLike => {
            if !store.is_int_like(ty) {
                return Err(type_vs_literal_clash(ty));
            }
            cluster[root].state = ResolveKind::Solved(ty);
            Ok(())
        }
        ResolveKind::FloatLike => {
            if !store.is_float_like(ty) {
                return Err(type_vs_literal_clash(ty));
            }
            cluster[root].state = ResolveKind::Solved(ty);
            Ok(())
        }
        ResolveKind::Func(call) => {
            unify_func_with_type(store, parent, cluster, call_sites, call, ty)?;
            cluster[root].state = ResolveKind::Solved(ty);
            Ok(())
        }
        ResolveKind::ExternRef(_) => {
            cluster[root].state = ResolveKind::Solved(ty);
            Ok(())
        }
    }
}

fn unify_func_with_type(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    call_sites: &mut Vec<CallSite>,
    call: CallSiteId,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let (params, ret) = match store.type_value(ty) {
        TypeValue::Func { params, ret } => (params.as_slice(), *ret),
        _ => {
            return Err(TypeClash {
                found: Some(BadTypeId(ty)),
                wanted: None,
            });
        }
    };

    let input_len = call_sites[call.0].inputs.len();
    if params.len() != input_len {
        return Err(TypeClash {
            found: Some(BadTypeId(ty)),
            wanted: None,
        });
    }

    for i in 0..input_len {
        let input = call_sites[call.0].inputs[i];

        //TODO (maybe): we constantly take the params again from the spot because borrow checker
        //              technically the Vec params points to never reallocs
        //              so theortically its possible to keep borowing this
        let param_ty = match store.type_value(ty) {
            TypeValue::Func { params, ret: _ } => params[i],
            _ => unreachable!(),
        };
        force_type(store, parent, cluster, call_sites, input, param_ty)?;
    }

    let output = call_sites[call.0].output;
    force_type(store, parent, cluster, call_sites, output, ret)?;

    Ok(())
}

fn simple_type_clash(a: TypeId, b: TypeId) -> TypeClash {
    TypeClash {
        found: Some(BadTypeId(a)),
        wanted: Some(BadTypeId(b)),
    }
}

fn type_vs_literal_clash(t: TypeId) -> TypeClash {
    TypeClash {
        found: Some(BadTypeId(t)),
        wanted: None,
    }
}

//TODO: this should actually check if some of the types are known
// we wana do recursive partial resolution
fn mock_type_from_cluster(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    call_sites: &Vec<CallSite>,
    cid: CId,
    visiting: &mut std::collections::HashSet<CId>,
) -> TypeId {
    let root = find_root(parent, cid);
    if !visiting.insert(root) {
        return UNKNOWN_TYPE;
    }

    let ty = match cluster[root].state {
        ResolveKind::Solved(t) => t,
        ResolveKind::IntLike => UNKNOWN_INT_SIZE,
        ResolveKind::FloatLike => UNKNOWN_FLOAT_SIZE,
        ResolveKind::Func(call) => {
            make_func_mock_inner(store, parent, cluster, call_sites, call, visiting)
        }
        ResolveKind::Nothing | ResolveKind::ExternRef(_) => UNKNOWN_TYPE,
    };

    visiting.remove(&root);
    ty
}

fn make_func_mock_inner(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    call_sites: &Vec<CallSite>,
    call: CallSiteId,
    visiting: &mut std::collections::HashSet<CId>,
) -> TypeId {
    let site = &call_sites[call.0];
    let params = site
        .inputs
        .iter()
        .map(|&input| mock_type_from_cluster(store, parent, cluster, call_sites, input, visiting))
        .collect::<Vec<_>>();
    let ret = mock_type_from_cluster(store, parent, cluster, call_sites, site.output, visiting);

    store.intern(TypeValue::Func { params, ret })
}

fn make_func_mock(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    call_sites: &Vec<CallSite>,
    call: CallSiteId,
) -> TypeId {
    let mut visiting = std::collections::HashSet::new();
    make_func_mock_inner(store, parent, cluster, call_sites, call, &mut visiting)
}

fn func_call_clash(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    call_sites: &Vec<CallSite>,
    dst_call: CallSiteId,
    src_call: CallSiteId,
) -> TypeClash {
    TypeClash {
        found: Some(BadTypeId(make_func_mock(
            store, parent, cluster, call_sites, src_call,
        ))),
        wanted: Some(BadTypeId(make_func_mock(
            store, parent, cluster, call_sites, dst_call,
        ))),
    }
}

fn extract_bad_type(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    call_sites: &Vec<CallSite>,
    cid: CId,
) -> Option<BadTypeId> {
    let root = find_root(parent, cid);
    match cluster[root].state {
        ResolveKind::Solved(t) => Some(BadTypeId(t)),
        ResolveKind::Nothing => None,
        ResolveKind::ExternRef(_) => None,
        ResolveKind::Func(call) => Some(BadTypeId(make_func_mock(
            store, parent, cluster, call_sites, call,
        ))),

        ResolveKind::IntLike => Some(BadTypeId(UNKNOWN_INT_SIZE)),
        ResolveKind::FloatLike=> Some(BadTypeId(UNKNOWN_FLOAT_SIZE)),
    }
}

// ===================================
// Constraint gathering (alias where possible)
// ===================================



fn gather_constraints(ctx: &mut InferState, v: ValId) -> CId {
    match ctx.program.value(v) {
        Value::Literal(Literal::Num(_)) => {
            let c = ctx.new_int_like(v);
            ctx.bind_val(v, c);
            c
        }

        Value::Literal(Literal::Float(_)) => {
            let c = ctx.new_float_like(v);
            ctx.bind_val(v, c);
            c
        }

        Value::Literal(Literal::Str(_)) => {
            let c = ctx.new_solved(BuiltinType::Str.into());

            ctx.bind_val(v, c);
            c
        }

        Value::Literal(Literal::Void) => {
            let c = ctx.new_solved(BuiltinType::Void.into());
            ctx.bind_val(v, c);
            c
        }

        Value::NameRef(n) => {
            if let Some(&c) = ctx.names.get(&n) {
                //names might refer to something that us generic in the local scope...
                //so this here is actually wrong for when users define local genric stuff
                ctx.bind_val(v, c);
                return c;
            }

            if ctx.program.definitions.contains_key(&n) {
                todo!("global name resolution / overload sets");
            }

            unreachable!("name used before binding");
        }

        Value::TypeAnnotation { value, ty } => {
            let rhs_cluster = gather_constraints(ctx, value);
            let ann_ty = compile_type_expr(ctx, ty);

            if let Err(clash) = ctx.unify(rhs_cluster, ann_ty) {
                ctx.push_error(TypeError::AnnotationMismatch {
                    annotation: v,
                    constrained: value,
                    clash,
                });
            }

            // Annotation does not introduce a new type identity: alias to the value
            ctx.bind_val(v, rhs_cluster);
            rhs_cluster
        }

        Value::Cast { value, ty } => {
            let _ = gather_constraints(ctx, value);
            // Cast produces a new type identity: the target type
            let c = compile_type_expr(ctx, ty);
            ctx.bind_val(v, c);
            c
        }

        Value::Let {
            pat,
            value,
            else_part,
        } => {
            let lhs = gather_pattern_constraints(ctx, pat);
            let rhs = gather_constraints(ctx, value);


            if let Err(clash) = ctx.unify(rhs,lhs) {
                ctx.push_error(TypeError::ValuesContradict {
                    expectation_reason: "let binding requires pattern and value to match",
                    site: v,
                    found: value,
                    expected_place: v,
                    clash,
                });
            }

            if let Some(e) = else_part {
                let ec = gather_constraints(ctx, e);
                if let Err(clash) = ctx.unify(ec, lhs) {
                    ctx.push_error(TypeError::ValuesContradict {
                        expectation_reason:
                            "let-else requires the else value to match the pattern type",
                        site: e,
                        found: e,
                        expected_place: v,
                        clash,
                    });
                }
            }

            // let-expr evaluates to the bound pattern value => alias
            ctx.bind_val(v, lhs);
            lhs
        }

        Value::Block {
            statements,
            return_value,
        } => {
            for s in statements.ids() {
                gather_constraints(ctx, s);
            }

            // block aliases its return value cluster (or void)
            let c = match return_value {
                Some(r) => gather_constraints(ctx, r),
                None => {
                    ctx.new_solved(BuiltinType::Void.into())
                }
            };

            ctx.bind_val(v, c);
            c
        }

        Value::BinOp { op, values } => {
            let (lhs, rhs) = values;

            let lc = gather_constraints(ctx, lhs);
            let rc = gather_constraints(ctx, rhs);


            let output = match op {
                //there is no legitmate reason to overload != == to have a diffrent signature
                //because of this we just hard assume this
                //we might take out Lt Gt later if thats a thing we need to handle it at resolve_operators
                BinOp::Eq | BinOp::Ne | BinOp::Le | BinOp::Ge | BinOp::Gt | BinOp::Lt => {
                    if let Err(clash) = ctx.unify(lc, rc) {
                        ctx.push_error(TypeError::ValuesContradict {
                            expectation_reason: "comparison operands must have the same type",
                            site: v,
                            found: lhs,
                            expected_place: rhs,
                            clash,
                        });
                    }
                    ctx.new_solved(BuiltinType::Bool.into())
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
                lhs_val: lhs,
                rhs_val: rhs,
                lhs: lc,
                rhs: rc,
                output,
                had_error: false,
            });
            return output;

           
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
            generics,
            params,
            output_type,
            body,
        } => {
            for (i,pat) in generics.ids().enumerate() {
                //TODO:actually use this for typing
                gather_generic_constraints(ctx,pat,GenId(i));
                
            }
            let inputs = params
                .ids()
                .map(|pat| gather_pattern_constraints(ctx, pat))
                .collect::<Vec<_>>();

            let output = if let Some(x) = output_type {
                compile_type_expr(ctx, x)
            } else {
                ctx.new_solved(BuiltinType::Void.into())
            };
            let f = ctx.new_func(CallSite {
                inputs,
                output,
                loc: v,
            });
            ctx.bind_val(v, f);

            let body_cluster = gather_constraints(ctx, body);

            if let Err(clash) = ctx.unify(body_cluster, output) {
                ctx.push_error(TypeError::ValuesContradict {
                    expectation_reason: "function body must match return type",
                    site: v,
                    found: body,
                    expected_place: v,
                    clash,
                });
            }

            //TODO limit f on params and out somehow
            //this might need to be done ahead of time globaly for all funcs
            //so that we can have weird type recursions
            //if thats the case this part might be just compiling cluster,(params need to be gathered so we get them in as vars we can use)
            f
        }
        _ => panic!("more expressions {:?}", ctx.program.value(v)),
    }
}

fn gather_pattern_constraints(ctx: &mut InferState, p: PatId) -> CId {
    match ctx.program.pattern(p) {
        Pattern::Wildcard=>{
            let c = ctx.new_cluster();
            ctx.bind_pat(p, c);
            c
        }
        Pattern::Bind(n) => {
            let c = ctx.new_cluster();
            ctx.names.insert(n, c);
            ctx.bind_pat(p, c);
            c
        }

        Pattern::TypeAnnotation { pat, ty } => {
            let c = gather_pattern_constraints(ctx, pat);
            let t = compile_type_expr(ctx, ty);

            if let Err(clash) = ctx.unify(c, t) {
                ctx.push_error(TypeError::PatternAnnotationMismatch {
                    annotation: p,
                    constrained: pat,
                    clash,
                });
            }

            ctx.bind_pat(p, c);
            c
        }

        _ => todo!(),
    }
}

fn gather_generic_constraints(ctx: &mut InferState, p: PatId,id:GenId) -> CId {
    match ctx.program.pattern(p) {
       
        Pattern::Bind(n) => {
            let t = ctx.store.intern(TypeValue::Generic(id));
            let c = ctx.new_solved(t);
            ctx.names.insert(n, c);
            ctx.bind_pat(p, c);
            c
        }

      

        _ => todo!(),
    }
}

fn compile_type_expr(ctx: &mut InferState, v: ValId) -> CId {
    match ctx.program.value(v) {
        Value::NameRef(n) => {
            let t = match ctx.program.definitions.get(&n) {
                Some(Defined::BuildinType(b)) => ctx.store.intern(b.clone()),
                Some(Defined::Type { ty, .. }) => *ty,
                _ => {
                    let c = ctx.new_cluster();
                    ctx.bind_val(v, c);
                    ctx.push_error(TypeError::ExpectedTypeExpr { type_expr: v });
                    return c;
                }
            };

            let c = ctx.new_solved(t);
            ctx.bind_val(v, c);
            c
        }
        Value::Wildcard => {
            let c = ctx.new_cluster();
            ctx.bind_val(v, c);
            c
        }
        _ => {
            let c = ctx.new_cluster();
            ctx.bind_val(v, c);
            ctx.push_error(TypeError::ExpectedTypeExpr { type_expr: v });
            c
        }
    }
}

// ===================================
// middle phase
// ===================================

#[inline(always)]
fn cluster_is_int_like(
    store: &TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    cid: CId,
) -> Option<bool> {
    let root = find_root(parent, cid);
    match cluster[root].state {
        ResolveKind::Solved(t) => Some(store.is_int_like(t)),
        ResolveKind::IntLike => Some(true),
        ResolveKind::FloatLike => Some(false),
        ResolveKind::Func(_) => Some(false),
        ResolveKind::Nothing | ResolveKind::ExternRef(_) => None,
    }
}

#[inline(always)]
fn cluster_is_float_like(
    store: &TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    cid: CId,
) -> Option<bool> {
    let root = find_root(parent, cid);
    match cluster[root].state {
        ResolveKind::Solved(t) => Some(store.is_float_like(t)),
        ResolveKind::FloatLike => Some(true),
        ResolveKind::IntLike => Some(false),
        ResolveKind::Func(_) => Some(false),
        ResolveKind::Nothing | ResolveKind::ExternRef(_) => None,
    }
}

/// Operator legality, tri-state:
///   Some(true)  = definitely allowed
///   Some(false) = definitely illegal
///   None        = insufficient info
#[inline(always)]
fn cluster_operator_applicable(
    store: &TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    op: BinOp,
    cid: CId,
) -> Option<bool> {
    use BinOp::*;
    match op {
        // Structural equality/comparison legality is handled elsewhere
        Eq | Ne | Lt | Le | Gt | Ge => Some(true),

        Add | Sub | Mul | Div | Mod => {
            match (
                cluster_is_int_like(store, parent, cluster, cid),
                cluster_is_float_like(store, parent, cluster, cid),
            ) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            }
        }

        BitAnd | BitOr | BitXor | Shl | Shr => cluster_is_int_like(store, parent, cluster, cid),
    }
}

/// Unify only if roots differ; report whether a merge happened.
#[inline]
fn unify_if_distinct(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    call_sites: &mut Vec<CallSite>,
    a: CId,
    b: CId,
) -> Result<bool, TypeClash> {
    let ra = find_root(parent, a);
    let rb = find_root(parent, b);
    if ra == rb {
        return Ok(false);
    }
    unify_clusters(store, parent, cluster, call_sites, ra, rb)?;
    Ok(true)
}

#[inline(always)]
fn resolve_operator_types(ctx: &mut InferState) -> bool {
    use BinOp::*;

    let mut progress = false;
    let (store, parent, cluster, call_sites, op_sites, errors) = (
        &mut ctx.store,
        &mut ctx.parent,
        &mut ctx.cluster,
        &mut ctx.call_sites,
        &mut ctx.op_sites,
        &mut ctx.errors,
    );

    for site in op_sites.iter_mut() {
        if site.had_error {
            continue;
        }
        let lhs = find_root(parent, site.lhs);
        let rhs = find_root(parent, site.rhs);
        let out = find_root(parent, site.output);
        let op = site.op;

        // ----------------------------------------------------
        // 1) Early legality rejection (single helper)
        // ----------------------------------------------------

        let lhs_ok = cluster_operator_applicable(store, parent, cluster, op, lhs);
        let rhs_ok = cluster_operator_applicable(store, parent, cluster, op, rhs);

        if lhs_ok == Some(false) || rhs_ok == Some(false) {
            errors.push(TypeError::ValuesContradict {
                expectation_reason: "operator cannot apply to this type",
                site: site.loc,
                found: site.lhs_val,
                expected_place: site.rhs_val,
                clash: TypeClash {
                    found: extract_bad_type(store, parent, cluster, call_sites, lhs),
                    wanted: extract_bad_type(store, parent, cluster, call_sites, rhs),
                },
            });
            site.had_error = true;
            continue;
        }

        // ----------------------------------------------------
        // 2) Equality / comparisons
        //
        // NOTE:
        // - operand equality is already enforced in gather
        // - output = bool is already enforced in gather
        // ----------------------------------------------------
        if matches!(op, Eq | Ne | Lt | Le | Gt | Ge) {
            continue;
        }

        // ----------------------------------------------------
        // 3) Arithmetic / bitwise
        //
        // - Only unify once both sides are known numeric
        // - Pointer arithmetic intentionally deferred
        // ----------------------------------------------------

        let lhs_numeric = matches!(cluster_is_int_like(store, parent, cluster, lhs), Some(true))
            || matches!(
                cluster_is_float_like(store, parent, cluster, lhs),
                Some(true)
            );

        let rhs_numeric = matches!(cluster_is_int_like(store, parent, cluster, rhs), Some(true))
            || matches!(
                cluster_is_float_like(store, parent, cluster, rhs),
                Some(true)
            );

        if !(lhs_numeric && rhs_numeric) {
            //TODO handle other cases
            continue;
        }

        // (a) unify operands
        match unify_if_distinct(store, parent, cluster, call_sites, lhs, rhs) {
            Ok(changed) => progress |= changed,
            Err(clash) => {
                errors.push(TypeError::ValuesContradict {
                    expectation_reason: "binary operator requires operands of the same type",
                    site: site.loc,
                    found: site.lhs_val,
                    expected_place: site.rhs_val,
                    clash,
                });
                site.had_error = true;
                continue;
            }
        }

        let operand = find_root(parent, lhs);

        // (b) unify output with operand
        match unify_if_distinct(store, parent, cluster, call_sites, out, operand) {
            Ok(changed) => progress |= changed,
            Err(clash) => {
                errors.push(TypeError::ValuesContradict {
                    expectation_reason: "operator result type must match operand type",
                    site: site.loc,
                    found: site.lhs_val,
                    expected_place: site.rhs_val,
                    clash,
                });
                site.had_error = true;
                continue;
            }
        }
    }

    progress
}

fn try_resolve_func_type(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    call_sites: &mut Vec<CallSite>,
    call: CallSiteId,
) -> Option<TypeId> {
    let mut params = Vec::with_capacity(call_sites[call.0].inputs.len());
    for i in 0..call_sites[call.0].inputs.len() {
        let input = call_sites[call.0].inputs[i];
        let root = find_root(parent, input);
        call_sites[call.0].inputs[i] = root;
        match cluster[root].state {
            ResolveKind::Solved(t) => params.push(t),
            _ => return None,
        }
    }

    let output = call_sites[call.0].output;
    let root = find_root(parent, output);
    call_sites[call.0].output = root;
    let ret = match cluster[root].state {
        ResolveKind::Solved(t) => t,
        _ => return None,
    };

    Some(store.intern(TypeValue::Func { params, ret }))
}

#[inline(always)]
fn resolve_func_types(ctx: &mut InferState) -> bool {
    let mut change = false;
    for cid in (0..ctx.cluster.len()).map(CId) {
        if let ResolveKind::Func(call) = ctx.cluster[cid].state {
            if let Some(t) = try_resolve_func_type(
                ctx.store,
                &mut ctx.parent,
                &mut ctx.cluster,
                &mut ctx.call_sites,
                call,
            ) {
                ctx.cluster[cid].state = ResolveKind::Solved(t);
                change = true;
            }
        }
    }
    change
}

#[inline(always)]
fn finalize(ctx: &mut InferState) -> LocalTypes {
    let (val_cluster, pat_cluster, parent, cluster, errors) = (
        &ctx.val_cluster,
        &ctx.pat_cluster,
        &mut ctx.parent,
        &ctx.cluster,
        &mut ctx.errors,
    );

    let mut reported = IdHashMap::default();
    let mut ans = LocalTypes::new();
    for (v, c) in val_cluster.iter() {
        let root = find_root(parent, *c);
        if let ResolveKind::Solved(t) = cluster[root].state {
            ans.val_types.insert(*v, t);
        } else if *c == root {
            errors.push(TypeError::Unresolved { value: *v });
            reported.insert(c,());
        }
    }
    for (p, c) in pat_cluster.iter() {
        let root = find_root(parent, *c);
        if let ResolveKind::Solved(t) = cluster[root].state {
            ans.pat_types.insert(*p, t);
        } else if *c == root &&reported.get(c).is_none(){
            errors.push(TypeError::UnresolvedPattern { pattern: *p });
        }
    }
    ans
}

// fn report_unresolved(ctx: &mut InferState){
//     let mut roots = Vec::with_capacity(ctx.cluster.len());
//     for i in 0..ctx.cluster.len(){
//         let c = CId(i);
//         if c==find_root(&mut ctx.parent,c){
//             roots.push(c);
//         }
//     }
// }

#[cfg(test)]
mod type_infer_tests {
    use super::*;
    use crate::parsing::Parser;
    use std::collections::HashSet;

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
    fn infer_fn(src: &str, store: &mut TypeStore) -> Result<TypeId, Vec<TypeError>> {
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
    fn infer_fn_body(src: &str, store: &mut TypeStore) -> Result<TypeId, Vec<TypeError>> {
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
        let errs = infer_fn_body("f = fn(){ let x = 1; x }", &mut store).unwrap_err();
        assert!(errs
            .iter()
            .any(|err| matches!(err, TypeError::Unresolved { .. })));
    }

    #[test]
    fn unresolved_clusters_report_once_and_stable() {
        let src = "f = fn(){ let x = 2; let y = x; let z = 2; }";
        let program = gather_program(src);
        let f = extract_single_fn(&program);
        let body = match program.value(f) {
            Value::Func { body, .. } => body,
            _ => panic!("expected function value"),
        };

        let body_val = program.value(body);
        let statements = match body_val {
            Value::Block { statements, .. } => statements,
            _ => panic!("expected block body"),
        };

        let first_let = statements
            .ids()
            .find(|id| matches!(program.value(*id), Value::Let { .. }))
            .expect("expected let statement");
        // let pat_x = match program.value(first_let) {
        //     Value::Let { pat, .. } => pat,
        //     _ => panic!("expected let value"),
        // };
        // let pat_x_loc = program.pattern_loc(pat_x);
        let let_x_loc = program.value_loc(first_let);

        let mut store = TypeStore::new();
        let errs = match infer_value_internals(&program, &mut store, body) {
            Ok(_) => panic!("expected type errors"),
            Err(errs) => errs,
        };

        let unresolved_locs = errs
            .iter()
            .filter_map(|err| match err {
                TypeError::Unresolved { value } => Some(program.value_loc(*value)),
                TypeError::UnresolvedPattern { pattern } => Some(program.pattern_loc(*pattern)),
                _ => None,
            })
            .collect::<Vec<_>>();

        let unique = unresolved_locs.iter().cloned().collect::<HashSet<_>>();
        let has_let_x = errs.iter().any(|err| match err {
            // TypeError::UnresolvedPattern { pattern } => program.pattern_loc(*pattern) == pat_x_loc,
            TypeError::Unresolved { value } => program.value_loc(*value) == let_x_loc,
            _ => false,
        });

        assert_eq!(errs.len(), 2);
        assert_eq!(unresolved_locs.len(), 2);
        assert_eq!(unique.len(), 2);
        assert!(has_let_x);
    }

    #[test]
    fn reports_multiple_hard_errors() {
        let src = "f = fn(){ let x:float = 2:int; let y:int = 2 + x; }";
        let program = gather_program(src);
        let f = extract_single_fn(&program);
        let body = match program.value(f) {
            Value::Func { body, .. } => body,
            _ => panic!("expected function value"),
        };

        let mut store = TypeStore::new();
        let errs = match infer_value_internals(&program, &mut store, body) {
            Ok(_) => panic!("expected type errors"),
            Err(errs) => errs,
        };
        assert_eq!(errs.len(), 2);
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
