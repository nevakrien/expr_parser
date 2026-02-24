//!
// ================================================================
// SCOPE (Closures and Rank)
// ================================================================
// for now we are trying to get a warking thing so we are not gona do proper closures
// closures are very hard to later compile AND they force rank-n types if we want lifetimes to work right
// so its actually too much of a hassle
//
//
// ================================================================
// CONTRACT
// ================================================================
// 1) we parse global type sinature and then internals of functions.
// 2) generics arent normalized and thus are only allowed globaly
//
// ================================================================
// THEORY
// ================================================================
// this is a modified hindly miller that uses kinds.
// we do have constraint solving but we only unify on equality.
// most constrains place some sort of pending task.
// and then later when enough type info is present we can apply unification.
// ================================================================
// use std::arch::asm;
use crate::global_type_inference::TypeExprCompileMode;
use crate::global_type_inference::compile_type_expr_with_mode;
use crate::global_type_inference::infer_global_types;
use crate::local_type_inference::gather_constraints;
use crate::local_type_inference::gather_func_constraints;
use crate::local_type_inference::infer_value_internals;
use crate::local_type_inference::local_solver;

use crate::ErrorReporter;
use crate::identity_hasher::IdHashMap;
use crate::ir::AccessKind;
use crate::ir::CallingConvention;
use crate::ir::LifeTimeId;
use crate::ir::StructLayoutSpec;
use crate::ir::VarKind;
use crate::ir::{BinOp, GenDec, NameId, PatId, Pattern, TExpId, TypeExpr, UnOp, ValId, Value};
use foldhash::HashMapExt;

// use crate::global_type_inference::*;
// use crate::local_type_inference::*;
use crate::parsing::Loc;
use crate::string_intern::StrId;
use foldhash::HashMap;
use std::fmt::Write as _;
use std::ops::{Index, IndexMut};

use crate::program::{Defined, Program};

use std::ffi::CStr;
unsafe extern "C" {
    fn perf_init();
    fn perf_begin();
    fn perf_done(name: *const std::os::raw::c_char);
}

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
pub const EXPANSION_STOPED: TypeId = TypeId(usize::MAX - 3);
pub const EXPANSION_LIMIT: usize = 100;

///this type specifically has internals containing UNKNOWN_TYPE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BadTypeId(pub TypeId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(pub usize);

///this is rank1 for now and so cant work for closures
///when we add them got to add rank and a normelization step and its a whole thing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LifeVar(pub usize);

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
    ///always last
    Type,
}
const BUILTIN_COUNT: usize = BuiltinType::Type as u8 as usize + 1;

// One place to update when adding builtin types.
// Note: `"float"` is an alias for `f64` in this sketch.
const BUILTINS: &[(&str, BuiltinType)] = {
    use BuiltinType::*;

    &[
        ("int", Int),
        ("uint", Uint),
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
    ]
};

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
        let x = id.0;
        if x < BUILTIN_COUNT {
            // SAFETY: repr(u8) + contiguous 0..BUILTIN_COUNT-1 by invariant.
            Ok(unsafe { core::mem::transmute::<u8, BuiltinType>(x as u8) })
        } else {
            Err(())
        }
    }
}

#[test]
fn try_from_buildin_works() {
    for (_, t) in BUILTINS.iter() {
        let tid: TypeId = (*t).into();
        assert_eq!(*t, BuiltinType::try_from(tid).unwrap());
    }

    for i in BUILTIN_COUNT..BUILTIN_COUNT + 10 {
        assert_eq!(Err(()), BuiltinType::try_from(TypeId(i)))
    }
}

/*const _: () = {
    if std::mem::size_of::<BuiltinType>() != 1 {
        panic!("BuiltinType must be 1 byte");
    }
};*/

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrayType {
    Sized(usize),
    Unsized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nullable {
    Yes,
    No,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerStyle {
    ///true means non null
    Raw(Nullable),
    Ref(LifeTime),
}

impl PointerStyle {
    pub fn is_fancy(&self) -> bool {
        !matches!(self, PointerStyle::Raw(Nullable::Yes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifeTime {
    //specifically infered internally.
    Local(LifeId),
    ///either from an argument or something we output
    //these can only really be created by type-inference level propagation
    //borrow checking itself looks at these as "longer than anything local okay sure"
    External(u32),
    ///for now no real use but it is technically a distinct category
    ///would probably add these for real once we actually have constants
    Static,

    ///ONLY MINTED AFTER MAIN SOLVER
    ///basically only happens when someone derefs a raw pointer
    ///or some signatures that are basically that
    ///this one may later get unified by the borrow checker into something
    Unknown(LifeId),
}

use std::cmp::Ordering;
impl PartialOrd for LifeTime {
    fn partial_cmp(&self, other: &LifeTime) -> Option<Ordering> {
        use LifeTime::*;

        if self == other {
            return Some(Ordering::Equal);
        }

        match (self, other) {
            // locals are shorter than externals and static
            (Local(_), External(_)) => Some(Ordering::Less),
            (Local(_), Static) => Some(Ordering::Less),

            // external shorter than static
            (External(_), Static) => Some(Ordering::Less),

            // reverse relations
            (External(_), Local(_)) => Some(Ordering::Greater),
            (Static, Local(_)) => Some(Ordering::Greater),
            (Static, External(_)) => Some(Ordering::Greater),

            // Unknown is incomparable
            (Unknown(..), _) | (_, Unknown(..)) => None,

            // different locals are incomparable
            (Local(_), Local(_)) => None,

            // different externals are incomparable
            (External(_), External(_)) => None,

            //static is static
            (LifeTime::Static, LifeTime::Static) => Some(Ordering::Equal),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeValue {
    Builtin(BuiltinType),
    Tuple(Vec<TypeId>),
    Array(TypeId, ArrayType),
    Func {
        calling_convention: CallingConvention,
        generics: usize,
        lifetimes: usize,
        params: Vec<TypeId>,
        ret: TypeId,
    },
    Ptr {
        tgt: TypeId,
        style: PointerStyle,
        mutable: bool,
    },
    Generic(GenId),
    // Specialized {
    //     base: TypeId,
    //     parts: Vec<TypeId>,
    // },
    Struct {
        id: StructId,
        generics: Vec<TypeId>,
        lifetimes: Vec<LifeTime>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LifeId(pub u32);

impl Program {
    //TODO make it so we can store TypeId here directly
    //or perhaps move type expressions to use some sort of global type context
    #[inline(always)]
    pub(crate) fn insert_builtin_types(&mut self) {
        for &(name, builtin) in BUILTINS {
            let name = self.str_intern.intern(name);
            let id = self.insert_value_in_current_scope(name);
            self.definitions
                .insert(id, Defined::BuildinType(TypeValue::Builtin(builtin)));
            // self.set_definition_loc(id, Program::placeholder_loc());
        }
    }
}

#[derive(Debug)]
pub struct TypeStore {
    pub(crate) values: Vec<TypeValue>,
    pub(crate) intern: HashMap<TypeValue, TypeId>,
    pub(crate) unused_function_generics: HashMap<TypeId, Vec<usize>>,

    pub(crate) structs: Vec<StructRep>,
    pub(crate) struct_overloads: IdHashMap<NameId, StructOverloadInfo>,
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
            unused_function_generics: HashMap::new(),

            structs: Vec::new(),
            struct_overloads: IdHashMap::default(),
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
    pub fn type_value(&self, id: TypeId) -> &TypeValue {
        &self.values[id.0]
    }

    #[inline]
    pub fn intern(&mut self, ty: TypeValue) -> TypeId {
        if let Some(&id) = self.intern.get(&ty) {
            return id;
        }

        let unused_generics = match &ty {
            TypeValue::Func {
                generics,
                params,
                ret,
                ..
            } if *generics != 0 => {
                let unused = self.compute_unused_function_generic_indexes(params, *ret, *generics);
                (!unused.is_empty()).then_some(unused)
            }
            _ => None,
        };

        let id = TypeId(self.values.len());
        self.values.push(ty.clone());
        self.intern.insert(ty, id);
        if let Some(unused_generics) = unused_generics {
            self.unused_function_generics.insert(id, unused_generics);
        }
        id
    }

    #[inline(always)]
    pub fn unused_function_generic_indexes(&self, ty: TypeId) -> &[usize] {
        self.unused_function_generics
            .get(&ty)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn unused_function_lifetime_indexes(
        &self,
        ty: TypeId,
        lifetime_count: usize,
    ) -> Vec<usize> {
        if lifetime_count == 0 {
            return Vec::new();
        }

        let mut used = vec![false; lifetime_count];
        self.mark_used_function_lifetime_indexes(ty, lifetime_count, &mut used);
        used.into_iter()
            .enumerate()
            .filter_map(|(i, is_used)| (!is_used).then_some(i))
            .collect()
    }

    fn compute_unused_function_generic_indexes(
        &self,
        params: &[TypeId],
        ret: TypeId,
        generic_count: usize,
    ) -> Vec<usize> {
        let mut used = vec![false; generic_count];
        for &param in params {
            self.mark_used_function_generic_indexes(param, generic_count, &mut used);
        }
        self.mark_used_function_generic_indexes(ret, generic_count, &mut used);

        used.into_iter()
            .enumerate()
            .filter_map(|(i, is_used)| (!is_used).then_some(i))
            .collect()
    }

    fn mark_used_function_generic_indexes(
        &self,
        ty: TypeId,
        generic_count: usize,
        used: &mut [bool],
    ) {
        match self.type_value(ty) {
            TypeValue::Builtin(_) => {}
            TypeValue::Generic(gid) => {
                if gid.0 < generic_count {
                    used[gid.0] = true;
                }
            }
            TypeValue::Tuple(items) => {
                for &item in items {
                    self.mark_used_function_generic_indexes(item, generic_count, used);
                }
            }
            TypeValue::Array(inner, _) => {
                self.mark_used_function_generic_indexes(*inner, generic_count, used);
            }
            TypeValue::Func { params, ret, .. } => {
                for &param in params {
                    self.mark_used_function_generic_indexes(param, generic_count, used);
                }
                self.mark_used_function_generic_indexes(*ret, generic_count, used);
            }
            TypeValue::Ptr { tgt, .. } => {
                self.mark_used_function_generic_indexes(*tgt, generic_count, used);
            }
            TypeValue::Struct { generics, .. } => {
                for &generic in generics {
                    self.mark_used_function_generic_indexes(generic, generic_count, used);
                }
            }
        }
    }

    fn mark_used_function_lifetime_indexes(
        &self,
        ty: TypeId,
        lifetime_count: usize,
        used: &mut [bool],
    ) {
        match self.type_value(ty) {
            TypeValue::Builtin(_) => {}
            TypeValue::Generic(_) => {}
            TypeValue::Tuple(items) => {
                for &item in items {
                    self.mark_used_function_lifetime_indexes(item, lifetime_count, used);
                }
            }
            TypeValue::Array(inner, _) => {
                self.mark_used_function_lifetime_indexes(*inner, lifetime_count, used);
            }
            TypeValue::Func { params, ret, .. } => {
                for &param in params {
                    self.mark_used_function_lifetime_indexes(param, lifetime_count, used);
                }
                self.mark_used_function_lifetime_indexes(*ret, lifetime_count, used);
            }
            TypeValue::Ptr { tgt, style, .. } => {
                if let PointerStyle::Ref(LifeTime::External(i)) = style
                    && (*i as usize) < lifetime_count
                {
                    used[*i as usize] = true;
                }
                self.mark_used_function_lifetime_indexes(*tgt, lifetime_count, used);
            }
            TypeValue::Struct {
                generics,
                lifetimes,
                ..
            } => {
                for &generic in generics {
                    self.mark_used_function_lifetime_indexes(generic, lifetime_count, used);
                }
                for lt in lifetimes {
                    if let LifeTime::External(i) = lt
                        && (*i as usize) < lifetime_count
                    {
                        used[*i as usize] = true;
                    }
                }
            }
        }
    }

    #[inline]
    pub fn new_struct(&mut self, rep: StructRep) -> StructId {
        let sid = StructId(self.structs.len());
        self.structs.push(rep);
        sid
    }

    #[cfg(test)]
    pub(crate) fn simple_struct(
        &mut self,
        name: Option<NameId>,
        fields: Vec<(NameId, TypeId)>,
    ) -> (StructId, TypeId) {
        let rep = StructRep {
            name,
            fields,
            gen_count: 0,
            life_count: 0,
            layout: StructLayoutSpec::Hot,
        };
        let sid = self.new_struct(rep);
        let tid = self.intern(TypeValue::Struct {
            id: sid,
            generics: Vec::new(),
            lifetimes: Vec::new(),
        });
        (sid, tid)
    }

    #[inline(always)]
    pub fn struct_value(&self, id: StructId) -> &StructRep {
        &self.structs[id.0]
    }

    #[inline]
    pub fn set_struct_fields(&mut self, id: StructId, fields: Vec<(NameId, TypeId)>) {
        self.structs[id.0].fields = fields;
    }

    #[inline(always)]
    pub fn struct_overload_info(&self, name: NameId) -> Option<&StructOverloadInfo> {
        self.struct_overloads.get(&name)
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
    pub fn get_bad_type_string(&self, program: &Program, t: BadTypeId) -> String {
        self.get_type_string(program, t.0)
    }
    pub fn get_type_string(&self, program: &Program, t: TypeId) -> String {
        self.get_type_string_nested(program, t, 0, 0)
    }
    pub fn get_type_string_nested(
        &self,
        program: &Program,
        t: TypeId,
        gen_count: usize,
        life_count: usize,
    ) -> String {
        if t == UNKNOWN_TYPE {
            return "_".to_string();
        }
        if t == UNKNOWN_INT_SIZE {
            return "int?".to_string();
        }
        if t == UNKNOWN_FLOAT_SIZE {
            return "float?".to_string();
        }

        if t == EXPANSION_STOPED {
            return "...".to_string();
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
                    .map(|id| self.get_type_string_nested(program, *id, gen_count, life_count))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({})", inner)
            }
            TypeValue::Func {
                calling_convention,
                generics,
                lifetimes,
                params,
                ret,
            } => {
                let params = params
                    .iter()
                    .map(|id| {
                        self.get_type_string_nested(
                            program,
                            *id,
                            gen_count + generics,
                            life_count + lifetimes,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let fn_kw = match calling_convention {
                    CallingConvention::Hot => "fn",
                    CallingConvention::C => "cfn",
                    CallingConvention::Unknown => "fn?",
                };
                let mut sig_parts = (life_count..(life_count + lifetimes))
                    .map(|i| format!("'a{i}"))
                    .collect::<Vec<_>>();
                sig_parts.extend((gen_count..(gen_count + generics)).map(|i| format!("T{i}")));
                let signature_params = if sig_parts.is_empty() {
                    String::new()
                } else {
                    format!("[{}]", sig_parts.join(", "))
                };
                format!(
                    "{}{}({}) -> {}",
                    fn_kw,
                    signature_params,
                    params,
                    self.get_type_string_nested(
                        program,
                        *ret,
                        gen_count + generics,
                        life_count + lifetimes,
                    )
                )
            }
            TypeValue::Ptr {
                tgt,
                style,
                mutable,
            } => {
                let inner = self.get_type_string_nested(program, *tgt, gen_count, life_count);

                match style {
                    // *T
                    PointerStyle::Raw(Nullable::Yes) => {
                        if *mutable {
                            format!("*{inner}")
                        } else {
                            format!("*const {inner}")
                        }
                    }

                    // &'raw T  (non-null raw pointer)
                    PointerStyle::Raw(Nullable::No) => {
                        if *mutable {
                            format!("&'raw {inner}")
                        } else {
                            format!("&'raw const {inner}")
                        }
                    }

                    // &'a T  (safe reference)
                    PointerStyle::Ref(lt) => {
                        let lt = self.format_lifetime(*lt);

                        if *mutable {
                            format!("&'{lt} mut {inner}")
                        } else {
                            format!("&'{lt} {inner}")
                        }
                    }
                }
            }

            TypeValue::Array(inner, ArrayType::Sized(n)) => {
                format!(
                    "[{};{n}]",
                    self.get_type_string_nested(program, *inner, gen_count, life_count)
                )
            }

            TypeValue::Array(inner, ArrayType::Unsized) => {
                format!(
                    "[{}]",
                    self.get_type_string_nested(program, *inner, gen_count, life_count)
                )
            }

            // TypeValue::Type => "Type".to_string(),
            TypeValue::Generic(g) => format!("T{}", g.0),

            //TODO cover cases where we do know the name
            TypeValue::Struct {
                id,
                generics,
                lifetimes,
            } => {
                self.format_struct_display(program, *id, generics, lifetimes, gen_count, life_count)
            }
        }
    }

    fn format_lifetime(&self, lt: LifeTime) -> String {
        match lt {
            LifeTime::Local(id) => format!("l{}", id.0),
            LifeTime::External(i) => format!("a{i}"),
            LifeTime::Static => "static".into(),
            LifeTime::Unknown(id) => format!("idk{}", id.0),
        }
    }

    fn format_struct_display(
        &self,
        program: &Program,
        sid: StructId,
        generics: &[TypeId],
        lifetimes: &[LifeTime],
        gen_count: usize,
        life_count: usize,
    ) -> String {
        let base = match self.struct_value(sid).name {
            Some(name) => program.name_string(name),
            None => "UnamedStruct",
        };
        let mut base = base.to_string();
        if !lifetimes.is_empty() || !generics.is_empty() {
            let mut args = lifetimes
                .iter()
                .map(|lt| format!("'{}", self.format_lifetime(*lt)))
                .collect::<Vec<_>>();
            args.extend(
                generics
                    .iter()
                    .map(|id| self.get_type_string_nested(program, *id, gen_count, life_count))
                    .collect::<Vec<_>>(),
            );
            base.push('[');
            base.push_str(&args.join(", "));
            base.push(']');
        }
        base
    }
}

pub struct SolvedTypes {
    pub typedef_types: IdHashMap<TExpId, TypeId>,
    pub function_values: IdHashMap<ValId, SolvedFunctionTypes>,
    pub function_types: IdHashMap<NameId, ValId>,
    pub member_function_types: HashMap<(NameId, StrId), ValId>,
}

impl SolvedTypes {
    pub fn new(program: &Program) -> Self {
        let mut typedef_types = IdHashMap::default();
        typedef_types.reserve(program.definitions.len());

        Self {
            typedef_types,
            function_values: IdHashMap::default(),
            function_types: IdHashMap::default(),
            member_function_types: HashMap::default(),
        }
    }

    #[inline]
    pub fn set_function_signature(&mut self, site: ValId, solved: SolvedFunctionTypes) {
        self.function_values.insert(site, solved);
    }

    #[inline]
    pub fn set_function_inner(&mut self, site: ValId, inner: InnerFunctionTypes) {
        if let Some(solved) = self.function_values.get_mut(&site) {
            solved.inner = Some(inner);
        }
    }

    #[inline(always)]
    pub fn type_of(&self, id: ValId) -> Option<TypeId> {
        self.function_values
            .get(&id)
            .map(|f| f.ty)
            .filter(|ty| *ty != UNKNOWN_TYPE)
    }

    #[inline(always)]
    pub fn pat_type(&self, id: PatId) -> Option<TypeId> {
        self.function_values
            .values()
            .find_map(|f| {
                f.arguments
                    .iter()
                    .find_map(|(p, _, t)| (*p == id).then_some(*t))
            })
            .filter(|ty| *ty != UNKNOWN_TYPE)
    }

    #[inline(always)]
    pub fn member_method_type(&self, id: ValId) -> Option<SolvedMemberMethodAccessType> {
        self.function_values.values().find_map(|f| {
            f.inner
                .as_ref()
                .and_then(|inner| inner.member_method_types.get(&id).copied())
        })
    }

    #[inline(always)]
    pub fn member_method_type_in_function(
        &self,
        function: ValId,
        id: ValId,
    ) -> Option<SolvedMemberMethodAccessType> {
        self.inner_types_of_function(function)
            .and_then(|inner| inner.member_method_types.get(&id).copied())
    }

    #[inline(always)]
    pub fn inner_types_of_function(&self, function: ValId) -> Option<&InnerFunctionTypes> {
        self.function_values
            .get(&function)
            .and_then(|f| f.inner.as_ref())
    }

    #[inline(always)]
    pub fn inner_value_type(&self, function: ValId, value: ValId) -> Option<TypeId> {
        self.inner_types_of_function(function)
            .and_then(|inner| inner.val_types.get(&value).copied())
            .filter(|ty| *ty != UNKNOWN_TYPE)
    }

    #[inline(always)]
    pub fn inner_pattern_type(&self, function: ValId, pattern: PatId) -> Option<TypeId> {
        self.inner_types_of_function(function)
            .and_then(|inner| inner.pat_types.get(&pattern).copied())
            .filter(|ty| *ty != UNKNOWN_TYPE)
    }

    #[inline(always)]
    pub fn function_types_by_name(&self, id: NameId) -> Option<&SolvedFunctionTypes> {
        self.function_types
            .get(&id)
            .and_then(|site| self.function_values.get(site))
    }

    #[inline(always)]
    pub fn member_function_types_by_name(
        &self,
        struct_name: NameId,
        member: StrId,
    ) -> Option<&SolvedFunctionTypes> {
        self.member_function_types
            .get(&(struct_name, member))
            .and_then(|site| self.function_values.get(site))
    }

    #[inline(always)]
    pub fn function_types_by_value(&self, id: ValId) -> Option<&SolvedFunctionTypes> {
        self.function_values.get(&id)
    }

    #[inline(always)]
    pub fn function_types_by_value_mut(&mut self, id: ValId) -> Option<&mut SolvedFunctionTypes> {
        self.function_values.get_mut(&id)
    }

    #[inline(always)]
    pub fn implicit_deref_chain(&self, id: ValId) -> Option<&[TypeId]> {
        self.function_values.values().find_map(|f| {
            f.inner
                .as_ref()
                .and_then(|inner| inner.implicit_derefs.get(&id).map(Vec::as_slice))
        })
    }

    #[inline(always)]
    pub fn implicit_deref_chain_in_function(
        &self,
        function: ValId,
        id: ValId,
    ) -> Option<&[TypeId]> {
        self.inner_types_of_function(function)
            .and_then(|inner| inner.implicit_derefs.get(&id).map(Vec::as_slice))
    }

    #[inline(always)]
    pub fn implicit_deref_count(&self, id: ValId) -> Option<usize> {
        self.implicit_deref_chain(id).map(|chain| chain.len())
    }

    #[inline(always)]
    pub fn member_access_implicit_deref_chain(&self, id: ValId) -> Option<&[TypeId]> {
        self.implicit_deref_chain(id)
    }

    #[inline(always)]
    pub fn member_access_implicit_deref_count(&self, id: ValId) -> Option<usize> {
        self.implicit_deref_count(id)
    }

    #[inline(always)]
    pub fn member_access_implicit_deref_count_in_function(
        &self,
        function: ValId,
        id: ValId,
    ) -> Option<usize> {
        self.implicit_deref_chain_in_function(function, id)
            .map(|chain| chain.len())
    }

    #[inline(always)]
    pub fn index_implicit_deref_chain(&self, id: ValId) -> Option<&[TypeId]> {
        self.implicit_deref_chain(id)
    }

    #[inline(always)]
    pub fn index_implicit_deref_count(&self, id: ValId) -> Option<usize> {
        self.implicit_deref_count(id)
    }
}

#[derive(Debug, Clone)]
pub struct SolvedFunctionTypes {
    pub ty: TypeId,
    pub impl_site: Option<ValId>,
    pub declaration_sites: Vec<ValId>,
    pub arguments: Vec<(PatId, Option<NameId>, TypeId)>,
    pub generic_parameters: Vec<(PatId, Option<NameId>)>,
    pub lifetime_parameters: Vec<(PatId, Option<LifeTimeId>)>,
    pub inner: Option<InnerFunctionTypes>,
}

#[derive(Debug, Clone, Default)]
pub struct InnerFunctionTypes {
    pub val_types: IdHashMap<ValId, TypeId>,
    // pub val_types: IdHashMap<ValId, (TypeId,ValueKind)>,
    // pub places: Vec<somestruct(Option<NameId>,some other info)>,
    pub pat_types: IdHashMap<PatId, TypeId>,
    pub member_method_types: IdHashMap<ValId, SolvedMemberMethodAccessType>,
    pub implicit_derefs: IdHashMap<ValId, Vec<TypeId>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaceId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueKind {
    LValue(PlaceId),
    RValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolvedMemberMethodAccessType {
    pub member: StrId,
    pub full_type: TypeId,
}

#[derive(Debug, Default, Clone)]
pub struct StructOverloadInfo {
    pub deref: Option<TypeId>,
    pub deref_site: Option<ValId>,
    pub deref_mut: Option<TypeId>,
    pub deref_mut_site: Option<ValId>,
    pub operators: IdHashMap<StrId, StructOperatorOverload>,
}

impl StructOverloadInfo {
    #[inline(always)]
    pub fn has_any(&self) -> bool {
        self.deref.is_some() || self.deref_mut.is_some() || !self.operators.is_empty()
    }
}

///todo add actual fields
#[derive(Debug)]
pub struct StructRep {
    pub name: Option<NameId>,
    pub fields: Vec<(NameId, TypeId)>,
    pub gen_count: usize,
    pub life_count: usize,
    pub layout: StructLayoutSpec,
}

// ==============================
// Errors
// ==============================

#[derive(Debug)]
pub enum TypeError {
    Simple {
        loc: Loc,
        message: &'static str,
    },
    UnknownBuiltinMemberMethod {
        site: ValId,
        method: StrId,
    },
    Unresolved {
        value: ValId,
        found: Option<String>,
    },
    UnresolvedPattern {
        pattern: PatId,
        found: Option<String>,
    },

    UnresolvedTypeExpr {
        expr: TExpId,
        found: Option<String>,
    },

    UnknownField {
        field: StrId,
        site: ValId,
    },

    DuplicateField {
        field: StrId,
        site: ValId,
    },

    FieldAlreadyPositional {
        field: StrId,
        site: ValId,
    },

    MissingField {
        field: NameId,
        site: ValId,
    },

    TooManyArguments {
        site: ValId,
        expected: usize,
        found: usize,
    },

    FieldTypeMismatch {
        field: StrId,
        value: ValId,
        clash: TypeClash,
    },

    IlegalMethod {
        member_name: StrId,
        access_site: ValId,
    },

    IlegalToImplMethod {
        method_name: StrId,
        method_site: ValId,
    },

    ConstructorBaseNotGlobal {
        site: ValId,
    },

    ConstructorBaseNotTypeName {
        site: ValId,
    },

    ConstructorBaseNotStruct {
        site: ValId,
        found: Option<String>,
    },

    TypeClashBeforeMentioned {
        name: NameId,
        expr: TExpId,
        clash: TypeClash,
    },

    /// Type expression (the RHS of `:` / `as`) wasn't a valid type
    ExpectedTypeExpr {
        type_expr: TExpId,
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

    /// No overload exists for the operator with the given operand types.
    BinOpOverloadNotFound {
        site: ValId,
        op: BinOp,
        lhs: ValId,
        rhs: ValId,
        lhs_type: Option<String>,
        rhs_type: Option<String>,
    },

    /// No overload exists for the operator with the given operand type.
    UnOpOverloadNotFound {
        site: ValId,
        op: UnOp,
        operand: ValId,
        operand_type: Option<String>,
    },

    CannotDeref {
        site: ValId,
        operand: ValId,
        operand_type: Option<String>,
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

    /// Function output type annotation conflicts with inferred body result.
    FunctionOutputAnnotationMismatch {
        output_type: Option<TExpId>,
        constrained: ValId,
        clash: TypeClash,
    },

    /// Pattern annotation mismatch
    PatternAnnotationMismatch {
        annotation: PatId,
        constrained: PatId,
        clash: TypeClash,
    },

    /// Type definition pattern must be a type name.
    TypeDefPatternMismatch {
        pattern: PatId,
        clash: TypeClash,
    },

    DuplicateFunctionImplementation {
        first_implementation: ValId,
        duplicate_implementation: ValId,
    },

    UnusedFunctionGeneric {
        function: ValId,
        generic_index: usize,
    },
    UnusedFunctionLifetime {
        function: ValId,
        lifetime_index: usize,
    },
    UnusedStructGeneric {
        type_expr: TExpId,
        generic_index: usize,
    },
    UnusedStructLifetime {
        type_expr: TExpId,
        lifetime_index: usize,
    },
}

#[derive(Debug)]
pub struct TypeClash {
    pub found: Option<String>,
    pub wanted: Option<String>,
}

pub(crate) const CLOSURES_UNSUPPORTED_MSG: &str = "sorry we dont support closures";

impl TypeClash {
    #[must_use]
    pub fn found(&self) -> Option<&str> {
        self.found.as_deref()
    }

    #[must_use]
    pub fn wanted(&self) -> Option<&str> {
        self.wanted.as_deref()
    }

    pub fn swap(self) -> Self {
        Self {
            found: self.wanted,
            wanted: self.found,
        }
    }
}

// ===================================
// Entry points
// ===================================

type TypecheckSummary = (Result<(TypeStore, SolvedTypes), usize>, usize);
type TypecheckResult = Result<TypecheckSummary, Box<dyn std::error::Error>>;

///runs the typechecker and reports all errors
///the rhs value is the total number of functions checked
///the lhs value is either the result or the number of errors found
pub fn run_typechecker(program: &Program, reporter: &mut ErrorReporter) -> TypecheckResult {
    let mut solved_types = SolvedTypes::new(program);
    let mut types = TypeStore::new();
    let mut err_count = 0;
    let mut function_checked = 0;

    unsafe {
        perf_init();
    }

    unsafe { perf_begin() }

    if let Err(errs) = infer_global_types(program, &mut types, &mut solved_types) {
        err_count += errs.len();

        for e in errs {
            reporter.report_type_error(program, &types, &e)?;
        }

        return Ok((Err(err_count), function_checked));
    }
    let name = CStr::from_bytes_with_nul(b"globals\0").unwrap();
    unsafe { perf_done(name.as_ptr()) };

    unsafe { perf_begin() }

    for (_n, methods) in program.member_methods.iter() {
        for (_s, method_set) in methods.iter() {
            for m in method_set.values() {
                function_checked += 1;

                let Err(errs) = infer_value_internals(program, &mut types, &mut solved_types, m)
                else {
                    continue;
                };
                err_count += errs.len();

                for e in errs {
                    reporter.report_type_error(program, &types, &e)?;
                }
            }
        }
    }

    for (_n, def) in program.definitions.iter() {
        let Defined::Func(funcs) = def else {
            continue;
        };
        for v in funcs.values() {
            function_checked += 1;

            let Err(errs) = infer_value_internals(program, &mut types, &mut solved_types, v) else {
                continue;
            };
            err_count += errs.len();

            for e in errs {
                reporter.report_type_error(program, &types, &e)?;
            }
        }
    }

    let name = CStr::from_bytes_with_nul(b"bodies\0").unwrap();
    unsafe { perf_done(name.as_ptr()) };

    if err_count > 0 {
        return Ok((Err(err_count), function_checked));
    }

    Ok((Ok((types, solved_types)), function_checked))
}

///this is just for tests we PURPOSFULLY ignore the global sig resolution
fn _infer_value_hacky<'a>(
    program: &'a Program,
    store: &'a mut TypeStore,
    ans: &'a mut SolvedTypes,

    value: ValId,
) -> Result<&'a mut SolvedTypes, Vec<TypeError>> {
    if !matches!(ans.function_values.get(&value), Some(_)) {
        ans.set_function_signature(
            value,
            SolvedFunctionTypes {
                ty: UNKNOWN_TYPE,
                impl_site: Some(value),
                declaration_sites: Vec::new(),
                arguments: Vec::new(),
                generic_parameters: Vec::new(),
                lifetime_parameters: Vec::new(),
                inner: None,
            },
        );
    }

    let mut ctx = InferState::new(store, program, ans);
    ctx.req.owner = Some(value);

    match ctx.ex.program.value(value) {
        Value::Func {
            calling_convention,
            generics,
            params,
            output_type,
            body,
        } => {
            let _ = gather_func_constraints::<true>(
                &mut ctx,
                value,
                calling_convention,
                generics,
                params,
                output_type,
                body,
            );
        }
        _ => {
            let c = gather_constraints(&mut ctx, value, None);
            ctx.bind_val(value, c);
        }
    }

    local_solver(&mut ctx);
    if ctx.ex.errors.is_empty() {
        Ok(ctx.ex.ans)
    } else {
        Err(ctx.ex.errors)
    }
}

pub fn main_solver(ctx: &mut InferState) {
    local_solver(ctx);
}

// ===================================
// Inference state + unify-find clusters
// ===================================
pub struct InferState<'a> {
    pub(crate) ex: ExternState<'a>,
    pub(crate) search: SearchState,
    pub(crate) types: TypeState,
    pub(crate) req: ReqState,
}
impl<'a> InferState<'a> {
    pub fn new(store: &'a mut TypeStore, program: &'a Program, ans: &'a mut SolvedTypes) -> Self {
        let mut types = TypeState::new();
        Self {
            ex: ExternState {
                store,
                program,
                name_render: GenLifeNameRender::Generate,
                errors: Vec::new(),
                ans,
            },

            search: SearchState::new(&mut types),
            types,
            req: ReqState::new(),
        }
    }

    pub(crate) fn new_cluster(&mut self) -> CId {
        self.types.new_cluster()
    }

    pub(crate) fn new_solved(&mut self, t: TypeId) -> CId {
        self.types.new_solved(t)
    }

    pub(crate) fn new_int_like(&mut self) -> CId {
        self.types.new_int_like()
    }

    pub(crate) fn new_float_like(&mut self) -> CId {
        self.types.new_float_like()
    }

    pub(crate) fn new_func(&mut self, call: FuncInfer) -> CId {
        self.types.new_func(call)
    }

    pub(crate) fn new_struct_instance(
        &mut self,
        sid: StructId,
        generics: Vec<CId>,
        lifetimes: Vec<LId>,
    ) -> CId {
        self.types.new_struct_instance(sid, generics, lifetimes)
    }

    pub(crate) fn new_tuple_instance(&mut self, items: Vec<CId>) -> CId {
        self.types.new_tuple_instance(items)
    }

    pub(crate) fn new_array_instance(&mut self, element: CId, size: ArrayType) -> CId {
        self.types.new_array_instance(element, size)
    }

    pub(crate) fn bind_val(&mut self, v: ValId, c: CId) {
        self.search.bind_val(v, c);
    }

    pub(crate) fn bind_pat(&mut self, p: PatId, c: CId) {
        self.search.bind_pat(p, c);
    }

    pub(crate) fn unify(&mut self, a: CId, b: CId) -> Result<CId, TypeClash> {
        self.types.unify(&mut self.ex, a, b)
    }

    pub(crate) fn force_type(&mut self, a: CId, t: TypeId) -> Result<(), TypeClash> {
        self.types.force_type(&mut self.ex, a, t)
    }

    pub fn push_error(&mut self, e: TypeError) {
        self.ex.push_error(e);
    }

    pub fn clear_local_state(&mut self) {
        self.types.clear_local_state();
        self.req.clear_local_state();
        self.ex.name_render = GenLifeNameRender::Generate;
        self.search.clear_local_state(&mut self.types);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CId(pub(crate) usize);

pub(crate) struct ClusterVec<T>(pub(crate) Vec<T>);
impl<T> ClusterVec<T> {
    pub(crate) fn new() -> Self {
        Self(Vec::new())
    }
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
    #[allow(dead_code)]
    pub(crate) fn swap(&mut self, a: CId, b: CId) {
        self.0.swap(a.0, b.0)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LId(pub(crate) usize);

pub(crate) struct LifeVec<T>(pub(crate) Vec<T>);
#[allow(dead_code)]
impl<T> LifeVec<T> {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    #[allow(dead_code)]
    fn swap(&mut self, a: LId, b: LId) {
        self.0.swap(a.0, b.0)
    }
}
impl<T> Index<LId> for LifeVec<T> {
    type Output = T;
    fn index(&self, id: LId) -> &T {
        &self.0[id.0]
    }
}

impl<T> IndexMut<LId> for LifeVec<T> {
    fn index_mut(&mut self, id: LId) -> &mut T {
        &mut self.0[id.0]
    }
}

#[derive(Debug)]
pub(crate) struct Cluster {
    pub(crate) state: ResolveKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FuncInferId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct StructInferId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TupleInferId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PtrKind {
    Solved(PointerStyle),
    RefInfer(LId),

    SafeRef,
    #[allow(dead_code)]
    SomeRef,
    Unknown,
}

impl PtrKind {
    pub fn is_fancy(self) -> Option<bool> {
        match self {
            PtrKind::Solved(s) => Some(s.is_fancy()),
            PtrKind::RefInfer(_) => Some(true),
            PtrKind::Unknown => None,
            _ => Some(true),
        }
    }

    // ///this is specifically for diagnostics
    // pub fn force_mock(&self) -> PointerStyle {
    //     match self {
    //         PtrKind::Solved(s) => *s,
    //         PtrKind::RefInfer(_) => PointerStyle::Ref(LifeTime::Unknown),
    //         PtrKind::Unknown => PointerStyle::Ref(LifeTime::Unknown),
    //         PtrKind::SafeRef | PtrKind::SomeRef => PointerStyle::Ref(LifeTime::Unknown),
    //     }
    // }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ResolveKind {
    Solved(TypeId),
    Nothing,

    // Specialized(SpecilizeId),
    ///the val is the last entity easily considered a lit like (2+1+3) in (let y = let x = 2+1+3)
    ///these lits can be used for error reporting
    IntLike,
    ///same as intlike but for float
    FloatLike,
    ///not all functions are like this but if something is declared as a function its this
    Func(FuncInferId),
    Struct(StructInferId),
    Tuple(TupleInferId),
    Array {
        element: CId,
        size: ArrayType,
    },

    Ptr {
        tgt: CId,
        kind: PtrKind,
        mutable: Option<bool>,
    },
}

#[derive(Debug)]
pub(crate) struct FuncInfer {
    pub(crate) calling_convention: CallingConvention,
    pub(crate) generics: usize,
    pub(crate) lifetimes: usize,
    pub(crate) inputs: Vec<CId>,
    pub(crate) output: CId,
}

#[derive(Debug)]
pub(crate) struct StructInfer {
    pub(crate) sid: StructId,
    pub(crate) generics: Vec<CId>,
    pub(crate) lifetimes: Vec<LId>,
}

#[derive(Debug)]
pub(crate) struct TupleInfer {
    pub(crate) items: Vec<CId>,
}

#[derive(Debug)]
pub(crate) struct StructDef {
    #[allow(dead_code)]
    pub(crate) loc: TExpId,
    pub(crate) fields: Vec<(NameId, CId)>,
    pub(crate) sid: StructId,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct Specialized {
    pub(crate) loc: Loc,
    pub(crate) base: CId,
    pub(crate) fields: Vec<CId>,
    pub(crate) output: CId,
}

#[derive(Debug)]
pub(crate) struct PendingSpecialization {
    pub(crate) name: NameId,
    pub(crate) global: TExpId,
    pub(crate) generics: Vec<CId>,
    pub(crate) lifetimes: Vec<LId>,
    pub(crate) output: CId,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct ComplexCallSite {
    pub(crate) loc: ValId,
    pub(crate) loc_called: ValId,

    pub(crate) called: CId,
    pub(crate) position_args: Vec<CId>,
    ///the strid can only be resolved once we know what we call;
    /// for structs thats just the type extra info
    /// for functions we need to know the actual specific one (which is a dependent type)
    pub(crate) named_args: Vec<(StrId, CId)>,
    pub(crate) output: CId,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BinOpSite {
    pub(crate) loc: ValId,
    pub(crate) op: BinOp,
    pub(crate) lhs_val: ValId,
    pub(crate) rhs_val: ValId,
    pub(crate) lhs: CId,
    pub(crate) rhs: CId,
    pub(crate) output: CId,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct UnOpSite {
    pub(crate) loc: ValId,
    pub(crate) op: UnOp,
    pub(crate) val: ValId,
    pub(crate) input: CId,
    pub(crate) output: CId,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum AssignIncDecFlavor {
    PreInc,
    PostInc,
    PreDec,
    PostDec,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AssignPrePostSite {
    pub(crate) loc: ValId,
    pub(crate) target_val: ValId,
    pub(crate) target: CId,
    pub(crate) implicit_rhs: CId,
    pub(crate) flavor: AssignIncDecFlavor,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingMemberMethodType {
    pub(crate) site: ValId,
    pub(crate) member: StrId,
    pub(crate) full_method: CId,
    pub(crate) receiver: CId,
    pub(crate) receiver_value: ValId,
}

#[derive(Debug, Clone)]
pub(crate) struct PendingMemberAccessImplicitDeref {
    pub(crate) site: ValId,
    pub(crate) receivers: Vec<CId>,
}

#[derive(Debug)]
pub(crate) struct PendingMemberAccess {
    pub(crate) site: ValId,
    pub(crate) base_value: ValId,

    // original base cluster
    pub(crate) source: CId,

    // resume cursor (VERY important)
    pub(crate) current: CId,

    // output cluster for the expression
    pub(crate) output: CId,

    pub(crate) member: StrId,
    pub(crate) kind: AccessKind,

    // autoderef state (persistent)
    pub(crate) implicit_receivers: Vec<CId>,
    pub(crate) deref_chain_lid: Option<LId>,
    pub(crate) deref_chain_is_mut: Option<bool>,
}

impl PendingMemberAccess {
    pub(crate) fn new(
        types: &mut TypeState,
        site: ValId,
        base_value: ValId,
        source: CId,
        output: CId,
        member: StrId,
        kind: AccessKind,
    ) -> Self {
        let source = types.root(source);

        Self {
            site,
            base_value,
            source,
            current: source, // CRITICAL: start cursor here
            output,
            member,
            kind,
            implicit_receivers: Vec::new(),
            deref_chain_lid: None,
            deref_chain_is_mut: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingIntAccess {
    pub(crate) site: ValId,
    pub(crate) source: CId,
    pub(crate) output: CId,
    pub(crate) id: usize,
    pub(crate) kind: AccessKind,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingDeref {
    pub(crate) site: ValId,
    pub(crate) source: CId,
    pub(crate) target: CId,
    pub(crate) source_value: ValId,
}

impl PendingDeref {
    #[inline(always)]
    pub(crate) fn new(
        types: &mut TypeState,
        site: ValId,
        source_value: ValId,
        source: CId,
        target: CId,
    ) -> Self {
        let source = types.root(source);

        Self {
            site,
            source_value,
            source,
            target,
        }
    }
}

#[derive(Debug)]
pub(crate) struct PendingIndex {
    pub(crate) site: ValId,
    pub(crate) base_value: ValId,
    pub(crate) index_value: ValId,

    pub(crate) base: CId,
    pub(crate) index: CId,
    pub(crate) output: CId,

    pub(crate) current: CId,
    pub(crate) implicit_receivers: Vec<CId>,

    pub(crate) deref_chain_lid: Option<LId>,
    pub(crate) deref_chain_mutability: Option<bool>,
}

impl PendingIndex {
    #[inline(always)]
    pub(crate) fn new(
        types: &mut TypeState,
        site: ValId,
        base_value: ValId,
        index_value: ValId,
        base: CId,
        index: CId,
        output: CId,
    ) -> Self {
        // canonicalize immediately (VERY important — prevents stale path issues)
        let base = types.root(base);
        let index = types.root(index);

        Self {
            site,
            base_value,
            index_value,

            base,
            index,
            output,

            // coroutine cursor: current autoderef position
            current: base,

            // persistent autoderef prefix
            implicit_receivers: Vec::new(),

            // smart-deref tracking
            deref_chain_lid: None,
            deref_chain_mutability: None,
        }
    }
}

pub(crate) enum GenLifeNameRender<'a> {
    TextNames {
        _decl: GenDec,
        generic_names: Vec<&'a str>,
        lifetime_names: Vec<&'a str>,
    },
    Generate,
}

impl<'a> GenLifeNameRender<'a> {
    pub(crate) fn from_decl(program: &'a Program, decl: GenDec) -> Self {
        let mut generic_names = Vec::with_capacity(decl.generics().len());
        for pat in decl.generics().ids() {
            match program.pattern(pat) {
                Pattern::Bind(name, _) => generic_names.push(program.name_string(name)),
                _ => generic_names.push("_"),
            }
        }

        let mut lifetime_names = Vec::with_capacity(decl.lifetimes().len());
        for pat in decl.lifetimes().ids() {
            match program.pattern(pat) {
                Pattern::LifeTime(id) => lifetime_names.push(program.lifetime_string(id)),
                _ => lifetime_names.push("_"),
            }
        }

        Self::TextNames {
            _decl: decl,
            generic_names,
            lifetime_names,
        }
    }

    fn generic_name(&self, idx: usize) -> String {
        match self {
            Self::TextNames { generic_names, .. } => generic_names
                .get(idx)
                .map(|s| (*s).to_string())
                .unwrap_or_else(|| generated_generic_name(idx)),
            Self::Generate => generated_generic_name(idx),
        }
    }

    fn external_lifetime_name(&self, idx: u32) -> String {
        match self {
            Self::TextNames { lifetime_names, .. } => lifetime_names
                .get(idx as usize)
                .map(|s| (*s).to_string())
                .unwrap_or_else(|| idx.to_string()),
            Self::Generate => generated_lifetime_name(idx),
        }
    }
}

// #[cold]
fn generated_generic_name(idx: usize) -> String {
    format!("T{idx}")
}

// #[cold]
fn generated_lifetime_name(idx: u32) -> String {
    format!("a{idx}")
}

pub(crate) struct ExternState<'a> {
    pub(crate) store: &'a mut TypeStore,
    pub(crate) program: &'a Program,
    pub(crate) name_render: GenLifeNameRender<'a>,

    //result
    pub(crate) errors: Vec<TypeError>,
    pub(crate) ans: &'a mut SolvedTypes,
}

impl<'a> ExternState<'a> {
    pub(crate) fn push_error(&mut self, err: TypeError) {
        self.errors.push(err);
    }
}

pub(crate) struct SearchState {
    //ir -> cid
    pub(crate) val_cluster: Vec<(ValId, CId)>,
    pub(crate) pat_cluster: Vec<(PatId, CId)>,
    pub(crate) typedef_cluster: Vec<(TExpId, CId)>,
    pub(crate) local_types: IdHashMap<NameId, CId>,
    pub(crate) names: IdHashMap<NameId, CId>,
    pub(crate) local_lifetimes: IdHashMap<LifeTimeId, (LifeTime, LId)>,
}

impl SearchState {
    fn new(types: &mut TypeState) -> Self {
        let mut ans = Self {
            val_cluster: Vec::default(),
            pat_cluster: Vec::default(),
            typedef_cluster: Vec::default(),
            local_types: IdHashMap::default(),
            names: IdHashMap::default(),
            local_lifetimes: IdHashMap::default(),
        };
        ans.populate_defaults(types);
        ans
    }

    fn clear_local_state(&mut self, types: &mut TypeState) {
        let SearchState {
            val_cluster,
            pat_cluster,
            typedef_cluster,
            local_types,
            names,
            local_lifetimes,
        } = self;

        val_cluster.clear();
        pat_cluster.clear();
        typedef_cluster.clear();
        local_types.clear();
        names.clear();
        local_lifetimes.clear();

        self.populate_defaults(types);
    }

    fn populate_defaults(&mut self, types: &mut TypeState) {
        let lid = types.new_lid_known(LifeTime::Static);
        self.local_lifetimes
            .insert(LifeTimeId::STATIC, (LifeTime::Static, lid));
    }

    pub(crate) fn bind_val(&mut self, v: ValId, c: CId) {
        self.val_cluster.push((v, c));
    }

    pub(crate) fn bind_pat(&mut self, p: PatId, c: CId) {
        self.pat_cluster.push((p, c));
    }
}

pub(crate) struct TypeCore {
    // unify-find
    pub(crate) parent: ClusterVec<CId>,
    pub(crate) cluster: ClusterVec<Cluster>,
}

impl TypeCore {
    pub(crate) fn find_root(&mut self, x: CId) -> CId {
        find_root(&mut self.parent, x)
    }

    #[allow(dead_code)]
    pub(crate) fn new_cluster(&mut self) -> CId {
        let id = CId(self.parent.len());
        self.parent.0.push(id);
        self.cluster.0.push(Cluster {
            state: ResolveKind::Nothing,
        });
        id
    }
}

pub(crate) struct TypeExtra {
    pub(crate) func_defs: Vec<FuncInfer>,
    pub(crate) struct_defs: Vec<StructDef>,
    pub(crate) struct_infers: Vec<StructInfer>,
    pub(crate) tuple_infers: Vec<TupleInfer>,
}

pub(crate) struct TypeState {
    pub(crate) core: TypeCore,
    pub(crate) extra: TypeExtra,
    pub(crate) life_parent: LifeVec<LId>,
    pub(crate) life_known: LifeVec<Option<LifeTime>>,
    pub(crate) next_undeclared_lifetime: u32,
}

#[inline(always)]
pub(crate) fn find_lid_root(life_parent: &mut LifeVec<LId>, lid: LId) -> LId {
    let p = life_parent[lid];
    if p == lid {
        return lid;
    }
    let root = find_lid_root(life_parent, p);
    life_parent[lid] = root;
    root
}

impl TypeState {
    fn new() -> Self {
        let mut ans = Self {
            core: TypeCore {
                parent: ClusterVec::new(),
                cluster: ClusterVec::new(),
            },
            extra: TypeExtra {
                func_defs: Vec::new(),
                struct_defs: Vec::new(),
                struct_infers: Vec::new(),
                tuple_infers: Vec::new(),
            },
            life_parent: LifeVec(Vec::new()),
            life_known: LifeVec(Vec::new()),
            next_undeclared_lifetime: 0,
        };

        ans.populate_defaults();
        ans
    }

    fn clear_local_state(&mut self) {
        let TypeCore { parent, cluster } = &mut self.core;

        let TypeExtra {
            func_defs,
            struct_defs,
            struct_infers,
            tuple_infers,
        } = &mut self.extra;

        // ---- union find ----
        parent.0.clear();
        cluster.0.clear();

        // ---- type database ----
        func_defs.clear();
        struct_defs.clear();
        struct_infers.clear();
        tuple_infers.clear();
        self.life_parent.0.clear();
        self.life_known.0.clear();
        self.next_undeclared_lifetime = 0;

        self.populate_defaults();
    }

    fn populate_defaults(&mut self) {
        for _i in 0..BUILTIN_COUNT {
            let id = self.new_cluster();
            debug_assert_eq!(id.0, _i);
            self.core.cluster[id].state = ResolveKind::Solved(TypeId(id.0));
        }
    }

    #[inline(always)]
    pub(crate) fn new_lid(&mut self) -> LId {
        let id = LId(self.life_parent.0.len());
        self.life_parent.0.push(id);
        self.life_known.0.push(None);
        id
    }

    #[inline(always)]
    pub(crate) fn new_lid_known(&mut self, known: LifeTime) -> LId {
        let id = self.new_lid();
        self.life_known[id] = Some(known);
        id
    }

    #[inline(always)]
    pub(crate) fn find_lid_root(&mut self, lid: LId) -> LId {
        find_lid_root(&mut self.life_parent, lid)
    }

    #[inline(always)]
    pub(crate) fn mint_undeclared_signature_lifetime(&mut self) -> LifeTime {
        let id = self.next_undeclared_lifetime;
        self.next_undeclared_lifetime += 1;
        LifeTime::External(id)
    }

    // =========================================================
    // cluster construction
    // =========================================================

    pub(crate) fn new_cluster(&mut self) -> CId {
        let id = CId(self.core.parent.len());
        self.core.parent.0.push(id);
        self.core.cluster.0.push(Cluster {
            state: ResolveKind::Nothing,
        });
        id
    }

    #[inline(always)]
    pub(crate) fn new_solved(&mut self, t: TypeId) -> CId {
        if let Ok(b) = BuiltinType::try_from(t) {
            let t: TypeId = b.into();
            return CId(t.0);
        }
        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::Solved(t);
        id
    }

    pub(crate) fn new_int_like(&mut self) -> CId {
        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::IntLike;
        id
    }

    pub(crate) fn new_float_like(&mut self) -> CId {
        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::FloatLike;
        id
    }

    pub(crate) fn new_func(&mut self, call: FuncInfer) -> CId {
        let call_id = FuncInferId(self.extra.func_defs.len());
        self.extra.func_defs.push(call);

        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::Func(call_id);
        id
    }

    pub(crate) fn new_struct_instance(
        &mut self,
        sid: StructId,
        generics: Vec<CId>,
        lifetimes: Vec<LId>,
    ) -> CId {
        let call_id = StructInferId(self.extra.struct_infers.len());
        self.extra.struct_infers.push(StructInfer {
            sid,
            generics,
            lifetimes,
        });

        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::Struct(call_id);
        id
    }

    pub(crate) fn new_tuple_instance(&mut self, items: Vec<CId>) -> CId {
        let tuple_id = TupleInferId(self.extra.tuple_infers.len());
        self.extra.tuple_infers.push(TupleInfer { items });

        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::Tuple(tuple_id);
        id
    }

    pub(crate) fn new_array_instance(&mut self, element: CId, size: ArrayType) -> CId {
        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::Array { element, size };
        id
    }

    // =========================================================
    // union-find operations
    // =========================================================

    pub(crate) fn unify(
        &mut self,
        ex: &mut ExternState<'_>,
        a: CId,
        b: CId,
    ) -> Result<CId, TypeClash> {
        unify_clusters(ex, self, a, b)
    }

    pub(crate) fn force_type(
        &mut self,
        ex: &mut ExternState<'_>,
        a: CId,
        t: TypeId,
    ) -> Result<(), TypeClash> {
        // thin wrapper over old function until we migrate it
        force_type(ex, self, a, t)
    }
}

#[inline(always)]
fn merge_lifetime_known_strict(
    a: Option<LifeTime>,
    b: Option<LifeTime>,
) -> Option<Option<LifeTime>> {
    match (a, b) {
        (None, None) => Some(None),
        (Some(x), None) | (None, Some(x)) => Some(Some(x)),
        (Some(x), Some(y)) if x == y => Some(Some(x)),
        _ => None,
    }
}

#[inline(always)]
pub(crate) fn unify_struct_lids(types: &mut TypeState, a: LId, b: LId) -> bool {
    let ra = types.find_lid_root(a);
    let rb = types.find_lid_root(b);
    if ra == rb {
        return true;
    }

    let Some(merged) = merge_lifetime_known_strict(types.life_known[ra], types.life_known[rb])
    else {
        return false;
    };

    types.life_parent[rb] = ra;
    types.life_known[ra] = merged;
    true
}

#[inline(always)]
pub(crate) fn bind_struct_lid_to_lifetime(
    types: &mut TypeState,
    lid: LId,
    target: LifeTime,
) -> bool {
    let root = types.find_lid_root(lid);
    let Some(merged) = merge_lifetime_known_strict(types.life_known[root], Some(target)) else {
        return false;
    };
    types.life_known[root] = merged;
    true
}

#[inline(always)]
pub(crate) fn unify_ptr_lifetimes(_types: &mut TypeState, a: LifeTime, b: LifeTime) -> bool {
    a == b
}

pub(crate) struct ReqState {
    pub(crate) owner: Option<ValId>,
    //requirments
    pub(crate) bin_op_sites: Vec<BinOpSite>,
    pub(crate) un_op_sites: Vec<UnOpSite>,
    pub(crate) assign_pre_post_sites: Vec<AssignPrePostSite>,

    //generic_func_values: Vec<(ValId, usize)>,
    pub(crate) pending_specializations: Vec<PendingSpecialization>,
    pub(crate) member_method_type_sites: Vec<PendingMemberMethodType>,
    pub(crate) member_access_implicit_deref_sites: Vec<PendingMemberAccessImplicitDeref>,
    pub(crate) index_implicit_deref_sites: Vec<PendingMemberAccessImplicitDeref>,
    pub(crate) pending_member_accesses: Vec<PendingMemberAccess>,
    pub(crate) pending_int_accesses: Vec<PendingIntAccess>,
    pub(crate) pending_indexes: Vec<PendingIndex>,
    pub(crate) pending_derefs: Vec<PendingDeref>,
}

impl ReqState {
    fn new() -> Self {
        Self {
            owner: None,
            bin_op_sites: Vec::new(),
            un_op_sites: Vec::new(),
            assign_pre_post_sites: Vec::new(),

            pending_specializations: Vec::new(),
            member_method_type_sites: Vec::new(),
            member_access_implicit_deref_sites: Vec::new(),
            index_implicit_deref_sites: Vec::new(),
            pending_member_accesses: Vec::new(),
            pending_int_accesses: Vec::new(),
            pending_indexes: Vec::new(),
            pending_derefs: Vec::new(),
        }
    }

    fn clear_local_state(&mut self) {
        let ReqState {
            owner,
            bin_op_sites,
            un_op_sites,
            assign_pre_post_sites,
            pending_specializations,
            member_method_type_sites,
            member_access_implicit_deref_sites,
            index_implicit_deref_sites,
            pending_member_accesses,
            pending_int_accesses,
            pending_indexes,
            pending_derefs,
        } = self;

        *owner = None;
        bin_op_sites.clear();
        un_op_sites.clear();
        assign_pre_post_sites.clear();

        pending_specializations.clear();
        member_method_type_sites.clear();
        member_access_implicit_deref_sites.clear();
        index_implicit_deref_sites.clear();
        pending_member_accesses.clear();
        pending_int_accesses.clear();
        pending_indexes.clear();
        pending_derefs.clear();
    }
}

// ===========================
// Keep this helper as-is
// (it is used by unify/force)
// ===========================

#[inline(always)]
pub(crate) fn find_root(parent: &mut ClusterVec<CId>, x: CId) -> CId {
    let p = parent[x];
    if p != x {
        let r = find_root(parent, p);
        parent[x] = r;
    }
    parent[x]
}

///tries to combine 2 clusters
///on fail produces a type_clash and keeps the 2 seprate
///inference can continue with the 2 types separated for the purpose of gathering more errors (obviously not Unresolved style errors)
pub(crate) fn unify_clusters(
    ex: &mut ExternState,
    types: &mut TypeState,
    found: CId,
    wanted: CId,
) -> Result<CId, TypeClash> {
    unify_clusters_inlined(ex, types, found, wanted)
}

impl TypeState {
    // =========================================================
    // union-find
    // =========================================================

    #[inline(always)]
    pub fn root(&mut self, c: CId) -> CId {
        find_root(&mut self.core.parent, c)
    }

    #[inline(always)]
    pub fn link(&mut self, child: CId, parent: CId) {
        debug_assert_eq!(self.core.parent[child], child);
        self.core.parent[child] = parent;
    }

    // =========================================================
    // cluster state (THIS is the important abstraction)
    // =========================================================

    #[inline(always)]
    pub fn cluster_state(&self, c: CId) -> ResolveKind {
        self.core.cluster[c].state
    }

    #[inline(always)]
    pub fn set_cluster_state(&mut self, c: CId, s: ResolveKind) {
        self.core.cluster[c].state = s;
    }

    #[inline(always)]
    pub fn copy_cluster_state(&mut self, dst: CId, src: CId) {
        self.core.cluster[dst].state = self.core.cluster[src].state;
    }

    // =========================================================
    // structural database access
    // =========================================================

    #[inline(always)]
    pub fn func(&self, id: FuncInferId) -> &FuncInfer {
        &self.extra.func_defs[id.0]
    }

    #[inline(always)]
    pub fn func_mut(&mut self, id: FuncInferId) -> &mut FuncInfer {
        &mut self.extra.func_defs[id.0]
    }

    #[inline(always)]
    pub fn struct_infer(&self, id: StructInferId) -> &StructInfer {
        &self.extra.struct_infers[id.0]
    }

    #[inline(always)]
    pub fn tuple_infer(&self, id: TupleInferId) -> &TupleInfer {
        &self.extra.tuple_infers[id.0]
    }

    // =========================================================
    // diagnostics
    // =========================================================

    pub fn bad_type(&mut self, ex: &mut ExternState, cid: CId) -> Option<String> {
        let mut limit = EXPANSION_LIMIT;
        extract_clash_type_string(ex, &mut self.core, &self.extra, cid, &mut limit)
    }

    pub fn clash(&mut self, ex: &mut ExternState, found: CId, wanted: CId) -> TypeClash {
        TypeClash {
            found: self.bad_type(ex, found),
            wanted: self.bad_type(ex, wanted),
        }
    }
}

#[inline(always)]
pub(crate) fn unify_clusters_inlined(
    ex: &mut ExternState,
    types: &mut TypeState,
    found: CId,
    wanted: CId,
) -> Result<CId, TypeClash> {
    let rf = types.root(found);
    let rw = types.root(wanted);

    if rf == rw {
        return Ok(rw);
    }

    // Try found <- wanted
    if __try_absorb(ex, types, rw, rf)? {
        types.link(rf, rw);
        return Ok(rw);
    }

    // Otherwise try wanted <- found
    if __try_absorb(ex, types, rf, rw).map_err(TypeClash::swap)? {
        types.link(rw, rf);
        return Ok(rf);
    }

    // real contradiction
    Err(TypeClash {
        found: types.bad_type(ex, found),
        wanted: types.bad_type(ex, wanted),
    })
}
#[inline(always)]
fn __try_absorb(
    ex: &mut ExternState,
    types: &mut TypeState,
    dst: CId,
    src: CId,
) -> Result<bool, TypeClash> {
    use ResolveKind::*;

    let dst_state = types.cluster_state(dst);
    let src_state = types.cluster_state(src);

    match (dst_state, src_state) {
        // =====================================================
        // this is a hack for making literals not apear in errors as much
        // =====================================================
        (Nothing, IntLike) | (Nothing, FloatLike) => {
            types.copy_cluster_state(dst, src);
            Ok(true)
        }

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
                    found: Some(type_string_from_type_id(ex, t2)),
                    wanted: Some(type_string_from_type_id(ex, t1)),
                })
            }
        }

        // =====================================================
        // Solved absorbs literals if compatible
        // =====================================================
        (Solved(t), IntLike) => {
            if !ex.store.is_int_like(t) {
                return Err(TypeClash {
                    found: Some(type_string_from_type_id(ex, UNKNOWN_INT_SIZE)),
                    wanted: Some(type_string_from_type_id(ex, t)),
                });
            }
            Ok(true)
        }

        (Solved(t), FloatLike) => {
            if !ex.store.is_float_like(t) {
                return Err(TypeClash {
                    found: Some(type_string_from_type_id(ex, UNKNOWN_FLOAT_SIZE)),
                    wanted: Some(type_string_from_type_id(ex, t)),
                });
            }
            Ok(true)
        }

        // =====================================================
        // Same-kind weak info: merge
        // =====================================================
        (IntLike, IntLike) | (FloatLike, FloatLike) => Ok(true),

        // =====================================================
        // Function placeholders
        // =====================================================
        (Func(dst_call), Func(src_call)) => {
            let dst_f = types.func(dst_call);
            let src_f = types.func(src_call);

            if dst_f.inputs.len() != src_f.inputs.len() {
                return Err(types.clash(ex, dst, src));
            }

            for i in 0..dst_f.inputs.len() {
                let d = types.func(dst_call).inputs[i];
                let s = types.func(src_call).inputs[i];
                if types.unify(ex, d, s).is_err() {
                    return Err(types.clash(ex, dst, src));
                }
            }

            let d = types.func(dst_call).output;
            let s = types.func(src_call).output;

            if types.unify(ex, d, s).is_err() {
                return Err(types.clash(ex, dst, src));
            }

            let dst_f = types.func(dst_call);
            let src_f = types.func(src_call);

            let Some(merged_cc) =
                merge_calling_convention(dst_f.calling_convention, src_f.calling_convention)
            else {
                return Err(types.clash(ex, dst, src));
            };

            types.func_mut(dst_call).calling_convention = merged_cc;
            types.func_mut(src_call).calling_convention = merged_cc;

            if let Some(t) = try_resolve_func_type(ex, types, dst_call) {
                types.set_cluster_state(dst, Solved(t));
            }

            Ok(true)
        }

        (Solved(t), Func(call)) => {
            unify_func_with_type(ex, types, call, t)?;
            Ok(true)
        }

        // =====================================================
        // Struct
        // =====================================================
        (Struct(dst_call), Struct(src_call)) => {
            let (dst_sid, dst_glen, dst_llen) = {
                let dst_s = types.struct_infer(dst_call);
                (dst_s.sid, dst_s.generics.len(), dst_s.lifetimes.len())
            };
            let (src_sid, src_glen, src_llen) = {
                let src_s = types.struct_infer(src_call);
                (src_s.sid, src_s.generics.len(), src_s.lifetimes.len())
            };

            if dst_sid != src_sid || dst_glen != src_glen || dst_llen != src_llen {
                return Err(types.clash(ex, dst, src));
            }

            for i in 0..dst_glen {
                let dst_s = types.struct_infer(dst_call);
                let src_s = types.struct_infer(src_call);

                if types
                    .unify(ex, dst_s.generics[i], src_s.generics[i])
                    .is_err()
                {
                    return Err(types.clash(ex, dst, src));
                }
            }

            for i in 0..dst_llen {
                let dst_s = types.struct_infer(dst_call);
                let src_s = types.struct_infer(src_call);
                if !unify_struct_lids(types, dst_s.lifetimes[i], src_s.lifetimes[i]) {
                    return Err(types.clash(ex, dst, src));
                }
            }

            if let Some(t) = try_resolve_struct_type(ex, types, dst_call) {
                types.set_cluster_state(dst, Solved(t));
            }

            Ok(true)
        }

        (Solved(t), Struct(call)) => {
            unify_struct_with_type(ex, types, call, t)?;
            Ok(true)
        }

        // =====================================================
        // Ptr
        // =====================================================
        (Solved(t), Ptr { tgt, kind, mutable }) => {
            unify_ptr_with_type(ex, types, tgt, kind, mutable, t)?;
            Ok(true)
        }

        (
            Ptr {
                tgt: dst_tgt,
                kind: dst_kind,
                mutable: dst_mut,
            },
            Ptr {
                tgt: src_tgt,
                kind: src_kind,
                mutable: src_mut,
            },
        ) => {
            let kind = merge_ptr_kind(types, dst_kind, src_kind)
                .ok_or_else(|| types.clash(ex, dst, src))?;

            let mutable =
                merge_ptr_flag(dst_mut, src_mut).ok_or_else(|| types.clash(ex, dst, src))?;

            let tgt = types.unify(ex, dst_tgt, src_tgt)?;

            types.set_cluster_state(dst, Ptr { tgt, kind, mutable });
            Ok(true)
        }

        // =====================================================
        // Tuple
        // =====================================================
        (Tuple(dst_tuple), Tuple(src_tuple)) => {
            let d = types.tuple_infer(dst_tuple);
            let s = types.tuple_infer(src_tuple);

            if d.items.len() != s.items.len() {
                return Err(types.clash(ex, dst, src));
            }

            for i in 0..d.items.len() {
                let d = types.tuple_infer(dst_tuple);
                let s = types.tuple_infer(src_tuple);
                if types.unify(ex, d.items[i], s.items[i]).is_err() {
                    return Err(types.clash(ex, dst, src));
                }
            }

            if let Some(t) = try_resolve_tuple_type(ex, types, dst_tuple) {
                types.set_cluster_state(dst, Solved(t));
            }

            Ok(true)
        }

        (Solved(t), Tuple(tuple_id)) => {
            unify_tuple_with_type(ex, types, tuple_id, t)?;
            Ok(true)
        }

        // =====================================================
        // Array
        // =====================================================
        (
            Array {
                element: dst_element,
                size: dst_len,
            },
            Array {
                element: src_element,
                size: src_len,
            },
        ) => {
            if dst_len != src_len {
                return Err(types.clash(ex, dst, src));
            }

            if types.unify(ex, dst_element, src_element).is_err() {
                return Err(types.clash(ex, dst, src));
            }

            if let Some(t) = try_resolve_array_type(ex, types, dst_element, dst_len) {
                types.set_cluster_state(dst, Solved(t));
            }

            Ok(true)
        }

        (Solved(t), Array { element, size }) => {
            unify_array_with_type(ex, types, element, size, t)?;
            Ok(true)
        }

        // =====================================================
        // Everything else: do not guess
        // =====================================================
        _ => Ok(false),
    }
}

pub(crate) fn force_type_if_distinct(
    ex: &mut ExternState,
    types: &mut TypeState,
    target: CId,
    ty: TypeId,
) -> Result<bool, TypeClash> {
    let root = types.root(target);

    if let ResolveKind::Solved(t) = types.cluster_state(root)
        && t == ty
    {
        return Ok(false);
    }

    force_type(ex, types, target, ty)?;
    Ok(true)
}

pub(crate) fn force_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    target: CId,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let root = types.root(target);
    let state = types.cluster_state(root);

    match state {
        ResolveKind::Nothing => {
            types.set_cluster_state(root, ResolveKind::Solved(ty));
            Ok(())
        }

        ResolveKind::Solved(t) if t == ty => Ok(()),

        ResolveKind::Solved(t) => Err(simple_type_clash(ex, t, ty)),

        ResolveKind::IntLike => {
            if !ex.store.is_int_like(ty) {
                return Err(TypeClash {
                    found: Some(type_string_from_type_id(ex, UNKNOWN_INT_SIZE)),
                    wanted: Some(type_string_from_type_id(ex, ty)),
                });
            }
            types.set_cluster_state(root, ResolveKind::Solved(ty));
            Ok(())
        }

        ResolveKind::FloatLike => {
            if !ex.store.is_float_like(ty) {
                return Err(TypeClash {
                    found: Some(type_string_from_type_id(ex, UNKNOWN_FLOAT_SIZE)),
                    wanted: Some(type_string_from_type_id(ex, ty)),
                });
            }
            types.set_cluster_state(root, ResolveKind::Solved(ty));
            Ok(())
        }

        ResolveKind::Func(call) => {
            unify_func_with_type(ex, types, call, ty)?;
            types.set_cluster_state(root, ResolveKind::Solved(ty));
            Ok(())
        }

        ResolveKind::Struct(call) => {
            unify_struct_with_type(ex, types, call, ty)?;
            types.set_cluster_state(root, ResolveKind::Solved(ty));
            Ok(())
        }

        ResolveKind::Tuple(call) => {
            unify_tuple_with_type(ex, types, call, ty)?;
            types.set_cluster_state(root, ResolveKind::Solved(ty));
            Ok(())
        }

        ResolveKind::Array { element, size } => {
            unify_array_with_type(ex, types, element, size, ty)?;
            types.set_cluster_state(root, ResolveKind::Solved(ty));
            Ok(())
        }

        ResolveKind::Ptr { tgt, kind, mutable } => {
            unify_ptr_with_type(ex, types, tgt, kind, mutable, ty)?;
            types.set_cluster_state(root, ResolveKind::Solved(ty));
            Ok(())
        }
    }
}

#[inline(always)]
fn merge_ptr_flag(a: Option<bool>, b: Option<bool>) -> Option<Option<bool>> {
    match (a, b) {
        (Some(x), Some(y)) if x != y => None,
        (Some(x), _) => Some(Some(x)),
        (_, Some(y)) => Some(Some(y)),
        (None, None) => Some(None),
    }
}

fn merge_ptr_kind(types: &mut TypeState, a: PtrKind, b: PtrKind) -> Option<PtrKind> {
    use PtrKind::*;

    if let (RefInfer(la), RefInfer(lb)) = (a, b) {
        if !unify_struct_lids(types, la, lb) {
            return None;
        }
        return Some(RefInfer(la));
    }

    if let (RefInfer(lid), Solved(PointerStyle::Ref(lt)))
    | (Solved(PointerStyle::Ref(lt)), RefInfer(lid)) = (a, b)
    {
        if !bind_struct_lid_to_lifetime(types, lid, lt) {
            return None;
        }
        return Some(RefInfer(lid));
    }

    if let (Solved(PointerStyle::Ref(la)), Solved(PointerStyle::Ref(lb))) = (a, b) {
        if !unify_ptr_lifetimes(types, la, lb) {
            return None;
        }
        return Some(Solved(PointerStyle::Ref(la)));
    }

    match (a, b) {
        // identical
        (x, y) if x == y => Some(x),

        // Unknown disappears
        (Unknown, x) | (x, Unknown) => Some(x),

        // partial info refinement
        (SomeRef, SafeRef) | (SafeRef, SomeRef) => Some(SafeRef),

        (RefInfer(lid), SafeRef) | (SafeRef, RefInfer(lid)) => Some(RefInfer(lid)),
        (RefInfer(lid), SomeRef) | (SomeRef, RefInfer(lid)) => Some(RefInfer(lid)),

        // solved vs partial
        (Solved(style), SafeRef) | (SafeRef, Solved(style)) => match style {
            PointerStyle::Ref(_lt) => Some(Solved(style)),
            PointerStyle::Raw(_) => None,
        },

        (Solved(style), SomeRef) | (SomeRef, Solved(style)) => {
            match style {
                PointerStyle::Ref(_) => Some(Solved(style)),
                PointerStyle::Raw(Nullable::No) => Some(Solved(style)), // &'raw
                PointerStyle::Raw(Nullable::Yes) => None,               // *T is nullable
            }
        }

        // solved vs solved
        (Solved(a), Solved(b)) => {
            if a == b {
                Some(Solved(a))
            } else {
                None
            }
        }

        // remaining cases are incompatible
        _ => None,
    }
}

#[inline(always)]
fn merge_calling_convention(
    a: CallingConvention,
    b: CallingConvention,
) -> Option<CallingConvention> {
    use CallingConvention::*;
    match (a, b) {
        (Unknown, x) | (x, Unknown) => Some(x),
        (x, y) if x == y => Some(x),
        _ => None,
    }
}

pub(crate) fn unify_ptr_with_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    tgt: CId,
    kind: PtrKind,
    mutable: Option<bool>,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let found_ptr = |ex, types: &mut TypeState| {
        let mut out = String::new();
        let mut limit = EXPANSION_LIMIT;
        write_ptr_mock_string_inner(
            ex,
            &mut types.core,
            &types.extra,
            tgt,
            kind,
            mutable,
            &mut out,
            &mut limit,
        );
        out
    };

    let TypeValue::Ptr {
        tgt: ty_tgt,
        style,
        mutable: ty_mut,
    } = *ex.store.type_value(ty)
    else {
        return Err(TypeClash {
            found: Some(found_ptr(ex, types)),
            wanted: Some(type_string_from_type_id(ex, ty)),
        });
    };
    match (kind, style) {
        (PtrKind::Solved(PointerStyle::Ref(a)), PointerStyle::Ref(b)) => {
            if !unify_ptr_lifetimes(types, a, b) {
                return Err(TypeClash {
                    found: Some(found_ptr(ex, types)),
                    wanted: Some(type_string_from_type_id(ex, ty)),
                });
            }
        }
        (PtrKind::RefInfer(lid), PointerStyle::Ref(b)) => {
            if !bind_struct_lid_to_lifetime(types, lid, b) {
                return Err(TypeClash {
                    found: Some(found_ptr(ex, types)),
                    wanted: Some(type_string_from_type_id(ex, ty)),
                });
            }
        }
        _ => {}
    }

    if matches!(kind.is_fancy(), Some(x) if x != style.is_fancy())
        || matches!(mutable, Some(x) if x != ty_mut)
    {
        return Err(TypeClash {
            found: Some(found_ptr(ex, types)),
            wanted: Some(type_string_from_type_id(ex, ty)),
        });
    }

    force_type(ex, types, tgt, ty_tgt)
}

pub(crate) fn unify_func_with_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    call: FuncInferId,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let found_func = |ex, types: &mut TypeState| {
        let mut out = String::new();
        let mut limit = EXPANSION_LIMIT;
        write_func_mock_string_inner(
            ex,
            &mut types.core,
            &types.extra,
            call,
            &mut out,
            &mut limit,
        );
        out
    };

    let (cc, generics, lifetimes, params, ret) = match ex.store.type_value(ty) {
        TypeValue::Func {
            calling_convention,
            generics,
            lifetimes,
            params,
            ret,
        } => (
            *calling_convention,
            *generics,
            *lifetimes,
            params.as_slice(),
            *ret,
        ),
        _ => {
            return Err(TypeClash {
                found: Some(found_func(ex, types)),
                wanted: Some(type_string_from_type_id(ex, ty)),
            });
        }
    };

    let infer_cc = types.extra.func_defs[call.0].calling_convention;
    let Some(merged_cc) = merge_calling_convention(infer_cc, cc) else {
        return Err(TypeClash {
            found: Some(found_func(ex, types)),
            wanted: Some(type_string_from_type_id(ex, ty)),
        });
    };

    types.extra.func_defs[call.0].calling_convention = merged_cc;

    if types.extra.func_defs[call.0].generics != generics
        || types.extra.func_defs[call.0].lifetimes != lifetimes
    {
        return Err(TypeClash {
            found: Some(found_func(ex, types)),
            wanted: Some(type_string_from_type_id(ex, ty)),
        });
    }

    let input_len = types.extra.func_defs[call.0].inputs.len();
    if params.len() != input_len {
        return Err(TypeClash {
            found: Some(found_func(ex, types)),
            wanted: Some(type_string_from_type_id(ex, ty)),
        });
    }

    for i in 0..input_len {
        let input = types.extra.func_defs[call.0].inputs[i];

        //TODO (maybe): we constantly take the params again from the spot because borrow checker
        //              technically the Vec params points to never reallocs
        //              so theortically its possible to keep borowing this
        let param_ty = match ex.store.type_value(ty) {
            TypeValue::Func { params, .. } => params[i],
            _ => unreachable!(),
        };

        force_type(ex, types, input, param_ty)?;
    }

    let output = types.extra.func_defs[call.0].output;
    force_type(ex, types, output, ret)?;

    Ok(())
}

pub(crate) fn unify_struct_with_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    call: StructInferId,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let found_struct = |ex, types: &mut TypeState| {
        let mut out = String::new();
        let mut limit = EXPANSION_LIMIT;
        write_struct_mock_string_inner(
            ex,
            &mut types.core,
            &types.extra,
            call,
            &mut out,
            &mut limit,
        );
        out
    };

    let (sid, glen, lifetimes) = match ex.store.type_value(ty) {
        TypeValue::Struct {
            id,
            generics,
            lifetimes,
        } => (*id, generics.len(), lifetimes.as_slice()),
        _ => {
            return Err(TypeClash {
                found: Some(found_struct(ex, types)),
                wanted: Some(type_string_from_type_id(ex, ty)),
            });
        }
    };

    let call_sid = types.extra.struct_infers[call.0].sid;
    if call_sid != sid
        || types.extra.struct_infers[call.0].generics.len() != glen
        || types.extra.struct_infers[call.0].lifetimes.len() != lifetimes.len()
    {
        return Err(TypeClash {
            found: Some(found_struct(ex, types)),
            wanted: Some(type_string_from_type_id(ex, ty)),
        });
    }

    for (i, target_lt) in lifetimes.iter().copied().enumerate() {
        let lid = types.extra.struct_infers[call.0].lifetimes[i];
        if !bind_struct_lid_to_lifetime(types, lid, target_lt) {
            return Err(TypeClash {
                found: Some(found_struct(ex, types)),
                wanted: Some(type_string_from_type_id(ex, ty)),
            });
        }
    }

    for i in 0..glen {
        let input = types.extra.struct_infers[call.0].generics[i];
        let TypeValue::Struct { generics, .. } = ex.store.type_value(ty) else {
            unreachable!();
        };
        let t = generics[i];
        force_type(ex, types, input, t)?;
    }

    Ok(())
}

pub(crate) fn unify_tuple_with_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    tuple: TupleInferId,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let found_tuple = |ex, types: &mut TypeState| {
        let mut out = String::new();
        let mut limit = EXPANSION_LIMIT;
        write_tuple_mock_string_inner(
            ex,
            &mut types.core,
            &types.extra,
            tuple,
            &mut out,
            &mut limit,
        );
        out
    };

    let ilen = types.extra.tuple_infers[tuple.0].items.len();

    let TypeValue::Tuple(items) = ex.store.type_value(ty) else {
        return Err(TypeClash {
            found: Some(found_tuple(ex, types)),
            wanted: Some(type_string_from_type_id(ex, ty)),
        });
    };

    if items.len() != ilen {
        return Err(TypeClash {
            found: Some(found_tuple(ex, types)),
            wanted: Some(type_string_from_type_id(ex, ty)),
        });
    }

    for i in 0..ilen {
        let TypeValue::Tuple(items) = ex.store.type_value(ty) else {
            unreachable!();
        };
        let item_ty = items[i];
        let item = types.extra.tuple_infers[tuple.0].items[i];
        force_type(ex, types, item, item_ty)?;
    }

    Ok(())
}

pub(crate) fn unify_array_with_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    element: CId,
    size: ArrayType,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let found_array = |ex, types: &mut TypeState| {
        let mut out = String::new();
        let mut limit = EXPANSION_LIMIT;
        write_array_mock_string_inner(
            ex,
            &mut types.core,
            &types.extra,
            element,
            size,
            &mut out,
            &mut limit,
        );
        out
    };

    let (ty_element, ty_size) = match ex.store.type_value(ty) {
        TypeValue::Array(item, n) => (*item, *n),
        _ => {
            return Err(TypeClash {
                found: Some(found_array(ex, types)),
                wanted: Some(type_string_from_type_id(ex, ty)),
            });
        }
    };

    if ty_size != size {
        return Err(TypeClash {
            found: Some(found_array(ex, types)),
            wanted: Some(type_string_from_type_id(ex, ty)),
        });
    }

    force_type(ex, types, element, ty_element)
}

fn try_resolve_func_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    call: FuncInferId,
) -> Option<TypeId> {
    let func = &mut types.extra.func_defs[call.0];

    let mut params = Vec::with_capacity(func.inputs.len());

    for i in 0..func.inputs.len() {
        let input = func.inputs[i];

        let root = find_root(&mut types.core.parent, input);
        func.inputs[i] = root; // canonicalize

        match types.core.cluster[root].state {
            ResolveKind::Solved(t) => params.push(t),
            _ => return None,
        }
    }

    let output = func.output;
    let root = find_root(&mut types.core.parent, output);
    func.output = root; // canonicalize

    let ret = match types.core.cluster[root].state {
        ResolveKind::Solved(t) => t,
        _ => return None,
    };

    Some(ex.store.intern(TypeValue::Func {
        calling_convention: func.calling_convention,
        generics: func.generics,
        lifetimes: func.lifetimes,
        params,
        ret,
    }))
}

fn try_resolve_struct_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    call: StructInferId,
) -> Option<TypeId> {
    let site = &mut types.extra.struct_infers[call.0];
    let sid = site.sid;

    let mut generics = Vec::with_capacity(site.generics.len());
    for input in site.generics.iter_mut() {
        let root = find_root(&mut types.core.parent, *input);
        *input = root;

        match types.core.cluster[root].state {
            ResolveKind::Solved(t) => generics.push(t),
            _ => return None,
        }
    }

    let mut lifetimes = Vec::with_capacity(site.lifetimes.len());
    for lid in site.lifetimes.iter_mut() {
        let root = find_lid_root(&mut types.life_parent, *lid);
        *lid = root;
        let Some(ans) = types.life_known[root] else {
            return None;
        };
        lifetimes.push(ans);
    }

    Some(ex.store.intern(TypeValue::Struct {
        id: sid,
        generics,
        lifetimes,
    }))
}

fn try_resolve_tuple_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    tuple: TupleInferId,
) -> Option<TypeId> {
    let site = &mut types.extra.tuple_infers[tuple.0];

    let mut items = Vec::with_capacity(site.items.len());

    for i in 0..site.items.len() {
        let input = site.items[i];

        let root = find_root(&mut types.core.parent, input);
        site.items[i] = root; // canonicalize

        match types.core.cluster[root].state {
            ResolveKind::Solved(t) => items.push(t),
            _ => return None,
        }
    }

    Some(ex.store.intern(TypeValue::Tuple(items)))
}

fn try_resolve_array_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    element: CId,
    size: ArrayType,
) -> Option<TypeId> {
    let root = types.root(element);

    let element = match types.cluster_state(root) {
        ResolveKind::Solved(t) => t,
        _ => return None,
    };

    Some(ex.store.intern(TypeValue::Array(element, size)))
}

fn try_resolve_ptr_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    tgt: CId,
    kind: PtrKind,
    mutable: Option<bool>,
) -> Option<TypeId> {
    let style = match kind {
        PtrKind::Solved(style) => style,
        PtrKind::RefInfer(lid) => {
            let root = find_lid_root(&mut types.life_parent, lid);
            let lt = types.life_known[root]?;
            PointerStyle::Ref(lt)
        }
        _ => return None,
    };
    let mutable = mutable?;

    let root = types.root(tgt);

    let tgt = match types.cluster_state(root) {
        ResolveKind::Solved(t) => t,
        _ => return None,
    };

    Some(ex.store.intern(TypeValue::Ptr {
        tgt,
        style,
        mutable,
    }))
}

fn type_string_from_type_id(ex: &ExternState<'_>, t: TypeId) -> String {
    type_string_from_type_id_nested(ex, t, 0, 0)
}

fn type_string_from_type_id_nested(
    ex: &ExternState<'_>,
    t: TypeId,
    gen_count: usize,
    life_count: usize,
) -> String {
    if t == UNKNOWN_TYPE {
        return "unknown".to_string();
    }
    if t == UNKNOWN_INT_SIZE {
        return "int?".to_string();
    }
    if t == UNKNOWN_FLOAT_SIZE {
        return "float?".to_string();
    }
    if t == EXPANSION_STOPED {
        return "...".to_string();
    }

    match ex.store.type_value(t) {
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
            let items = items
                .iter()
                .map(|id| type_string_from_type_id_nested(ex, *id, gen_count, life_count))
                .collect::<Vec<_>>()
                .join(", ");
            format!("({items})")
        }
        TypeValue::Func {
            calling_convention,
            generics,
            lifetimes,
            params,
            ret,
        } => {
            let params = params
                .iter()
                .map(|id| {
                    type_string_from_type_id_nested(
                        ex,
                        *id,
                        gen_count + *generics,
                        life_count + *lifetimes,
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");

            let mut sig_parts = (life_count..(life_count + *lifetimes))
                .map(|i| format!("'{}", ex.name_render.external_lifetime_name(i as u32)))
                .collect::<Vec<_>>();
            sig_parts.extend(
                (gen_count..(gen_count + *generics)).map(|i| ex.name_render.generic_name(i)),
            );
            let signature_params = if sig_parts.is_empty() {
                String::new()
            } else {
                format!("[{}]", sig_parts.join(", "))
            };

            format!(
                "{}{}({}) -> {}",
                calling_convention_keyword(*calling_convention),
                signature_params,
                params,
                type_string_from_type_id_nested(
                    ex,
                    *ret,
                    gen_count + *generics,
                    life_count + *lifetimes,
                )
            )
        }
        TypeValue::Ptr {
            tgt,
            style,
            mutable,
        } => {
            let inner = type_string_from_type_id_nested(ex, *tgt, gen_count, life_count);
            match style {
                PointerStyle::Raw(Nullable::Yes) => {
                    if *mutable {
                        format!("*{inner}")
                    } else {
                        format!("*const {inner}")
                    }
                }
                PointerStyle::Raw(Nullable::No) => {
                    if *mutable {
                        format!("&'raw {inner}")
                    } else {
                        format!("&'raw const {inner}")
                    }
                }
                PointerStyle::Ref(lt) => {
                    let lt = lifetime_for_display(ex, *lt);
                    if *mutable {
                        format!("&'{lt} mut {inner}")
                    } else {
                        format!("&'{lt} {inner}")
                    }
                }
            }
        }
        TypeValue::Array(inner, ArrayType::Sized(n)) => {
            format!(
                "[{};{n}]",
                type_string_from_type_id_nested(ex, *inner, gen_count, life_count)
            )
        }
        TypeValue::Array(inner, ArrayType::Unsized) => {
            format!(
                "[{}]",
                type_string_from_type_id_nested(ex, *inner, gen_count, life_count)
            )
        }
        TypeValue::Generic(g) => ex.name_render.generic_name(g.0),
        TypeValue::Struct {
            id,
            generics,
            lifetimes,
        } => {
            let mut base = match ex.store.struct_value(*id).name {
                Some(name) => ex.program.name_string(name).to_string(),
                None => "UnamedStruct".to_string(),
            };
            if !lifetimes.is_empty() || !generics.is_empty() {
                let mut args = lifetimes
                    .iter()
                    .map(|lt| format!("'{}", lifetime_for_display(ex, *lt)))
                    .collect::<Vec<_>>();
                args.extend(
                    generics
                        .iter()
                        .map(|id| type_string_from_type_id_nested(ex, *id, gen_count, life_count)),
                );
                base.push('[');
                base.push_str(&args.join(", "));
                base.push(']');
            }
            base
        }
    }
}

pub(crate) fn simple_type_clash(ex: &ExternState<'_>, a: TypeId, b: TypeId) -> TypeClash {
    TypeClash {
        found: Some(type_string_from_type_id(ex, a)),
        wanted: Some(type_string_from_type_id(ex, b)),
    }
}

fn lifetime_for_display(ex: &ExternState<'_>, lt: LifeTime) -> String {
    match lt {
        LifeTime::Local(id) => format!("l{}", id.0),
        LifeTime::Unknown(id) => format!("idk{}", id.0),
        LifeTime::External(i) => ex.name_render.external_lifetime_name(i),
        LifeTime::Static => "static".to_string(),
    }
}

fn write_lifetime_for_display(ex: &ExternState<'_>, out: &mut String, lt: LifeTime) {
    let _ = out.write_str(&lifetime_for_display(ex, lt));
}

fn calling_convention_keyword(cc: CallingConvention) -> &'static str {
    match cc {
        CallingConvention::Hot => "fn",
        CallingConvention::C => "cfn",
        CallingConvention::Unknown => "fn?",
    }
}

fn write_mock_type_from_cluster(
    ex: &ExternState<'_>,
    core: &mut TypeCore,
    extra: &TypeExtra,
    cid: CId,
    out: &mut String,
    limit: &mut usize,
) {
    if *limit == 0 {
        let _ = out.write_str("...");
        return;
    }
    *limit -= 1;

    let root = find_root(&mut core.parent, cid);
    match core.cluster[root].state {
        ResolveKind::Solved(t) => {
            let _ = out.write_str(&type_string_from_type_id(ex, t));
        }
        ResolveKind::IntLike => {
            let _ = out.write_str("int?");
        }
        ResolveKind::FloatLike => {
            let _ = out.write_str("float?");
        }
        ResolveKind::Func(call) => write_func_mock_string_inner(ex, core, extra, call, out, limit),
        ResolveKind::Struct(call) => {
            write_struct_mock_string_inner(ex, core, extra, call, out, limit)
        }
        ResolveKind::Tuple(tuple) => {
            write_tuple_mock_string_inner(ex, core, extra, tuple, out, limit)
        }
        ResolveKind::Array { element, size } => {
            write_array_mock_string_inner(ex, core, extra, element, size, out, limit)
        }
        ResolveKind::Ptr { tgt, kind, mutable } => {
            write_ptr_mock_string_inner(ex, core, extra, tgt, kind, mutable, out, limit)
        }
        ResolveKind::Nothing => {
            let _ = out.write_char('_');
        }
    }

    *limit += 1;
}

fn write_func_mock_string_inner(
    ex: &ExternState<'_>,
    core: &mut TypeCore,
    extra: &TypeExtra,
    call: FuncInferId,
    out: &mut String,
    limit: &mut usize,
) {
    let site = &extra.func_defs[call.0];
    let _ = out.write_str(calling_convention_keyword(site.calling_convention));
    if site.generics > 0 {
        let _ = out.write_char('[');
        for i in 0..site.generics {
            if i > 0 {
                let _ = out.write_str(", ");
            }
            let _ = write!(out, "T{i}");
        }
        let _ = out.write_char(']');
    }
    let _ = out.write_char('(');
    for (i, input) in site.inputs.iter().copied().enumerate() {
        if i > 0 {
            let _ = out.write_str(", ");
        }
        write_mock_type_from_cluster(ex, core, extra, input, out, limit);
    }
    let _ = out.write_str(") -> ");
    write_mock_type_from_cluster(ex, core, extra, site.output, out, limit);
}

fn write_struct_mock_string_inner(
    ex: &ExternState<'_>,
    core: &mut TypeCore,
    extra: &TypeExtra,
    call: StructInferId,
    out: &mut String,
    limit: &mut usize,
) {
    let site = &extra.struct_infers[call.0];
    let rep = ex.store.struct_value(site.sid);
    match rep.name {
        Some(name) => {
            let _ = write!(out, "{}", ex.program.name_string(name));
        }
        None => {
            let _ = out.write_str("UnamedStruct");
        }
    }

    if site.lifetimes.is_empty() && site.generics.is_empty() {
        return;
    }

    let _ = out.write_char('[');
    let mut wrote_any = false;
    for _ in &site.lifetimes {
        if wrote_any {
            let _ = out.write_str(", ");
        }
        wrote_any = true;
        let _ = out.write_str("\'idx");
    }
    for input in site.generics.iter().copied() {
        if wrote_any {
            let _ = out.write_str(", ");
        }
        wrote_any = true;
        write_mock_type_from_cluster(ex, core, extra, input, out, limit);
    }
    let _ = out.write_char(']');
}

fn write_tuple_mock_string_inner(
    ex: &ExternState<'_>,
    core: &mut TypeCore,
    extra: &TypeExtra,
    tuple: TupleInferId,
    out: &mut String,
    limit: &mut usize,
) {
    let _ = out.write_char('(');
    for (i, item) in extra.tuple_infers[tuple.0]
        .items
        .iter()
        .copied()
        .enumerate()
    {
        if i > 0 {
            let _ = out.write_str(", ");
        }
        write_mock_type_from_cluster(ex, core, extra, item, out, limit);
    }
    let _ = out.write_char(')');
}

fn write_array_mock_string_inner(
    ex: &ExternState<'_>,
    core: &mut TypeCore,
    extra: &TypeExtra,
    element: CId,
    size: ArrayType,
    out: &mut String,
    limit: &mut usize,
) {
    let _ = out.write_char('[');
    write_mock_type_from_cluster(ex, core, extra, element, out, limit);
    if let ArrayType::Sized(n) = size {
        let _ = out.write_char(';');
        let _ = write!(out, "{n}");
    }
    let _ = out.write_char(']');
}

fn write_ptr_mock_string_inner(
    ex: &ExternState<'_>,
    core: &mut TypeCore,
    extra: &TypeExtra,
    tgt: CId,
    kind: PtrKind,
    mutable: Option<bool>,
    out: &mut String,
    limit: &mut usize,
) {
    let PtrKind::Solved(style) = kind else {
        let _ = out.write_str("&? ");

        match mutable {
            Some(true) => {
                let _ = out.write_str("mut ");
            }
            Some(false) => {
                let _ = out.write_str("const ");
            }
            None => {
                let _ = out.write_str("mut? ");
            }
        }

        write_mock_type_from_cluster(ex, core, extra, tgt, out, limit);
        return;
    };

    match style {
        // raw nullable pointer (*const / *mut)
        PointerStyle::Raw(Nullable::Yes) => {
            match mutable {
                Some(true) => {
                    let _ = out.write_str("* ");
                }
                Some(false) => {
                    let _ = out.write_str("*const ");
                }
                None => {
                    let _ = out.write_str("*mut? ");
                }
            }
            write_mock_type_from_cluster(ex, core, extra, tgt, out, limit);
        }

        // non-null raw reference (&'raw)
        PointerStyle::Raw(Nullable::No) => {
            let _ = out.write_str("&'raw ");

            match mutable {
                Some(true) => {
                    let _ = out.write_str("");
                }
                Some(false) => {
                    let _ = out.write_str("const ");
                }
                None => {
                    let _ = out.write_str("mut? ");
                }
            }

            write_mock_type_from_cluster(ex, core, extra, tgt, out, limit);
        }

        // normal reference (&'a)
        PointerStyle::Ref(lt) => {
            let _ = out.write_char('&');
            let _ = out.write_char('\'');
            write_lifetime_for_display(ex, out, lt);
            let _ = out.write_char(' ');

            match mutable {
                Some(true) => {
                    let _ = out.write_str("mut ");
                }
                Some(false) => { /* shared ref, print nothing */ }
                None => {
                    let _ = out.write_str("const? ");
                }
            }

            write_mock_type_from_cluster(ex, core, extra, tgt, out, limit);
        }
    }
}

pub(crate) fn extract_clash_type_string(
    ex: &ExternState<'_>,
    core: &mut TypeCore,
    extra: &TypeExtra,
    cid: CId,
    limit: &mut usize,
) -> Option<String> {
    if *limit == 0 {
        return Some("...".to_string());
    }

    let root = find_root(&mut core.parent, cid);
    if matches!(core.cluster[root].state, ResolveKind::Nothing) {
        return None;
    }

    let mut out = String::new();
    write_mock_type_from_cluster(ex, core, extra, cid, &mut out, limit);
    Some(out)
}

// ============================================================
// Specialization (monomorphisation into local clusters)
// ============================================================

pub(crate) struct SpecializeCtx<'a> {
    pub(crate) generics: &'a [CId],
    pub(crate) lifetimes: &'a [LId],
    pub(crate) loc: ValId,
}

impl<'a> SpecializeCtx<'a> {
    fn new(generics: &'a [CId], lifetimes: &'a [LId], loc: ValId) -> Self {
        Self {
            generics,
            lifetimes,
            loc,
        }
    }
}

fn specialize_lifetime(types: &mut TypeState, ctx: &mut SpecializeCtx<'_>, lt: LifeTime) -> LId {
    match lt {
        LifeTime::Static => types.new_lid_known(LifeTime::Static),
        LifeTime::External(i) => *ctx.lifetimes.get(i as usize).unwrap(),
        _ => {
            debug_assert!(false, "bad lifetime");
            types.new_lid()
        }
    }
}

fn specialize_type_inner(
    ex: &mut ExternState,
    types: &mut TypeState,
    ty: TypeId,
    ctx: &mut SpecializeCtx<'_>,
) -> CId {
    match ex.store.type_value(ty) {
        TypeValue::Generic(id) => ctx.generics.get(id.0).copied().unwrap(),

        TypeValue::Func {
            calling_convention,
            generics: _,
            lifetimes: _,
            params,
            ret,
        } => {
            let ret = *ret;
            let calling_convention = *calling_convention;

            let mut inputs = Vec::with_capacity(params.len());
            for i in 0..params.len() {
                let TypeValue::Func { params, .. } = ex.store.type_value(ty) else {
                    unreachable!()
                };
                inputs.push(specialize_type_inner(ex, types, params[i], ctx))
            }

            let output = specialize_type_inner(ex, types, ret, ctx);

            // create FuncInfer
            let call_id = FuncInferId(types.extra.func_defs.len());
            types.extra.func_defs.push(FuncInfer {
                generics: 0,
                lifetimes: 0,
                inputs,
                output,
                calling_convention,
            });

            let id = CId(types.core.parent.len());
            types.core.parent.0.push(id);
            types.core.cluster.0.push(Cluster {
                state: ResolveKind::Func(call_id),
            });
            id
        }

        TypeValue::Struct {
            id,
            generics,
            lifetimes,
        } => {
            let id = *id;
            if generics.is_empty() && lifetimes.is_empty() {
                let idc = CId(types.core.parent.len());
                types.core.parent.0.push(idc);
                types.core.cluster.0.push(Cluster {
                    state: ResolveKind::Solved(ty),
                });
                return idc;
            }
            let glen = generics.len();
            let llen = lifetimes.len();

            let mut resolved = Vec::with_capacity(glen);
            for i in 0..glen {
                let TypeValue::Struct { generics, .. } = ex.store.type_value(ty) else {
                    unreachable!()
                };
                resolved.push(specialize_type_inner(ex, types, generics[i], ctx))
            }
            let mut resolved_lifetimes = Vec::with_capacity(llen);
            for i in 0..llen {
                let TypeValue::Struct { lifetimes, .. } = ex.store.type_value(ty) else {
                    unreachable!()
                };
                resolved_lifetimes.push(specialize_lifetime(types, ctx, lifetimes[i]))
            }

            let call_id = StructInferId(types.extra.struct_infers.len());
            types.extra.struct_infers.push(StructInfer {
                sid: id,
                generics: resolved,
                lifetimes: resolved_lifetimes,
            });

            let idc = CId(types.core.parent.len());
            types.core.parent.0.push(idc);
            types.core.cluster.0.push(Cluster {
                state: ResolveKind::Struct(call_id),
            });
            idc
        }

        TypeValue::Ptr {
            tgt,
            style,
            mutable,
        } => {
            let tgt = *tgt;
            let mutable = *mutable;
            let style = *style;
            let target = specialize_type_inner(ex, types, tgt, ctx);
            let kind = match style {
                PointerStyle::Ref(lt) => {
                    let lid = specialize_lifetime(types, ctx, lt);
                    PtrKind::RefInfer(lid)
                }
                x => PtrKind::Solved(x),
            };

            let id = CId(types.core.parent.len());
            types.core.parent.0.push(id);
            types.core.cluster.0.push(Cluster {
                state: ResolveKind::Ptr {
                    tgt: target,
                    kind,
                    mutable: Some(mutable),
                },
            });
            id
        }

        TypeValue::Tuple(items) => {
            let mut new_items = Vec::with_capacity(items.len());
            for i in 0..items.len() {
                let TypeValue::Tuple(items) = ex.store.type_value(ty) else {
                    unreachable!()
                };
                new_items.push(specialize_type_inner(ex, types, items[i], ctx));
            }

            let tuple_id = TupleInferId(types.extra.tuple_infers.len());
            types
                .extra
                .tuple_infers
                .push(TupleInfer { items: new_items });

            let id = CId(types.core.parent.len());
            types.core.parent.0.push(id);
            types.core.cluster.0.push(Cluster {
                state: ResolveKind::Tuple(tuple_id),
            });
            id
        }

        TypeValue::Array(inner, len) => {
            let inner = *inner;
            let len = *len;
            let inner = specialize_type_inner(ex, types, inner, ctx);

            let id = CId(types.core.parent.len());
            types.core.parent.0.push(id);
            types.core.cluster.0.push(Cluster {
                state: ResolveKind::Array {
                    element: inner,
                    size: len,
                },
            });
            id
        }

        TypeValue::Builtin(_) => {
            let id = CId(types.core.parent.len());
            types.core.parent.0.push(id);
            types.core.cluster.0.push(Cluster {
                state: ResolveKind::Solved(ty),
            });
            id
        }
    }
}

#[inline(always)]
pub(crate) fn specialize_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    ty: TypeId,
    generics: &[CId],
    lifetimes: &[LId],
    loc: ValId,
) -> CId {
    let mut ctx = SpecializeCtx::new(generics, lifetimes, loc);
    specialize_type_inner(ex, types, ty, &mut ctx)
}

// ===================================
// Constraint gathering (alias where possible)
// ===================================

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedStructDerefMethod {
    pub(crate) self_param: CId,
    pub(crate) self_kind: PtrKind,
    pub(crate) self_mutable: Option<bool>,
    pub(crate) target: CId,
    pub(crate) ret_kind: PtrKind,
    pub(crate) ret_mutable: Option<bool>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedStructDerefTarget {
    pub(crate) target: CId,
    pub(crate) deref_receiver_ptr: CId,
    pub(crate) deref_result_ptr: CId,
}

#[derive(Debug)]
pub(crate) enum MemberAccessResolve {
    Resolved {
        result: CId,
        implicit_receivers: Vec<CId>,
    },
    Pending {
        source: CId,
    },
    Error(TypeError),
}

#[derive(Debug)]
pub(crate) enum IntAccessResolve {
    Resolved {
        result: CId,
        implicit_receivers: Vec<CId>,
    },
    Pending {
        source: CId,
    },
    Error(TypeError),
}

// ///this tries to resolve specifically a from a module.
// ///if what we have is a member of a struct it wont give a name
// fn try_func_and_member<G:GlobalHandler>(ctx: &mut InferState, v: ValId)->(CId,NameId){
//     match ctx.program.value(v){
//         Value::NameRef(n)=>Some(n),
//         Value::Access { base: _, name: _, kind: _ }=>todo!{},
//         _ => {
//             None
//         }
//     }
// }

#[inline(always)]
pub(crate) fn gather_pattern_constraints(ctx: &mut InferState, p: PatId) -> CId {
    gather_pattern_constraints_with_generics::<false>(ctx, p)
}

#[inline(always)]
pub(crate) fn gather_pattern_constraints_and_name(
    ctx: &mut InferState,
    p: PatId,
) -> (CId, Option<NameId>) {
    gather_pattern_constraints_and_name_with_generics::<false>(ctx, p)
}

pub(crate) fn pattern_bind_name(program: &Program, p: PatId) -> Option<NameId> {
    match program.pattern(p) {
        Pattern::Bind(n, _) => Some(n),
        Pattern::TypeAnnotation { pat, .. } => pattern_bind_name(program, pat),
        _ => None,
    }
}

#[inline(always)]
pub(crate) fn gather_pattern_constraints_with_generics<const GLOBAL_SCOPE: bool>(
    ctx: &mut InferState,
    p: PatId,
) -> CId {
    let (x, _) = gather_pattern_constraints_and_name_with_generics::<GLOBAL_SCOPE>(ctx, p);
    x
}

fn gather_pattern_constraints_and_name_with_generics<const GLOBAL_SCOPE: bool>(
    ctx: &mut InferState,
    p: PatId,
) -> (CId, Option<NameId>) {
    match ctx.ex.program.pattern(p) {
        Pattern::Wildcard(_) => {
            let c = ctx.new_cluster();
            ctx.bind_pat(p, c);
            (c, None)
        }
        Pattern::Bind(n, _) => {
            let c = ctx.new_cluster();
            ctx.search.names.insert(n, c);
            ctx.bind_pat(p, c);
            (c, Some(n))
        }

        Pattern::AddrOf(base, kind) => {
            let (tgt, n) =
                gather_pattern_constraints_and_name_with_generics::<GLOBAL_SCOPE>(ctx, base);
            let mutable = matches!(kind, VarKind::Mut);
            let c = ctx.new_cluster();
            ctx.types.core.cluster[c].state = ResolveKind::Ptr {
                mutable: Some(mutable),
                kind: PtrKind::SafeRef,
                tgt,
            };
            ctx.bind_pat(p, c);
            (c, n)
        }

        Pattern::TypeAnnotation { pat, ty } => {
            let (c, n) =
                gather_pattern_constraints_and_name_with_generics::<GLOBAL_SCOPE>(ctx, pat);
            let t = if GLOBAL_SCOPE {
                compile_signature_type_expr(ctx, ty)
            } else {
                compile_type_expr(ctx, ty)
            };

            if let Err(clash) = ctx.unify(c, t) {
                ctx.push_error(TypeError::PatternAnnotationMismatch {
                    annotation: p,
                    constrained: pat,
                    clash,
                });
            }

            ctx.bind_pat(p, c);
            (c, n)
        }

        Pattern::Tuple(items) => {
            let item_clusters = items
                .ids()
                .map(|item| gather_pattern_constraints_with_generics::<GLOBAL_SCOPE>(ctx, item))
                .collect::<Vec<_>>();
            let tuple = ctx.new_tuple_instance(item_clusters);
            ctx.bind_pat(p, tuple);
            (tuple, None)
        }

        _ => todo!(),
    }
}

#[inline(always)]
pub(crate) fn compile_type_expr(ctx: &mut InferState, texpr: TExpId) -> CId {
    compile_type_expr_with_mode(ctx, texpr, TypeExprCompileMode::Local)
}

#[inline(always)]
pub(crate) fn compile_signature_type_expr(ctx: &mut InferState, texpr: TExpId) -> CId {
    compile_type_expr_with_mode(ctx, texpr, TypeExprCompileMode::Signature)
}

pub(crate) fn get_type_name(prog: &Program, t: TExpId) -> Option<NameId> {
    match prog.type_expr(t) {
        TypeExpr::NameRef(n) => Some(n),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StructOperatorOverload {
    pub(crate) method_type: TypeId,
    #[allow(dead_code)]
    pub(crate) method_site: ValId,
    #[allow(dead_code)]
    pub(crate) self_pointer_style: Option<PointerStyle>,
}

// ===================================
// middle phase
// ===================================

#[inline(always)]
pub(crate) fn cluster_is_int_like(
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
        ResolveKind::Struct(_) => Some(false),
        ResolveKind::Tuple(_) => Some(false),
        ResolveKind::Array { .. } => Some(false),
        ResolveKind::Ptr { .. } => Some(false),
        ResolveKind::Nothing => None,
    }
}

#[inline(always)]
pub(crate) fn cluster_is_float_like(
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
        ResolveKind::Struct(_) => Some(false),
        ResolveKind::Tuple(_) => Some(false),
        ResolveKind::Array { .. } => Some(false),
        ResolveKind::Ptr { .. } => Some(false),
        ResolveKind::Nothing => None,
    }
}

#[inline(always)]
pub(crate) fn cluster_is_bool(
    store: &TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    cid: CId,
) -> Option<bool> {
    let root = find_root(parent, cid);
    match cluster[root].state {
        ResolveKind::Solved(t) => Some(store.as_builtin(t) == Some(BuiltinType::Bool)),
        ResolveKind::IntLike => Some(false),
        ResolveKind::FloatLike => Some(false),
        ResolveKind::Func(_) => Some(false),
        ResolveKind::Struct(_) => Some(false),
        ResolveKind::Tuple(_) => Some(false),
        ResolveKind::Array { .. } => Some(false),
        ResolveKind::Ptr { .. } => Some(false),
        ResolveKind::Nothing => None,
    }
}

/// Unify only if roots differ; report whether a merge happened.
#[inline]
pub(crate) fn unify_if_distinct(
    ex: &mut ExternState,
    types: &mut TypeState,
    a: CId,
    b: CId,
) -> Result<bool, TypeClash> {
    let ra = types.root(a);
    let rb = types.root(b);

    if ra == rb {
        return Ok(false);
    }

    unify_clusters_inlined(ex, types, ra, rb)?;
    Ok(true)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolveOutcome {
    pub(crate) progress: bool,
    pub(crate) retain: bool,
}

impl ResolveOutcome {
    #[inline(always)]
    pub(crate) fn keep(progress: bool) -> Self {
        Self {
            progress,
            retain: true,
        }
    }

    #[inline(always)]
    pub(crate) fn drop(progress: bool) -> Self {
        Self {
            progress,
            retain: false,
        }
    }
}

pub(crate) const OP_OVERLOAD_SIGNATURE_MISMATCH: &str =
    "operator overload arguments and result must match overload signature";

#[derive(Debug)]
pub(crate) struct ResolvedMemberOverload {
    pub(crate) params: Vec<CId>,
    pub(crate) ret: CId,
    pub(crate) full_method: CId,
}

pub(crate) fn resolve_deferred_types(ctx: &mut InferState) -> bool {
    let mut change = false;
    for cid in (0..ctx.types.core.cluster.len()).map(CId) {
        if ctx.types.core.parent[cid] != cid {
            continue;
        }
        let resolved = match ctx.types.core.cluster[cid].state {
            ResolveKind::Func(call) => try_resolve_func_type(&mut ctx.ex, &mut ctx.types, call),
            ResolveKind::Struct(call) => try_resolve_struct_type(&mut ctx.ex, &mut ctx.types, call),
            ResolveKind::Tuple(call) => try_resolve_tuple_type(&mut ctx.ex, &mut ctx.types, call),
            ResolveKind::Array { element, size } => {
                try_resolve_array_type(&mut ctx.ex, &mut ctx.types, element, size)
            }
            ResolveKind::Ptr { tgt, kind, mutable } => {
                try_resolve_ptr_type(&mut ctx.ex, &mut ctx.types, tgt, kind, mutable)
            }
            _ => None,
        };

        if let Some(t) = resolved {
            ctx.types.core.cluster[cid].state = ResolveKind::Solved(t);
            change = true;
        }
    }
    change
}

pub(crate) fn resolve_pending_specializations(ctx: &mut InferState) -> bool {
    let mut change = false;

    let types = &mut ctx.types;
    let ex = &mut ctx.ex;

    ctx.req.pending_specializations.retain_mut(|p| {
        let Some(base_type) = ex.ans.typedef_types.get(&p.global).copied() else {
            return true;
        };

        let sid = match ex.store.type_value(base_type) {
            TypeValue::Struct { id: sid, .. } => *sid,
            _ => {
                let loc = ex.program.type_expr_loc(p.global);
                ex.push_error(TypeError::Simple {
                    loc,
                    message: "only struct types can be specialized",
                });
                change = true;
                return false;
            }
        };

        let expected = ex.store.struct_value(sid).gen_count;
        if p.generics.len() != expected {
            let loc = ex.program.type_expr_loc(p.global);
            ex.push_error(TypeError::Simple {
                loc,
                message: "wrong number of generic arguments for struct type",
            });
            change = true;
            return false;
        }

        let expected_lifetimes = match ex.store.type_value(base_type) {
            TypeValue::Struct { lifetimes, .. } => lifetimes.len(),
            _ => unreachable!(),
        };

        if p.lifetimes.is_empty() && expected_lifetimes != 0 {
            p.lifetimes = (0..expected_lifetimes).map(|_| types.new_lid()).collect();
        }

        if p.lifetimes.len() != expected_lifetimes {
            let loc = ex.program.type_expr_loc(p.global);
            ex.push_error(TypeError::Simple {
                loc,
                message: "wrong number of lifetime arguments for struct type",
            });
            change = true;
            return false;
        }

        let found = types.new_struct_instance(sid, p.generics.clone(), p.lifetimes.clone());
        if let Err(clash) = types.unify(ex, found, p.output) {
            ex.push_error(TypeError::TypeClashBeforeMentioned {
                name: p.name,
                expr: p.global,
                clash,
            });
        }

        change = true;
        false
    });

    change
}

// fn report_unresolved(ctx: &mut InferState){
//     let mut roots = Vec::with_capacity(ctx.cluster.len());
//     for i in 0..ctx.cluster.len(){
//         let c = CId(i);
//         if c==ctx.types.root(c){
//             roots.push(c);
//         }
//     }
// }

#[cfg(test)]
mod type_infer_tests {
    use super::*;
    use crate::parsing::Parser;
    use crate::string_intern::*;
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
                    program.gather_definition(expr);
                }
                Ok(None) => break,
                Err(e) => panic!("parse error: {:?}", e),
            }
        }

        program.check_pending_names();

        if !program.lowering_errors.is_empty() {
            panic!("lowering errors: {:?}", program.lowering_errors);
        }

        program
    }

    /// Extract the implementation value of the *single* function in the program.
    fn extract_single_fn(program: &Program) -> ValId {
        program
            .definitions
            .iter()
            .find_map(|(_, def)| match def {
                Defined::Func(funcs) => funcs.implementations.first().copied(),
                _ => None,
            })
            .expect("expected a function implementation")
    }

    fn find_value_by_name(program: &Program, name: &str) -> ValId {
        program
            .definitions
            .iter()
            .find_map(|(n, def)| match def {
                Defined::Func(funcs) if program.name_string(*n) == name => {
                    funcs.implementations.first().copied()
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("implementation `{}` not found", name))
    }

    fn find_declaration_by_name(program: &Program, name: &str) -> ValId {
        program
            .definitions
            .iter()
            .find_map(|(n, def)| match def {
                Defined::Func(funcs) if program.name_string(*n) == name => {
                    funcs.declarations.first().copied()
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("declaration `{}` not found", name))
    }

    fn extract_bind_name(program: &Program, pat: PatId) -> Option<NameId> {
        match program.pattern(pat) {
            Pattern::Bind(n, _) => Some(n),
            Pattern::TypeAnnotation { pat, ty: _ } => extract_bind_name(program, pat),
            _ => None,
        }
    }

    fn find_let_stmt_type(
        program: &Program,
        solved: &SolvedTypes,
        func: ValId,
        name: &str,
    ) -> TypeId {
        let Value::Func { body, .. } = program.value(func) else {
            panic!("expected function value")
        };
        let body = body.expect("expected function body");
        let Value::Block {
            statements,
            return_value: _,
        } = program.value(body)
        else {
            panic!("expected block body")
        };

        for stmt in statements.ids() {
            let Value::Let { pat, .. } = program.value(stmt) else {
                continue;
            };
            let Some(n) = extract_bind_name(program, pat) else {
                continue;
            };
            if program.name_string(n) == name {
                return solved
                    .inner_value_type(func, stmt)
                    .unwrap_or_else(|| panic!("missing type for let `{}`", name));
            }
        }

        panic!("let binding `{}` not found", name)
    }

    fn find_let_stmt_value(program: &Program, func: ValId, name: &str) -> ValId {
        let Value::Func { body, .. } = program.value(func) else {
            panic!("expected function value")
        };
        let body = body.expect("expected function body");
        let Value::Block {
            statements,
            return_value: _,
        } = program.value(body)
        else {
            panic!("expected block body")
        };

        for stmt in statements.ids() {
            let Value::Let { pat, value, .. } = program.value(stmt) else {
                continue;
            };
            let Some(n) = extract_bind_name(program, pat) else {
                continue;
            };
            if program.name_string(n) == name {
                return value;
            }
        }

        panic!("let binding `{}` not found", name)
    }

    fn find_typedef_type_by_name(program: &Program, solved: &SolvedTypes, name: &str) -> TypeId {
        let texp = program
            .definitions
            .iter()
            .find_map(|(n, def)| match def {
                Defined::Type(texp) if program.name_string(*n) == name => Some(*texp),
                _ => None,
            })
            .unwrap_or_else(|| panic!("type `{}` not found", name));
        solved
            .typedef_types
            .get(&texp)
            .copied()
            .unwrap_or_else(|| panic!("type `{}` did not resolve", name))
    }

    fn find_signature_param_type(solved: &SolvedTypes, function: ValId, pat: PatId) -> TypeId {
        solved
            .function_types_by_value(function)
            .and_then(|f| {
                f.arguments
                    .iter()
                    .find_map(|(param_pat, _, ty)| (*param_pat == pat).then_some(*ty))
            })
            .unwrap_or_else(|| panic!("missing signature parameter type for pattern {:?}", pat))
    }

    fn find_member_access_and_result_types(
        program: &Program,
        solved: &SolvedTypes,
        func: ValId,
        name: &str,
    ) -> (ValId, TypeId, TypeId) {
        let Value::Func { body, .. } = program.value(func) else {
            panic!("expected function value")
        };
        let body = body.expect("expected function body");
        let Value::Block {
            statements,
            return_value: _,
        } = program.value(body)
        else {
            panic!("expected block body")
        };

        for stmt in statements.ids() {
            let Value::Let { pat, value, .. } = program.value(stmt) else {
                continue;
            };
            let Some(n) = extract_bind_name(program, pat) else {
                continue;
            };
            if program.name_string(n) != name {
                continue;
            }

            let Value::Call(call) = program.value(value) else {
                panic!("expected let value to be a call")
            };
            let Value::Access { .. } = program.value(call.base) else {
                panic!("expected call base to be member access")
            };

            let access_ty = solved
                .inner_value_type(func, call.base)
                .unwrap_or_else(|| panic!("missing type for member access in `{}`", name));
            let result_ty = solved
                .inner_value_type(func, stmt)
                .unwrap_or_else(|| panic!("missing type for let statement `{}`", name));
            return (call.base, access_ty, result_ty);
        }

        panic!("let binding `{}` not found", name)
    }

    fn implicit_deref_chain_type_strings(
        program: &Program,
        store: &TypeStore,
        solved: &SolvedTypes,
        function: ValId,
        site: ValId,
    ) -> Option<Vec<String>> {
        solved
            .implicit_deref_chain_in_function(function, site)
            .map(|chain| {
                chain
                    .iter()
                    .map(|t| store.get_type_string(program, *t))
                    .collect()
            })
    }

    /// Run inference on a single function body.
    fn infer_fn(src: &str, store: &mut TypeStore) -> Result<TypeId, Vec<TypeError>> {
        let program = gather_program(src);
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, store, &mut solved_types)?;
        let f = extract_single_fn(&program);
        let body = match program.value(f) {
            Value::Func { body, .. } => body.expect("expected function body"),
            _ => panic!("expected function value"),
        };

        let solved_types = infer_value_internals(&program, store, &mut solved_types, f)?;
        Ok(solved_types.inner_value_type(f, body).unwrap())
    }

    //this is a hack for just testing
    fn infer_fn_body(src: &str, store: &mut TypeStore) -> Result<TypeId, Vec<TypeError>> {
        let program = gather_program(src);
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, store, &mut solved_types)?;

        let f = extract_single_fn(&program);
        let body = match program.value(f) {
            Value::Func { body, .. } => body.expect("expected function body"),
            _ => panic!("expected function value"),
        };

        let solved_types = _infer_value_hacky(&program, store, &mut solved_types, body)?;
        Ok(solved_types.inner_value_type(body, body).unwrap())
    }

    fn infer_global_errs(src: &str) -> Vec<TypeError> {
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        match infer_global_types(&program, &mut store, &mut solved_types) {
            Ok(_) => panic!("expected global type inference to fail"),
            Err(errs) => errs,
        }
    }

    fn assert_has_simple_error(errs: &[TypeError], message: &'static str) {
        assert!(
            errs.iter().any(|err| matches!(
                err,
                TypeError::Simple {
                    message: found_message,
                    ..
                } if *found_message == message
            )),
            "expected simple error `{message}`, got {errs:?}"
        );
    }

    fn assert_fn_body_simple_error(src: &str, message: &'static str) {
        let mut store = TypeStore::new();
        let errs = infer_fn_body(src, &mut store).unwrap_err();
        assert_has_simple_error(&errs, message);
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
    fn tuple_expression_and_annotation_typecheck() {
        let src = "f = fn(){ let t:(int, float) = (1:int, 2.0:float); t }";
        let mut store = TypeStore::new();
        let ty = infer_fn_body(src, &mut store).unwrap();
        let TypeValue::Tuple(items) = store.type_value(ty) else {
            panic!("expected tuple type")
        };
        assert_eq!(items.len(), 2);
        assert!(matches!(
            store.type_value(items[0]),
            TypeValue::Builtin(BuiltinType::Int)
        ));
        assert!(matches!(
            store.type_value(items[1]),
            TypeValue::Builtin(BuiltinType::F64)
        ));
    }

    #[test]
    fn tuple_pattern_destructure_typechecks() {
        assert_fn_type!(
            "f = fn(){ let (x, y):(int, float) = (1:int, 2.0:float); x }",
            BuiltinType::Int
        );
    }

    #[test]
    fn array_expression_typechecks_and_tracks_length() {
        let src = "f = fn(){ [1:int, 2:int, 3:int] }";
        let mut store = TypeStore::new();
        let ty = infer_fn_body(src, &mut store).unwrap();
        let TypeValue::Array(item, size) = *store.type_value(ty) else {
            panic!("expected array type")
        };
        assert_eq!(size, ArrayType::Sized(3));
        assert!(matches!(
            store.type_value(item),
            TypeValue::Builtin(BuiltinType::Int)
        ));
    }

    #[test]
    fn array_assignment_with_different_length_fails() {
        let mut store = TypeStore::new();
        let errs = infer_fn_body(
            "f = fn(){ let a = [1:int, 2:int]; a = [3:int]; }",
            &mut store,
        )
        .unwrap_err();
        assert!(errs.iter().any(|err| matches!(
            err,
            TypeError::ValuesContradict {
                expectation_reason: "assignment requires both sides match",
                ..
            }
        )));
    }

    #[test]
    fn sized_array_type_expression_typechecks() {
        let src = "f = fn(){ let a:[int;3] = [1:int, 2:int, 3:int]; a }";
        let mut store = TypeStore::new();
        let ty = infer_fn_body(src, &mut store).unwrap();
        let TypeValue::Array(item, size) = *store.type_value(ty) else {
            panic!("expected array type")
        };
        assert_eq!(size, ArrayType::Sized(3));
        assert!(matches!(
            store.type_value(item),
            TypeValue::Builtin(BuiltinType::Int)
        ));
    }

    #[test]
    fn unsized_array_type_expression_resolves_in_global_types() {
        let src = "type A = [int]; f = fn(){}";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let ty = find_typedef_type_by_name(&program, &solved_types, "A");
        let TypeValue::Array(item, size) = *store.type_value(ty) else {
            panic!("expected array typedef")
        };
        assert_eq!(size, ArrayType::Unsized);
        assert!(matches!(
            store.type_value(item),
            TypeValue::Builtin(BuiltinType::Int)
        ));
    }

    #[test]
    fn array_index_expression_typechecks() {
        assert_fn_type!(
            "f = fn(){ let a = [1:int, 2:int, 3:int]; a[0:usize] }",
            BuiltinType::Int
        );
    }

    #[test]
    fn pointer_to_array_index_expression_typechecks() {
        assert_fn_type!(
            "f = fn(){ let a:[int;2] = [1:int, 2:int]; let p:*[int;2] = &a; p[1:usize] }",
            BuiltinType::Int
        );
    }

    #[test]
    fn struct_deref_to_array_index_expression_typechecks() {
        let src = "Box=struct['a]{inner:&'a [int;2]}; Box.__deref_mut = fn['a](self:&mut Box['a])->&mut &'a [int;2] { &mut self.inner }; f = fn['rand,'a](b:Box['a],random:&'rand int)->int { let y:int = b[1:usize]; y };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let y_ty = find_let_stmt_type(&program, &solved_types, f, "y");
        assert!(matches!(
            store.type_value(y_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));

        let index_site = find_let_stmt_value(&program, f, "y");
        let chain = solved_types
            .index_implicit_deref_chain(index_site)
            .map(|chain| {
                chain
                    .iter()
                    .map(|t| store.get_type_string(&program, *t))
                    .collect::<Vec<_>>()
            })
            .expect("expected implicit deref chain for index site");
        assert_eq!(
            chain,
            vec![
                "Box['a1]".to_string(),
                "&'idk0 mut Box['a1]".to_string(),
                "&'idk0 mut &'a1 [int;2]".to_string(),
                "&'a1 [int;2]".to_string(),
                "[int;2]".to_string(),
            ],
            "unexpected implicit deref chain for struct-deref indexing"
        );
    }
    const BOX_EXAMPLE: &str = r#"
        Box = struct[T]{ptr:&'raw T};

        free = cfn(p:*void);
        no_fail_alloc = cfn(s:usize)->*void;
        Box.new = fn[T](x:T)->Box[T] {
          let p=no_fail_alloc(x.__size_of());
          Box{p as &'raw _}
        }
        Box.__free = fn[T](b:&mut Box[T]){
        (*b.ptr).__free()
        free(b->ptr as *void)
        }


        Box.__deref = fn[T](b:&const Box[T])->&T{&*b.ptr}
        Box.__deref_mut = fn[T](b:&mut Box[T])->&mut T{&*b.ptr}

        f=fn(x:int)->Box[int] { Box::new(x) };
        "#;

    #[test]

    //currently fails over not doing places in f
    fn generic_box_array_index_chain_includes_box_step() {
        let source = r#"
        Box = struct[T]{ptr:&'raw T};

        free = cfn(p:*void);
        no_fail_alloc = cfn(s:usize)->*void;
        Box.new = fn[T](x:T)->Box[T] {
          let p=no_fail_alloc(x.__size_of());
          Box{p as &'raw _}
        }
        Box.__free = fn[T](b:&mut Box[T]){
        (&*b.ptr).__free()
        free(b->ptr as *void)
        }


        Box.__deref = fn[T](b:&const Box[T])->&T{&*b.ptr}
        Box.__deref_mut = fn[T](b:&mut Box[T])->&mut T{&*b.ptr}

        f=fn(b:Box[[int]])->int { let y:int = b[0]; y };
            
        "#;
        let program = gather_program(source);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let index_site = find_let_stmt_value(&program, f, "y");
        let chain = solved_types
            .index_implicit_deref_chain(index_site)
            .map(|chain| {
                chain
                    .iter()
                    .map(|t| store.get_type_string(&program, *t))
                    .collect::<Vec<_>>()
            })
            .expect("expected implicit deref chain for index site");
        assert_eq!(
            chain,
            vec![
                "Box[[int]]".to_string(),
                "&'idk0 Box[[int]]".to_string(),
                "&'idk0 [int]".to_string(),
                "[int]".to_string(),
            ],
            "unexpected implicit deref chain for generic Box indexing"
        );
    }

    #[test]
    fn readme_style_free_and_user_free_box_example_typechecks_exact_snippet() {
        let program = gather_program(BOX_EXAMPLE);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
    }

    #[test]
    fn array_index_requires_usize() {
        let mut store = TypeStore::new();
        let errs =
            infer_fn_body("f = fn(){ let a = [1:int, 2:int]; a[0:int] }", &mut store).unwrap_err();
        assert!(errs.iter().any(|err| matches!(
            err,
            TypeError::ValuesContradict {
                expectation_reason: "array indexing requires an index of type usize",
                ..
            }
        )));
    }

    #[test]
    fn if_condition_bool_mismatch_reports_found_pointer_expected_bool() {
        let src = "f = fn(){ let x:int = 1; if &x { 1:int } else { 2:int } }";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let f = find_value_by_name(&program, "f");
        let errs = infer_value_internals(&program, &mut store, &mut solved_types, f)
            .err()
            .unwrap_or_default();

        let clash = errs.iter().find_map(|err| match err {
            TypeError::ValuesContradict {
                expectation_reason: "if condition must be bool",
                clash,
                ..
            } => Some(clash),
            _ => None,
        });
        let clash = clash.expect("expected if-condition type mismatch");

        let found = clash.found().expect("missing found type");
        let wanted = clash.wanted().expect("missing expected type");

        assert!(found.starts_with('&') || found.starts_with('*'));
        assert_eq!(wanted, "bool");
    }

    #[test]
    fn calling_non_function_reports_found_function_expected_target_type() {
        let src = "f = fn(){ let x:int = 1; x(2:int) }";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let f = find_value_by_name(&program, "f");
        let errs = infer_value_internals(&program, &mut store, &mut solved_types, f)
            .err()
            .unwrap_or_default();

        let clash = errs.iter().find_map(|err| match err {
            TypeError::ValuesContradict {
                expectation_reason: "called function with wrong signature",
                clash,
                ..
            } => Some(clash),
            _ => None,
        });
        let clash = clash.expect("expected call-signature mismatch");

        let found = clash.found().expect("missing found type");
        let wanted = clash.wanted().expect("missing expected type");

        assert!(found.starts_with("fn") || found.starts_with("cfn"));
        assert_eq!(wanted, "int");
    }

    #[test]
    fn annotation_mismatch_reports_found_tuple_expected_annotation_type() {
        let src = "f = fn(){ let t = (1:int, 2:int); t : int }";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let f = find_value_by_name(&program, "f");
        let errs = infer_value_internals(&program, &mut store, &mut solved_types, f)
            .err()
            .unwrap_or_default();

        let clash = errs.iter().find_map(|err| match err {
            TypeError::AnnotationMismatch { clash, .. } => Some(clash),
            _ => None,
        });
        let clash = clash.expect("expected annotation mismatch");

        let found = clash.found().expect("missing found type");
        let wanted = clash.wanted().expect("missing expected type");

        assert!(found.starts_with('('));
        assert_eq!(wanted, "int");
    }

    #[test]
    fn pointer_type_expr_forms_resolve() {
        let src = r#"
            f = fn(x:int) {
                let a:&int = &x;
                let b:&mut int = &x;
                let c:*int = &x;
                let d:*mut int = &x;
                let e:*const int = &x;
            }
        "#;

        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let f = find_value_by_name(&program, "f");
        let solved_types =
            infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let a_ty = find_let_stmt_type(&program, &solved_types, f, "a");
        let b_ty = find_let_stmt_type(&program, &solved_types, f, "b");
        let c_ty = find_let_stmt_type(&program, &solved_types, f, "c");
        let d_ty = find_let_stmt_type(&program, &solved_types, f, "d");
        let e_ty = find_let_stmt_type(&program, &solved_types, f, "e");

        let is_int_ptr = |store: &TypeStore, ty: TypeId, raw: bool, mutable: bool| {
            let TypeValue::Ptr {
                tgt,
                style,
                mutable: got_mut,
            } = *store.type_value(ty)
            else {
                return false;
            };
            style.is_fancy() == !raw
                && got_mut == mutable
                && matches!(store.type_value(tgt), TypeValue::Builtin(BuiltinType::Int))
        };

        assert!(is_int_ptr(&store, a_ty, false, false));
        assert!(is_int_ptr(&store, b_ty, false, true));
        assert!(is_int_ptr(&store, c_ty, true, true));
        assert!(is_int_ptr(&store, d_ty, true, true));
        assert!(is_int_ptr(&store, e_ty, true, false));
    }

    #[test]
    fn infer_block_return() {
        assert_fn_type!("f = fn(){ { let x : usize = 1; x } }", BuiltinType::Usize);
    }

    #[test]
    fn explicit_return_statement_typechecks() {
        let mut store = TypeStore::new();
        let ty = infer_fn("f = fn()->int { return 1; 2 }", &mut store).unwrap();
        match store.type_value(ty) {
            TypeValue::Builtin(b) => assert_eq!(*b, BuiltinType::Int),
            other => panic!("expected builtin type, got {:?}", other),
        }
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

    #[test]
    fn infer_external_cfn_signature() {
        let src = "f = cfn(x:int)->int;";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_declaration_by_name(&program, "f");
        let f_ty = solved_types.type_of(f).expect("missing f type");
        let TypeValue::Func {
            calling_convention,
            generics: _,
            lifetimes: _,
            params,
            ret,
        } = store.type_value(f_ty)
        else {
            panic!("expected function type")
        };

        assert_eq!(*calling_convention, CallingConvention::C);
        assert_eq!(params.len(), 1);
        assert!(matches!(
            store.type_value(params[0]),
            TypeValue::Builtin(BuiltinType::Int)
        ));
        assert!(matches!(
            store.type_value(*ret),
            TypeValue::Builtin(BuiltinType::Int)
        ));
    }

    #[test]
    fn repeated_function_declarations_must_exactly_match_first() {
        let errs = infer_global_errs("f = fn(x:int)->int; f = fn(x:int)->str;");
        assert!(errs.iter().any(|e| {
            matches!(
                e,
                TypeError::ValuesContradict {
                    expectation_reason:
                        "all declarations must exactly match the first declaration signature",
                    ..
                }
            )
        }));
    }

    #[test]
    fn repeated_function_declaration_exact_match_is_allowed() {
        let src = "f = fn[T](x:T)->T; f = fn[T](x:T)->T;";
        let mut program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f_name = program.str_intern.intern("f");
        let f_id = *program
            .scopes
            .first()
            .and_then(|scope| scope.0.get(&f_name))
            .expect("missing f name id");

        let solved = solved_types
            .function_types_by_name(f_id)
            .expect("missing solved function types");

        let reference = solved.ty;
        assert_ne!(reference, UNKNOWN_TYPE);
    }

    #[test]
    fn function_implementation_must_exactly_match_declaration() {
        let errs = infer_global_errs("f = fn[T](x:T)->int; f = fn(x:int)->str { \"nope\" };");
        assert!(errs.iter().any(|e| {
            matches!(
                e,
                TypeError::ValuesContradict {
                    expectation_reason: "function implementation must exactly match the declared signature",
                    ..
                }
            )
        }));
    }

    #[test]
    fn duplicate_function_implementation_errors() {
        let errs = infer_global_errs(
            "f = fn[T](x:T)->T; f = fn(x:int)->int { x }; f = fn(x:int)->int { x };",
        );
        assert!(
            errs.iter()
                .any(|e| { matches!(e, TypeError::DuplicateFunctionImplementation { .. }) })
        );
    }

    #[test]
    fn calling_convention_mismatch_is_a_type_error() {
        let src = r#"
            a = fn(x:int)->int { x }
            b = cfn(x:int)->int;
            c = fn() { let x = a; x = b; }
        "#;

        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let c = find_value_by_name(&program, "c");
        let errs = infer_value_internals(&program, &mut store, &mut solved_types, c)
            .err()
            .unwrap_or_default();

        let clash = errs.iter().find_map(|err| match err {
            TypeError::ValuesContradict { clash, .. } => Some(clash),
            _ => None,
        });
        let clash = clash.expect("expected mismatch error");

        let mut saw_hot = false;
        let mut saw_c = false;
        for side in [clash.found(), clash.wanted()] {
            let Some(side) = side else {
                continue;
            };
            saw_hot |= side.starts_with("fn(") || side.starts_with("fn[");
            saw_c |= side.starts_with("cfn(") || side.starts_with("cfn[");
        }

        assert!(
            saw_hot && saw_c,
            "expected fn/cfn mismatch in clash payload"
        );
    }

    #[test]
    fn unknown_calling_convention_prints_fn_question_mark() {
        let program = Program::new();
        let mut store = TypeStore::new();
        let ty = store.intern(TypeValue::Func {
            calling_convention: CallingConvention::Unknown,
            generics: 0,
            lifetimes: 0,
            params: vec![BuiltinType::Int.into()],
            ret: BuiltinType::Int.into(),
        });

        assert_eq!(store.get_type_string(&program, ty), "fn?(int) -> int");
    }

    #[test]
    fn cstruct_layout_is_tracked_in_struct_representation() {
        let src = "type S = cstruct { x:int };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let s_ty = find_typedef_type_by_name(&program, &solved_types, "S");
        let TypeValue::Struct { id, .. } = store.type_value(s_ty) else {
            panic!("expected struct type")
        };
        assert_eq!(store.struct_value(*id).layout, StructLayoutSpec::C);
    }

    #[test]
    fn generic_function_solves_with_generics() {
        let src = "f = fn[T](x:T)->T { x }";
        let mut store = TypeStore::new();
        let program = gather_program(src);
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let f = extract_single_fn(&program);
        let f_ty = solved_types.type_of(f).unwrap();

        match store.type_value(f_ty) {
            TypeValue::Func {
                generics,
                params,
                ret,
                ..
            } => {
                assert_eq!(*generics, 1);
                assert_eq!(params.len(), 1);
                assert_eq!(params[0], *ret);
                match store.type_value(params[0]) {
                    TypeValue::Generic(gid) => assert_eq!(gid.0, 0),
                    other => panic!("expected generic param, got {:?}", other),
                }
            }
            other => panic!("expected func type, got {:?}", other),
        }
    }

    #[test]
    fn generic_function_prints_generic_arity_on_func_type() {
        let src = "f = fn[T](x:T)->T { x }";
        let mut store = TypeStore::new();
        let program = gather_program(src);
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let f = extract_single_fn(&program);
        let f_ty = solved_types.type_of(f).unwrap();

        assert_eq!(store.get_type_string(&program, f_ty), "fn[T0](T0) -> T0");
    }

    #[test]
    fn elided_output_lifetime_picks_single_implicit_input_lifetime() {
        let src = "f = fn['a0](y:&'a0 int, x:&int)->&int { x }";
        let mut store = TypeStore::new();
        let program = gather_program(src);
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let f = extract_single_fn(&program);
        let f_ty = solved_types.type_of(f).unwrap();

        assert_eq!(
            store.get_type_string(&program, f_ty),
            "fn['a0, 'a1](&'a0 int, &'a1 int) -> &'a1 int"
        );
    }

    #[test]
    fn function_lifetime_generic_annotation_is_preserved() {
        let src = "f = fn['a0](x:&'a0 int)->&'a0 int { x }";
        let mut store = TypeStore::new();
        let program = gather_program(src);
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let f = extract_single_fn(&program);
        let f_ty = solved_types.type_of(f).unwrap();

        assert_eq!(
            store.get_type_string(&program, f_ty),
            "fn['a0](&'a0 int) -> &'a0 int"
        );
    }

    #[test]
    fn generic_function_reports_unused_lifetime_indexes() {
        let errs = infer_global_errs("f = fn['a0](x:int)->int { x }");
        assert!(errs.iter().any(|err| matches!(
            err,
            TypeError::UnusedFunctionLifetime {
                lifetime_index: 0,
                ..
            }
        )));
    }

    #[test]
    fn generic_function_reports_each_unused_lifetime_once() {
        let errs = infer_global_errs("f = fn['a0, 'a1](x:&'a0 int)->&'a0 int { x }");
        let mut unused = errs
            .iter()
            .filter_map(|err| match err {
                TypeError::UnusedFunctionLifetime { lifetime_index, .. } => Some(*lifetime_index),
                _ => None,
            })
            .collect::<Vec<_>>();
        unused.sort_unstable();
        assert_eq!(unused, vec![1]);
    }

    #[test]
    fn struct_field_elided_lifetime_must_be_declared_in_struct_generics() {
        let errs = infer_global_errs("Box=struct{inner:&[int;2]}; f=fn(){};");
        assert_has_simple_error(
            &errs,
            "struct fields cannot use elided lifetimes; declare them in struct lifetime parameters",
        );
    }

    #[test]
    fn struct_specialization_handles_explicit_and_implicit_lifetime_args() {
        let src = r#"
            type Pair = struct['a, 'b, T] {
                left: &'a T,
                right: &'b T,
            }

            f = fn['x](x:&'x int, y:&int)->Pair['x, '_, int] {
                Pair{ left = x, right = y }
            }
        "#;

        let mut store = TypeStore::new();
        let program = gather_program(src);
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let f = find_value_by_name(&program, "f");
        infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let f_ty = solved_types.type_of(f).unwrap();
        assert_eq!(
            store.get_type_string(&program, f_ty),
            "fn['a0, 'a1](&'a0 int, &'a1 int) -> Pair['a0, 'a1, int]"
        );
    }

    #[test]
    fn generic_function_reports_unused_generic_indexes() {
        let errs = infer_global_errs("f = fn[T, U](x:T)->T { x }");
        assert!(errs.iter().any(|err| matches!(
            err,
            TypeError::UnusedFunctionGeneric {
                generic_index: 1,
                ..
            }
        )));
    }

    #[test]
    fn generic_function_reports_each_unused_generic_once() {
        let errs = infer_global_errs("f = fn[T, U, V](x:U)->U { x }");
        let mut unused = errs
            .iter()
            .filter_map(|err| match err {
                TypeError::UnusedFunctionGeneric { generic_index, .. } => Some(*generic_index),
                _ => None,
            })
            .collect::<Vec<_>>();
        unused.sort_unstable();
        assert_eq!(unused, vec![0, 2]);
    }

    #[test]
    fn struct_reports_unused_generic_indexes() {
        let errs = infer_global_errs("S = struct[T]{x:int}; f=fn(){}");
        assert!(errs.iter().any(|err| matches!(
            err,
            TypeError::UnusedStructGeneric {
                generic_index: 0,
                ..
            }
        )));
    }

    #[test]
    fn struct_reports_unused_lifetime_indexes() {
        let errs = infer_global_errs("S = struct['a0]{x:int}; f=fn(){}");
        assert!(errs.iter().any(|err| matches!(
            err,
            TypeError::UnusedStructLifetime {
                lifetime_index: 0,
                ..
            }
        )));
    }

    #[test]
    fn generic_function_specializes_on_use() {
        let src = r#"
            f = fn[T](x:T)->T { x }
            g = fn() {
                let a = f(1:int);
                let b = f(2.0:float);
            }
        "#;

        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let g = find_value_by_name(&program, "g");
        let solved_types =
            infer_value_internals(&program, &mut store, &mut solved_types, g).unwrap();

        let a_ty = find_let_stmt_type(&program, &solved_types, g, "a");
        let b_ty = find_let_stmt_type(&program, &solved_types, g, "b");

        match store.type_value(a_ty) {
            TypeValue::Builtin(b) => assert_eq!(*b, BuiltinType::Int),
            other => panic!("expected int type, got {:?}", other),
        }
        match store.type_value(b_ty) {
            TypeValue::Builtin(b) => assert_eq!(*b, BuiltinType::F64),
            other => panic!("expected float type, got {:?}", other),
        }
    }

    #[test]
    fn generic_struct_specializes_on_construction() {
        let src = r#"
            type Point = struct[T] { x:T }
            g = fn() {
                let a = Point{ x = 1:int };
                let b = Point{ x = 2.0:float };
            }
        "#;

        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let g = find_value_by_name(&program, "g");
        let solved_types =
            infer_value_internals(&program, &mut store, &mut solved_types, g).unwrap();

        let a_ty = find_let_stmt_type(&program, &solved_types, g, "a");
        let b_ty = find_let_stmt_type(&program, &solved_types, g, "b");

        let (a_generics, b_generics) = match (store.type_value(a_ty), store.type_value(b_ty)) {
            (
                TypeValue::Struct {
                    id: a_id,
                    generics: a_generics,
                    ..
                },
                TypeValue::Struct {
                    id: b_id,
                    generics: b_generics,
                    ..
                },
            ) => {
                assert_eq!(a_id, b_id);
                (a_generics, b_generics)
            }
            (other, _) => panic!("expected struct type, got {:?}", other),
        };

        assert_eq!(a_generics.len(), 1);
        assert_eq!(b_generics.len(), 1);

        match store.type_value(a_generics[0]) {
            TypeValue::Builtin(b) => assert_eq!(*b, BuiltinType::Int),
            other => panic!("expected int generic, got {:?}", other),
        }
        match store.type_value(b_generics[0]) {
            TypeValue::Builtin(b) => assert_eq!(*b, BuiltinType::F64),
            other => panic!("expected float generic, got {:?}", other),
        }
    }

    #[test]
    fn typedef_used_in_function_return() {
        let mut store = TypeStore::new();
        let ty = infer_fn("type i = int; f = fn() -> i { 2 }", &mut store).unwrap();
        match store.type_value(ty) {
            TypeValue::Builtin(b) => assert_eq!(*b, BuiltinType::Int),
            other => panic!("expected builtin type, got {:?}", other),
        }
    }

    #[test]
    fn recursive_struct_typedef_self_reference() {
        let src = r#"
            type s = struct { next: s }
        "#;

        let program = gather_program(src);
        let mut store = TypeStore::new();

        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types)
            .expect("recursive struct typedef should typecheck");

        // Find the typedef for `s` via program.definitions
        let (_name, texp) = program
            .definitions
            .iter()
            .find_map(|(name, def)| match def {
                Defined::Type(texp) => {
                    let name_str = program.name_string(*name);
                    if name_str == "s" {
                        Some((*name, *texp))
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .expect("typedef `s` not found");

        // Get resolved TypeId
        let ty = solved_types
            .typedef_types
            .get(&texp)
            .copied()
            .expect("typedef did not resolve");

        // Ensure it resolved to a struct
        let sid = match store.type_value(ty) {
            TypeValue::Struct {
                id, generics: _, ..
            } => *id,
            other => panic!("expected struct type, got {:?}", other),
        };

        let rep = store.struct_value(sid);

        // Must have exactly one field
        assert_eq!(rep.fields.len(), 1);

        let (_field_name, field_ty) = rep.fields[0];

        // Critical check: field points back to the struct type itself
        assert_eq!(
            field_ty, ty,
            "recursive field should point to the struct type itself"
        );
    }
    #[test]
    fn recursive_structs_with_generics() {
        let mut store = TypeStore::new();
        infer_fn(
            r#"
                type A = struct[T]{next:*A[T],other:B[T]}
                type B = struct[T]{value:T,parent:*A[T]}
                f=fn[T](a:A)->T {
                    let b = a.other;
                    b.value
                }
            "#,
            &mut store,
        )
        .unwrap();
    }

    #[test]
    fn infer_construct() {
        let mut store = TypeStore::new();
        infer_fn(
            "type S = struct{a:int,b:float,c:int} f=fn(){S{1,c=1,b=2.0}; S{1,2.1,3};}",
            &mut store,
        )
        .unwrap();
    }

    #[test]
    fn calling_a_function() {
        let mut store = TypeStore::new();
        infer_fn(
            "type S = struct{}; f=fn()->S S{}; g=fn()->S{f()}",
            &mut store,
        )
        .unwrap();
    }

    #[test]
    fn if_inside_while_typechecks() {
        assert_fn_type!(
            r#"
            f = fn() {
                let z = 0:int;
                let x: bool = true;
                while x {
                    z = z + if x { 1 } else { 2 }
                }
            }
            "#,
            BuiltinType::Bool
        );
    }

    #[test]
    fn if_branch_with_break_unifies_with_concrete_branch() {
        let mut store = TypeStore::new();
        infer_fn_body(
            r#"
            f = fn() {
                let keep_going: bool = true;
                let z: int = 0;
                while keep_going {
                    z = z + if keep_going { 1:int } else { break }
                }
            }
            "#,
            &mut store,
        )
        .unwrap();
    }

    #[test]
    fn nested_if_with_never_branch_avoids_branch_mismatch_errors() {
        let mut store = TypeStore::new();
        infer_fn_body(
            r#"
            f = fn() {
                let keep_going: bool = true;
                let z: int = 0;
                while keep_going {
                    let step = if keep_going {
                        if keep_going { 2:int } else { continue }
                    } else {
                        3:int
                    };
                    z = z + step;
                }
            }
            "#,
            &mut store,
        )
        .unwrap();
    }

    #[test]
    fn can_understand_function_type_exprs() {
        let mut store = TypeStore::new();
        infer_fn("f = fn(g:fn(int)->int)->int {g(2)}", &mut store).unwrap();
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
        assert!(
            errs.iter()
                .any(|err| matches!(err, TypeError::Unresolved { .. }))
        );
    }

    #[test]
    fn unresolved_clusters_report_once() {
        let src = "f = fn(){ let x = 2; let y = x; let z = 2; }";
        let program = gather_program(src);
        let f = extract_single_fn(&program);

        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let errs = match infer_value_internals(&program, &mut store, &mut solved_types, f) {
            Ok(_) => panic!("expected type errors"),
            Err(errs) => errs,
        };

        let unresolved_locs = errs
            .iter()
            .filter_map(|err| match err {
                TypeError::Unresolved { value, .. } => Some(program.value_loc(*value)),
                TypeError::UnresolvedPattern { pattern, .. } => Some(program.pattern_loc(*pattern)),
                _ => None,
            })
            .collect::<Vec<_>>();

        let unique = unresolved_locs.iter().cloned().collect::<HashSet<_>>();

        assert_eq!(errs.len(), 2);
        assert_eq!(unresolved_locs.len(), 2);
        assert_eq!(unique.len(), 2);
    }

    // #[test]
    // fn unresolved_clusters_report_on_first_let() {
    //     let src = "f = fn(){ let x = 2; let y = x;}";
    //     let program = gather_program(src);
    //     let f = extract_single_fn(&program);
    //     let body = match program.value(f) {
    //         Value::Func { body, .. } => body,
    //         _ => panic!("expected function value"),
    //     };

    //     let body_val = program.value(body);
    //     let statements = match body_val {
    //         Value::Block { statements, .. } => statements,
    //         _ => panic!("expected block body"),
    //     };

    //     let first_let = statements
    //         .ids()
    //         .find(|id| matches!(program.value(*id), Value::Let { .. }))
    //         .expect("expected let statement");
    //     // let pat_x = match program.value(first_let) {
    //     //     Value::Let { pat, .. } => pat,
    //     //     _ => panic!("expected let value"),
    //     // };
    //     // let pat_x_loc = program.pattern_loc(pat_x);
    //     let let_x_loc = program.value_loc(first_let);

    //     let mut store = TypeStore::new();
    //     let errs = match infer_value_internals(&program, &mut store, body) {
    //         Ok(_) => panic!("expected type errors"),
    //         Err(errs) => errs,
    //     };

    //     let has_let_x = errs.iter().any(|err| match err {
    //         // TypeError::UnresolvedPattern { pattern } => program.pattern_loc(*pattern) == pat_x_loc,
    //         TypeError::Unresolved { value } => program.value_loc(*value) == let_x_loc,
    //         _ => false,
    //     });

    //     assert_eq!(errs.len(), 1);
    //     assert!(has_let_x);
    // }

    #[test]
    fn reports_multiple_hard_errors() {
        let src = "f = fn(){ let x:float = 2:int; let y:int = 2 + x; }";
        let program = gather_program(src);
        let f = extract_single_fn(&program);

        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let errs = match infer_value_internals(&program, &mut store, &mut solved_types, f) {
            Ok(_) => panic!("expected type errors"),
            Err(errs) => errs,
        };
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn if_condition_must_be_bool() {
        let mut store = TypeStore::new();
        let errs = infer_fn_body("f = fn(){ if 1 { 2 } else { 3 } }", &mut store).unwrap_err();
        assert!(errs.iter().any(|err| matches!(
            err,
            TypeError::ValuesContradict {
                expectation_reason: "if condition must be bool",
                ..
            }
        )));
    }

    #[test]
    fn if_branches_must_match() {
        let mut store = TypeStore::new();
        let errs = infer_fn_body("f = fn(){ if true { 1 } else { 2.0 } }", &mut store).unwrap_err();
        assert!(errs.iter().any(|err| matches!(
            err,
            TypeError::ValuesContradict {
                expectation_reason: "if branches must have the same type",
                ..
            }
        )));
    }

    #[test]
    fn deref_return_mismatch_reports_concrete_box_type() {
        let src = "Box = struct[T]{ptr:*T}; Box.__deref = fn[T](b:&Box[T])->&T{&*b.ptr}; f = fn(b:Box[Box[Box[int]]])->int { *b };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let errs = infer_value_internals(&program, &mut store, &mut solved_types, f)
            .err()
            .unwrap_or_default();

        let clash = errs
            .iter()
            .find_map(|err| match err {
                TypeError::FunctionOutputAnnotationMismatch { clash, .. } => Some(clash),
                _ => None,
            })
            .unwrap_or_else(|| panic!("expected type mismatch, got errs={errs:?}"));

        let found_s = clash
            .found()
            .unwrap_or_else(|| panic!("expected found side in clash: {clash:?}"));
        let wanted_s = clash
            .wanted()
            .unwrap_or_else(|| panic!("expected wanted side in clash: {clash:?}"));
        assert!(
            (found_s == "int" && wanted_s.contains("Box"))
                || (wanted_s == "int" && found_s.contains("Box")),
            "expected int-vs-Box mismatch, got found={found_s}, wanted={wanted_s}"
        );
    }

    #[test]
    fn generic_deref_return_mismatch_uses_correct_generic_slot() {
        let src = "Box = struct[GenBox]{ptr:*GenBox}; Box.__deref = fn[GenBox](b:&Box[GenBox])->&GenBox{&*b.ptr}; f = fn[Output,Input](b:Box[Box[Input]])->Output { *b };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let errs = infer_value_internals(&program, &mut store, &mut solved_types, f)
            .err()
            .unwrap_or_default();

        let clash = errs
            .iter()
            .find_map(|err| match err {
                TypeError::FunctionOutputAnnotationMismatch { clash, .. } => Some(clash),
                _ => None,
            })
            .unwrap_or_else(|| panic!("expected type mismatch, got errs={errs:?}"));

        let found_s = clash
            .found()
            .unwrap_or_else(|| panic!("expected found side in clash: {clash:?}"));
        let wanted_s = clash
            .wanted()
            .unwrap_or_else(|| panic!("expected wanted side in clash: {clash:?}"));

        let (ret_side, box_side) = if found_s == "Output" {
            (found_s, wanted_s)
        } else if wanted_s == "Output" {
            (wanted_s, found_s)
        } else {
            panic!("expected one side to be Output, got found={found_s}, wanted={wanted_s}");
        };

        assert_eq!(ret_side, "Output");
        assert!(
            box_side.contains("Box") && box_side.contains("Input"),
            "expected box side to mention Box[Input], got {box_side}"
        );
        assert!(
            !box_side.contains("Output"),
            "box side should not use wrong generic slot, got {box_side}"
        );
    }

    #[test]
    fn operator_overload_not_found_for_structs() {
        let mut store = TypeStore::new();
        let errs = infer_fn_body("S=struct{}; f=fn(){ S{} + S{}; }", &mut store).unwrap_err();
        assert!(
            errs.iter()
                .any(|err| matches!(err, TypeError::BinOpOverloadNotFound { op: BinOp::Add, .. }))
        );
    }

    #[test]
    fn raw_pointer_addition_resolves_to_pointer_type() {
        let src = "f = fn(x:int){ let p:*const int = &x; let y:*const int = p + 1; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let solved_types =
            infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();
        let y_ty = find_let_stmt_type(&program, solved_types, f, "y");

        let TypeValue::Ptr {
            tgt,
            style,
            mutable,
        } = *store.type_value(y_ty)
        else {
            panic!("expected raw pointer result type")
        };
        assert_eq!(style, PointerStyle::Raw(Nullable::Yes));
        assert!(!mutable);
        assert!(matches!(
            store.type_value(tgt),
            TypeValue::Builtin(BuiltinType::Int)
        ));
    }

    #[test]
    fn raw_pointer_subtraction_resolves_to_pointer_type() {
        let src = "f = fn(x:int){ let p:*int = &x; let y:*int = p - 1; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let solved_types =
            infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();
        let y_ty = find_let_stmt_type(&program, solved_types, f, "y");

        let TypeValue::Ptr {
            tgt,
            style,
            mutable,
        } = *store.type_value(y_ty)
        else {
            panic!("expected raw pointer result type")
        };
        assert_eq!(style, PointerStyle::Raw(Nullable::Yes));
        assert!(mutable);
        assert!(matches!(
            store.type_value(tgt),
            TypeValue::Builtin(BuiltinType::Int)
        ));
    }

    #[test]
    fn raw_pointer_minus_raw_pointer_returns_isize() {
        let src =
            "f = fn(x:int){ let p:*const int = &x; let q:*const int = &x; let d:isize = p - q; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let solved_types =
            infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();
        let d_ty = find_let_stmt_type(&program, solved_types, f, "d");

        assert!(matches!(
            store.type_value(d_ty),
            TypeValue::Builtin(BuiltinType::Isize)
        ));
    }

    #[test]
    fn reference_addition_still_rejected() {
        let src = "f = fn(x:int){ let p:&int = &x; let y = p + 1:int; }";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let errs = match infer_value_internals(&program, &mut store, &mut solved_types, f) {
            Ok(_) => panic!("expected type errors"),
            Err(errs) => errs,
        };
        assert!(
            errs.iter()
                .any(|err| matches!(err, TypeError::BinOpOverloadNotFound { op: BinOp::Add, .. }))
        );
    }

    #[test]
    fn binary_member_overload_specializes_generic_signature() {
        let src = "type S = struct{}; S.__add = fn[T](s:S, y:T)->T { y }; f=fn(){ let s = S{}; let x:int = s + 2; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();
        let x_ty = find_let_stmt_type(&program, &solved_types, f, "x");
        let bin_op_site = find_let_stmt_value(&program, f, "x");

        assert!(matches!(
            store.type_value(x_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));

        let called = solved_types
            .member_method_type(bin_op_site)
            .expect("missing member overload signature for binary operator");
        assert_eq!(program.str_intern.resolve(called.member), "__add");
        let TypeValue::Func { params, ret, .. } = store.type_value(called.full_type) else {
            panic!("expected full overload type to be function")
        };
        assert_eq!(params.len(), 2);
        assert!(matches!(
            store.type_value(params[1]),
            TypeValue::Builtin(BuiltinType::Int)
        ));
        assert!(matches!(
            store.type_value(*ret),
            TypeValue::Builtin(BuiltinType::Int)
        ));
    }

    #[test]
    fn unary_member_overload_implicit_ref_self() {
        let src = "S=struct{}; S.__bitnot = fn(self:&S)->int { 1 }; f=fn(){ let s = S{}; let x:int = ~s; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let solved_types =
            infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();
        let x_ty = find_let_stmt_type(&program, solved_types, f, "x");

        assert!(matches!(
            store.type_value(x_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));
    }

    #[test]
    fn compound_assignment_reuses_binary_operator_resolution() {
        let src = "S=struct{}; S.__add = fn(self:S, rhs:int)->S { self }; f=fn(){ let s = S{}; s += 1:int; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();
    }

    #[test]
    fn inc_dec_assign_falls_back_to_add_sub_overloads_with_implicit_int_rhs() {
        let src = "S=struct{}; S.__add = fn(self:S, rhs:usize)->S { self }; S.__sub = fn(self:S, rhs:int)->S { self }; f=fn(){ let s = S{}; ++s; s--; --s; s++; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();
    }

    #[test]
    fn inc_dec_assign_can_use_dedicated_pre_post_overloads() {
        let src = "S=struct{}; S.__pre_inc = fn(self:S)->S { self }; S.__post_inc = fn(self:S)->S { self }; S.__pre_dec = fn(self:S)->S { self }; S.__post_dec = fn(self:S)->S { self }; f=fn(){ let s = S{}; ++s; s++; --s; s--; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();
    }

    #[test]
    fn member_access_uses_curried_type_and_tracks_full_signature_for_by_value_self() {
        let src = "S=struct{}; S.add_5 = fn(self:S)->S { self }; f=fn(x:S){ let y = x.add_5(); };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let Value::Func { params, .. } = program.value(f) else {
            panic!("expected function value");
        };
        let x_ty = find_signature_param_type(&solved_types, f, params.at(0));
        let s_ty = find_typedef_type_by_name(&program, &solved_types, "S");
        assert_eq!(x_ty, s_ty);
        let call_site = find_let_stmt_value(&program, f, "y");
        let (access_site, access_ty, call_ty) =
            find_member_access_and_result_types(&program, &solved_types, f, "y");
        assert_ne!(access_site, call_site);
        assert!(
            solved_types
                .member_method_type_in_function(f, call_site)
                .is_none()
        );

        let TypeValue::Func { params, ret, .. } = store.type_value(access_ty) else {
            panic!("expected member access to be curried function")
        };
        assert_eq!(params.len(), 0);
        assert_eq!(*ret, x_ty);
        assert_eq!(call_ty, x_ty);

        let called = solved_types
            .member_method_type_in_function(f, access_site)
            .expect("missing solved member method signature for access site");
        assert_eq!(program.str_intern.resolve(called.member), "add_5");
        let TypeValue::Func { params, ret, .. } = store.type_value(called.full_type) else {
            panic!("expected tracked full member method type to be a function")
        };
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], x_ty);
        assert_eq!(*ret, x_ty);
    }

    #[test]
    fn member_access_curried_ref_self_and_tracks_full_signature() {
        let src = "S=struct{}; S.add_5 = fn(self:&S)->int { 1 }; f=fn(x:S){ let y = x.add_5(); };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let Value::Func { params, .. } = program.value(f) else {
            panic!("expected function value");
        };
        let x_ty = find_signature_param_type(&solved_types, f, params.at(0));
        let s_ty = find_typedef_type_by_name(&program, &solved_types, "S");
        assert_eq!(x_ty, s_ty);
        let call_site = find_let_stmt_value(&program, f, "y");
        let (access_site, access_ty, call_ty) =
            find_member_access_and_result_types(&program, &solved_types, f, "y");
        assert_ne!(access_site, call_site);
        assert!(
            solved_types
                .member_method_type_in_function(f, call_site)
                .is_none()
        );

        let TypeValue::Func { params, ret, .. } = store.type_value(access_ty) else {
            panic!("expected member access to be curried function")
        };
        assert_eq!(params.len(), 0);
        assert!(matches!(
            store.type_value(*ret),
            TypeValue::Builtin(BuiltinType::Int)
        ));
        assert!(matches!(
            store.type_value(call_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));

        let called = solved_types
            .member_method_type_in_function(f, access_site)
            .expect("missing solved member method signature for access site");
        assert_eq!(program.str_intern.resolve(called.member), "add_5");
        let TypeValue::Func {
            params: full_params,
            ret: full_ret,
            ..
        } = store.type_value(called.full_type)
        else {
            panic!("expected tracked full member method type to be a function")
        };
        assert_eq!(full_params.len(), 1);
        let TypeValue::Ptr {
            tgt,
            style: _,
            mutable,
        } = store.type_value(full_params[0])
        else {
            panic!("expected tracked full self parameter to stay as pointer")
        };
        //TODO figure out what exactly we expect here
        // assert!(matches!(style, PointerStyle::Ref(LifeTime::External(_))));
        assert!(!*mutable);
        assert_eq!(*tgt, s_ty);
        assert!(matches!(
            store.type_value(*full_ret),
            TypeValue::Builtin(BuiltinType::Int)
        ));
    }

    #[test]
    fn member_access_autoderefs_plain_pointer_like_base() {
        let src = "S=struct{x:int}; f=fn(p:&S){ let y:int = p.x; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let errs = infer_value_internals(&program, &mut store, &mut solved_types, f)
            .err()
            .unwrap_or_default();
        assert!(errs.iter().all(|err| {
            !matches!(
                err,
                TypeError::UnknownField { .. } | TypeError::CannotDeref { .. }
            )
        }));

        let y_ty = find_let_stmt_type(&program, &solved_types, f, "y");
        assert!(matches!(
            store.type_value(y_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));

        let access_site = find_let_stmt_value(&program, f, "y");
        assert_eq!(
            solved_types.member_access_implicit_deref_count_in_function(f, access_site),
            Some(2),
            "plain pointer-like implicit deref should be tracked"
        );
        let chain =
            implicit_deref_chain_type_strings(&program, &store, &solved_types, f, access_site)
                .expect("expected implicit deref chain");
        assert_eq!(
            chain,
            vec!["&'a0 S".to_string(), "S".to_string()],
            "unexpected implicit deref chain for plain pointer-like base"
        );
    }

    #[test]
    fn smart_pointer_member_access_falls_back_to_deref_and_tracks_count() {
        let src = "Inner=struct{x:int}; Box=struct{inner:Inner}; Box.__deref = fn(self:&Box)->&Inner { &self.inner }; f=fn(b:Box){ let y:int = b.x; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let y_ty = find_let_stmt_type(&program, &solved_types, f, "y");
        assert!(matches!(
            store.type_value(y_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));

        let access_site = find_let_stmt_value(&program, f, "y");
        assert_eq!(
            solved_types.member_access_implicit_deref_count_in_function(f, access_site),
            Some(4)
        );
        let chain =
            implicit_deref_chain_type_strings(&program, &store, &solved_types, f, access_site)
                .expect("expected implicit deref chain");
        assert_eq!(
            chain,
            vec![
                "Box".to_string(),
                "&'idk0 Box".to_string(),
                "&'idk0 Inner".to_string(),
                "Inner".to_string(),
            ],
            "unexpected smart-deref chain for member access"
        );
    }

    #[test]
    fn ptr_member_access_single_smart_deref_chain_tracks_ref_steps() {
        let src = "S=struct{x:int}; Box=struct{inner:S}; Box.__deref = fn(self:&Box)->&S { &self.inner }; f=fn(b:Box){ let y:int = b->x; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let y_ty = find_let_stmt_type(&program, &solved_types, f, "y");
        assert!(matches!(
            store.type_value(y_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));

        let access_site = find_let_stmt_value(&program, f, "y");
        assert_eq!(
            solved_types.member_access_implicit_deref_count_in_function(f, access_site),
            Some(4)
        );

        let chain =
            implicit_deref_chain_type_strings(&program, &store, &solved_types, f, access_site)
                .expect("expected implicit deref chain");
        assert_eq!(
            chain,
            vec![
                "Box".to_string(),
                "&'idk0 Box".to_string(),
                "&'idk0 S".to_string(),
                "S".to_string(),
            ],
            "unexpected single smart-deref chain for `->` member access"
        );
    }

    #[test]
    fn smart_pointer_member_access_prefers_direct_member_before_deref() {
        let src = "Inner=struct{x:int}; Box=struct{x:bool, inner:Inner}; Box.__deref = fn(self:&Box)->&Inner { &self.inner }; f=fn(b:Box){ let y:bool = b.x; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let y_ty = find_let_stmt_type(&program, &solved_types, f, "y");
        assert!(matches!(
            store.type_value(y_ty),
            TypeValue::Builtin(BuiltinType::Bool)
        ));

        let access_site = find_let_stmt_value(&program, f, "y");
        assert_eq!(
            solved_types.member_access_implicit_deref_count_in_function(f, access_site),
            None
        );
    }

    #[test]
    fn dot_member_access_does_not_chain_multiple_smart_derefs() {
        let src = "Inner=struct{x:int}; Box=struct{inner:Inner}; Wrap=struct{boxed:Box}; Box.__deref = fn(self:&Box)->&Inner { &self.inner }; Wrap.__deref = fn(self:&Wrap)->&Box { &self.boxed }; f=fn(w:Wrap){ let y:int = w.x; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let errs = match infer_value_internals(&program, &mut store, &mut solved_types, f) {
            Ok(_) => panic!("expected member access to fail without multi-hop dot autoderef"),
            Err(errs) => errs,
        };

        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::UnknownField {
                    field,
                    ..
                } if program.str_intern.resolve(*field) == "x"
            )
        }));
    }

    #[test]
    fn ptr_member_access_can_chain_smart_derefs() {
        let src = "Inner=struct{x:int}; Box=struct{inner:Inner}; Wrap=struct{boxed:Box}; Box.__deref = fn(self:&Box)->&Inner { &self.inner }; Wrap.__deref = fn(self:&Wrap)->&Box { &self.boxed }; f=fn(w:Wrap){ let y:int = w->x; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let y_ty = find_let_stmt_type(&program, &solved_types, f, "y");
        assert!(matches!(
            store.type_value(y_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));

        let access_site = find_let_stmt_value(&program, f, "y");
        assert_eq!(
            solved_types.member_access_implicit_deref_count_in_function(f, access_site),
            Some(7)
        );
        let chain =
            implicit_deref_chain_type_strings(&program, &store, &solved_types, f, access_site)
                .expect("expected implicit deref chain");
        assert_eq!(
            chain,
            vec![
                "Wrap".to_string(),
                "&'idk0 Wrap".to_string(),
                "&'idk0 Box".to_string(),
                "Box".to_string(),
                "&'idk1 Box".to_string(),
                "&'idk1 Inner".to_string(),
                "Inner".to_string(),
            ],
            "unexpected multi-hop smart-deref chain for `->` member access"
        );
    }

    //breaks because we dont wait our turn enough in derefs resolution
    #[test]
    fn deref_chain_supports_all_four_style_transitions_with_raw_links() {
        let src = "
            Wrapper = struct {inner:int};
            Wrapper.get = fn(self:&mut Wrapper)->&mut int {&mut self.inner}

            Unsafe = struct { inner: &'raw Wrapper };
            Unsafe.__deref_mut = fn['a](self: &'raw mut Unsafe) -> &'a mut Wrapper  { &*self.inner };

            RawCalc = struct { inner: &'raw Unsafe };
            RawCalc.__deref_mut = fn(self: &'raw mut RawCalc) -> &'raw Unsafe { self.inner };

            Raw = struct { inner: &'raw RawCalc };
            Raw.__deref_mut = fn(self: &mut Raw) -> &'raw RawCalc { self.inner };

            Safe = struct { inner: &'raw Raw };
            Safe.__deref_mut = fn(self: &mut Safe) -> &mut Raw { &*self.inner };

            f = fn(s: &mut Safe) {
                let out : &mut int = s->get();
            };

        ";

        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let body_ty = find_let_stmt_type(&program, &solved_types, f, "out");

        let TypeValue::Ptr {
            tgt,
            style,
            mutable,
        } = store.type_value(body_ty)
        else {
            panic!("expected function body to infer as pointer")
        };
        assert!(style.is_fancy());
        assert!(*mutable);
        assert!(matches!(
            store.type_value(*tgt),
            TypeValue::Builtin(BuiltinType::Int)
        ));

        let (access_site, _, _) =
            find_member_access_and_result_types(&program, &solved_types, f, "out");
        let chain =
            implicit_deref_chain_type_strings(&program, &store, &solved_types, f, access_site)
                .expect("expected implicit deref chain");
        assert_eq!(
            chain,
            vec![
                "&'a0 mut Safe".to_string(),
                "&'idk1 mut Safe".to_string(),
                "&'idk1 mut Raw".to_string(),
                "&'raw RawCalc".to_string(),
                "&'raw RawCalc".to_string(),
                "&'raw Unsafe".to_string(),
                "&'raw Unsafe".to_string(),
                "&'idk3 mut Wrapper".to_string(),
                "Wrapper".to_string(),
            ],
            "unexpected full deref chain for four-style transition case"
        );
    }

    #[test]
    fn smart_deref_chain_can_drop_mutability_through_nested_ref_targets() {
        let src = "S=struct{x:int}; Box=struct[T]{inner:T}; Box.__deref_mut = fn[T](self:&mut Box[T])->&mut T { &mut self.inner }; f=fn['a](b:Box[&'raw const &'a S])->int { let y:int = b->x; y };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let y_ty = find_let_stmt_type(&program, &solved_types, f, "y");
        assert!(matches!(
            store.type_value(y_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));

        let access_site = find_let_stmt_value(&program, f, "y");
        let chain =
            implicit_deref_chain_type_strings(&program, &store, &solved_types, f, access_site)
                .expect("expected implicit deref chain");
        assert_eq!(
            chain,
            vec![
                "Box[&'raw const &'a0 S]".to_string(),
                "&'idk0 mut Box[&'raw const &'a0 S]".to_string(),
                "&'idk0 mut &'raw const &'a0 S".to_string(),
                "&'raw const &'a0 S".to_string(),
                "&'a0 S".to_string(),
                "S".to_string(),
            ],
            "unexpected smart-deref chain through nested references"
        );
    }

    #[test]
    fn tuple_int_access_resolves_tuple_element_types() {
        let src = "f=fn(t:(int,bool)){ let a:int = t.0; let b:bool = t.1; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let a_ty = find_let_stmt_type(&program, &solved_types, f, "a");
        assert!(matches!(
            store.type_value(a_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));
        let b_ty = find_let_stmt_type(&program, &solved_types, f, "b");
        assert!(matches!(
            store.type_value(b_ty),
            TypeValue::Builtin(BuiltinType::Bool)
        ));

        let access_site = find_let_stmt_value(&program, f, "a");
        assert_eq!(
            solved_types.member_access_implicit_deref_count_in_function(f, access_site),
            None
        );
    }

    #[test]
    fn ptr_tuple_int_access_can_chain_derefs() {
        let src = "f=fn(pp:& &(int,bool)){ let a:int = pp->0; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let a_ty = find_let_stmt_type(&program, &solved_types, f, "a");
        assert!(matches!(
            store.type_value(a_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));

        let access_site = find_let_stmt_value(&program, f, "a");
        assert_eq!(
            solved_types.member_access_implicit_deref_count_in_function(f, access_site),
            Some(3)
        );
    }

    #[test]
    fn dot_tuple_int_access_does_not_chain_multiple_derefs() {
        let src = "f=fn(pp:& &(int,bool)){ let a:int = pp.0; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let errs = match infer_value_internals(&program, &mut store, &mut solved_types, f) {
            Ok(_) => panic!("expected tuple dot access to fail on multi-hop autoderef"),
            Err(errs) => errs,
        };

        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple { message, .. }
                if *message == "`.` tuple access performs at most one implicit dereference"
            )
        }));
    }

    #[test]
    fn pending_member_access_resolves_after_source_type_becomes_known() {
        let src = "Inner=struct{x:int}; Box=struct{inner:Inner}; Box.__deref = fn(self:&Box)->&Inner { &self.inner }; f=fn(b:Box)->void{ let v = b as _; let y:int = v.x; v:Box; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let y_ty = find_let_stmt_type(&program, &solved_types, f, "y");
        assert!(matches!(
            store.type_value(y_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));

        let access_site = find_let_stmt_value(&program, f, "y");
        assert_eq!(
            solved_types.member_access_implicit_deref_count_in_function(f, access_site),
            Some(4)
        );
        let chain =
            implicit_deref_chain_type_strings(&program, &store, &solved_types, f, access_site)
                .expect("expected implicit deref chain");
        assert_eq!(
            chain,
            vec![
                "Box".to_string(),
                "&'idk0 Box".to_string(),
                "&'idk0 Inner".to_string(),
                "Inner".to_string(),
            ],
            "unexpected deferred-source smart-deref chain"
        );
    }

    #[test]
    fn implicit_deref_chain_includes_pointer_and_smart_hops() {
        let src = "Inner=struct{x:int}; Box=struct{inner:Inner}; Wrap=struct{boxed:Box}; Box.__deref = fn(self:&Box)->&Inner { &self.inner }; Wrap.__deref = fn(self:&Wrap)->& &Box { & &self.boxed }; f=fn(w:Wrap){ let y:int = w->x; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let access_site = find_let_stmt_value(&program, f, "y");
        assert_eq!(
            solved_types.member_access_implicit_deref_count_in_function(f, access_site),
            Some(8)
        );

        let chain =
            implicit_deref_chain_type_strings(&program, &store, &solved_types, f, access_site)
                .expect("expected implicit deref chain");
        assert_eq!(chain.len(), 8);
        assert_eq!(chain[0], "Wrap");
        assert_eq!(chain[1], "&'idk0 Wrap");
        assert!(
            chain[2].starts_with("&'idk0 &") && chain[2].contains("Box"),
            "expected nested reference output from first smart deref, got {:?}",
            chain[2]
        );
        assert!(chain[3].starts_with("&") && chain[3].contains("Box"));
        assert_eq!(chain[4], "Box");
        assert_eq!(chain[5], "&'idk1 Box");
        assert_eq!(chain[6], "&'idk1 Inner");
        assert_eq!(chain[7], "Inner");
    }

    #[test]
    fn declaration_specialization_is_rejected_under_unique_signature_rule() {
        let src = r#"
            f = fn[T](x:T)->T;
            f = fn(x:int)->int;
        "#;

        let errs = infer_global_errs(src);
        assert!(errs.iter().any(|e| {
            matches!(
                e,
                TypeError::ValuesContradict {
                    expectation_reason:
                        "all declarations must exactly match the first declaration signature",
                    ..
                }
            )
        }));
    }

    #[test]
    fn binary_member_overload_requires_one_extra_param() {
        let errs = infer_global_errs("S=struct{}; S.__add = fn(self:S){ }; f=fn(){};");
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple {
                    message:
                        "binary operator overloads must take exactly one parameter after `self`",
                    ..
                }
            )
        }));
    }

    #[test]
    fn binary_member_overload_requires_self_like_first_param() {
        let errs = infer_global_errs("S=struct{}; S.__add = fn(x:int, rhs:int){ }; f=fn(){};");
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple {
                    message:
                        "binary operator overloads must take `self` as the first parameter type",
                    ..
                }
            )
        }));
    }

    #[test]
    fn unary_member_overload_disallows_extra_params() {
        let errs = infer_global_errs("S=struct{}; S.__bitnot = fn(self:S, x:int){ }; f=fn(){};");
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple {
                    message: "unary operator overloads must not take parameters after `self`",
                    ..
                }
            )
        }));
    }

    #[test]
    fn inc_dec_member_overloads_share_unary_signature_requirements() {
        let errs = infer_global_errs("S=struct{}; S.__pre_inc = fn(self:S, x:int){ }; f=fn(){};");
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple {
                    message: "unary operator overloads must not take parameters after `self`",
                    ..
                }
            )
        }));
    }

    #[test]
    fn free_member_signature_is_checked() {
        let errs = infer_global_errs("S=struct{}; S.__free = fn(self:&S)->int { 1 }; f=fn(){};");
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple {
                    message: "`__free` must take `&mut self` as the first parameter",
                    ..
                }
            )
        }));
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple {
                    message: "`__free` must return `void`",
                    ..
                }
            )
        }));
    }

    #[test]
    fn free_member_with_mut_ref_self_and_void_output_is_allowed() {
        let src = "S=struct{}; S.__free = fn(self:&mut S){}; f=fn(x:S){ x.__free(); };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let f = find_value_by_name(&program, "f");
        infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();
    }

    #[test]
    fn user_free_member_name_is_allowed_to_implement() {
        let src = "S=struct{}; S.__user_free = fn(self:&mut S){}; f=fn(){};";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
    }

    #[test]
    fn free_member_on_generic_struct_requires_all_generics_free_in_order() {
        let errs = infer_global_errs(
            "S=struct[T,U]{x:T,y:U}; S.__free = fn[T,U](self:&mut S[U,T]){}; f=fn(){};",
        );
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple {
                    message: "`__free` must take `&mut self` as the first parameter",
                    ..
                }
            )
        }));
    }

    #[test]
    fn free_member_on_generic_struct_accepts_all_generics_free_in_order() {
        let src = "S=struct[T,U]{x:T,y:U}; S.__free = fn[T,U](self:&mut S[T,U]){}; f=fn(){};";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
    }

    #[test]
    fn any_type_builtin_member_methods_are_available_on_primitives_refs_and_generic_t() {
        let src = r#"
            g = fn[T](x:T)->usize { x.__size_of() + x.__align_of() }
            f = fn(x:int)->usize {
                let y:int = 1;
                let p:&int = &y;
                let a:usize = x.__size_of();
                let b:usize = p.__align_of();
                a + b + g(x) + (1:int).__size_of()
            }
        "#;
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let Value::Func { body, .. } = program.value(f) else {
            panic!("expected function value")
        };
        let body = body.expect("expected function body");
        let ty = solved_types
            .inner_value_type(f, body)
            .expect("missing inferred body type");
        assert!(matches!(
            store.type_value(ty),
            TypeValue::Builtin(BuiltinType::Usize)
        ));
    }

    #[test]
    fn size_of_member_signature_requires_reference_self() {
        let errs =
            infer_global_errs("S=struct{}; S.__size_of = fn(self:S)->usize { 1:usize }; f=fn(){};");
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::IlegalToImplMethod {
                    method_name,
                    ..
                } if *method_name == SIZE_OF_STR
            )
        }));
    }

    #[test]
    fn any_type_builtin_free_is_available_on_generic_t_and_literals() {
        let src = r#"
            g = fn[T](x:T){ x.__free(); }
            f = fn(){
                let x:int = 1;
                x.__free();
                (2:int).__free();
                g(x);
            }
        "#;
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let Value::Func { body, .. } = program.value(f) else {
            panic!("expected function value")
        };
        let body = body.expect("expected function body");
        let ty = solved_types
            .inner_value_type(f, body)
            .expect("missing inferred body type");
        assert!(matches!(
            store.type_value(ty),
            TypeValue::Builtin(BuiltinType::Void)
        ));
    }

    #[test]
    fn deref_member_requires_shared_ref_self_and_shared_ref_output() {
        let errs = infer_global_errs("S=struct{}; S.__deref = fn(self:S)->int { 1 }; f=fn(){};");
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple {
                    message: "`__deref` must take `&self` as the first parameter",
                    ..
                }
            )
        }));
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple {
                    message: "`__deref` must return a non-raw shared reference `&T`",
                    ..
                }
            )
        }));
    }

    #[test]
    fn deref_mut_member_requires_mut_ref_self_and_mut_ref_output() {
        let errs =
            infer_global_errs("S=struct{}; S.__deref_mut = fn(self:&S)->&int { }; f=fn(){};");
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple {
                    message: "`__deref_mut` must take `&mut self` as the first parameter",
                    ..
                }
            )
        }));
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple {
                    message: "`__deref_mut` must return a non-raw mutable reference `&mut T`",
                    ..
                }
            )
        }));
    }

    #[test]
    fn deref_and_deref_mut_must_have_same_target() {
        let errs = infer_global_errs(
            "S=struct{}; S.__deref = fn(self:&S)->&int { }; S.__deref_mut = fn(self:&mut S)->&mut bool { }; f=fn(){};",
        );
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::Simple {
                    message: "`__deref` and `__deref_mut` must dereference to the same target type",
                    ..
                }
            )
        }));
    }

    #[test]
    fn value_deref_uses_special_deref_method_target() {
        let src =
            "type S=struct{x:int}; S.__deref = fn(self:&S)->&int { }; f = fn(){ let s = S{1}; *s }";
        let mut store = TypeStore::new();
        let ty = infer_fn_body(src, &mut store).unwrap();
        assert!(matches!(
            store.type_value(ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));
    }

    #[test]
    fn value_deref_from_unknown_is_deferred_and_can_resolve_to_struct_deref() {
        let src = "
            type S=struct{p:*int};
            S.__deref = fn(self:&S)->&int { &*self.p };
            f=fn(){
                let x = 0:int;
                let y = x as _;
                let r = *y;
                y = S{&x};
                r
            }
        ";
        let mut store = TypeStore::new();
        let ty = infer_fn_body(src, &mut store).unwrap();
        assert!(matches!(
            store.type_value(ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));
    }

    ///currently fails because we dont do places properly
    #[test]
    fn generic_struct_deref_methods_specialize_from_receiver_type() {
        let src = r#"
            Box = struct[T]{p:*T}
            Box.__deref = fn[T](b:&Box[T])->&T { &*(*b).p }
            Box.__deref_mut = fn[T](b:&mut Box[T])->&mut T { &*(*b).p }
            f = fn(b:Box[int])->int { *b }
        "#;

        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let f = find_value_by_name(&program, "f");
        infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let Value::Func { body, .. } = program.value(f) else {
            panic!("expected function value")
        };
        let body = body.expect("expected function body");
        let body_ty = solved_types
            .inner_value_type(f, body)
            .unwrap_or_else(|| panic!("missing body type for `f`"));

        assert!(matches!(
            store.type_value(body_ty),
            TypeValue::Builtin(BuiltinType::Int)
        ));
    }

    #[test]
    fn cannot_deref_error_reports_operand_type() {
        let mut store = TypeStore::new();
        let errs = infer_fn_body("f=fn(){ let x:int = 1; *x }", &mut store).unwrap_err();
        assert!(errs.iter().any(|err| {
            matches!(
                err,
                TypeError::CannotDeref {
                    operand_type: Some(t),
                    ..
                } if t == "int"
            )
        }));
    }

    #[test]
    fn unknown_builtin_member_method_name_errors() {
        let errs = infer_global_errs("S=struct{}; S.__derefed = fn(self:S){ }; f=fn(){};");
        assert!(
            errs.iter()
                .any(|err| matches!(err, TypeError::UnknownBuiltinMemberMethod { .. }))
        );
    }

    ///structs in body open us up to qualified structures having generics
    ///we also dont do de bjurn ids so it would be a mess
    #[test]
    fn reject_structs_in_body() {
        let mut store = TypeStore::new();
        let _errs = infer_fn_body("f=fn(){ type S = struct[T]{x:T} }", &mut store).unwrap_err();
    }

    #[test]
    fn closures_in_body_are_rejected() {
        assert_fn_body_simple_error(
            "f=fn(){ let g = fn(x:int)->int{x}; }",
            CLOSURES_UNSUPPORTED_MSG,
        );
        assert_fn_body_simple_error(
            "f=fn(){ let g = fn[T](x:T)->T{x}; }",
            CLOSURES_UNSUPPORTED_MSG,
        );
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
