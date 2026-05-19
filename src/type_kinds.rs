use crate::error_reporting::ErrorReporter;
use crate::identity_hasher::IdHashMap;
/**!
 * this is specifically a storage of KINDs and not TYPES
 *
 * the reason for this is ultimatly that borrow checking needs to be factored out
 * and that means we need to solve the general shape independently ie the kind
 *
 * the previous version of this code failed because it was badly structured.
 * this time we are hoping to seperate it out so that latice style solvers like (ie stuff that does a < instead of = relations)
 * would have seprate ids from kinds that can be solved with a union find
 *
 * for figuring out things like operators generics and auto derefs an HM style solver is enough.
 * thus having it work on the rough shape while other parts solve lifetimes seems correct.
 */
use crate::index::{Idx, IndexVec, UnionFind};
use crate::ir::{BinOp, LifeTimeId, NameId, PatId, TExpId, UnOp, ValId};
use crate::operator_solver::Projection;
use crate::parsing::Loc;
use crate::program::Program;
use crate::string_intern::StrId;
use std::collections::HashMap;
use std::error::Error;
use std::ops::Index;
use std::ops::IndexMut;
use thin_vec::ThinVec;

macro_rules! impl_idx {
    ($($id:ty),* $(,)?) => {
        $(
            impl Idx for $id {
                fn new(idx: usize) -> Self {
                    assert!(idx <= u32::MAX as usize);
                    Self(idx as u32)
                }

                fn index(self) -> usize {
                    self.0 as usize
                }
            }
        )*
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KindId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtrId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LifeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MutId(pub u32);

impl_idx!(StructId, GenId, KindId, PtrId, LifeId, MutId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntKind {
    pub size: Option<IntSize>,
    pub sign: Option<IntSign>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FloatKind {
    pub size: Option<FloatSize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntSize {
    Int,
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntSign {
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatSize {
    F16,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinKind {
    Int(IntKind),
    Float(FloatKind),
    Bool,
    Str,
    Void,
    Type,
}

pub const BUILTINS: &[(&str, BuiltinKind)] = {
    use FloatSize::*;
    use IntSign::*;
    use IntSize::*;

    &[
        (
            "int",
            BuiltinKind::Int(IntKind {
                size: Some(Int),
                sign: Some(Signed),
            }),
        ),
        (
            "uint",
            BuiltinKind::Int(IntKind {
                size: Some(Int),
                sign: Some(Unsigned),
            }),
        ),
        (
            "i8",
            BuiltinKind::Int(IntKind {
                size: Some(I8),
                sign: Some(Signed),
            }),
        ),
        (
            "i16",
            BuiltinKind::Int(IntKind {
                size: Some(I16),
                sign: Some(Signed),
            }),
        ),
        (
            "i32",
            BuiltinKind::Int(IntKind {
                size: Some(I32),
                sign: Some(Signed),
            }),
        ),
        (
            "i64",
            BuiltinKind::Int(IntKind {
                size: Some(I64),
                sign: Some(Signed),
            }),
        ),
        (
            "i128",
            BuiltinKind::Int(IntKind {
                size: Some(I128),
                sign: Some(Signed),
            }),
        ),
        (
            "isize",
            BuiltinKind::Int(IntKind {
                size: Some(Isize),
                sign: Some(Signed),
            }),
        ),
        (
            "u8",
            BuiltinKind::Int(IntKind {
                size: Some(I8),
                sign: Some(Unsigned),
            }),
        ),
        (
            "u16",
            BuiltinKind::Int(IntKind {
                size: Some(I16),
                sign: Some(Unsigned),
            }),
        ),
        (
            "u32",
            BuiltinKind::Int(IntKind {
                size: Some(I32),
                sign: Some(Unsigned),
            }),
        ),
        (
            "u64",
            BuiltinKind::Int(IntKind {
                size: Some(I64),
                sign: Some(Unsigned),
            }),
        ),
        (
            "u128",
            BuiltinKind::Int(IntKind {
                size: Some(I128),
                sign: Some(Unsigned),
            }),
        ),
        (
            "usize",
            BuiltinKind::Int(IntKind {
                size: Some(Isize),
                sign: Some(Unsigned),
            }),
        ),
        ("f16", BuiltinKind::Float(FloatKind { size: Some(F16) })),
        ("f32", BuiltinKind::Float(FloatKind { size: Some(F32) })),
        ("f64", BuiltinKind::Float(FloatKind { size: Some(F64) })),
        ("float", BuiltinKind::Float(FloatKind { size: Some(F64) })),
        ("bool", BuiltinKind::Bool),
        ("str", BuiltinKind::Str),
        ("void", BuiltinKind::Void),
        ("Type", BuiltinKind::Type),
    ]
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArraySize {
    Sized(usize),
    Unsized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nullable {
    Yes,
    No,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifeKind {
    Static,
    Univeral(Option<u32>),
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerStyle {
    Raw(Option<Nullable>),
    Ref(LifeId),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeKind {
    Builtin(BuiltinKind),
    Generic(GenId),

    Tuple(ThinVec<KindId>),
    Struct {
        id: StructId,
        gens: Option<ThinVec<KindId>>,
        lifes: Option<ThinVec<LifeId>>,
    },
    Func {
        params: ThinVec<KindId>,
        ret: KindId,
    },
    Array {
        inner: KindId,
        size: Option<ArraySize>,
    },
    Ptr {
        tgt: KindId,
        style: PtrId,
        mutable: MutId,
    },
}

pub struct TypeUniverse {
    pub look: KindLookUp,
    pub storage: KindStorage,
}

pub struct KindLookUp {
    pub kinds: UnionFind<KindId>,
    pub ptr: UnionFind<PtrId>,
    pub mutable: UnionFind<MutId>,
    pub life: UnionFind<LifeId>,
}

pub struct TypeIntern {
    map: IdHashMap<TypeKind, KindId>,
    pub storage: IndexVec<KindId, Option<TypeKind>>,
}

impl TypeIntern {
    pub fn new() -> Self {
        Self {
            map: IdHashMap::default(),
            storage: IndexVec::new(),
        }
    }

    pub fn intern(&mut self, ty: TypeKind) -> KindId {
        if let Some(id) = self.map.get(&ty).copied() {
            return id;
        }

        let id = self.storage.push(Some(ty.clone()));
        self.map.insert(ty, id);
        id
    }

    pub fn add_empty(&mut self) -> KindId {
        self.storage.push(None)
    }
}

impl Default for TypeIntern {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<KindId> for TypeIntern {
    type Output = Option<TypeKind>;

    #[inline]
    fn index(&self, i: KindId) -> &Self::Output {
        &self.storage[i]
    }
}

impl IndexMut<KindId> for TypeIntern {
    #[inline]
    fn index_mut(&mut self, i: KindId) -> &mut Self::Output {
        &mut self.storage[i]
    }
}

pub struct KindStorage {
    pub types: TypeIntern,
    pub ptr: IndexVec<PtrId, Option<PointerStyle>>,
    pub life: IndexVec<LifeId, Option<LifeKind>>,
    pub mutable: IndexVec<MutId, Option<bool>>,
}

impl KindLookUp {
    pub fn new() -> Self {
        Self {
            kinds: UnionFind::new(),
            ptr: UnionFind::new(),
            mutable: UnionFind::new(),
            life: UnionFind::new(),
        }
    }
}

impl Default for KindLookUp {
    fn default() -> Self {
        Self::new()
    }
}

impl KindStorage {
    pub fn new() -> Self {
        Self {
            types: TypeIntern::new(),
            ptr: IndexVec::new(),
            life: IndexVec::new(),
            mutable: IndexVec::new(),
        }
    }
}

impl Default for KindStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeUniverse {
    pub fn new() -> Self {
        Self {
            look: KindLookUp::new(),
            storage: KindStorage::new(),
        }
    }

    pub fn intern(&mut self, ty: TypeKind) -> KindId {
        if let Some(id) = self.storage.types.map.get(&ty).copied() {
            return id;
        }

        let id = self.storage.types.intern(ty);
        let uf_id = self.look.kinds.push_singleton();
        debug_assert_eq!(id, uf_id);
        id
    }

    pub fn intern_builtin(&mut self, builtin: BuiltinKind) -> KindId {
        self.intern(TypeKind::Builtin(builtin))
    }

    pub fn add_empty(&mut self) -> KindId {
        let id = self.storage.types.add_empty();
        let uf_id = self.look.kinds.push_singleton();
        debug_assert_eq!(id, uf_id);
        id
    }

    pub fn get(&self, id: KindId) -> Option<&TypeKind> {
        self.storage.types[id].as_ref()
    }

    pub fn kind_to_string(&self, program: &Program, id: KindId) -> String {
        let mut out = String::new();
        self.write_kind(program, id, &mut out);
        out
    }

    pub fn deref_chain_to_string(
        &self,
        program: &Program,
        chain: &[(KindId, Projection)],
    ) -> String {
        chain
            .iter()
            .map(|(id, projection)| {
                format!(
                    "{} [{}]",
                    self.kind_to_string(program, *id),
                    projection_name(*projection)
                )
            })
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    fn write_kind(&self, program: &Program, id: KindId, out: &mut String) {
        match self.get(id) {
            None => out.push('_'),
            Some(TypeKind::Builtin(builtin)) => out.push_str(builtin.name()),
            Some(TypeKind::Generic(gen_id)) => out.push_str(&format!("T{}", gen_id.0)),
            Some(TypeKind::Tuple(items)) => {
                out.push('(');
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(", ");
                    }
                    self.write_kind(program, *item, out);
                }
                out.push(')');
            }
            Some(TypeKind::Struct { id, gens, lifes }) => {
                out.push_str(&format!("struct#{}", id.0));
                if let Some(gens) = gens {
                    write_angle_list(out, gens.iter().copied(), |out, item| {
                        self.write_kind(program, item, out);
                    });
                }
                if let Some(lifes) = lifes {
                    write_angle_list(out, lifes.iter().copied(), |out, life| {
                        self.write_life(program, life, out);
                    });
                }
            }
            Some(TypeKind::Func { params, ret }) => {
                out.push_str("fn(");
                for (idx, param) in params.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(", ");
                    }
                    self.write_kind(program, *param, out);
                }
                out.push_str(") -> ");
                self.write_kind(program, *ret, out);
            }
            Some(TypeKind::Array { inner, size }) => {
                out.push('[');
                self.write_kind(program, *inner, out);
                match size {
                    Some(ArraySize::Sized(size)) => out.push_str(&format!("; {size}")),
                    Some(ArraySize::Unsized) => out.push_str("; _"),
                    None => {}
                }
                out.push(']');
            }
            Some(TypeKind::Ptr {
                tgt,
                style,
                mutable,
            }) => {
                match self.storage.ptr.get(*style).and_then(|style| *style) {
                    Some(PointerStyle::Ref(life)) => {
                        out.push('&');
                        self.write_life(program, life, out);
                        out.push(' ');
                    }
                    Some(PointerStyle::Raw(nullable)) => {
                        out.push('*');
                        if matches!(nullable, Some(Nullable::Yes)) {
                            out.push('?');
                        }
                    }
                    None => out.push_str("ptr "),
                }
                if self
                    .storage
                    .mutable
                    .get(*mutable)
                    .copied()
                    .flatten()
                    .unwrap_or(false)
                {
                    out.push_str("mut ");
                }
                self.write_kind(program, *tgt, out);
            }
        }
    }

    fn write_life(&self, _program: &Program, id: LifeId, out: &mut String) {
        match self.storage.life.get(id).copied().flatten() {
            Some(LifeKind::Static) => out.push_str("'static"),
            Some(LifeKind::Univeral(Some(id))) => out.push_str(&format!("'u{id}")),
            Some(LifeKind::Univeral(None)) => out.push_str("'u"),
            Some(LifeKind::Local) => out.push_str("'local"),
            None => out.push_str("'_"),
        }
    }
}

impl Default for TypeUniverse {
    fn default() -> Self {
        Self::new()
    }
}

impl BuiltinKind {
    pub fn name(self) -> &'static str {
        match self {
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::Int),
                sign: Some(IntSign::Signed),
            }) => "int",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::Int),
                sign: Some(IntSign::Unsigned),
            }) => "uint",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::I8),
                sign: Some(IntSign::Signed),
            }) => "i8",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::I16),
                sign: Some(IntSign::Signed),
            }) => "i16",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::I32),
                sign: Some(IntSign::Signed),
            }) => "i32",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::I64),
                sign: Some(IntSign::Signed),
            }) => "i64",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::I128),
                sign: Some(IntSign::Signed),
            }) => "i128",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::Isize),
                sign: Some(IntSign::Signed),
            }) => "isize",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::I8),
                sign: Some(IntSign::Unsigned),
            }) => "u8",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::I16),
                sign: Some(IntSign::Unsigned),
            }) => "u16",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::I32),
                sign: Some(IntSign::Unsigned),
            }) => "u32",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::I64),
                sign: Some(IntSign::Unsigned),
            }) => "u64",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::I128),
                sign: Some(IntSign::Unsigned),
            }) => "u128",
            BuiltinKind::Int(IntKind {
                size: Some(IntSize::Isize),
                sign: Some(IntSign::Unsigned),
            }) => "usize",
            BuiltinKind::Int(_) => "int",
            BuiltinKind::Float(FloatKind {
                size: Some(FloatSize::F16),
            }) => "f16",
            BuiltinKind::Float(FloatKind {
                size: Some(FloatSize::F32),
            }) => "f32",
            BuiltinKind::Float(FloatKind {
                size: Some(FloatSize::F64),
            }) => "f64",
            BuiltinKind::Float(_) => "float",
            BuiltinKind::Bool => "bool",
            BuiltinKind::Str => "str",
            BuiltinKind::Void => "void",
            BuiltinKind::Type => "Type",
        }
    }
}

fn projection_name(projection: Projection) -> &'static str {
    match projection {
        Projection::SimpleDeref => "deref",
        Projection::FieldReref(_) => "field_reref",
        Projection::RawReref => "raw_reref",
        Projection::ForgetSafe => "forget_safe",
        Projection::SmartCall => "smart_call",
        Projection::Casted => "casted",
    }
}

fn write_angle_list<T, F>(out: &mut String, items: impl Iterator<Item = T>, mut write: F)
where
    F: FnMut(&mut String, T),
{
    out.push('<');
    for (idx, item) in items.enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        write(out, item);
    }
    out.push('>');
}

#[derive(Debug)]
pub struct SolvedTypes {
    pub typedef_types: IdHashMap<TExpId, KindId>,
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
            member_function_types: HashMap::new(),
        }
    }

    pub fn type_of(&self, value: ValId) -> Option<KindId> {
        self.function_values.get(&value).map(|f| f.ty)
    }

    pub fn pat_type(&self, id: PatId) -> Option<KindId> {
        self.function_values.values().find_map(|f| {
            f.arguments
                .iter()
                .find_map(|(p, _, t)| (*p == id).then_some(*t))
        })
    }

    pub fn function_types_by_name(&self, id: NameId) -> Option<&SolvedFunctionTypes> {
        self.function_types
            .get(&id)
            .and_then(|site| self.function_values.get(site))
    }

    pub fn inner_types_of_function(&self, function: ValId) -> Option<&InnerFunctionTypes> {
        self.function_values
            .get(&function)
            .and_then(|f| f.inner.as_ref())
    }

    pub fn implicit_deref_chain(&self, id: ValId) -> Option<&[(KindId, Projection)]> {
        self.function_values.values().find_map(|f| {
            f.inner
                .as_ref()
                .and_then(|inner| inner.implicit_derefs.get(&id).map(Vec::as_slice))
        })
    }

    pub fn implicit_deref_chain_in_function(
        &self,
        function: ValId,
        id: ValId,
    ) -> Option<&[(KindId, Projection)]> {
        self.inner_types_of_function(function)
            .and_then(|inner| inner.implicit_derefs.get(&id).map(Vec::as_slice))
    }

    pub fn implicit_deref_count(&self, id: ValId) -> Option<usize> {
        self.implicit_deref_chain(id).map(|chain| chain.len())
    }

    pub fn member_access_implicit_deref_chain(&self, id: ValId) -> Option<&[(KindId, Projection)]> {
        self.implicit_deref_chain(id)
    }

    pub fn member_access_implicit_deref_count(&self, id: ValId) -> Option<usize> {
        self.implicit_deref_count(id)
    }

    pub fn member_access_implicit_deref_count_in_function(
        &self,
        function: ValId,
        id: ValId,
    ) -> Option<usize> {
        self.implicit_deref_chain_in_function(function, id)
            .map(|chain| chain.len())
    }

    pub fn index_implicit_deref_chain(&self, id: ValId) -> Option<&[(KindId, Projection)]> {
        self.implicit_deref_chain(id)
    }

    pub fn index_implicit_deref_count(&self, id: ValId) -> Option<usize> {
        self.implicit_deref_count(id)
    }
}

#[derive(Debug, Clone)]
pub struct SolvedFunctionTypes {
    pub ty: KindId,
    pub impl_site: Option<ValId>,
    pub declaration_sites: Vec<ValId>,
    pub arguments: Vec<(PatId, Option<NameId>, KindId)>,
    pub generic_parameters: Vec<(PatId, Option<NameId>)>,
    pub lifetime_parameters: Vec<(PatId, Option<LifeTimeId>)>,
    pub inner: Option<InnerFunctionTypes>,
}

#[derive(Debug, Clone, Default)]
pub struct InnerFunctionTypes {
    pub val_types: IdHashMap<ValId, KindId>,
    pub pat_types: IdHashMap<PatId, KindId>,
    pub member_method_types: IdHashMap<ValId, SolvedMemberMethodAccessType>,
    pub implicit_derefs: IdHashMap<ValId, Vec<(KindId, Projection)>>,
    pub origins: OriginVec<OriginNode>,
    pub value_origins: IdHashMap<ValId, OriginId>,
    pub pattern_origins: IdHashMap<PatId, OriginId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolvedMemberMethodAccessType {
    pub member: StrId,
    pub full_type: KindId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OriginId(pub u32);

#[derive(Debug, Clone)]
pub struct OriginVec<T>(pub Vec<T>);

impl<T> Default for OriginVec<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> OriginVec<T> {
    pub fn get(&self, id: OriginId) -> Option<&T> {
        self.0.get(id.0 as usize)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OriginNode {
    pub effective_mutability: Option<bool>,
}

#[derive(Debug)]
pub enum TypeError {
    Simple {
        loc: Loc,
        message: &'static str,
    },
    SimpleRelated {
        loc: Loc,
        message: &'static str,
        related: Loc,
        related_message: &'static str,
    },
    LifetimeError {
        loc: Loc,
        message: String,
        label: String,
        related: Option<Loc>,
        related_label: Option<String>,
    },
    LifetimeOrderingConflict {
        loc: Loc,
        operation: &'static str,
        shorter: String,
        longer: String,
        related: Option<Loc>,
    },
    IllegalGlobalLifetimeOrdering {
        loc: Loc,
        operation: &'static str,
        shorter: String,
        longer: String,
        related: Option<Loc>,
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
    ExpectedTypeExpr {
        type_expr: TExpId,
    },
    ValuesContradict {
        expectation_reason: &'static str,
        site: ValId,
        found: ValId,
        expected_place: ValId,
        clash: TypeClash,
    },
    BinOpOverloadNotFound {
        site: ValId,
        op: BinOp,
        lhs: ValId,
        rhs: ValId,
        lhs_type: Option<String>,
        rhs_type: Option<String>,
    },
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
    AnnotationMismatch {
        annotation: ValId,
        constrained: ValId,
        clash: TypeClash,
    },
    FunctionOutputAnnotationMismatch {
        output_type: Option<TExpId>,
        constrained: ValId,
        clash: TypeClash,
    },
    PatternAnnotationMismatch {
        annotation: PatId,
        constrained: PatId,
        clash: TypeClash,
    },
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

impl TypeClash {
    pub fn found(&self) -> Option<&str> {
        self.found.as_deref()
    }

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

pub fn run_typechecker(
    program: &Program,
    _reporter: &mut ErrorReporter,
) -> Result<(Result<(TypeUniverse, SolvedTypes), usize>, usize), Box<dyn Error>> {
    Ok((Ok((TypeUniverse::new(), SolvedTypes::new(program))), 0))
}
