use crate::data_structures::index::Idx;
use crate::data_structures::index::IndexSpan;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MutId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OriginId(pub u32);

// #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
// pub struct FKId(pub u32);

impl_idx!(StructId, OriginId, GenId, KindId, PtrId, LifeId, MutId);

pub type KindSpan = IndexSpan<KindId>;
pub type LifeSpan = IndexSpan<LifeId>;

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

impl KindId {
    pub const INT: Self = Self(0);
    pub const UINT: Self = Self(1);
    pub const I8: Self = Self(2);
    pub const I16: Self = Self(3);
    pub const I32: Self = Self(4);
    pub const I64: Self = Self(5);
    pub const I128: Self = Self(6);
    pub const ISIZE: Self = Self(7);
    pub const U8: Self = Self(8);
    pub const U16: Self = Self(9);
    pub const U32: Self = Self(10);
    pub const U64: Self = Self(11);
    pub const U128: Self = Self(12);
    pub const USIZE: Self = Self(13);
    pub const F16: Self = Self(14);
    pub const F32: Self = Self(15);
    pub const F64: Self = Self(16);
    pub const FLOAT: Self = Self::F64;
    pub const BOOL: Self = Self(17);
    pub const STR: Self = Self(18);
    pub const VOID: Self = Self(19);
    pub const TYPE: Self = Self(20);
}

pub const HARD_CODED_BUILTIN_KINDS: &[BuiltinKind] = {
    use FloatSize::*;
    use IntSign::*;
    use IntSize::*;

    // Hardcoded ids are global constants. Do not put partial builtin shapes here:
    // `None` fields are solver variables and can be refined by unification.
    &[
        BuiltinKind::Int(IntKind {
            size: Some(Int),
            sign: Some(Signed),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(Int),
            sign: Some(Unsigned),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(I8),
            sign: Some(Signed),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(I16),
            sign: Some(Signed),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(I32),
            sign: Some(Signed),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(I64),
            sign: Some(Signed),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(I128),
            sign: Some(Signed),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(Isize),
            sign: Some(Signed),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(I8),
            sign: Some(Unsigned),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(I16),
            sign: Some(Unsigned),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(I32),
            sign: Some(Unsigned),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(I64),
            sign: Some(Unsigned),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(I128),
            sign: Some(Unsigned),
        }),
        BuiltinKind::Int(IntKind {
            size: Some(Isize),
            sign: Some(Unsigned),
        }),
        BuiltinKind::Float(FloatKind { size: Some(F16) }),
        BuiltinKind::Float(FloatKind { size: Some(F32) }),
        BuiltinKind::Float(FloatKind { size: Some(F64) }),
        BuiltinKind::Bool,
        BuiltinKind::Str,
        BuiltinKind::Void,
        BuiltinKind::Type,
    ]
};

pub const BUILTINS: &[(&str, KindId)] = {
    &[
        ("int", KindId::INT),
        ("uint", KindId::UINT),
        ("i8", KindId::I8),
        ("i16", KindId::I16),
        ("i32", KindId::I32),
        ("i64", KindId::I64),
        ("i128", KindId::I128),
        ("isize", KindId::ISIZE),
        ("u8", KindId::U8),
        ("u16", KindId::U16),
        ("u32", KindId::U32),
        ("u64", KindId::U64),
        ("u128", KindId::U128),
        ("usize", KindId::USIZE),
        ("f16", KindId::F16),
        ("f32", KindId::F32),
        ("f64", KindId::F64),
        ("float", KindId::FLOAT),
        ("bool", KindId::BOOL),
        ("str", KindId::STR),
        ("void", KindId::VOID),
        ("Type", KindId::TYPE),
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

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// pub enum FuncStyle {
//     Simple,
//     Closure(ValId)
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeKind {
    Builtin(BuiltinKind),
    Generic(GenId),

    Tuple(KindSpan),
    Struct {
        id: StructId,
        gens: KindSpan,
        lifes: LifeSpan,
    },
    Func {
        params: KindSpan,
        ret: KindId,
        // call_style:FKId,
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
