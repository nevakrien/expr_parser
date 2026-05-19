use crate::index::Idx;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
