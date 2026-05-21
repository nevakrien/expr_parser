use crate::ir::PatId;
use crate::ir::ValId;
use crate::type_system::MutId;
use crate::type_system::OrigId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Projection {
    /// &T -> T
    SimpleDeref,

    /// &s -> &s.x
    FieldReref(u32),

    /// *T -> &T
    RawReref,

    /// &T -> *T
    ForgetSafe,

    /// T -> f(T)
    SmartCall,

    /// T -> S
    Casted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OriginKind {
    FuncArg {
        name: PatId,
    },
    Local {
        name: PatId,
    },

    Global {
        val: ValId,
    },

    ///for cases like let x = foo().bar;
    ///this is ALWAYS the result of a function call in IR
    Transient {
        val: ValId,
    },
    Derived {
        parent: OrigId,
        proj: Projection,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Origin {
    pub kind: Option<OriginKind>,
    pub mutability: MutId,
}
