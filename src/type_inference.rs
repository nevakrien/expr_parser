//! Type inference sketch
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

use crate::ErrorReporter;
use crate::identity_hasher::IdHashMap;
use crate::ir::AccessKind;
use crate::ir::CallingConvention;
use crate::ir::StructLayoutSpec;
use crate::ir::StructLike;
use crate::ir::VarKind;
use crate::ir::{
    AssignOp, BinOp, Dir, Literal, NameId, PatId, Pattern, PatternSpan, TExpId, TypeExpr, UnOp,
    ValId, Value,
};
use crate::parsing::Loc;
use crate::string_intern::{
    ADD_STR, BITAND_STR, BITNOT_STR, BITOR_STR, BITXOR_STR, DEREF_MUT_STR, DEREF_STR, DIV_STR,
    EQ_STR, FREE_STR, GE_STR, GT_STR, LE_STR, LT_STR, MOD_STR, MUL_STR, NE_STR, NEG_STR, NOT_STR,
    POST_DEC_STR, POST_INC_STR, PRE_DEC_STR, PRE_INC_STR, SHL_STR, SHR_STR, SUB_STR, StrId,
};
use std::collections::HashMap;
use std::ops::{Index, IndexMut};

use crate::program::{Defined, Program};

// use std::ffi::CStr;
// unsafe extern "C" {
//     fn perf_init();
//     fn perf_begin();
//     fn perf_done(name: *const std::os::raw::c_char);
// }

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
const EXPANSION_LIMIT: usize = 100;

///this type specifically has internals containing UNKNOWN_TYPE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BadTypeId(pub TypeId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(pub usize);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrayType {
    Sized(usize),
    Unsized,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeValue {
    Builtin(BuiltinType),
    Tuple(Vec<TypeId>),
    Array(TypeId, ArrayType),
    Func {
        calling_convention: CallingConvention,
        generics: usize,
        params: Vec<TypeId>,
        ret: TypeId,
    },
    Ptr {
        tgt: TypeId,
        raw: bool,
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
    },
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
        ];

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

    pub(crate) structs: Vec<StructRep>,
}

///todo add actual fields
#[derive(Debug)]
pub struct StructRep {
    pub name: Option<NameId>,
    pub fields: Vec<(NameId, TypeId)>,
    pub gen_count: usize,
    pub layout: StructLayoutSpec,
}

impl StructRep {
    fn new(
        names: impl Iterator<Item = NameId>,
        gen_count: usize,
        layout: StructLayoutSpec,
    ) -> Self {
        Self {
            //TODO: when solving typedefs in finalize we want to set this value
            //for anonymous structs it wont exist but those are rare
            name: None,
            fields: names.map(|x| (x, UNKNOWN_TYPE)).collect(),
            gen_count,
            layout,
        }
    }

    // pub(crate) fn with_fields(name: Option<NameId>, fields: Vec<(NameId, TypeId)>) -> Self {
    //     Self { name, fields,gen_count:0 }
    // }
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

            structs: Vec::new(),
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
        let id = TypeId(self.values.len());
        self.values.push(ty.clone());
        self.intern.insert(ty, id);
        id
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
            layout: StructLayoutSpec::Hot,
        };
        let sid = self.new_struct(rep);
        let tid = self.intern(TypeValue::Struct {
            id: sid,
            generics: Vec::new(),
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
        self.get_type_string_nested(program, t, 0)
    }
    pub fn get_type_string_nested(&self, program: &Program, t: TypeId, gen_count: usize) -> String {
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
                    .map(|id| self.get_type_string_nested(program, *id, gen_count))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({})", inner)
            }
            TypeValue::Func {
                calling_convention,
                generics,
                params,
                ret,
            } => {
                let params = params
                    .iter()
                    .map(|id| self.get_type_string_nested(program, *id, gen_count))
                    .collect::<Vec<_>>()
                    .join(", ");
                let fn_kw = match calling_convention {
                    CallingConvention::Hot => "fn",
                    CallingConvention::C => "cfn",
                    CallingConvention::Unknown => "fn?",
                };
                let generic_params = if *generics == 0 {
                    String::new()
                } else {
                    let pars = (gen_count..(gen_count + generics))
                        .map(|i| format!("T{i}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("[{pars}]")
                };
                format!(
                    "{}{}({}) -> {}",
                    fn_kw,
                    generic_params,
                    params,
                    self.get_type_string_nested(program, *ret, gen_count)
                )
            }
            TypeValue::Ptr { tgt, raw, mutable } => {
                let inner = self.get_type_string_nested(program, *tgt, gen_count);

                match (*raw, *mutable) {
                    (true, true) => format!("*{inner}"),
                    (true, false) => format!("*const {inner}"),
                    (false, true) => format!("&mut {inner}"),
                    (false, false) => format!("&{inner}"),
                }
            }

            TypeValue::Array(inner, ArrayType::Sized(n)) => {
                format!(
                    "[{};{n}]",
                    self.get_type_string_nested(program, *inner, gen_count)
                )
            }

            TypeValue::Array(inner, ArrayType::Unsized) => {
                format!(
                    "[{}]",
                    self.get_type_string_nested(program, *inner, gen_count)
                )
            }

            // TypeValue::Type => "Type".to_string(),
            TypeValue::Generic(g) => format!("T{}", g.0),

            //TODO cover cases where we do know the name
            TypeValue::Struct { id, generics } => {
                self.format_struct_display(program, *id, generics, gen_count)
            }
        }
    }

    fn format_struct_display(
        &self,
        program: &Program,
        sid: StructId,
        generics: &[TypeId],
        gen_count: usize,
    ) -> String {
        let base = match self.struct_value(sid).name {
            Some(name) => program.name_string(name),
            None => "UnamedStruct",
        };
        let mut base = format!("{}{}", base, subscript_id(sid.0));
        if !generics.is_empty() {
            let args = generics
                .iter()
                .map(|id| self.get_type_string_nested(program, *id, gen_count))
                .collect::<Vec<_>>()
                .join(", ");
            base.push('[');
            base.push_str(&args);
            base.push(']');
        }
        base
    }
}

fn subscript_id(id: usize) -> String {
    const SUBS: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
    if id == 0 {
        return SUBS[0].to_string();
    }
    let mut digits = Vec::new();
    let mut n = id;
    while n > 0 {
        digits.push(SUBS[n % 10]);
        n /= 10;
    }
    digits.reverse();
    digits.into_iter().collect()
}

pub struct SolvedTypes {
    pub val_types: Vec<TypeId>,
    pub typedef_types: IdHashMap<TExpId, TypeId>,
    pub pat_types: Vec<TypeId>,
    pub member_method_types: IdHashMap<ValId, SolvedMemberMethodType>,
    pub implicit_derefs: IdHashMap<ValId, Vec<TypeId>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolvedMemberMethodType {
    pub member: StrId,
    pub full_type: TypeId,
}

impl SolvedTypes {
    pub fn new(program: &Program) -> Self {
        let mut typedef_types = IdHashMap::default();
        typedef_types.reserve(program.definitions.len());

        Self {
            pat_types: vec![UNKNOWN_TYPE; program.patterns.len()],
            typedef_types,
            val_types: vec![UNKNOWN_TYPE; program.values.len()],
            member_method_types: IdHashMap::default(),
            implicit_derefs: IdHashMap::default(),
        }
    }

    #[inline]
    pub fn set_val(&mut self, id: ValId, t: TypeId) {
        if self.val_types.len() <= id.0 {
            self.val_types.resize(id.0, UNKNOWN_TYPE);
        }

        self.val_types[id.0] = t;
    }

    #[inline]
    pub fn set_pat(&mut self, id: PatId, t: TypeId) {
        if self.pat_types.len() <= id.0 {
            self.pat_types.resize(id.0, UNKNOWN_TYPE);
        }

        self.pat_types[id.0] = t;
    }

    #[inline(always)]
    pub fn type_of(&self, id: ValId) -> Option<TypeId> {
        let ans = *self.val_types.get(id.0)?;
        if ans == UNKNOWN_TYPE { None } else { Some(ans) }
    }

    #[inline(always)]
    pub fn pat_type(&self, id: PatId) -> Option<TypeId> {
        let ans = *self.pat_types.get(id.0)?;
        if ans == UNKNOWN_TYPE { None } else { Some(ans) }
    }

    #[inline(always)]
    pub fn member_method_type(&self, id: ValId) -> Option<SolvedMemberMethodType> {
        self.member_method_types.get(&id).copied()
    }

    #[inline(always)]
    pub fn implicit_deref_chain(&self, id: ValId) -> Option<&[TypeId]> {
        self.implicit_derefs.get(&id).map(Vec::as_slice)
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
    pub fn index_implicit_deref_chain(&self, id: ValId) -> Option<&[TypeId]> {
        self.implicit_deref_chain(id)
    }

    #[inline(always)]
    pub fn index_implicit_deref_count(&self, id: ValId) -> Option<usize> {
        self.implicit_deref_count(id)
    }
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
    },
    UnresolvedPattern {
        pattern: PatId,
    },

    UnresolvedTypeExpr {
        expr: TExpId,
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

    ConstructorBaseNotGlobal {
        site: ValId,
    },

    ConstructorBaseNotTypeName {
        site: ValId,
    },

    ConstructorBaseNotStruct {
        site: ValId,
        found: Option<BadTypeId>,
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
        lhs_type: Option<BadTypeId>,
        rhs_type: Option<BadTypeId>,
    },

    /// No overload exists for the operator with the given operand type.
    UnOpOverloadNotFound {
        site: ValId,
        op: UnOp,
        operand: ValId,
        operand_type: Option<BadTypeId>,
    },

    CannotDeref {
        site: ValId,
        operand: ValId,
        operand_type: Option<BadTypeId>,
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
}

#[derive(Debug, Clone, Copy)]
pub struct TypeClash {
    pub found: Option<BadTypeId>,
    pub wanted: Option<BadTypeId>,
}

impl TypeClash {
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

    // unsafe{perf_init();}

    if let Err(errs) = infer_global_types(program, &mut types, &mut solved_types) {
        err_count += errs.len();

        for e in errs {
            reporter.report_type_error(program, &types, &e)?;
        }

        return Ok((Err(err_count), function_checked));
    }

    for (_n, methods) in program.member_methods.iter() {
        for (_s, m) in methods.iter() {
            function_checked += 1;

            let Err(errs) = infer_value_internals(program, &mut types, &mut solved_types, *m)
            else {
                continue;
            };
            err_count += errs.len();

            for e in errs {
                reporter.report_type_error(program, &types, &e)?;
            }
        }
    }

    for (_n, def) in program.definitions.iter() {
        let Defined::Func(v) = def else {
            continue;
        };
        function_checked += 1;

        // println!("working on {}",program.name_string(*n));
        // unsafe { std::arch::asm!("int3"); }

        let Err(errs) = infer_value_internals(program, &mut types, &mut solved_types, *v) else {
            continue;
        };
        err_count += errs.len();

        for e in errs {
            reporter.report_type_error(program, &types, &e)?;
        }
    }

    if err_count > 0 {
        return Ok((Err(err_count), function_checked));
    }

    Ok((Ok((types, solved_types)), function_checked))
}

///this function gathers global typedefs/structs
///and just the signature part of global functions
///we dont monomorphise here so its important to do so later
pub fn infer_global_types<'a>(
    program: &'a Program,
    store: &'a mut TypeStore,
    ans: &'a mut SolvedTypes,
) -> Result<&'a mut SolvedTypes, Vec<TypeError>> {
    let mut ctx = InferState::new(store, program, ans);

    for (n, def) in program.definitions.iter() {
        let Defined::Type(texp) = def else {
            continue;
        };

        let t = do_typedef::<true>(&mut ctx, *n, *texp);
        if let Some(previous) = ctx.search.local_types.insert(*n, t)
            && let Err(clash) = ctx.unify(previous, t)
        {
            ctx.push_error(TypeError::TypeClashBeforeMentioned {
                name: *n,
                expr: *texp,
                clash,
            });
        }
        if let ResolveKind::Solved(ty) = ctx.types.core.cluster[t].state {
            ctx.ex.ans.typedef_types.insert(*texp, ty);
        } else {
            ctx.search.typedef_cluster.push((*texp, t));
        }
    }

    main_solver(&mut ctx);
    if !ctx.ex.errors.is_empty() {
        return Err(ctx.ex.errors);
    }

    for (struct_name, methods) in program.member_methods.iter() {
        for (method_name, m) in methods.iter() {
            //each function must solve by itself.
            //since there isnt a body its fine to solve in order
            //note that namespace on generics gurntees this works for the most outer scope
            if let Value::Func {
                calling_convention,
                generics,
                params,
                output_type,
                body: _,
            } = ctx.ex.program.value(*m)
            {
                ctx.clear_local_state();
                let _ = type_check_func_signature(
                    &mut ctx,
                    *m,
                    calling_convention,
                    generics,
                    params,
                    output_type,
                );
                check_special_member_method_signature(&mut ctx, *m, *struct_name, *method_name);
            };
        }
        check_struct_deref_targets_compatible(&mut ctx, *struct_name);
    }

    for (_n, def) in program.definitions.iter() {
        let Defined::Func(v) = def else {
            continue;
        };

        //each function must solve by itself.
        //since there isnt a body its fine to solve in order
        //note that namespace on generics gurntees this works for the most outer scope
        if let Value::Func {
            calling_convention,
            generics,
            params,
            output_type,
            body: _,
        } = ctx.ex.program.value(*v)
        {
            ctx.clear_local_state();
            type_check_func_signature(
                &mut ctx,
                *v,
                calling_convention,
                generics,
                params,
                output_type,
            );
        };
    }

    if ctx.ex.errors.is_empty() {
        Ok(ctx.ex.ans)
    } else {
        Err(ctx.ex.errors)
    }
}

pub fn infer_value_internals<'a>(
    program: &'a Program,
    store: &'a mut TypeStore,
    ans: &'a mut SolvedTypes,
    value: ValId,
) -> Result<&'a mut SolvedTypes, Vec<TypeError>> {
    let known = ans.type_of(value);
    let mut ctx = InferState::new(store, program, ans);

    match ctx.ex.program.value(value) {
        Value::Func {
            calling_convention,
            generics,
            params,
            output_type,
            body,
        } => {
            let (found_sig, output) = gather_func_signature::<true>(
                &mut ctx,
                value,
                calling_convention,
                generics,
                params,
                output_type,
            );

            if let Some(known) = known {
                let known_sig = ctx.new_solved(known);
                if let Err(clash) = ctx.unify(found_sig, known_sig) {
                    ctx.push_error(TypeError::ValuesContradict {
                        expectation_reason:
                            "function signature should match previously solved global signature",
                        site: value,
                        found: value,
                        expected_place: value,
                        clash,
                    });
                }
            }

            if let Some(body) = body {
                let body_cluster = gather_constraints(&mut ctx, body, Some(output));
                if let Err(clash) = ctx.unify(body_cluster, output) {
                    let found = match ctx.ex.program.value(body) {
                        Value::Block {
                            statements: _,
                            return_value: Some(x),
                        } => x,
                        _ => body,
                    };
                    ctx.push_error(TypeError::FunctionOutputAnnotationMismatch {
                        output_type,
                        constrained: found,
                        clash,
                    });
                }
            }

            found_sig
        }
        _ => {
            let found = gather_constraints(&mut ctx, value, None);
            if let Some(known) = known {
                let known = ctx.new_solved(known);

                if let Err(clash) = ctx.unify(found, known) {
                    ctx.push_error(TypeError::ValuesContradict{
                        expectation_reason: "expected value signature to match global signature (this is likely ALSO an internal bug in error reporting)",
                        site:value,
                        found:value,
                        expected_place:value,
                        clash,
                    })
                }
            }

            found
        }
    };

    main_solver(&mut ctx);

    //this debug assert is mostly meaningless
    //it shouldnt even be SET by us in the firstplace
    //we specifically do NOT bind_val and finalize cant handle generics
    //so this trigers as soon as we fuckup and bind_val ourselvs on anything with generics
    if let Some(known) = known {
        debug_assert_eq!(known, ctx.ex.ans.type_of(value).unwrap())
    }

    if ctx.ex.errors.is_empty() {
        Ok(ctx.ex.ans)
    } else {
        Err(ctx.ex.errors)
    }
}

///this is just for tests we PURPOSFULLY ignore the global sig resolution
fn _infer_value_hacky<'a>(
    program: &'a Program,
    store: &'a mut TypeStore,
    ans: &'a mut SolvedTypes,

    value: ValId,
) -> Result<&'a mut SolvedTypes, Vec<TypeError>> {
    let mut ctx = InferState::new(store, program, ans);

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
            gather_constraints(&mut ctx, value, None);
        }
    }

    main_solver(&mut ctx);
    if ctx.ex.errors.is_empty() {
        Ok(ctx.ex.ans)
    } else {
        Err(ctx.ex.errors)
    }
}

fn main_solver(ctx: &mut InferState) {
    //this loop only exists once ALL requirments have checked and didnt complain
    //on the state we are gona release. since there was no change
    //this is SUPER important because they are not just progressions
    loop {
        let mut progress = false;
        progress |= resolve_operator_types(ctx);
        progress |= resolve_deferred_types(ctx);
        progress |= resolve_pointer_likes(ctx);
        progress |= resolve_pending_indexes(ctx);
        progress |= resolve_pending_member_accesses(ctx);
        progress |= resolve_pending_specializations(ctx);
        if !progress {
            break;
        }
    }

    if !ctx.ex.errors.is_empty() {
        return;
    }

    finalize(ctx);
}

// ===================================
// Inference state + unify-find clusters
// ===================================
pub struct InferState<'a> {
    ex: ExternState<'a>,
    search: SearchState,
    types: TypeState,
    req: ReqState,
}
impl<'a> InferState<'a> {
    pub fn new(store: &'a mut TypeStore, program: &'a Program, ans: &'a mut SolvedTypes) -> Self {
        Self {
            ex: ExternState {
                store,
                program,
                errors: Vec::new(),
                ans,
            },

            search: SearchState::new(),
            types: TypeState::new(),
            req: ReqState::new(),
        }
    }

    fn new_cluster(&mut self) -> CId {
        self.types.new_cluster()
    }

    fn new_solved(&mut self, t: TypeId) -> CId {
        self.types.new_solved(t)
    }

    fn new_int_like(&mut self) -> CId {
        self.types.new_int_like()
    }

    fn new_float_like(&mut self) -> CId {
        self.types.new_float_like()
    }

    fn new_func(&mut self, call: FuncInfer) -> CId {
        self.types.new_func(call)
    }

    fn new_struct_instance(&mut self, sid: StructId, generics: Vec<CId>) -> CId {
        self.types.new_struct_instance(sid, generics)
    }

    fn new_tuple_instance(&mut self, items: Vec<CId>) -> CId {
        self.types.new_tuple_instance(items)
    }

    fn new_array_instance(&mut self, element: CId, size: ArrayType) -> CId {
        self.types.new_array_instance(element, size)
    }

    fn bind_val(&mut self, v: ValId, c: CId) {
        self.search.bind_val(v, c);
    }

    fn bind_pat(&mut self, p: PatId, c: CId) {
        self.search.bind_pat(p, c);
    }

    fn unify(&mut self, a: CId, b: CId) -> Result<CId, TypeClash> {
        self.types.unify(&mut self.ex, a, b)
    }

    fn force_type(&mut self, a: CId, t: TypeId) -> Result<(), TypeClash> {
        self.types.force_type(&mut self.ex, a, t)
    }

    pub fn push_error(&mut self, e: TypeError) {
        self.ex.push_error(e);
    }

    pub fn clear_local_state(&mut self) {
        self.search.clear_local_state();
        self.types.clear_local_state();
        self.req.clear_local_state();
    }
}

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
    #[allow(dead_code)]
    fn swap(&mut self, a: CId, b: CId) {
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

#[derive(Debug)]
struct Cluster {
    state: ResolveKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FuncInferId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StructInferId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TupleInferId(usize);

#[derive(Debug, Clone, Copy)]
enum ResolveKind {
    Solved(TypeId),
    Nothing,
    Never,

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
        raw: Option<bool>,
        mutable: Option<bool>,
    },
}

#[derive(Debug)]
struct FuncInfer {
    calling_convention: CallingConvention,
    generics: usize,
    inputs: Vec<CId>,
    output: CId,
}

#[derive(Debug)]
struct StructInfer {
    sid: StructId,
    generics: Vec<CId>,
}

#[derive(Debug)]
struct TupleInfer {
    items: Vec<CId>,
}

#[derive(Debug)]
struct StructDef {
    #[allow(dead_code)]
    loc: TExpId,
    fields: Vec<(NameId, CId)>,
    sid: StructId,
}

#[allow(dead_code)]
#[derive(Debug)]
struct Specialized {
    loc: Loc,
    base: CId,
    fields: Vec<CId>,
    output: CId,
}

#[derive(Debug)]
struct PendingSpecialization {
    name: NameId,
    global: TExpId,
    generics: Vec<CId>,
    output: CId,
}

#[allow(dead_code)]
#[derive(Debug)]
struct ComplexCallSite {
    loc: ValId,
    loc_called: ValId,

    called: CId,
    position_args: Vec<CId>,
    ///the strid can only be resolved once we know what we call;
    /// for structs thats just the type extra info
    /// for functions we need to know the actual specific one (which is a dependent type)
    named_args: Vec<(StrId, CId)>,
    output: CId,
}

#[derive(Debug, Clone, Copy)]
struct BinOpSite {
    loc: ValId,
    op: BinOp,
    lhs_val: ValId,
    rhs_val: ValId,
    lhs: CId,
    rhs: CId,
    output: CId,
}

#[derive(Debug, Clone, Copy)]
struct UnOpSite {
    loc: ValId,
    op: UnOp,
    val: ValId,
    input: CId,
    output: CId,
}

#[derive(Debug, Clone, Copy)]
enum AssignIncDecFlavor {
    PreInc,
    PostInc,
    PreDec,
    PostDec,
}

#[derive(Debug, Clone, Copy)]
struct AssignPrePostSite {
    loc: ValId,
    target_val: ValId,
    target: CId,
    implicit_rhs: CId,
    flavor: AssignIncDecFlavor,
}

#[derive(Debug, Clone, Copy)]
struct PendingMemberMethodType {
    site: ValId,
    member: StrId,
    full_method: CId,
    receiver: CId,
    receiver_value: ValId,
}

#[derive(Debug, Clone)]
struct PendingMemberAccessImplicitDeref {
    site: ValId,
    receivers: Vec<CId>,
}

#[derive(Debug, Clone, Copy)]
struct PendingMemberAccess {
    site: ValId,
    base_value: ValId,
    source: CId,
    output: CId,
    member: StrId,
    kind: AccessKind,
}

#[derive(Debug, Clone, Copy)]
struct PendingPointerLike {
    site: ValId,
    source: CId,
    target: CId,
    source_value: ValId,
}

#[derive(Debug, Clone)]
struct PendingIndex {
    site: ValId,
    base_value: ValId,
    index_value: ValId,
    base: CId,
    index: CId,
    output: CId,
    implicit_receivers: Vec<CId>,
}

struct ExternState<'a> {
    store: &'a mut TypeStore,
    program: &'a Program,

    //result
    errors: Vec<TypeError>,
    ans: &'a mut SolvedTypes,
}

impl<'a> ExternState<'a> {
    fn push_error(&mut self, err: TypeError) {
        self.errors.push(err);
    }
}

struct SearchState {
    //ir -> cid
    val_cluster: Vec<(ValId, CId)>,
    pat_cluster: Vec<(PatId, CId)>,
    typedef_cluster: Vec<(TExpId, CId)>,
    local_types: IdHashMap<NameId, CId>,
    names: IdHashMap<NameId, CId>,
}

impl SearchState {
    fn new() -> Self {
        Self {
            val_cluster: Vec::default(),
            pat_cluster: Vec::default(),
            typedef_cluster: Vec::default(),
            local_types: IdHashMap::default(),
            names: IdHashMap::default(),
        }
    }

    fn clear_local_state(&mut self) {
        let SearchState {
            val_cluster,
            pat_cluster,
            typedef_cluster,
            local_types,
            names,
        } = self;

        val_cluster.clear();
        pat_cluster.clear();
        typedef_cluster.clear();
        local_types.clear();
        names.clear();
    }

    fn bind_val(&mut self, v: ValId, c: CId) {
        self.val_cluster.push((v, c));
    }

    fn bind_pat(&mut self, p: PatId, c: CId) {
        self.pat_cluster.push((p, c));
    }
}
struct TypeCore {
    // unify-find
    parent: ClusterVec<CId>,
    cluster: ClusterVec<Cluster>,
}

impl TypeCore {
    fn find_root(&mut self, x: CId) -> CId {
        find_root(&mut self.parent, x)
    }

    fn new_cluster(&mut self) -> CId {
        let id = CId(self.parent.len());
        self.parent.0.push(id);
        self.cluster.0.push(Cluster {
            state: ResolveKind::Nothing,
        });
        id
    }
}

struct TypeExtra {
    func_defs: Vec<FuncInfer>,
    struct_defs: Vec<StructDef>,
    struct_infers: Vec<StructInfer>,
    tuple_infers: Vec<TupleInfer>,
}

struct TypeState {
    core: TypeCore,
    extra: TypeExtra,
}

impl TypeState {
    fn new() -> Self {
        Self {
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
        }
    }

    fn clear_local_state(&mut self) {
        // ---- union find ----
        self.core.parent.0.clear();
        self.core.cluster.0.clear();

        // ---- type database ----
        self.extra.func_defs.clear();
        self.extra.struct_defs.clear();
        self.extra.struct_infers.clear();
        self.extra.tuple_infers.clear();
    }

    // =========================================================
    // cluster construction
    // =========================================================

    fn new_cluster(&mut self) -> CId {
        let id = CId(self.core.parent.len());
        self.core.parent.0.push(id);
        self.core.cluster.0.push(Cluster {
            state: ResolveKind::Nothing,
        });
        id
    }

    fn new_solved(&mut self, t: TypeId) -> CId {
        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::Solved(t);
        id
    }

    fn new_int_like(&mut self) -> CId {
        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::IntLike;
        id
    }

    fn new_float_like(&mut self) -> CId {
        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::FloatLike;
        id
    }

    fn new_func(&mut self, call: FuncInfer) -> CId {
        let call_id = FuncInferId(self.extra.func_defs.len());
        self.extra.func_defs.push(call);

        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::Func(call_id);
        id
    }

    fn new_struct_instance(&mut self, sid: StructId, generics: Vec<CId>) -> CId {
        let call_id = StructInferId(self.extra.struct_infers.len());
        self.extra.struct_infers.push(StructInfer { sid, generics });

        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::Struct(call_id);
        id
    }

    fn new_tuple_instance(&mut self, items: Vec<CId>) -> CId {
        let tuple_id = TupleInferId(self.extra.tuple_infers.len());
        self.extra.tuple_infers.push(TupleInfer { items });

        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::Tuple(tuple_id);
        id
    }

    fn new_array_instance(&mut self, element: CId, size: ArrayType) -> CId {
        let id = self.new_cluster();
        self.core.cluster[id].state = ResolveKind::Array { element, size };
        id
    }

    // =========================================================
    // union-find operations
    // =========================================================

    fn unify(&mut self, ex: &mut ExternState<'_>, a: CId, b: CId) -> Result<CId, TypeClash> {
        unify_clusters(ex, self, a, b)
    }

    fn force_type(&mut self, ex: &mut ExternState<'_>, a: CId, t: TypeId) -> Result<(), TypeClash> {
        // thin wrapper over old function until we migrate it
        force_type(ex, self, a, t)
    }
}

struct ReqState {
    //requirments
    bin_op_sites: Vec<BinOpSite>,
    un_op_sites: Vec<UnOpSite>,
    assign_pre_post_sites: Vec<AssignPrePostSite>,

    //generic_func_values: Vec<(ValId, usize)>,
    pending_specializations: Vec<PendingSpecialization>,
    member_method_type_sites: Vec<PendingMemberMethodType>,
    member_access_implicit_deref_sites: Vec<PendingMemberAccessImplicitDeref>,
    index_implicit_deref_sites: Vec<PendingMemberAccessImplicitDeref>,
    pending_member_accesses: Vec<PendingMemberAccess>,
    pending_indexes: Vec<PendingIndex>,
    pointer_likes: Vec<PendingPointerLike>,
}

impl ReqState {
    fn new() -> Self {
        Self {
            bin_op_sites: Vec::new(),
            un_op_sites: Vec::new(),
            assign_pre_post_sites: Vec::new(),

            pending_specializations: Vec::new(),
            member_method_type_sites: Vec::new(),
            member_access_implicit_deref_sites: Vec::new(),
            index_implicit_deref_sites: Vec::new(),
            pending_member_accesses: Vec::new(),
            pending_indexes: Vec::new(),
            pointer_likes: Vec::new(),
        }
    }

    fn clear_local_state(&mut self) {
        let ReqState {
            bin_op_sites,
            un_op_sites,
            assign_pre_post_sites,
            pending_specializations,
            member_method_type_sites,
            member_access_implicit_deref_sites,
            index_implicit_deref_sites,
            pending_member_accesses,
            pending_indexes,
            pointer_likes,
        } = self;

        bin_op_sites.clear();
        un_op_sites.clear();
        assign_pre_post_sites.clear();

        pending_specializations.clear();
        member_method_type_sites.clear();
        member_access_implicit_deref_sites.clear();
        index_implicit_deref_sites.clear();
        pending_member_accesses.clear();
        pending_indexes.clear();
        pointer_likes.clear();
    }
}

// ===========================
// Keep this helper as-is
// (it is used by unify/force)
// ===========================

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
///inference can continue with the 2 types separated for the purpose of gathering more errors (obviously not Unresolved style errors)
fn unify_clusters(
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

    #[inline(always)]
    pub fn is_never(&self, c: CId) -> bool {
        matches!(self.cluster_state(c), ResolveKind::Never)
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

    pub fn bad_type(&mut self, ex: &mut ExternState, cid: CId) -> Option<BadTypeId> {
        let mut limit = EXPANSION_LIMIT;
        extract_bad_type(ex, &mut self.core, &self.extra, cid, &mut limit)
    }

    pub fn clash(&mut self, ex: &mut ExternState, found: CId, wanted: CId) -> TypeClash {
        TypeClash {
            found: self.bad_type(ex, found),
            wanted: self.bad_type(ex, wanted),
        }
    }
}

#[inline(always)]
fn unify_clusters_inlined(
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

    if types.is_never(rw) {
        return Ok(rf);
    }
    if types.is_never(rf) {
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
                    found: Some(BadTypeId(t2)),
                    wanted: Some(BadTypeId(t1)),
                })
            }
        }

        // =====================================================
        // Solved absorbs literals if compatible
        // =====================================================
        (Solved(t), IntLike) => {
            if !ex.store.is_int_like(t) {
                return Err(TypeClash {
                    found: Some(BadTypeId(UNKNOWN_INT_SIZE)),
                    wanted: Some(BadTypeId(t)),
                });
            }
            Ok(true)
        }

        (Solved(t), FloatLike) => {
            if !ex.store.is_float_like(t) {
                return Err(TypeClash {
                    found: Some(BadTypeId(UNKNOWN_FLOAT_SIZE)),
                    wanted: Some(BadTypeId(t)),
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
            let dst_s = types.struct_infer(dst_call);
            let src_s = types.struct_infer(src_call);

            if dst_s.sid != src_s.sid || dst_s.generics.len() != src_s.generics.len() {
                return Err(types.clash(ex, dst, src));
            }

            for i in 0..dst_s.generics.len() {
                let dst_s = types.struct_infer(dst_call);
                let src_s = types.struct_infer(src_call);

                if types
                    .unify(ex, dst_s.generics[i], src_s.generics[i])
                    .is_err()
                {
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
        (Solved(t), Ptr { tgt, raw, mutable }) => {
            unify_ptr_with_type(ex, types, tgt, raw, mutable, t)?;
            Ok(true)
        }

        (
            Ptr {
                tgt: dst_tgt,
                raw: dst_raw,
                mutable: dst_mut,
            },
            Ptr {
                tgt: src_tgt,
                raw: src_raw,
                mutable: src_mut,
            },
        ) => {
            let raw = merge_ptr_flag(dst_raw, src_raw).ok_or_else(|| types.clash(ex, dst, src))?;

            let mutable =
                merge_ptr_flag(dst_mut, src_mut).ok_or_else(|| types.clash(ex, dst, src))?;

            let tgt = types.unify(ex, dst_tgt, src_tgt)?;

            types.set_cluster_state(dst, Ptr { tgt, raw, mutable });
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

fn force_type_if_distinct(
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

fn force_type(
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

        ResolveKind::Solved(t) => Err(simple_type_clash(t, ty)),

        ResolveKind::IntLike => {
            if !ex.store.is_int_like(ty) {
                return Err(TypeClash {
                    found: Some(BadTypeId(UNKNOWN_INT_SIZE)),
                    wanted: Some(BadTypeId(ty)),
                });
            }
            types.set_cluster_state(root, ResolveKind::Solved(ty));
            Ok(())
        }

        ResolveKind::FloatLike => {
            if !ex.store.is_float_like(ty) {
                return Err(TypeClash {
                    found: Some(BadTypeId(UNKNOWN_FLOAT_SIZE)),
                    wanted: Some(BadTypeId(ty)),
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

        ResolveKind::Ptr { tgt, raw, mutable } => {
            unify_ptr_with_type(ex, types, tgt, raw, mutable, ty)?;
            types.set_cluster_state(root, ResolveKind::Solved(ty));
            Ok(())
        }

        ResolveKind::Never => Ok(()),
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

fn unify_ptr_with_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    tgt: CId,
    raw: Option<bool>,
    mutable: Option<bool>,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let found_ptr = BadTypeId(make_ptr_mock(
        ex,
        &mut types.core,
        &types.extra,
        tgt,
        raw,
        mutable,
    ));

    let TypeValue::Ptr {
        tgt: ty_tgt,
        raw: ty_raw,
        mutable: ty_mut,
    } = *ex.store.type_value(ty)
    else {
        return Err(TypeClash {
            found: Some(found_ptr),
            wanted: Some(BadTypeId(ty)),
        });
    };

    if matches!(raw, Some(x) if x != ty_raw) || matches!(mutable, Some(x) if x != ty_mut) {
        return Err(TypeClash {
            found: Some(found_ptr),
            wanted: Some(BadTypeId(ty)),
        });
    }

    force_type(ex, types, tgt, ty_tgt)
}

fn unify_func_with_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    call: FuncInferId,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let found_func = BadTypeId(make_func_mock(ex, &mut types.core, &types.extra, call));

    let (cc, generics, params, ret) = match ex.store.type_value(ty) {
        TypeValue::Func {
            calling_convention,
            generics,
            params,
            ret,
        } => (*calling_convention, *generics, params.as_slice(), *ret),
        _ => {
            return Err(TypeClash {
                found: Some(found_func),
                wanted: Some(BadTypeId(ty)),
            });
        }
    };

    let infer_cc = types.extra.func_defs[call.0].calling_convention;
    let Some(merged_cc) = merge_calling_convention(infer_cc, cc) else {
        return Err(TypeClash {
            found: Some(found_func),
            wanted: Some(BadTypeId(ty)),
        });
    };

    types.extra.func_defs[call.0].calling_convention = merged_cc;

    if types.extra.func_defs[call.0].generics != generics {
        return Err(TypeClash {
            found: Some(found_func),
            wanted: Some(BadTypeId(ty)),
        });
    }

    let input_len = types.extra.func_defs[call.0].inputs.len();
    if params.len() != input_len {
        return Err(TypeClash {
            found: Some(found_func),
            wanted: Some(BadTypeId(ty)),
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

fn unify_struct_with_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    call: StructInferId,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let found_struct = BadTypeId(make_struct_mock(ex, &mut types.core, &types.extra, call));

    let (sid, glen) = match ex.store.type_value(ty) {
        TypeValue::Struct { id, generics } => (*id, generics.len()),
        _ => {
            return Err(TypeClash {
                found: Some(found_struct),
                wanted: Some(BadTypeId(ty)),
            });
        }
    };

    let call_sid = types.extra.struct_infers[call.0].sid;

    if call_sid != sid || types.extra.struct_infers[call.0].generics.len() != glen {
        return Err(TypeClash {
            found: Some(found_struct),
            wanted: Some(BadTypeId(ty)),
        });
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

fn unify_tuple_with_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    tuple: TupleInferId,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let found_tuple = BadTypeId(make_tuple_mock(ex, &mut types.core, &types.extra, tuple));

    let ilen = types.extra.tuple_infers[tuple.0].items.len();

    let TypeValue::Tuple(items) = ex.store.type_value(ty) else {
        return Err(TypeClash {
            found: Some(found_tuple),
            wanted: Some(BadTypeId(ty)),
        });
    };

    if items.len() != ilen {
        return Err(TypeClash {
            found: Some(found_tuple),
            wanted: Some(BadTypeId(ty)),
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

fn unify_array_with_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    element: CId,
    size: ArrayType,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let found_array = BadTypeId(make_array_mock(
        ex,
        &mut types.core,
        &types.extra,
        element,
        size,
    ));

    let (ty_element, ty_size) = match ex.store.type_value(ty) {
        TypeValue::Array(item, n) => (*item, *n),
        _ => {
            return Err(TypeClash {
                found: Some(found_array),
                wanted: Some(BadTypeId(ty)),
            });
        }
    };

    if ty_size != size {
        return Err(TypeClash {
            found: Some(found_array),
            wanted: Some(BadTypeId(ty)),
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

    let mut generics = Vec::with_capacity(site.generics.len());

    for i in 0..site.generics.len() {
        let input = site.generics[i];

        let root = find_root(&mut types.core.parent, input);
        site.generics[i] = root; // canonicalize

        match types.core.cluster[root].state {
            ResolveKind::Solved(t) => generics.push(t),
            _ => return None,
        }
    }

    Some(ex.store.intern(TypeValue::Struct {
        id: site.sid,
        generics,
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
    raw: Option<bool>,
    mutable: Option<bool>,
) -> Option<TypeId> {
    let raw = raw?;
    let mutable = mutable?;

    let root = types.root(tgt);

    let tgt = match types.cluster_state(root) {
        ResolveKind::Solved(t) => t,
        _ => return None,
    };

    Some(ex.store.intern(TypeValue::Ptr { tgt, raw, mutable }))
}

fn simple_type_clash(a: TypeId, b: TypeId) -> TypeClash {
    TypeClash {
        found: Some(BadTypeId(a)),
        wanted: Some(BadTypeId(b)),
    }
}

//TODO: this should actually check if some of the types are known
// we wana do recursive partial resolution
fn mock_type_from_cluster(
    ex: &mut ExternState,
    core: &mut TypeCore,
    extra: &TypeExtra,
    cid: CId,
    limit: &mut usize,
) -> TypeId {
    if *limit == 0 {
        return EXPANSION_STOPED;
    }
    *limit -= 1;

    let root = find_root(&mut core.parent, cid);

    let ty = match core.cluster[root].state {
        ResolveKind::Solved(t) => t,
        ResolveKind::IntLike => UNKNOWN_INT_SIZE,
        ResolveKind::FloatLike => UNKNOWN_FLOAT_SIZE,
        ResolveKind::Func(call) => make_func_mock_inner(ex, core, extra, call, limit),
        ResolveKind::Struct(call) => make_struct_mock_inner(ex, core, extra, call, limit),
        ResolveKind::Tuple(call) => make_tuple_mock_inner(ex, core, extra, call, limit),
        ResolveKind::Array { element, size } => {
            make_array_mock_inner(ex, core, extra, element, size, limit)
        }
        ResolveKind::Ptr { tgt, raw, mutable } => {
            make_ptr_mock_inner(ex, core, extra, tgt, raw, mutable, limit)
        }
        ResolveKind::Nothing | ResolveKind::Never => UNKNOWN_TYPE,
    };

    *limit += 1;
    ty
}

fn make_func_mock_inner(
    ex: &mut ExternState,
    core: &mut TypeCore,
    extra: &TypeExtra,
    call: FuncInferId,
    limit: &mut usize,
) -> TypeId {
    let site = &extra.func_defs[call.0];
    let params = site
        .inputs
        .iter()
        .map(|&input| mock_type_from_cluster(ex, core, extra, input, limit))
        .collect::<Vec<_>>();
    let ret = mock_type_from_cluster(ex, core, extra, site.output, limit);

    ex.store.intern(TypeValue::Func {
        calling_convention: site.calling_convention,
        generics: site.generics,
        params,
        ret,
    })
}

fn make_func_mock(
    ex: &mut ExternState,
    core: &mut TypeCore,
    extra: &TypeExtra,
    call: FuncInferId,
) -> TypeId {
    let mut limit = EXPANSION_LIMIT;
    make_func_mock_inner(ex, core, extra, call, &mut limit)
}

fn make_struct_mock_inner(
    ex: &mut ExternState,
    core: &mut TypeCore,
    extra: &TypeExtra,
    call: StructInferId,
    limit: &mut usize,
) -> TypeId {
    let site = &extra.struct_infers[call.0];
    let generics = site
        .generics
        .iter()
        .map(|&input| mock_type_from_cluster(ex, core, extra, input, limit))
        .collect::<Vec<_>>();
    ex.store.intern(TypeValue::Struct {
        id: site.sid,
        generics,
    })
}

fn make_struct_mock(
    ex: &mut ExternState,
    core: &mut TypeCore,
    extra: &TypeExtra,
    call: StructInferId,
) -> TypeId {
    let mut limit = EXPANSION_LIMIT;
    make_struct_mock_inner(ex, core, extra, call, &mut limit)
}

fn make_tuple_mock_inner(
    ex: &mut ExternState,
    core: &mut TypeCore,
    extra: &TypeExtra,
    tuple: TupleInferId,
    limit: &mut usize,
) -> TypeId {
    let items = extra.tuple_infers[tuple.0]
        .items
        .iter()
        .map(|&item| mock_type_from_cluster(ex, core, extra, item, limit))
        .collect::<Vec<_>>();
    ex.store.intern(TypeValue::Tuple(items))
}

fn make_tuple_mock(
    ex: &mut ExternState,
    core: &mut TypeCore,
    extra: &TypeExtra,
    tuple: TupleInferId,
) -> TypeId {
    let mut limit = EXPANSION_LIMIT;
    make_tuple_mock_inner(ex, core, extra, tuple, &mut limit)
}

fn make_array_mock_inner(
    ex: &mut ExternState,
    core: &mut TypeCore,
    extra: &TypeExtra,
    element: CId,
    size: ArrayType,
    limit: &mut usize,
) -> TypeId {
    let element = mock_type_from_cluster(ex, core, extra, element, limit);
    ex.store.intern(TypeValue::Array(element, size))
}

fn make_array_mock(
    ex: &mut ExternState,
    core: &mut TypeCore,
    extra: &TypeExtra,
    element: CId,
    size: ArrayType,
) -> TypeId {
    let mut limit = EXPANSION_LIMIT;
    make_array_mock_inner(ex, core, extra, element, size, &mut limit)
}

fn make_ptr_mock_inner(
    ex: &mut ExternState,
    core: &mut TypeCore,
    extra: &TypeExtra,
    tgt: CId,
    raw: Option<bool>,
    mutable: Option<bool>,
    limit: &mut usize,
) -> TypeId {
    let tgt = mock_type_from_cluster(ex, core, extra, tgt, limit);
    ex.store.intern(TypeValue::Ptr {
        tgt,
        raw: raw.unwrap_or(false),
        mutable: mutable.unwrap_or(false),
    })
}

fn make_ptr_mock(
    ex: &mut ExternState,
    core: &mut TypeCore,
    extra: &TypeExtra,
    tgt: CId,
    raw: Option<bool>,
    mutable: Option<bool>,
) -> TypeId {
    let mut limit = EXPANSION_LIMIT;
    make_ptr_mock_inner(ex, core, extra, tgt, raw, mutable, &mut limit)
}

fn extract_bad_type(
    ex: &mut ExternState,
    core: &mut TypeCore,
    extra: &TypeExtra,
    cid: CId,
    limit: &mut usize,
) -> Option<BadTypeId> {
    if *limit == 0 {
        return Some(BadTypeId(EXPANSION_STOPED));
    }
    *limit -= 1;

    let root = find_root(&mut core.parent, cid);
    let out = match core.cluster[root].state {
        ResolveKind::Solved(t) => Some(BadTypeId(t)),
        ResolveKind::Nothing | ResolveKind::Never => None,

        ResolveKind::Func(call) => Some(BadTypeId(make_func_mock(ex, core, extra, call))),
        ResolveKind::Struct(call) => Some(BadTypeId(make_struct_mock(ex, core, extra, call))),
        ResolveKind::Tuple(call) => Some(BadTypeId(make_tuple_mock(ex, core, extra, call))),
        ResolveKind::Array { element, size } => {
            Some(BadTypeId(make_array_mock(ex, core, extra, element, size)))
        }
        ResolveKind::Ptr { tgt, raw, mutable } => {
            Some(BadTypeId(make_ptr_mock(ex, core, extra, tgt, raw, mutable)))
        }

        ResolveKind::IntLike => Some(BadTypeId(UNKNOWN_INT_SIZE)),
        ResolveKind::FloatLike => Some(BadTypeId(UNKNOWN_FLOAT_SIZE)),
    };

    *limit += 1;
    out
}

// ============================================================
// Specialization (monomorphisation into local clusters)
// ============================================================

fn specialize_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    ty: TypeId,
    generics: &[CId],
    loc: ValId,
) -> CId {
    match ex.store.type_value(ty).clone() {
        TypeValue::Generic(id) => generics.get(id.0).copied().unwrap(),

        TypeValue::Func {
            calling_convention,
            generics: _,
            params,
            ret,
        } => {
            let inputs = params
                .into_iter()
                .map(|t| specialize_type(ex, types, t, generics, loc))
                .collect::<Vec<_>>();

            let output = specialize_type(ex, types, ret, generics, loc);

            // create FuncInfer
            let call_id = FuncInferId(types.extra.func_defs.len());
            types.extra.func_defs.push(FuncInfer {
                generics: 0,
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
            generics: parts,
        } => {
            if parts.is_empty() {
                let idc = CId(types.core.parent.len());
                types.core.parent.0.push(idc);
                types.core.cluster.0.push(Cluster {
                    state: ResolveKind::Solved(ty),
                });
                return idc;
            }

            let resolved = parts
                .into_iter()
                .map(|t| specialize_type(ex, types, t, generics, loc))
                .collect::<Vec<_>>();

            let call_id = StructInferId(types.extra.struct_infers.len());
            types.extra.struct_infers.push(StructInfer {
                sid: id,
                generics: resolved,
            });

            let idc = CId(types.core.parent.len());
            types.core.parent.0.push(idc);
            types.core.cluster.0.push(Cluster {
                state: ResolveKind::Struct(call_id),
            });
            idc
        }

        TypeValue::Ptr { tgt, raw, mutable } => {
            let target = specialize_type(ex, types, tgt, generics, loc);

            let id = CId(types.core.parent.len());
            types.core.parent.0.push(id);
            types.core.cluster.0.push(Cluster {
                state: ResolveKind::Ptr {
                    tgt: target,
                    raw: Some(raw),
                    mutable: Some(mutable),
                },
            });
            id
        }

        TypeValue::Tuple(items) => {
            let items = items
                .into_iter()
                .map(|item| specialize_type(ex, types, item, generics, loc))
                .collect::<Vec<_>>();

            let tuple_id = TupleInferId(types.extra.tuple_infers.len());
            types.extra.tuple_infers.push(TupleInfer { items });

            let id = CId(types.core.parent.len());
            types.core.parent.0.push(id);
            types.core.cluster.0.push(Cluster {
                state: ResolveKind::Tuple(tuple_id),
            });
            id
        }

        TypeValue::Array(inner, len) => {
            let inner = specialize_type(ex, types, inner, generics, loc);

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

// ===================================
// Constraint gathering (alias where possible)
// ===================================

fn solved_type_to_specialized_local(
    ex: &mut ExternState,
    types: &mut TypeState,
    t: TypeId,
    loc: ValId,
) -> CId {
    if let TypeValue::Func { generics, .. } = *ex.store.type_value(t)
        && generics != 0
    {
        let gens: Vec<_> = (0..generics)
            .map(|_| {
                let id = CId(types.core.parent.len());
                types.core.parent.0.push(id);
                types.core.cluster.0.push(Cluster {
                    state: ResolveKind::Nothing,
                });
                id
            })
            .collect();

        return specialize_type(ex, types, t, &gens, loc);
    }

    let id = CId(types.core.parent.len());
    types.core.parent.0.push(id);
    types.core.cluster.0.push(Cluster {
        state: ResolveKind::Solved(t),
    });
    id
}

fn global_to_specialized_local(
    ex: &mut ExternState,
    search: &mut SearchState,
    types: &mut TypeState,
    def_val: &ValId,
    v: ValId,
) -> CId {
    let Some(t) = ex.ans.type_of(*def_val) else {
        let loc = ex.program.value_loc(v);
        let c = types.new_cluster();
        ex.push_error(TypeError::Simple {
            loc,
            message: "global value has no inferred type",
        });
        search.bind_val(v, c);
        return c;
    };

    //TODO this check is actually CURRENTLY non exustive
    //we wana make sure that we add a good way to run this
    //would be done as some normlization function somewhere
    //structs especially are weird with this
    let ans = solved_type_to_specialized_local(ex, types, t, v);
    search.bind_val(v, ans);
    ans
}

fn resolve_member_method_access(
    ex: &mut ExternState,
    types: &mut TypeState,
    search: &mut SearchState,
    member_method_type_sites: &mut Vec<PendingMemberMethodType>,
    access_site: ValId,
    base_value: ValId,
    base_cluster: CId,
    struct_name: NameId,
    member_name: StrId,
) -> CId {
    let Some(method) = ex
        .program
        .member_methods
        .get(&struct_name)
        .and_then(|methods| methods.get(&member_name))
        .copied()
    else {
        let unresolved = types.new_cluster();
        search.bind_val(access_site, unresolved);
        ex.push_error(TypeError::UnknownField {
            field: member_name,
            site: access_site,
        });
        return unresolved;
    };

    let Some(method_ty) = ex.ans.type_of(method) else {
        unreachable!(
            "global member method signatures must be solved before body inference; missing type for member access"
        );
    };

    let method_local = solved_type_to_specialized_local(ex, types, method_ty, access_site);

    let Some(self_style) = get_member_self_style(ex.store, method_ty, struct_name) else {
        member_method_type_sites.push(PendingMemberMethodType {
            site: access_site,
            member: member_name,
            full_method: method_local,
            receiver: base_cluster,
            receiver_value: base_value,
        });
        search.bind_val(access_site, method_local);
        return method_local;
    };

    let Some((params, ret)) = function_parts_from_cluster(ex, types, method_local) else {
        unreachable!("specialized member access method must resolve to a function shape");
    };

    let curried_method = make_member_closure(
        ex,
        types,
        base_cluster,
        ResolvedMemberOverload {
            params,
            ret,
            self_style,
            full_method: method_local,
        },
        access_site,
    );

    match curried_method {
        Ok(curried) => {
            member_method_type_sites.push(PendingMemberMethodType {
                site: access_site,
                member: member_name,
                full_method: method_local,
                receiver: base_cluster,
                receiver_value: base_value,
            });
            search.bind_val(access_site, curried);
            curried
        }
        Err(clash) => {
            let unresolved = types.new_cluster();
            member_method_type_sites.push(PendingMemberMethodType {
                site: access_site,
                member: member_name,
                full_method: method_local,
                receiver: base_cluster,
                receiver_value: base_value,
            });
            search.bind_val(access_site, unresolved);
            ex.push_error(TypeError::ValuesContradict {
                expectation_reason: "member method receiver must match method self parameter",
                site: access_site,
                found: base_value,
                expected_place: access_site,
                clash,
            });
            unresolved
        }
    }
}

fn try_resolve_struct_deref_method(
    ex: &mut ExternState,
    types: &mut TypeState,
    site: ValId,
    base_value: ValId,
    base_cluster: CId,
    struct_name: NameId,
    method_name: StrId,
    expected_self_mutable: bool,
    expected_output_mutable: bool,
) -> Option<ResolvedStructDerefTarget> {
    let method = ex
        .program
        .member_methods
        .get(&struct_name)
        .and_then(|methods| methods.get(&method_name))
        .copied()?;

    let Some(method_ty) = ex.ans.type_of(method) else {
        unreachable!(
            "global member method signatures must be solved before body inference; missing type for deref"
        );
    };

    let method_local = solved_type_to_specialized_local(ex, types, method_ty, site);

    let Some((params, ret)) = function_parts_from_cluster(ex, types, method_local) else {
        unreachable!("specialized deref method must resolve to a function shape");
    };

    let self_param = params.first().copied()?;
    if params.len() != 1 {
        return None;
    }

    let self_input = member_self_input_cluster(
        &mut types.core,
        base_cluster,
        MemberSelfStyle::Ref {
            mutable: expected_self_mutable,
        },
    );

    if let Err(clash) = unify_if_distinct(ex, types, self_param, self_input) {
        ex.push_error(TypeError::ValuesContradict {
            expectation_reason: "deref receiver must match special deref method self parameter",
            site,
            found: base_value,
            expected_place: site,
            clash,
        });
        return None;
    }

    let ret_root = types.root(ret);
    match types.cluster_state(ret_root) {
        ResolveKind::Ptr { tgt, raw, mutable }
            if raw == Some(false) && mutable == Some(expected_output_mutable) =>
        {
            Some(ResolvedStructDerefTarget {
                target: tgt,
                deref_result_ptr: ret_root,
            })
        }

        ResolveKind::Solved(ty) => match ex.store.type_value(ty) {
            TypeValue::Ptr { tgt, raw, mutable }
                if !*raw && *mutable == expected_output_mutable =>
            {
                Some(ResolvedStructDerefTarget {
                    target: types.new_solved(*tgt),
                    deref_result_ptr: types.new_solved(ty),
                })
            }
            _ => None,
        },

        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
struct ResolvedStructDerefTarget {
    target: CId,
    deref_result_ptr: CId,
}
fn resolve_struct_deref_target(
    ex: &mut ExternState,
    types: &mut TypeState,
    site: ValId,
    base_value: ValId,
    base_cluster: CId,
    struct_name: NameId,
) -> Option<ResolvedStructDerefTarget> {
    let deref_target = try_resolve_struct_deref_method(
        ex,
        types,
        site,
        base_value,
        base_cluster,
        struct_name,
        DEREF_STR,
        false,
        false,
    );
    let deref_mut_target = try_resolve_struct_deref_method(
        ex,
        types,
        site,
        base_value,
        base_cluster,
        struct_name,
        DEREF_MUT_STR,
        true,
        true,
    );

    match (deref_target, deref_mut_target) {
        (None, None) => None,
        (Some(target), None) | (None, Some(target)) => Some(target),
        (Some(a), Some(b)) => {
            if let Err(clash) = unify_if_distinct(ex, types, a.target, b.target) {
                ex.push_error(TypeError::ValuesContradict {
                    expectation_reason:
                        "`__deref` and `__deref_mut` must produce the same dereference target type",
                    site,
                    found: base_value,
                    expected_place: site,
                    clash,
                });
                return None;
            }
            Some(b)
        }
    }
}

fn push_cannot_deref_error(
    ex: &mut ExternState,
    types: &mut TypeState,
    site: ValId,
    source_value: ValId,
    source: CId,
) {
    let mut limit = EXPANSION_LIMIT;
    let source_type = extract_bad_type(
        /*ex=*/ ex,
        /*core=*/ &mut types.core,
        /*extra=*/ &types.extra,
        source,
        &mut limit,
    );

    ex.push_error(TypeError::CannotDeref {
        site,
        operand: source_value,
        operand_type: source_type,
    });
}

#[derive(Debug)]
enum MemberAccessResolve {
    Resolved {
        result: CId,
        implicit_receivers: Vec<CId>,
    },
    Pending {
        source: CId,
    },
    Error(TypeError),
}

#[inline(always)]
fn finalize_member_access_implicit_chain(
    mut chain: Vec<CId>,
    used_implicit_deref_steps: usize,
    resolved_base: CId,
) -> Vec<CId> {
    if used_implicit_deref_steps > 0 {
        chain.push(resolved_base);
    }
    chain
}

#[inline(always)]
fn try_resolve_member_access(
    ex: &mut ExternState,
    types: &mut TypeState,
    search: &mut SearchState,
    member_method_type_sites: &mut Vec<PendingMemberMethodType>,
    site: ValId,
    base_value: ValId,
    source: CId,
    member_name: StrId,
    kind: AccessKind,
) -> MemberAccessResolve {
    let mut current = types.root(source);
    let mut implicit_receivers = Vec::new();
    let max_implicit_deref_steps = match kind {
        AccessKind::Dot => 1usize,
        AccessKind::Ptr => 64usize,
        AccessKind::Static => 0usize,
    };
    let implicit_deref_limit_message = match kind {
        AccessKind::Dot => "`.` member access performs at most one implicit dereference",
        AccessKind::Ptr => "member access autoderef recursion exceeded safety limit",
        AccessKind::Static => "static member access does not support implicit dereference",
    };
    let mut used_implicit_deref_steps = 0usize;

    loop {
        match types.core.cluster[current].state {
            ResolveKind::Nothing => return MemberAccessResolve::Pending { source: current },
            ResolveKind::Ptr { tgt, .. } => {
                if used_implicit_deref_steps >= max_implicit_deref_steps {
                    return MemberAccessResolve::Error(TypeError::Simple {
                        loc: ex.program.value_loc(site),
                        message: implicit_deref_limit_message,
                    });
                }
                let next = types.root(tgt);
                implicit_receivers.push(current);
                used_implicit_deref_steps += 1;
                current = next;
            }
            ResolveKind::Solved(t) => {
                let solved = ex.store.type_value(t).clone();
                match solved {
                    TypeValue::Ptr { tgt, .. } => {
                        if used_implicit_deref_steps >= max_implicit_deref_steps {
                            return MemberAccessResolve::Error(TypeError::Simple {
                                loc: ex.program.value_loc(site),
                                message: implicit_deref_limit_message,
                            });
                        }
                        let next = types.new_solved(tgt);
                        let next = types.root(next);
                        implicit_receivers.push(current);
                        used_implicit_deref_steps += 1;
                        current = next;
                    }
                    TypeValue::Struct { id: sid, generics } => {
                        let (field_ty, struct_name) = {
                            let rep = ex.store.struct_value(sid);
                            let field_ty = rep
                                .fields
                                .iter()
                                .find(|(n, _)| ex.program.name_str_id(*n) == member_name)
                                .map(|(_, t)| *t);
                            (field_ty, rep.name)
                        };

                        if let Some(field_ty) = field_ty {
                            match kind {
                                AccessKind::Dot | AccessKind::Ptr => {}
                                AccessKind::Static => {
                                    return MemberAccessResolve::Error(TypeError::Simple {
                                        loc: ex.program.value_loc(site),
                                        message: "some error on it not making sense",
                                    });
                                }
                            }

                            let result = match ex.store.type_value(field_ty) {
                                TypeValue::Generic(i) => types.new_solved(generics[i.0]),
                                _ => types.new_solved(field_ty),
                            };
                            return MemberAccessResolve::Resolved {
                                result,
                                implicit_receivers: finalize_member_access_implicit_chain(
                                    implicit_receivers,
                                    used_implicit_deref_steps,
                                    current,
                                ),
                            };
                        }

                        let has_member_method = struct_name
                            .and_then(|sn| ex.program.member_methods.get(&sn))
                            .is_some_and(|methods| methods.contains_key(&member_name));
                        if let Some(struct_name) = struct_name {
                            if has_member_method {
                                let result = resolve_member_method_access(
                                    ex,
                                    types,
                                    search,
                                    member_method_type_sites,
                                    site,
                                    base_value,
                                    current,
                                    struct_name,
                                    member_name,
                                );
                                return MemberAccessResolve::Resolved {
                                    result,
                                    implicit_receivers: finalize_member_access_implicit_chain(
                                        implicit_receivers,
                                        used_implicit_deref_steps,
                                        current,
                                    ),
                                };
                            }

                            if used_implicit_deref_steps < max_implicit_deref_steps
                                && let Some(target) = resolve_struct_deref_target(
                                    ex,
                                    types,
                                    site,
                                    base_value,
                                    current,
                                    struct_name,
                                )
                            {
                                let next = types.root(target.target);
                                implicit_receivers.push(current);
                                used_implicit_deref_steps += 1;
                                current = next;
                                continue;
                            }
                        }

                        return MemberAccessResolve::Error(TypeError::UnknownField {
                            field: member_name,
                            site,
                        });
                    }
                    _ => {
                        return MemberAccessResolve::Error(TypeError::Simple {
                            loc: ex.program.value_loc(site),
                            message: "member access requires a struct or pointer-like base",
                        });
                    }
                }
            }
            ResolveKind::Struct(rid) => {
                let sid = types.extra.struct_infers[rid.0].sid;
                let (field_ty, struct_name) = {
                    let rep = ex.store.struct_value(sid);
                    let field_ty = rep
                        .fields
                        .iter()
                        .find(|(n, _)| ex.program.name_str_id(*n) == member_name)
                        .map(|(_, t)| *t);
                    (field_ty, rep.name)
                };

                if let Some(field_ty) = field_ty {
                    match kind {
                        AccessKind::Dot | AccessKind::Ptr => {}
                        AccessKind::Static => {
                            return MemberAccessResolve::Error(TypeError::Simple {
                                loc: ex.program.value_loc(site),
                                message: "some error on it not making sense",
                            });
                        }
                    }

                    let result = match ex.store.type_value(field_ty) {
                        TypeValue::Generic(i) => types.extra.struct_infers[rid.0].generics[i.0],
                        _ => types.new_solved(field_ty),
                    };
                    return MemberAccessResolve::Resolved {
                        result,
                        implicit_receivers: finalize_member_access_implicit_chain(
                            implicit_receivers,
                            used_implicit_deref_steps,
                            current,
                        ),
                    };
                }

                let has_member_method = struct_name
                    .and_then(|sn| ex.program.member_methods.get(&sn))
                    .is_some_and(|methods| methods.contains_key(&member_name));
                if let Some(struct_name) = struct_name {
                    if has_member_method {
                        let result = resolve_member_method_access(
                            ex,
                            types,
                            search,
                            member_method_type_sites,
                            site,
                            base_value,
                            current,
                            struct_name,
                            member_name,
                        );
                        return MemberAccessResolve::Resolved {
                            result,
                            implicit_receivers: finalize_member_access_implicit_chain(
                                implicit_receivers,
                                used_implicit_deref_steps,
                                current,
                            ),
                        };
                    }

                    if used_implicit_deref_steps < max_implicit_deref_steps
                        && let Some(target) = resolve_struct_deref_target(
                            ex,
                            types,
                            site,
                            base_value,
                            current,
                            struct_name,
                        )
                    {
                        let next = types.root(target.target);
                        implicit_receivers.push(current);
                        used_implicit_deref_steps += 1;
                        current = next;
                        continue;
                    }
                }

                return MemberAccessResolve::Error(TypeError::UnknownField {
                    field: member_name,
                    site,
                });
            }
            _ => {
                return MemberAccessResolve::Error(TypeError::Simple {
                    loc: ex.program.value_loc(site),
                    message: "member access requires a struct or pointer-like base",
                });
            }
        }
    }
}

fn gather_constraints(ctx: &mut InferState, v: ValId, current_output: Option<CId>) -> CId {
    match ctx.ex.program.value(v) {
        Value::Literal(Literal::Num(_)) => {
            let c = ctx.new_int_like();
            ctx.bind_val(v, c);
            c
        }

        Value::Literal(Literal::Float(_)) => {
            let c = ctx.new_float_like();
            ctx.bind_val(v, c);
            c
        }

        Value::Literal(Literal::Str(_)) => {
            let c = ctx.new_solved(BuiltinType::Str.into());

            ctx.bind_val(v, c);
            c
        }

        Value::Literal(Literal::Bool(_)) => {
            let c = ctx.new_solved(BuiltinType::Bool.into());

            ctx.bind_val(v, c);
            c
        }

        Value::Literal(Literal::Void) => {
            let c = ctx.new_solved(BuiltinType::Void.into());
            ctx.bind_val(v, c);
            c
        }

        Value::Wildcard => {
            let c = ctx.new_cluster();
            ctx.bind_val(v, c);
            c
        }
        Value::NameRef(n) => {
            if let Some(base) = ctx.search.names.get_mut(&n) {
                //names might refer to something that us generic in the local scope...
                //so this here is actually wrong for when users define local genric stuff
                let c = ctx.types.root(*base);
                *base = c;
                ctx.bind_val(v, c);
                return c;
            }

            let Some(def) = ctx.ex.program.definitions.get(&n) else {
                unreachable!("name used before binding");
            };

            match def {
                Defined::Type(_t) => {
                    let ans = ctx.new_solved(BuiltinType::Type.into());
                    ctx.bind_val(v, ans);
                    ans
                }
                Defined::Func(def_val) => global_to_specialized_local(
                    &mut ctx.ex,
                    &mut ctx.search,
                    &mut ctx.types,
                    def_val,
                    v,
                ),
                _ => todo!("global name resolution / overload sets"),
            }
        }

        Value::Let {
            pat,
            value,
            else_part,
        } => {
            let lhs = gather_pattern_constraints(ctx, pat);
            // let-expr evaluates to the bound pattern value => alias
            ctx.bind_val(v, lhs);

            let rhs = gather_constraints(ctx, value, current_output);

            if let Err(clash) = ctx.unify(rhs, lhs) {
                ctx.push_error(TypeError::ValuesContradict {
                    expectation_reason: "let binding requires pattern and value to match",
                    site: v,
                    found: value,
                    expected_place: v,
                    clash,
                });
            }

            if let Some(e) = else_part {
                let ec = gather_constraints(ctx, e, current_output);
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

            lhs
        }

        Value::TypeAnnotation { value, ty } => {
            let rhs_cluster = gather_constraints(ctx, value, current_output);
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
            let _ = gather_constraints(ctx, value, current_output);
            // Cast produces a new type identity: the target type
            let c = compile_type_expr(ctx, ty);
            ctx.bind_val(v, c);
            c
        }

        Value::TypeDef { pat, ty } => {
            let (p, n) = gather_pattern_constraints_and_name(ctx, pat);
            if let Err(clash) = ctx.force_type(p, BuiltinType::Type.into()) {
                ctx.push_error(TypeError::TypeDefPatternMismatch {
                    pattern: pat,
                    clash,
                });
            }
            let t = if let Some(n) = n {
                let t = do_typedef::<false>(ctx, n, ty);
                ctx.search.local_types.insert(n, t);
                t
            } else {
                compile_type_expr(ctx, ty)
            };
            ctx.search.typedef_cluster.push((ty, t));
            p
        }

        Value::AddrOf(base, kind) => {
            let tgt = gather_constraints(ctx, base, current_output);
            let mutable = kind.map(|x| matches!(x, VarKind::Mut));
            let ans = ctx.new_cluster();
            ctx.types.core.cluster[ans].state = ResolveKind::Ptr {
                tgt,
                raw: None,
                mutable,
            };
            ctx.bind_val(v, ans);
            ans
        }

        Value::Deref(base) => {
            let output = ctx.new_cluster();
            ctx.bind_val(v, output);

            let src = gather_constraints(ctx, base, current_output);
            let src = ctx.types.root(src);
            let resolved_target = match ctx.types.core.cluster[src].state {
                ResolveKind::Ptr { tgt, .. } => Some(tgt),
                ResolveKind::Nothing => {
                    ctx.req.pointer_likes.push(PendingPointerLike {
                        site: v,
                        source: src,
                        target: output,
                        source_value: base,
                    });
                    None
                }
                ResolveKind::Struct(rid) => {
                    let sid = ctx.types.extra.struct_infers[rid.0].sid;
                    let Some(struct_name) = ctx.ex.store.struct_value(sid).name else {
                        push_cannot_deref_error(&mut ctx.ex, &mut ctx.types, v, base, src);
                        return output;
                    };

                    let Some(target) = resolve_struct_deref_target(
                        &mut ctx.ex,
                        &mut ctx.types,
                        v,
                        base,
                        src,
                        struct_name,
                    ) else {
                        push_cannot_deref_error(&mut ctx.ex, &mut ctx.types, v, base, src);
                        return output;
                    };

                    Some(target.target)
                }
                ResolveKind::Solved(t) => {
                    let solved = ctx.ex.store.type_value(t).clone();
                    match solved {
                        TypeValue::Ptr { tgt, .. } => Some(ctx.new_solved(tgt)),
                        TypeValue::Struct { id, .. } => {
                            let Some(struct_name) = ctx.ex.store.struct_value(id).name else {
                                push_cannot_deref_error(&mut ctx.ex, &mut ctx.types, v, base, src);
                                return output;
                            };

                            let Some(target) = resolve_struct_deref_target(
                                &mut ctx.ex,
                                &mut ctx.types,
                                v,
                                base,
                                src,
                                struct_name,
                            ) else {
                                push_cannot_deref_error(&mut ctx.ex, &mut ctx.types, v, base, src);
                                return output;
                            };

                            Some(target.target)
                        }
                        _ => {
                            push_cannot_deref_error(&mut ctx.ex, &mut ctx.types, v, base, src);
                            return output;
                        }
                    }
                }
                _ => {
                    push_cannot_deref_error(&mut ctx.ex, &mut ctx.types, v, base, src);
                    return output;
                }
            };

            if let Some(tgt) = resolved_target
                && let Err(clash) = ctx.unify(tgt, output)
            {
                ctx.push_error(TypeError::ValuesContradict {
                    expectation_reason: "dereference result must match pointee type",
                    site: v,
                    found: base,
                    expected_place: v,
                    clash,
                });
            }

            output
        }

        Value::Assign { op, target } => {
            let lhs = gather_constraints(ctx, target, current_output);
            ctx.bind_val(v, lhs);

            match op {
                AssignOp::Nothing(value) => {
                    let rhs = gather_constraints(ctx, value, current_output);
                    if let Err(clash) = ctx.unify(rhs, lhs) {
                        ctx.push_error(TypeError::ValuesContradict {
                            expectation_reason: "assignment requires both sides match",
                            site: v,
                            found: value,
                            expected_place: target,
                            clash,
                        });
                    }
                }
                AssignOp::Bin(bin_op, value) => {
                    let rhs = gather_constraints(ctx, value, current_output);
                    let mut site = BinOpSite {
                        loc: v,
                        op: bin_op,
                        lhs_val: target,
                        rhs_val: value,
                        lhs,
                        rhs,
                        output: lhs,
                    };
                    let outcome = resolve_operator_site(
                        &mut ctx.ex,
                        &mut ctx.types,
                        &mut ctx.req.member_method_type_sites,
                        &mut site,
                    );
                    if outcome.retain {
                        ctx.req.bin_op_sites.push(site);
                    }
                }
                AssignOp::Pre(dir) | AssignOp::Post(dir) => {
                    let implicit_rhs = ctx.new_int_like();
                    let flavor = match (matches!(op, AssignOp::Post(_)), dir) {
                        (false, Dir::Inc) => AssignIncDecFlavor::PreInc,
                        (true, Dir::Inc) => AssignIncDecFlavor::PostInc,
                        (false, Dir::Dec) => AssignIncDecFlavor::PreDec,
                        (true, Dir::Dec) => AssignIncDecFlavor::PostDec,
                    };
                    let mut site = AssignPrePostSite {
                        loc: v,
                        target_val: target,
                        target: lhs,
                        implicit_rhs,
                        flavor,
                    };
                    let outcome = resolve_assign_pre_post_site(
                        &mut ctx.ex,
                        &mut ctx.types,
                        &mut ctx.req.member_method_type_sites,
                        &mut site,
                    );
                    if outcome.retain {
                        ctx.req.assign_pre_post_sites.push(site);
                    }
                }
            }

            lhs
        }

        Value::Block {
            statements,
            return_value,
        } => {
            for s in statements.ids() {
                gather_constraints(ctx, s, current_output);
            }

            // block aliases its return value cluster (or void)
            let c = match return_value {
                Some(r) => gather_constraints(ctx, r, current_output),
                None => ctx.new_solved(BuiltinType::Void.into()),
            };

            ctx.bind_val(v, c);
            c
        }

        Value::BinOp { op, values } => {
            let (lhs, rhs) = values;

            let lc = gather_constraints(ctx, lhs, current_output);
            let rc = gather_constraints(ctx, rhs, current_output);

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
            {
                let mut site = BinOpSite {
                    loc: v,
                    op,
                    lhs_val: lhs,
                    rhs_val: rhs,
                    lhs: lc,
                    rhs: rc,
                    output,
                };
                let outcome = resolve_operator_site(
                    &mut ctx.ex,
                    &mut ctx.types,
                    &mut ctx.req.member_method_type_sites,
                    &mut site,
                );
                if outcome.retain {
                    ctx.req.bin_op_sites.push(site);
                }
            }
            output
        }
        Value::UnOp { op, value } => {
            let input = gather_constraints(ctx, value, current_output);
            let output = match op {
                UnOp::Not => ctx.new_solved(BuiltinType::Bool.into()),
                _ => ctx.new_cluster(),
            };

            ctx.bind_val(v, output);
            {
                let mut site = UnOpSite {
                    loc: v,
                    op,
                    val: value,
                    input,
                    output,
                };
                let outcome = resolve_unary_operator_site(
                    &mut ctx.ex,
                    &mut ctx.types,
                    &mut ctx.req.member_method_type_sites,
                    &mut site,
                );
                if outcome.retain {
                    ctx.req.un_op_sites.push(site);
                }
            }
            output
        }
        Value::While { cond, body } => {
            let cond_cluster = gather_constraints(ctx, cond, current_output);
            if let Err(clash) = ctx.force_type(cond_cluster, BuiltinType::Bool.into()) {
                ctx.push_error(TypeError::ValuesContradict {
                    expectation_reason: "while condition must be bool",
                    site: v,
                    found: cond,
                    expected_place: cond,
                    clash,
                });
            }

            let _body_cluster = gather_constraints(ctx, body, current_output);

            let output = ctx.new_solved(BuiltinType::Bool.into());
            ctx.bind_val(v, output);
            output
        }
        Value::If { cond, then, els } => {
            let cond_cluster = gather_constraints(ctx, cond, current_output);
            if let Err(clash) = ctx.force_type(cond_cluster, BuiltinType::Bool.into()) {
                ctx.push_error(TypeError::ValuesContradict {
                    expectation_reason: "if condition must be bool",
                    site: v,
                    found: cond,
                    expected_place: cond,
                    clash,
                });
            }

            let then_cluster = gather_constraints(ctx, then, current_output);

            let output = if let Some(els) = els {
                let else_cluster = gather_constraints(ctx, els, current_output);
                if let Err(clash) = ctx.unify(then_cluster, else_cluster) {
                    ctx.push_error(TypeError::ValuesContradict {
                        expectation_reason: "if branches must have the same type",
                        site: v,
                        found: then,
                        expected_place: els,
                        clash,
                    });
                }
                then_cluster
            } else {
                ctx.new_solved(BuiltinType::Void.into())
            };

            ctx.bind_val(v, output);
            output
        }
        Value::Func {
            calling_convention,
            generics,
            params,
            output_type,
            body,
        } => gather_func_constraints::<false>(
            ctx,
            v,
            calling_convention,
            generics,
            params,
            output_type,
            body,
        ),
        Value::Call(call) => {
            if call.named_args().is_empty() {
                //we can try derive the type of base directly
                //this makes life SOOOO much easier than named args

                let base = gather_constraints(ctx, call.base, current_output);
                let inputs: Vec<_> = call
                    .args
                    .ids()
                    .map(|a| gather_constraints(ctx, a, current_output))
                    .collect();
                let output = ctx.new_cluster();

                let found = ctx.new_func(FuncInfer {
                    calling_convention: CallingConvention::Unknown,
                    generics: 0,
                    inputs,
                    output,
                });
                if let Err(clash) = ctx.unify(found, base) {
                    ctx.push_error(TypeError::ValuesContradict {
                        expectation_reason: "called function with wrong signature",
                        site: v,
                        found: call.base,
                        expected_place: v,
                        clash,
                    });
                }
                output
            } else {
                //we have to get exact function here because we need to figure out arg order
                if let Some(_n) = try_get_name(ctx, call.base) {
                    todo!("easy case not a member function")
                } else {
                    //CAN  be a member function.
                    //we need the thing calling its member function
                    //and we need the functions value

                    //we might also just have a closure being called immidiatly
                    //or maybe a function returned from somewhere
                    //if thats the case thats an error as we dont permit named args there
                    todo!(
                        "for now this isnt a thing since we dont do member functions yet in ir.rs"
                    )
                }
            }
        }

        Value::Construct(cons) => {
            //we dont gather the base because we just care about the name
            let Some(base_name) = try_get_name(ctx, cons.base) else {
                ctx.push_error(TypeError::ConstructorBaseNotGlobal { site: cons.base });
                for arg in cons.args.ids() {
                    gather_constraints(ctx, arg, current_output);
                }
                let ans = ctx.new_cluster();
                ctx.bind_val(v, ans);
                return ans;
            };
            let Some(def) = ctx.ex.program.definitions.get(&base_name) else {
                ctx.push_error(TypeError::ConstructorBaseNotGlobal { site: cons.base });
                for arg in cons.args.ids() {
                    gather_constraints(ctx, arg, current_output);
                }
                let ans = ctx.new_cluster();
                ctx.bind_val(v, ans);
                return ans;
            };

            let Defined::Type(texp) = def else {
                ctx.push_error(TypeError::ConstructorBaseNotTypeName { site: cons.base });
                for arg in cons.args.ids() {
                    gather_constraints(ctx, arg, current_output);
                }
                let ans = ctx.new_cluster();
                ctx.bind_val(v, ans);
                return ans;
            };

            let Some(base_type) = ctx.ex.ans.typedef_types.get(texp) else {
                ctx.push_error(TypeError::UnresolvedTypeExpr { expr: *texp });
                for arg in cons.args.ids() {
                    gather_constraints(ctx, arg, current_output);
                }
                let ans = ctx.new_cluster();
                ctx.bind_val(v, ans);
                return ans;
            };
            let base_type = *base_type;

            let sid = match ctx.ex.store.type_value(base_type) {
                TypeValue::Struct { id, generics: _ } => *id,
                // TypeValue::Specialized { base, .. } => {
                //     match ctx.ex.store.type_value(*base) {
                //         TypeValue::Struct(sid) => *sid,
                //         _ => {
                //             ctx.push_error(TypeError::ConstructorBaseNotStruct {
                //                 site: cons.base,
                //                 found: Some(BadTypeId(*base)),
                //             });
                //             for arg in cons.args.ids() {
                //                 gather_constraints(ctx, arg);
                //             }
                //             let ans = ctx.new_cluster();
                //             ctx.bind_val(v, ans);
                //             return ans;
                //         }
                //     }
                // }
                _ => {
                    ctx.push_error(TypeError::ConstructorBaseNotStruct {
                        site: cons.base,
                        found: Some(BadTypeId(base_type)),
                    });
                    for arg in cons.args.ids() {
                        gather_constraints(ctx, arg, current_output);
                    }
                    let ans = ctx.new_cluster();
                    ctx.bind_val(v, ans);
                    return ans;
                }
            };

            // let fields = &ctx.ex.store.struct_value(sid).fields;
            let expected = ctx.ex.store.struct_value(sid).fields.len();
            let provided = cons.args.len();
            if provided > expected {
                ctx.push_error(TypeError::TooManyArguments {
                    site: v,
                    expected,
                    found: provided,
                });
            }

            let TypeValue::Struct { id: _, generics } = ctx.ex.store.type_value(base_type) else {
                unreachable!("verified above");
            };

            let glen = generics.len();

            let mut generic_clusters = Vec::new();
            let mut field_type_clusters = None;
            if glen != 0 {
                generic_clusters = (0..glen).map(|_| ctx.new_cluster()).collect();
                let flen = ctx.ex.store.struct_value(sid).fields.len();

                field_type_clusters = Some(
                    (0..flen)
                        .map(|f| {
                            let (_, t) = ctx.ex.store.struct_value(sid).fields[f];
                            specialize_type(&mut ctx.ex, &mut ctx.types, t, &generic_clusters, v)
                        })
                        .collect::<Vec<_>>(),
                );
            }

            let missing = CId(usize::MAX);
            let mut args = Vec::with_capacity(expected.max(provided));
            for (i, a) in cons.pos_args().ids().enumerate() {
                let c = gather_constraints(ctx, a, current_output);
                args.push(c);

                let (nid, t) = ctx.ex.store.struct_value(sid).fields[i];
                debug_assert!(t != UNKNOWN_TYPE);
                if let Some(field_types) = &field_type_clusters {
                    let expected = field_types[i];
                    if let Err(clash) = ctx.unify(c, expected) {
                        let name = ctx.ex.program.name_str_id(nid);
                        ctx.push_error(TypeError::FieldTypeMismatch {
                            field: name,
                            value: a,
                            clash,
                        });
                    }
                } else if let Err(clash) = ctx.force_type(c, t) {
                    let name = ctx.ex.program.name_str_id(nid);
                    ctx.push_error(TypeError::FieldTypeMismatch {
                        field: name,
                        value: a,
                        clash,
                    });
                }
            }

            //add a place for all the named args to go
            args.extend(cons.named_args().ids().map(|_| missing));
            if args.len() < expected {
                args.resize(expected, missing);
            }

            for na in cons.named_args().ids() {
                let Value::Labeled { name, value } = ctx.ex.program.value(na) else {
                    unreachable!()
                };

                let value_c = gather_constraints(ctx, value, current_output);

                let spot = ctx
                    .ex
                    .store
                    .struct_value(sid)
                    .fields
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_i, (n, _t))| ctx.ex.program.name_str_id(*n) == name);

                let Some((i, (_n, t))) = spot else {
                    ctx.push_error(TypeError::UnknownField {
                        field: name,
                        site: na,
                    });
                    continue;
                };

                if i < cons.pos_args().len() {
                    ctx.push_error(TypeError::FieldAlreadyPositional {
                        field: name,
                        site: na,
                    });
                    continue;
                }
                if args[i] != missing {
                    ctx.push_error(TypeError::DuplicateField {
                        field: name,
                        site: na,
                    });
                    continue;
                }

                args[i] = value_c;

                debug_assert!(*t != UNKNOWN_TYPE);
                if let Some(field_types) = &field_type_clusters {
                    let expected = field_types[i];
                    if let Err(clash) = ctx.unify(value_c, expected) {
                        ctx.push_error(TypeError::FieldTypeMismatch {
                            field: name,
                            value,
                            clash,
                        });
                    }
                } else if let Err(clash) = ctx.force_type(value_c, *t) {
                    ctx.push_error(TypeError::FieldTypeMismatch {
                        field: name,
                        value,
                        clash,
                    });
                }
            }

            let fields = &ctx.ex.store.struct_value(sid).fields;
            for ((field, _t), c) in fields.iter().zip(args.iter()) {
                if *c == missing {
                    ctx.ex.errors.push(TypeError::MissingField {
                        field: *field,
                        site: v,
                    });
                }
            }

            if glen == 0 {
                let t = ctx.ex.store.intern(TypeValue::Struct {
                    id: sid,
                    generics: Vec::new(),
                });
                let ans = ctx.new_solved(t);
                ctx.bind_val(v, ans);
                return ans;
            }

            let ans = ctx.new_struct_instance(sid, generic_clusters);
            ctx.bind_val(v, ans);
            ans
        }

        Value::Access { base, name, kind } => {
            let source = gather_constraints(ctx, base, current_output);
            match try_resolve_member_access(
                &mut ctx.ex,
                &mut ctx.types,
                &mut ctx.search,
                &mut ctx.req.member_method_type_sites,
                v,
                base,
                source,
                name,
                kind,
            ) {
                MemberAccessResolve::Resolved {
                    result,
                    implicit_receivers,
                } => {
                    ctx.bind_val(v, result);
                    if !implicit_receivers.is_empty() {
                        ctx.req.member_access_implicit_deref_sites.push(
                            PendingMemberAccessImplicitDeref {
                                site: v,
                                receivers: implicit_receivers,
                            },
                        );
                    }
                    result
                }
                MemberAccessResolve::Pending { source } => {
                    let result = ctx.new_cluster();
                    ctx.bind_val(v, result);
                    ctx.req.pending_member_accesses.push(PendingMemberAccess {
                        site: v,
                        base_value: base,
                        source,
                        output: result,
                        member: name,
                        kind,
                    });
                    result
                }
                MemberAccessResolve::Error(err) => {
                    ctx.push_error(err);
                    let result = ctx.new_cluster();
                    ctx.bind_val(v, result);
                    result
                }
            }
        }
        Value::Goto(_) | Value::Break | Value::Continue | Value::LabelDecl(_) => {
            let c = ctx.new_cluster();
            ctx.types.core.cluster[c].state = ResolveKind::Never;
            c
        }
        Value::Return(op) => {
            if let Some(output) = current_output {
                match op {
                    Some(ret_value) => {
                        let ret_cluster = gather_constraints(ctx, ret_value, current_output);
                        if let Err(clash) = ctx.unify(ret_cluster, output) {
                            ctx.push_error(TypeError::ValuesContradict {
                                expectation_reason: "return value must match function return type",
                                site: v,
                                found: ret_value,
                                expected_place: v,
                                clash,
                            });
                        }
                    }
                    None => {
                        let void = ctx.new_solved(BuiltinType::Void.into());
                        if let Err(clash) = ctx.unify(void, output) {
                            ctx.push_error(TypeError::ValuesContradict {
                                expectation_reason:
                                    "bare return requires function return type void",
                                site: v,
                                found: v,
                                expected_place: v,
                                clash,
                            });
                        }
                    }
                }
            } else {
                if let Some(ret_value) = op {
                    let _ = gather_constraints(ctx, ret_value, None);
                }
                let loc = ctx.ex.program.value_loc(v);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "return used outside of function body",
                });
            }

            let c = ctx.new_cluster();
            ctx.types.core.cluster[c].state = ResolveKind::Never;
            c
        }
        Value::LogicOp { op: _, values: _ } => todo!(),

        Value::Tuple(items) => {
            let item_clusters = items
                .ids()
                .map(|item| gather_constraints(ctx, item, current_output))
                .collect::<Vec<_>>();
            let tuple = ctx.new_tuple_instance(item_clusters);
            ctx.bind_val(v, tuple);
            tuple
        }
        Value::Array(items) => {
            let values = items.ids().collect::<Vec<_>>();
            let element = if let Some(first) = values.first().copied() {
                let element = gather_constraints(ctx, first, current_output);
                for item in values.iter().copied().skip(1) {
                    let item_c = gather_constraints(ctx, item, current_output);
                    if let Err(clash) = ctx.unify(item_c, element) {
                        ctx.push_error(TypeError::ValuesContradict {
                            expectation_reason: "array elements must all have the same type",
                            site: v,
                            found: item,
                            expected_place: first,
                            clash,
                        });
                    }
                }
                element
            } else {
                ctx.new_cluster()
            };

            let array = ctx.new_array_instance(element, ArrayType::Sized(values.len()));
            ctx.bind_val(v, array);
            array
        }
        Value::Index(call) => {
            let base = gather_constraints(ctx, call.base, current_output);
            let pos_args = call.pos_args().ids().collect::<Vec<_>>();
            let pos_arg_clusters = pos_args
                .iter()
                .copied()
                .map(|arg| gather_constraints(ctx, arg, current_output))
                .collect::<Vec<_>>();
            let named_args = call.named_args().ids().collect::<Vec<_>>();
            for arg in named_args.iter().copied() {
                let _ = gather_constraints(ctx, arg, current_output);
            }

            let output = ctx.new_cluster();
            ctx.bind_val(v, output);

            if !named_args.is_empty() || pos_args.len() != 1 {
                let loc = ctx.ex.program.value_loc(v);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "indexing currently expects exactly one positional argument",
                });
                return output;
            }

            let mut site = PendingIndex {
                site: v,
                base_value: call.base,
                index_value: pos_args[0],
                base,
                index: pos_arg_clusters[0],
                output,
                implicit_receivers: Vec::new(),
            };
            let outcome = resolve_index_site(&mut ctx.ex, &mut ctx.types, &mut site);
            if outcome.retain {
                ctx.req.pending_indexes.push(site);
            } else if !site.implicit_receivers.is_empty() {
                ctx.req
                    .index_implicit_deref_sites
                    .push(PendingMemberAccessImplicitDeref {
                        site: v,
                        receivers: site.implicit_receivers,
                    });
            }

            output
        }
        Value::Match { .. } => todo!(),

        Value::Labeled { .. } => unreachable!("bug tried compiling labeled normally"),
        Value::MatchArm(_) => unreachable!("bug tried compiling match arm normally"),
        // Value::LifeTime(_) => todo!("some sort of error? maybe we actualy have a type for lifetime"),
    }
}

///this tries to resolve specifically a from a module.
///if what we have is a member of a struct it wont give a name
fn try_get_name(ctx: &mut InferState, v: ValId) -> Option<NameId> {
    match ctx.ex.program.value(v) {
        Value::NameRef(n) => Some(n),
        Value::Access {
            base: _,
            name: _,
            kind: _,
        } => todo! {},
        _ => None,
    }
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
fn gather_pattern_constraints(ctx: &mut InferState, p: PatId) -> CId {
    gather_pattern_constraints_with_generics::<false>(ctx, p)
}

#[inline(always)]
fn gather_pattern_constraints_and_name(ctx: &mut InferState, p: PatId) -> (CId, Option<NameId>) {
    gather_pattern_constraints_and_name_with_generics::<false>(ctx, p)
}

#[inline(always)]
fn gather_pattern_constraints_with_generics<const GLOBAL_SCOPE: bool>(
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

        Pattern::TypeAnnotation { pat, ty } => {
            let (c, n) =
                gather_pattern_constraints_and_name_with_generics::<GLOBAL_SCOPE>(ctx, pat);
            let t = compile_type_expr(ctx, ty);

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

///this method is kinda weird and ill formed
///currently when compiling type expressions we give them a type other the Type::Type
///we dont have a good destinction between the type of THE VALUE ITSELF and the type IT REFERS TO
///and this means that fn[T](){let x=T;} is technically legal and x has type Generic(0).
fn gather_generic_constraints(ctx: &mut InferState, p: PatId, id: GenId) -> CId {
    match ctx.ex.program.pattern(p) {
        Pattern::Bind(n, m) => {
            if m != VarKind::Const {
                let loc = ctx.ex.program.pattern_loc(p);
                ctx.ex.push_error(TypeError::Simple {
                    loc,
                    message: "generic parameters must be const bindings",
                });
            }
            let t = ctx.ex.store.intern(TypeValue::Generic(id));
            let c = ctx.new_solved(t);
            ctx.search.names.insert(n, c);
            ctx.search.local_types.insert(n, c);
            ctx.bind_pat(p, c);
            c
        }

        //hack for  now
        Pattern::LifeTime(_id) => ctx.new_cluster(),

        _ => todo!(),
    }
}

///in order to break recursion this function MUST return a concrete type
///the returned struct is not fully realized yet and its fields are gona be handeled later
fn compile_struct_type<const GLOBAL_SCOPE: bool>(
    ctx: &mut InferState,
    texpr: TExpId,
    StructLike {
        layout,
        generics,
        fields,
    }: StructLike,
) -> CId {
    if !GLOBAL_SCOPE {
        let loc = ctx.ex.program.type_expr_loc(texpr);
        ctx.ex.push_error(TypeError::Simple {
            loc,
            message: "struct types are only allowed at the top level",
        });
    }

    for (i, g) in generics.ids().enumerate() {
        let gid = GenId(i);
        let _c = gather_generic_constraints(ctx, g, gid);
        // todo!()
        //TODO: we probably wana do something with generics that are ints here if we have them
    }

    let mut field_info = Vec::with_capacity(fields.len());
    for p in fields.ids() {
        match ctx.ex.program.pattern(p) {
            Pattern::Bind(n, _) => {
                let c = ctx.new_cluster();
                field_info.push((n, c));
            }
            Pattern::TypeAnnotation { pat, ty } => {
                let Pattern::Bind(n, _) = ctx.ex.program.pattern(pat) else {
                    let loc = ctx.ex.program.pattern_loc(pat);
                    ctx.ex.push_error(TypeError::Simple {
                        loc,
                        message: "struct field must be a named binding",
                    });
                    continue;
                };
                let c = compile_type_expr(ctx, ty);
                field_info.push((n, c));
            }
            _ => {
                let loc = ctx.ex.program.pattern_loc(p);
                ctx.ex.push_error(TypeError::Simple {
                    loc,
                    message: "struct field must be a named binding",
                });
                continue;
            }
        }
    }

    let rep = StructRep::new(field_info.iter().map(|(n, _)| *n), generics.len(), layout);
    let sid = ctx.ex.store.new_struct(rep);
    let generics = (0..generics.len())
        .map(|x| ctx.ex.store.intern(TypeValue::Generic(GenId(x))))
        .collect();
    let t = ctx.ex.store.intern(TypeValue::Struct { id: sid, generics });
    let output = ctx.new_solved(t);

    ctx.types.extra.struct_defs.push(StructDef {
        loc: texpr,
        fields: field_info,
        sid,
    });
    output
}

fn do_typedef<const ALLOW_STRUCT_GENERICS: bool>(
    ctx: &mut InferState,
    typedef_name: NameId,
    texpr: TExpId,
) -> CId {
    match ctx.ex.program.type_expr(texpr) {
        TypeExpr::Struct(def) => {
            let cid = compile_struct_type::<ALLOW_STRUCT_GENERICS>(ctx, texpr, def);
            let sid = match ctx.types.core.cluster[cid].state {
                ResolveKind::Struct(rid) => ctx.types.extra.struct_infers[rid.0].sid,
                ResolveKind::Solved(t) => match ctx.ex.store.type_value(t) {
                    TypeValue::Struct { id, .. } => *id,
                    _ => unreachable!("struct def didnt return struct"),
                },
                _ => unreachable!("struct def didnt return struct"),
            };

            debug_assert_eq!(ctx.ex.store.structs[sid.0].name, None);
            ctx.ex.store.structs[sid.0].name = Some(typedef_name);

            cid
        }
        _ => compile_type_expr(ctx, texpr),
    }
}

fn compile_type_expr(ctx: &mut InferState, texpr: TExpId) -> CId {
    match ctx.ex.program.type_expr(texpr) {
        TypeExpr::NameRef(n) => {
            if let Some(ans) = ctx.search.local_types.get(&n) {
                return *ans;
            }
            let t = match ctx.ex.program.definitions.get(&n) {
                Some(Defined::BuildinType(b)) => ctx.ex.store.intern(b.clone()),
                Some(Defined::Type(texp)) => {
                    // return ctx.global_types.handle_global(
                    //     n,
                    //     &mut ctx.local_types,
                    //     *texp,
                    //     &mut ctx.parent,
                    //     &mut ctx.cluster,
                    // );
                    let Some(t) = ctx.ex.ans.typedef_types.get(texp) else {
                        let id = ctx.new_cluster();
                        ctx.search.local_types.insert(n, id);
                        return id;
                    };

                    *t
                }
                _ => {
                    let c = ctx.new_cluster();
                    ctx.push_error(TypeError::ExpectedTypeExpr { type_expr: texpr });
                    return c;
                }
            };

            ctx.new_solved(t)
        }
        TypeExpr::Wildcard => ctx.new_cluster(),

        TypeExpr::Tuple(items) => {
            let item_clusters = items
                .ids()
                .map(|item| compile_type_expr(ctx, item))
                .collect::<Vec<_>>();
            ctx.new_tuple_instance(item_clusters)
        }

        TypeExpr::Struct(def) => compile_struct_type::<false>(ctx, texpr, def),
        TypeExpr::Ptr {
            base,
            raw,
            mutable,
            lifetime: _,
        } => {
            let tgt = compile_type_expr(ctx, base);
            let ans = ctx.new_cluster();
            ctx.types.core.cluster[ans].state = ResolveKind::Ptr {
                tgt,
                raw: Some(raw),
                mutable: Some(mutable),
            };
            ans
        }
        TypeExpr::Func {
            calling_convention,
            params,
            output_type,
        } => {
            let inputs = params
                .ids()
                .map(|arg| compile_type_expr(ctx, arg))
                .collect::<Vec<_>>();
            let output = output_type
                .map(|o| compile_type_expr(ctx, o))
                .unwrap_or_else(|| ctx.new_solved(BuiltinType::Void.into()));
            ctx.new_func(FuncInfer {
                calling_convention,
                generics: 0,
                inputs,
                output,
            })
        }
        TypeExpr::Array(element, len) => {
            let element = compile_type_expr(ctx, element);
            let size = len.map_or(ArrayType::Unsized, ArrayType::Sized);
            ctx.new_array_instance(element, size)
        }
        TypeExpr::Index { base, args } => {
            let generics = args
                .ids()
                .map(|arg| compile_type_expr(ctx, arg))
                .collect::<Vec<_>>();

            // let ans = ctx.new_cluster();
            let Some(name) = get_type_name(ctx.ex.program, base) else {
                let loc = ctx.ex.program.type_expr_loc(base);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "type specialization base must be a type name",
                });
                return ctx.new_cluster();
            };

            let Some(def) = ctx.ex.program.definitions.get(&name) else {
                //we dont allow generics on local structs
                //so this is either not a struct at all
                //or a struct with no generics
                let loc = ctx.ex.program.type_expr_loc(base);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "type specialization base must be a global type",
                });
                return ctx.new_cluster();
            };

            let Defined::Type(g) = def else {
                let loc = ctx.ex.program.type_expr_loc(base);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "type specialization expects a type definition",
                });
                return ctx.new_cluster();
            };

            let Some(t) = ctx.ex.ans.typedef_types.get(g) else {
                //this happens only in global context
                //and so it only happens when we specifically solve for global structs
                //because of this to break the recursion we are gona cheat
                //but with a tiny bit of class

                let Some(_cid) = ctx.search.local_types.get(&name) else {
                    let output = ctx.new_cluster();
                    ctx.req.pending_specializations.push(PendingSpecialization {
                        name,
                        global: *g,
                        generics,
                        output,
                    });
                    return output;
                };

                //we would need to double check here that its not a side speciliztion.
                //that acually ends up being a bunch of work
                //instead we can make sure that all structs defined globally are inserted ASAP into ans.typedef_types
                //and this saves us the hassle
                let loc = ctx.ex.program.type_expr_loc(texpr);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "currently we only support specilizing struct definitions directly",
                });

                return ctx.new_cluster();
            };

            let TypeValue::Struct { id: sid, .. } = ctx.ex.store.type_value(*t) else {
                let loc = ctx.ex.program.type_expr_loc(base);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "type specialization expects a struct type",
                });
                return ctx.new_cluster();
            };
            let sid = *sid;

            let expected = ctx.ex.store.struct_value(sid).gen_count;
            if generics.len() != expected {
                let loc = ctx.ex.program.type_expr_loc(texpr);
                ctx.ex.push_error(TypeError::Simple {
                    loc,
                    message: "wrong number of generic arguments for struct type",
                });
                return ctx.new_cluster();
            }

            ctx.new_struct_instance(sid, generics)
        }
        _ => {
            let c = ctx.new_cluster();
            ctx.push_error(TypeError::ExpectedTypeExpr { type_expr: texpr });
            c
        }
    }
}

fn get_type_name(prog: &Program, t: TExpId) -> Option<NameId> {
    match prog.type_expr(t) {
        TypeExpr::NameRef(n) => Some(n),
        _ => None,
    }
}

fn type_check_func_signature(
    ctx: &mut InferState,
    v: ValId,
    calling_convention: CallingConvention,
    generics: PatternSpan,
    params: PatternSpan,
    output_type: Option<TExpId>,
) {
    for (i, pat) in generics.ids().enumerate() {
        gather_generic_constraints(ctx, pat, GenId(i));
    }

    let inputs = params
        .ids()
        .map(|pat| gather_pattern_constraints_with_generics::<true>(ctx, pat))
        .collect::<Vec<_>>();

    let output = if let Some(x) = output_type {
        compile_type_expr(ctx, x)
    } else {
        ctx.new_solved(BuiltinType::Void.into())
    };

    let f = ctx.new_func(FuncInfer {
        calling_convention,
        generics: generics.len(),
        inputs,
        output,
    });
    ctx.bind_val(v, f);
    main_solver(ctx);
}

fn gather_func_signature<const GLOBAL_SCOPE: bool>(
    ctx: &mut InferState,
    v: ValId,
    calling_convention: CallingConvention,
    generics: PatternSpan,
    params: PatternSpan,
    output_type: Option<TExpId>,
) -> (CId, CId) {
    if !GLOBAL_SCOPE && !generics.is_empty() {
        let loc = generics
            .ids()
            .next()
            .map(|pat| ctx.ex.program.pattern_loc(pat))
            .unwrap_or_else(|| ctx.ex.program.value_loc(v));
        ctx.push_error(TypeError::Simple {
            loc,
            message: "generic functions are only allowed at the top level",
        });
    }

    for (i, pat) in generics.ids().enumerate() {
        gather_generic_constraints(ctx, pat, GenId(i));
    }

    let inputs = params
        .ids()
        .map(|pat| gather_pattern_constraints_with_generics::<GLOBAL_SCOPE>(ctx, pat))
        .collect::<Vec<_>>();

    let output = if let Some(x) = output_type {
        compile_type_expr(ctx, x)
    } else {
        ctx.new_solved(BuiltinType::Void.into())
    };

    let f = ctx.new_func(FuncInfer {
        calling_convention,
        generics: if GLOBAL_SCOPE { generics.len() } else { 0 },
        inputs,
        output,
    });

    if !GLOBAL_SCOPE {
        ctx.bind_val(v, f);
    }
    (f, output)
}

fn gather_func_constraints<const GLOBAL_SCOPE: bool>(
    ctx: &mut InferState,
    v: ValId,
    calling_convention: CallingConvention,
    generics: PatternSpan,
    params: PatternSpan,
    output_type: Option<TExpId>,
    body: Option<ValId>,
) -> CId {
    let (f, output) = gather_func_signature::<GLOBAL_SCOPE>(
        ctx,
        v,
        calling_convention,
        generics,
        params,
        output_type,
    );

    let Some(body) = body else {
        return f;
    };

    let body_cluster = gather_constraints(ctx, body, Some(output));

    if let Err(clash) = ctx.unify(body_cluster, output) {
        let found = match ctx.ex.program.value(body) {
            Value::Block {
                statements: _,
                return_value: Some(x),
            } => x,
            _ => body,
        };
        ctx.push_error(TypeError::FunctionOutputAnnotationMismatch {
            output_type,
            constrained: found,
            clash,
        });
    }

    //TODO limit f on params and out somehow
    //this might need to be done ahead of time globaly for all funcs
    //so that we can have weird type recursions
    //if thats the case this part might be just compiling cluster,(params need to be gathered so we get them in as vars we can use)
    f
}

#[inline(always)]
fn is_binary_operator_overload_name(name: StrId) -> bool {
    matches!(
        name,
        ADD_STR
            | SUB_STR
            | MUL_STR
            | DIV_STR
            | MOD_STR
            | BITAND_STR
            | BITOR_STR
            | BITXOR_STR
            | SHL_STR
            | SHR_STR
            | EQ_STR
            | NE_STR
            | LT_STR
            | LE_STR
            | GT_STR
            | GE_STR
    )
}

#[inline(always)]
fn is_unary_operator_overload_name(name: StrId) -> bool {
    matches!(
        name,
        NEG_STR | NOT_STR | BITNOT_STR | PRE_INC_STR | POST_INC_STR | PRE_DEC_STR | POST_DEC_STR
    )
}

#[inline(always)]
fn is_known_special_member_method_name(name: StrId) -> bool {
    is_binary_operator_overload_name(name)
        || is_unary_operator_overload_name(name)
        || name == FREE_STR
        || name == DEREF_STR
        || name == DEREF_MUT_STR
}

#[inline(always)]
fn is_reserved_builtin_member_name(program: &Program, method_name: StrId) -> bool {
    let method_name = program.str_intern.resolve(method_name);
    method_name.starts_with("__") && !method_name.ends_with('_')
}

#[inline(always)]
fn is_named_struct_type(store: &TypeStore, ty: TypeId, struct_name: NameId) -> bool {
    match store.type_value(ty) {
        TypeValue::Struct { id, .. } => store.struct_value(*id).name == Some(struct_name),
        _ => false,
    }
}

#[inline(always)]
fn method_signature_type_parts(store: &TypeStore, ty: TypeId) -> Option<(&[TypeId], TypeId)> {
    match store.type_value(ty) {
        TypeValue::Func { params, ret, .. } => Some((params.as_slice(), *ret)),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemberSelfStyle {
    Value,
    Ref { mutable: bool },
}

#[inline(always)]
fn get_member_self_style(
    store: &TypeStore,
    method_ty: TypeId,
    struct_name: NameId,
) -> Option<MemberSelfStyle> {
    let (inputs, _) = method_signature_type_parts(store, method_ty)?;
    let first_input = *inputs.first()?;
    match store.type_value(first_input) {
        TypeValue::Struct { .. } if is_named_struct_type(store, first_input, struct_name) => {
            Some(MemberSelfStyle::Value)
        }
        TypeValue::Ptr { tgt, raw, mutable }
            if !*raw && is_named_struct_type(store, *tgt, struct_name) =>
        {
            Some(MemberSelfStyle::Ref { mutable: *mutable })
        }
        _ => None,
    }
}

#[inline(always)]
fn member_self_input_cluster(core: &mut TypeCore, lhs: CId, style: MemberSelfStyle) -> CId {
    match style {
        MemberSelfStyle::Value => lhs,
        MemberSelfStyle::Ref { mutable } => {
            let ans = core.new_cluster();
            core.cluster[ans].state = ResolveKind::Ptr {
                tgt: lhs,
                raw: Some(false),
                mutable: Some(mutable),
            };
            ans
        }
    }
}

#[inline(always)]
fn is_self_like_member_input_type(store: &TypeStore, input: TypeId, struct_name: NameId) -> bool {
    match store.type_value(input) {
        TypeValue::Struct { .. } => is_named_struct_type(store, input, struct_name),
        TypeValue::Ptr {
            tgt,
            raw,
            mutable: _,
        } => !*raw && is_named_struct_type(store, *tgt, struct_name),
        _ => false,
    }
}

#[inline(always)]
fn is_mut_ref_to_named_struct_input_type(
    store: &TypeStore,
    input: TypeId,
    struct_name: NameId,
) -> bool {
    match store.type_value(input) {
        TypeValue::Ptr { tgt, raw, mutable } => {
            !*raw && *mutable && is_named_struct_type(store, *tgt, struct_name)
        }
        _ => false,
    }
}

#[inline(always)]
fn is_ref_to_named_struct_input_type(
    store: &TypeStore,
    input: TypeId,
    struct_name: NameId,
    mutable: bool,
) -> bool {
    match store.type_value(input) {
        TypeValue::Ptr {
            tgt,
            raw,
            mutable: is_mut,
        } => !*raw && *is_mut == mutable && is_named_struct_type(store, *tgt, struct_name),
        _ => false,
    }
}

#[inline(always)]
fn get_ref_target_type_if_kind(store: &TypeStore, ty: TypeId, mutable: bool) -> Option<TypeId> {
    match store.type_value(ty) {
        TypeValue::Ptr {
            tgt,
            raw,
            mutable: is_mut,
        } if !*raw && *is_mut == mutable => Some(*tgt),
        _ => None,
    }
}

#[inline(always)]
fn get_deref_method_target_type(
    store: &TypeStore,
    method_ty: TypeId,
    struct_name: NameId,
    self_mutable: bool,
    output_mutable: bool,
) -> Option<TypeId> {
    let (inputs, output) = method_signature_type_parts(store, method_ty)?;
    if inputs.len() != 1 {
        return None;
    }

    let first = inputs[0];
    if !is_ref_to_named_struct_input_type(store, first, struct_name, self_mutable) {
        return None;
    }

    get_ref_target_type_if_kind(store, output, output_mutable)
}

fn check_struct_deref_targets_compatible(ctx: &mut InferState, struct_name: NameId) {
    let Some(methods) = ctx.ex.program.member_methods.get(&struct_name) else {
        return;
    };
    let (Some(deref_method), Some(deref_mut_method)) =
        (methods.get(&DEREF_STR), methods.get(&DEREF_MUT_STR))
    else {
        return;
    };

    let (Some(deref_ty), Some(deref_mut_ty)) = (
        ctx.ex.ans.type_of(*deref_method),
        ctx.ex.ans.type_of(*deref_mut_method),
    ) else {
        // This runs after per-method signature solving in the global pass.
        // If either type is still missing here, the signature is unresolved for
        // this pass and should already have produced primary diagnostics.
        // Skipping avoids layering secondary mismatch noise.
        debug_assert!(
            !ctx.ex.errors.is_empty(),
            "missing deref method type without prior diagnostics"
        );
        return;
    };

    let Some(deref_target) =
        get_deref_method_target_type(ctx.ex.store, deref_ty, struct_name, false, false)
    else {
        return;
    };
    let Some(deref_mut_target) =
        get_deref_method_target_type(ctx.ex.store, deref_mut_ty, struct_name, true, true)
    else {
        return;
    };

    if deref_target != deref_mut_target {
        ctx.push_error(TypeError::Simple {
            loc: ctx.ex.program.value_loc(*deref_mut_method),
            message: "`__deref` and `__deref_mut` must dereference to the same target type",
        });
    }
}

fn check_special_member_method_signature(
    ctx: &mut InferState,
    method_val: ValId,
    struct_name: NameId,
    method_name: StrId,
) {
    let loc = ctx.ex.program.value_loc(method_val);

    if is_reserved_builtin_member_name(ctx.ex.program, method_name)
        && !is_known_special_member_method_name(method_name)
    {
        ctx.push_error(TypeError::UnknownBuiltinMemberMethod {
            site: method_val,
            method: method_name,
        });
    }

    if !is_known_special_member_method_name(method_name) {
        return;
    }

    let Some(method_ty) = ctx.ex.ans.type_of(method_val) else {
        // This check runs after `main_solver` for signature-only validation.
        // If a special member method type is still missing now, we rely on the
        // unresolved/signature diagnostics already produced by that solve pass.
        // Emitting follow-up shape checks here would only add noisy cascades.
        debug_assert!(
            !ctx.ex.errors.is_empty(),
            "missing special member method type without prior diagnostics"
        );
        return;
    };
    let Some((inputs, output)) = method_signature_type_parts(ctx.ex.store, method_ty) else {
        return;
    };
    let inputs = inputs.to_vec();

    if method_name == FREE_STR {
        let Some(first_input) = inputs.first().copied() else {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "special member methods must take `self` as the first parameter",
            });
            return;
        };

        if !is_mut_ref_to_named_struct_input_type(ctx.ex.store, first_input, struct_name) {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "`__free` must take `&mut self` as the first parameter",
            });
            return;
        }

        let additional_args = inputs.len() - 1;
        if additional_args != 0 {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "`__free` must not take parameters after `self`",
            });
            return;
        }

        if output != BuiltinType::Void.into() {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "`__free` must return `void`",
            });
        }
        return;
    }

    if method_name == DEREF_STR {
        let Some(first_input) = inputs.first().copied() else {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "special member methods must take `self` as the first parameter",
            });
            return;
        };

        if !is_ref_to_named_struct_input_type(ctx.ex.store, first_input, struct_name, false) {
            ctx.push_error(TypeError::Simple {
                loc: loc.clone(),
                message: "`__deref` must take `&self` as the first parameter",
            });
        }

        if inputs.len() != 1 {
            ctx.push_error(TypeError::Simple {
                loc: loc.clone(),
                message: "`__deref` must not take parameters after `self`",
            });
        }

        if get_ref_target_type_if_kind(ctx.ex.store, output, false).is_none() {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "`__deref` must return a non-raw shared reference `&T`",
            });
        }
        return;
    }

    if method_name == DEREF_MUT_STR {
        let Some(first_input) = inputs.first().copied() else {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "special member methods must take `self` as the first parameter",
            });
            return;
        };

        if !is_ref_to_named_struct_input_type(ctx.ex.store, first_input, struct_name, true) {
            ctx.push_error(TypeError::Simple {
                loc: loc.clone(),
                message: "`__deref_mut` must take `&mut self` as the first parameter",
            });
        }

        if inputs.len() != 1 {
            ctx.push_error(TypeError::Simple {
                loc: loc.clone(),
                message: "`__deref_mut` must not take parameters after `self`",
            });
        }

        if get_ref_target_type_if_kind(ctx.ex.store, output, true).is_none() {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "`__deref_mut` must return a non-raw mutable reference `&mut T`",
            });
        }
        return;
    }

    let Some(first_input) = inputs.first().copied() else {
        ctx.push_error(TypeError::Simple {
            loc,
            message: "special member methods must take `self` as the first parameter",
        });
        return;
    };

    let additional_args = inputs.len() - 1;

    if is_binary_operator_overload_name(method_name) {
        if !is_self_like_member_input_type(ctx.ex.store, first_input, struct_name) {
            ctx.push_error(TypeError::Simple {
                loc: loc.clone(),
                message: "binary operator overloads must take `self` as the first parameter type",
            });
        }

        if additional_args != 1 {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "binary operator overloads must take exactly one parameter after `self`",
            });
        }
    } else if is_unary_operator_overload_name(method_name) {
        if !is_self_like_member_input_type(ctx.ex.store, first_input, struct_name) {
            ctx.push_error(TypeError::Simple {
                loc: loc.clone(),
                message: "unary operator overloads must take `self` as the first parameter type",
            });
        }

        if additional_args != 0 {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "unary operator overloads must not take parameters after `self`",
            });
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
        ResolveKind::Struct(_) => Some(false),
        ResolveKind::Tuple(_) => Some(false),
        ResolveKind::Array { .. } => Some(false),
        ResolveKind::Ptr { .. } => Some(false),
        ResolveKind::Never => Some(false),
        ResolveKind::Nothing => None,
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
        ResolveKind::Struct(_) => Some(false),
        ResolveKind::Tuple(_) => Some(false),
        ResolveKind::Array { .. } => Some(false),
        ResolveKind::Ptr { .. } => Some(false),
        ResolveKind::Never => Some(false),
        ResolveKind::Nothing => None,
    }
}

#[inline(always)]
fn cluster_is_bool(
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
        ResolveKind::Never => Some(false),
        ResolveKind::Nothing => None,
    }
}

/// WARNING: this function is only intended for when lhs+rhs are not user defined
/// we specifically do not check for user overloading
/// Operator legality, tri-state:
///   Some(true)  = definitely allowed
///   Some(false) = definitely illegal
///   None        = insufficient info
#[inline(always)]
fn system_types_operator_applicable(
    ex: &mut ExternState,
    types: &mut TypeState,
    op: BinOp,
    cid: CId,
) -> Option<bool> {
    use BinOp::*;
    let store = &ex.store;
    let parent = &mut types.core.parent;
    let cluster = &types.core.cluster;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperandKind {
    KnownNonUser,
    UserStruct(Option<NameId>),
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawPointerOperandKind {
    RawPointer(CId),
    UnknownRawPointer(CId),
    NonRawPointer,
    NotPointer,
    Unknown,
}

#[inline(always)]
fn classify_raw_pointer_operand(
    ex: &mut ExternState,
    core: &mut TypeCore,
    cid: CId,
) -> RawPointerOperandKind {
    let root = core.find_root(cid);
    match core.cluster[root].state {
        ResolveKind::Solved(t) => match ex.store.type_value(t) {
            TypeValue::Ptr { raw: true, .. } => RawPointerOperandKind::RawPointer(root),
            TypeValue::Ptr { raw: false, .. } => RawPointerOperandKind::NonRawPointer,
            _ => RawPointerOperandKind::NotPointer,
        },
        ResolveKind::Ptr {
            raw: Some(true), ..
        } => RawPointerOperandKind::RawPointer(root),
        ResolveKind::Ptr {
            raw: Some(false), ..
        } => RawPointerOperandKind::NonRawPointer,
        ResolveKind::Ptr { raw: None, .. } => RawPointerOperandKind::UnknownRawPointer(root),
        ResolveKind::Nothing | ResolveKind::Never => RawPointerOperandKind::Unknown,
        _ => RawPointerOperandKind::NotPointer,
    }
}

#[inline(always)]
fn classify_operand(ex: &mut ExternState, types: &mut TypeState, cid: CId) -> OperandKind {
    let root = types.root(cid);
    match types.core.cluster[root].state {
        ResolveKind::IntLike
        | ResolveKind::FloatLike
        | ResolveKind::Func(_)
        | ResolveKind::Array { .. }
        | ResolveKind::Tuple(_) => OperandKind::KnownNonUser,

        ResolveKind::Solved(t) => match ex.store.type_value(t) {
            TypeValue::Struct { id, generics: _ } => {
                OperandKind::UserStruct(ex.store.struct_value(*id).name)
            }
            _ => OperandKind::KnownNonUser,
        },
        ResolveKind::Struct(call_id) => {
            let sid = types.extra.struct_infers[call_id.0].sid;
            OperandKind::UserStruct(ex.store.struct_value(sid).name)
        }

        ResolveKind::Ptr {
            raw: Some(true), ..
        } => OperandKind::KnownNonUser,
        ResolveKind::Ptr {
            tgt,
            raw: Some(false),
            ..
        } => classify_operand(ex, types, tgt),
        ResolveKind::Ptr { tgt, .. } => match classify_operand(ex, types, tgt) {
            OperandKind::KnownNonUser => OperandKind::KnownNonUser,
            _ => OperandKind::Unknown,
        },

        ResolveKind::Nothing => OperandKind::Unknown,

        //for never its probably best if we treat it as an unresolved in the end
        //we can do something more fancy but that would just be confusing
        ResolveKind::Never => OperandKind::Unknown,
    }
}

#[inline(always)]
fn bin_op_overload_name(op: BinOp) -> StrId {
    match op {
        BinOp::Add => ADD_STR,
        BinOp::Sub => SUB_STR,
        BinOp::Mul => MUL_STR,
        BinOp::Div => DIV_STR,
        BinOp::Mod => MOD_STR,
        BinOp::BitAnd => BITAND_STR,
        BinOp::BitOr => BITOR_STR,
        BinOp::BitXor => BITXOR_STR,
        BinOp::Shl => SHL_STR,
        BinOp::Shr => SHR_STR,
        BinOp::Eq => EQ_STR,
        BinOp::Ne => NE_STR,
        BinOp::Lt => LT_STR,
        BinOp::Le => LE_STR,
        BinOp::Gt => GT_STR,
        BinOp::Ge => GE_STR,
    }
}

#[inline(always)]
fn un_op_overload_name(op: UnOp) -> StrId {
    match op {
        UnOp::Neg => NEG_STR,
        UnOp::Not => NOT_STR,
        UnOp::BitNot => BITNOT_STR,
    }
}

#[inline(always)]
fn assign_inc_dec_overload_name(flavor: AssignIncDecFlavor) -> StrId {
    match flavor {
        AssignIncDecFlavor::PreInc => PRE_INC_STR,
        AssignIncDecFlavor::PostInc => POST_INC_STR,
        AssignIncDecFlavor::PreDec => PRE_DEC_STR,
        AssignIncDecFlavor::PostDec => POST_DEC_STR,
    }
}

#[inline(always)]
fn assign_inc_dec_fallback_bin_op(flavor: AssignIncDecFlavor) -> BinOp {
    match flavor {
        AssignIncDecFlavor::PreInc | AssignIncDecFlavor::PostInc => BinOp::Add,
        AssignIncDecFlavor::PreDec | AssignIncDecFlavor::PostDec => BinOp::Sub,
    }
}

/// Unify only if roots differ; report whether a merge happened.
#[inline]
fn unify_if_distinct(
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

    if types.is_never(ra) {
        return Ok(false);
    }
    if types.is_never(rb) {
        return Ok(false);
    }

    unify_clusters_inlined(ex, types, ra, rb)?;
    Ok(true)
}

#[inline(always)]
fn function_parts_from_cluster(
    ex: &ExternState,
    types: &mut TypeState,
    cid: CId,
) -> Option<(Vec<CId>, CId)> {
    let root = types.root(cid);

    match types.cluster_state(root) {
        ResolveKind::Func(call) => {
            // IMPORTANT:
            // clone inputs because unify may mutate graph later
            let inputs = types.func(call).inputs.clone();
            let output = types.func(call).output;
            Some((inputs, output))
        }

        ResolveKind::Solved(t) => {
            let TypeValue::Func { params, ret, .. } = ex.store.type_value(t) else {
                return None;
            };

            // Reify solved function type into fresh local clusters
            let inputs = params
                .iter()
                .map(|p| types.new_solved(*p))
                .collect::<Vec<_>>();

            let output = types.new_solved(*ret);
            Some((inputs, output))
        }

        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
struct ResolveOutcome {
    progress: bool,
    retain: bool,
}

impl ResolveOutcome {
    #[inline(always)]
    fn keep(progress: bool) -> Self {
        Self {
            progress,
            retain: true,
        }
    }

    #[inline(always)]
    fn drop(progress: bool) -> Self {
        Self {
            progress,
            retain: false,
        }
    }
}

const OP_OVERLOAD_SIGNATURE_MISMATCH: &str =
    "operator overload arguments and result must match overload signature";

#[derive(Debug)]
struct ResolvedMemberOverload {
    params: Vec<CId>,
    ret: CId,
    self_style: MemberSelfStyle,
    full_method: CId,
}

#[inline(always)]
fn bin_op_overload_not_found_error(
    ex: &mut ExternState,
    types: &mut TypeState,
    site: &BinOpSite,
    lhs: CId,
    rhs: CId,
) -> TypeError {
    TypeError::BinOpOverloadNotFound {
        site: site.loc,
        op: site.op,
        lhs: site.lhs_val,
        rhs: site.rhs_val,
        lhs_type: types.bad_type(ex, lhs),
        rhs_type: types.bad_type(ex, rhs),
    }
}

#[inline(always)]
fn un_op_overload_not_found_error(
    ex: &mut ExternState,
    types: &mut TypeState,
    site: &UnOpSite,
    input: CId,
) -> TypeError {
    TypeError::UnOpOverloadNotFound {
        site: site.loc,
        op: site.op,
        operand: site.val,
        operand_type: types.bad_type(ex, input),
    }
}

#[inline(always)]
fn resolve_member_overload_signature(
    ex: &mut ExternState,
    types: &mut TypeState,
    method: ValId,
    struct_name: NameId,
    loc: ValId,
) -> Option<ResolvedMemberOverload> {
    let Some(method_ty) = ex.ans.type_of(method) else {
        unreachable!(
            "global member method signatures must be solved before body inference; missing type for operator overload"
        );
    };

    let self_style = get_member_self_style(ex.store, method_ty, struct_name)?;
    let method_local = solved_type_to_specialized_local(ex, types, method_ty, loc);
    let Some((params, ret)) = function_parts_from_cluster(ex, types, method_local) else {
        unreachable!("specialized operator overload method must resolve to a function shape");
    };

    Some(ResolvedMemberOverload {
        params,
        ret,
        self_style,
        full_method: method_local,
    })
}

#[inline(always)]
fn make_member_closure(
    ex: &mut ExternState,
    types: &mut TypeState,
    receiver: CId,
    method: ResolvedMemberOverload,
    _loc: ValId,
) -> Result<CId, TypeClash> {
    let ResolvedMemberOverload {
        mut params,
        ret,
        self_style,
        full_method: _,
    } = method;
    debug_assert!(!params.is_empty());

    let self_param = params.remove(0);
    let self_input = member_self_input_cluster(&mut types.core, receiver, self_style);
    unify_if_distinct(ex, types, self_param, self_input)?;

    Ok(types.new_func(FuncInfer {
        calling_convention: CallingConvention::Unknown,
        generics: 0,
        inputs: params,
        output: ret,
    }))
}

#[inline(always)]
fn resolve_operator_site(
    ex: &mut ExternState,
    types: &mut TypeState,
    member_method_type_sites: &mut Vec<PendingMemberMethodType>,
    site: &mut BinOpSite,
) -> ResolveOutcome {
    use BinOp::*;

    let mut progress = false;
    let lhs = types.root(site.lhs);
    let rhs = types.root(site.rhs);
    let out = types.root(site.output);
    let op = site.op;

    let lhs_kind = classify_operand(ex, types, lhs);
    // let rhs_kind = classify_operand(ex,types, rhs);

    if let OperandKind::UserStruct(struct_name) = lhs_kind {
        let Some(struct_name) = struct_name else {
            unreachable!(
                "operator overload lookup produced a method only when struct_name is present"
            );
        };

        let method = ex
            .program
            .member_methods
            .get(&struct_name)
            .and_then(|methods| methods.get(&bin_op_overload_name(op)))
            .copied();

        if let Some(method) = method {
            let Some(overload_sig) =
                resolve_member_overload_signature(ex, types, method, struct_name, site.loc)
            else {
                let err = bin_op_overload_not_found_error(ex, types, site, lhs, rhs);
                ex.push_error(err);
                return ResolveOutcome::drop(progress);
            };

            if overload_sig.params.len() != 2 {
                let err = bin_op_overload_not_found_error(ex, types, site, lhs, rhs);
                ex.push_error(err);
                return ResolveOutcome::drop(progress);
            }

            let overload_mismatch: Result<(), TypeClash> = (|| {
                let method_name = bin_op_overload_name(op);
                let full_method = overload_sig.full_method;
                let method_closure = make_member_closure(ex, types, lhs, overload_sig, site.loc)?;
                let expected_fn = types.new_func(FuncInfer {
                    calling_convention: CallingConvention::Unknown,
                    generics: 0,
                    inputs: vec![rhs],
                    output: out,
                });
                progress |= unify_if_distinct(ex, types, method_closure, expected_fn)?;
                member_method_type_sites.push(PendingMemberMethodType {
                    site: site.loc,
                    member: method_name,
                    full_method,
                    receiver: lhs,
                    receiver_value: site.lhs_val,
                });
                Ok(())
            })();

            if let Err(clash) = overload_mismatch {
                ex.push_error(TypeError::ValuesContradict {
                    expectation_reason: OP_OVERLOAD_SIGNATURE_MISMATCH,
                    site: site.loc,
                    found: site.lhs_val,
                    expected_place: site.rhs_val,
                    clash,
                });
                return ResolveOutcome::drop(progress);
            }

            return ResolveOutcome::drop(progress);
        }

        let err = bin_op_overload_not_found_error(ex, types, site, lhs, rhs);
        ex.push_error(err);
        return ResolveOutcome::drop(progress);
    }

    // if matches!(rhs_kind, OperandKind::UserStruct(_)) {
    //     //TODO we need to enforce the constraint.
    //     return ResolveOutcome::keep(progress);
    // }

    if matches!(op, Add | Sub) {
        //there simply isnt any intresting operator on non user types other than pointer arithmetic
        if matches!(lhs_kind, OperandKind::KnownNonUser)
            && let ResolveKind::Ptr { ref mut raw, .. } = types.core.cluster[lhs].state
        {
            if raw.is_none() {
                progress = true;
                *raw = Some(true);
            } else if matches!(raw, Some(false)) {
                //todo!("error")
            }
        }

        let lhs_ptr = classify_raw_pointer_operand(ex, &mut types.core, lhs);
        let rhs_ptr = classify_raw_pointer_operand(ex, &mut types.core, rhs);
        let rhs_int =
            cluster_is_int_like(ex.store, &mut types.core.parent, &types.core.cluster, rhs);

        if op == Sub {
            match (lhs_ptr, rhs_ptr) {
                (
                    RawPointerOperandKind::RawPointer(lhs_raw),
                    RawPointerOperandKind::RawPointer(rhs_raw),
                )
                | (
                    RawPointerOperandKind::RawPointer(lhs_raw),
                    RawPointerOperandKind::UnknownRawPointer(rhs_raw),
                )

                //todo if lhs is non user and rhs is a bit pointer like we can hard force both to be raw and the same
                => {
                    match unify_if_distinct(ex, types, lhs_raw, rhs_raw) {
                        Ok(changed) => progress |= changed,
                        Err(clash) => {
                            ex.push_error(TypeError::ValuesContradict {
                                expectation_reason:
                                    "pointer subtraction requires both operands have the same pointer type",
                                site: site.loc,
                                found: site.lhs_val,
                                expected_place: site.rhs_val,
                                clash,
                            });
                            return ResolveOutcome::drop(progress);
                        }
                    }

                    match force_type(
                        ex, types,
                        out,
                        BuiltinType::Isize.into(),
                    ) {
                        Ok(()) => {}
                        Err(clash) => {
                            ex.push_error(TypeError::ValuesContradict {
                                expectation_reason: "pointer subtraction result must be isize",
                                site: site.loc,
                                found: site.lhs_val,
                                expected_place: site.rhs_val,
                                clash,
                            });
                            return ResolveOutcome::drop(progress);
                        }
                    }

                    return ResolveOutcome::drop(progress);
                }
                _ => {}
            }
        }

        match (lhs_ptr, rhs_int, op) {
            (RawPointerOperandKind::RawPointer(ptr), Some(true), _)
            | (RawPointerOperandKind::RawPointer(ptr), _, Add) => {
                match force_type_if_distinct(ex, types, rhs, BuiltinType::Usize.into()) {
                    Ok(changed) => progress |= changed,
                    Err(clash) => {
                        ex.push_error(TypeError::ValuesContradict {
                            expectation_reason: "pointer add may only happen with usize",
                            site: site.loc,
                            found: site.lhs_val,
                            expected_place: site.rhs_val,
                            clash,
                        });
                    }
                }
                match unify_if_distinct(ex, types, out, ptr) {
                    Ok(changed) => progress |= changed,
                    Err(clash) => {
                        ex.push_error(TypeError::ValuesContradict {
                            expectation_reason: "pointer arithmetic preserves type",
                            site: site.loc,
                            found: site.lhs_val,
                            expected_place: site.rhs_val,
                            clash,
                        });
                    }
                }
                return ResolveOutcome::drop(progress);
            }
            _ => {}
        }

        if matches!(
            lhs_ptr,
            RawPointerOperandKind::UnknownRawPointer(_) | RawPointerOperandKind::Unknown
        ) {
            return ResolveOutcome::keep(progress);
        }
    }

    if matches!(lhs_kind, OperandKind::Unknown) {
        return ResolveOutcome::keep(progress);
    }

    // basic lit like operands
    // ----------------------------------------------------
    // 1) Early legality rejection (single helper)
    // ----------------------------------------------------
    let lhs_ok = system_types_operator_applicable(ex, types, op, lhs);
    let rhs_ok = system_types_operator_applicable(ex, types, op, rhs);

    if lhs_ok == Some(false) || rhs_ok == Some(false) {
        let err = bin_op_overload_not_found_error(ex, types, site, lhs, rhs);
        ex.push_error(err);
        return ResolveOutcome::drop(progress);
    }

    // ----------------------------------------------------
    // 2) Equality / comparisons
    //
    // NOTE:
    // - operand equality is already enforced in gather
    // - output = bool is already enforced in gather
    // ----------------------------------------------------
    if matches!(op, Eq | Ne | Lt | Le | Gt | Ge) {
        return ResolveOutcome::drop(progress);
    }

    // ----------------------------------------------------
    // 3) Arithmetic / bitwise
    //
    // - Only unify once both sides are known numeric
    // ----------------------------------------------------
    let (store, parent, cluster) = (&ex.store, &mut types.core.parent, &mut types.core.cluster);
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
        return ResolveOutcome::keep(progress);
    }

    // (a) unify operands
    match unify_if_distinct(ex, types, lhs, rhs) {
        Ok(changed) => progress |= changed,
        Err(clash) => {
            ex.push_error(TypeError::ValuesContradict {
                expectation_reason: "binary operator requires operands of the same type",
                site: site.loc,
                found: site.lhs_val,
                expected_place: site.rhs_val,
                clash,
            });
            return ResolveOutcome::drop(progress);
        }
    }

    let operand = types.root(lhs);

    // (b) unify output with operand
    match unify_if_distinct(ex, types, out, operand) {
        Ok(changed) => progress |= changed,
        Err(clash) => {
            ex.push_error(TypeError::ValuesContradict {
                expectation_reason: "operator result type must match operand type",
                site: site.loc,
                found: site.lhs_val,
                expected_place: site.rhs_val,
                clash,
            });
            return ResolveOutcome::drop(progress);
        }
    }

    ResolveOutcome::drop(progress)
}

#[inline(always)]
fn resolve_unary_operator_site(
    ex: &mut ExternState,
    types: &mut TypeState,
    member_method_type_sites: &mut Vec<PendingMemberMethodType>,
    site: &mut UnOpSite,
) -> ResolveOutcome {
    use UnOp::*;

    let mut progress = false;
    let input = types.root(site.input);
    let out = types.root(site.output);
    let op = site.op;

    let operand_kind = classify_operand(ex, types, input);
    if let OperandKind::UserStruct(struct_name) = operand_kind {
        let Some(struct_name) = struct_name else {
            unreachable!(
                "operator overload lookup produced a method only when struct_name is present"
            );
        };

        let method = ex
            .program
            .member_methods
            .get(&struct_name)
            .and_then(|methods| methods.get(&un_op_overload_name(op)))
            .copied();
        if let Some(method) = method {
            let Some(overload_sig) =
                resolve_member_overload_signature(ex, types, method, struct_name, site.loc)
            else {
                let err = un_op_overload_not_found_error(ex, types, site, input);
                ex.push_error(err);
                return ResolveOutcome::drop(progress);
            };

            if overload_sig.params.len() != 1 {
                let err = un_op_overload_not_found_error(ex, types, site, input);
                ex.push_error(err);
                return ResolveOutcome::drop(progress);
            }

            let overload_mismatch: Result<(), TypeClash> = (|| {
                let method_name = un_op_overload_name(op);
                let full_method = overload_sig.full_method;
                let method_closure = make_member_closure(ex, types, input, overload_sig, site.loc)?;
                let expected_fn = types.new_func(FuncInfer {
                    calling_convention: CallingConvention::Unknown,
                    generics: 0,
                    inputs: Vec::new(),
                    output: out,
                });
                progress |= unify_if_distinct(ex, types, method_closure, expected_fn)?;
                member_method_type_sites.push(PendingMemberMethodType {
                    site: site.loc,
                    member: method_name,
                    full_method,
                    receiver: input,
                    receiver_value: site.val,
                });
                Ok(())
            })();

            if let Err(clash) = overload_mismatch {
                ex.push_error(TypeError::ValuesContradict {
                    expectation_reason: OP_OVERLOAD_SIGNATURE_MISMATCH,
                    site: site.loc,
                    found: site.val,
                    expected_place: site.loc,
                    clash,
                });
            }

            return ResolveOutcome::drop(progress);
        }

        let err = un_op_overload_not_found_error(ex, types, site, input);
        ex.push_error(err);
        return ResolveOutcome::drop(progress);
    }

    if operand_kind == OperandKind::Unknown {
        return ResolveOutcome::keep(progress);
    }

    let (store, parent, cluster) = (&ex.store, &mut types.core.parent, &mut types.core.cluster);
    match op {
        Not => {
            if let Some(false) = cluster_is_bool(store, parent, cluster, input) {
                let err = un_op_overload_not_found_error(ex, types, site, input);
                ex.push_error(err);
                return ResolveOutcome::drop(progress);
            }
            match unify_if_distinct(ex, types, input, out) {
                Ok(changed) => progress |= changed,
                Err(clash) => {
                    ex.push_error(TypeError::ValuesContradict {
                        expectation_reason: "logical not requires a bool operand",
                        site: site.loc,
                        found: site.val,
                        expected_place: site.loc,
                        clash,
                    });
                    return ResolveOutcome::drop(progress);
                }
            }
        }
        Neg => {
            match (
                cluster_is_int_like(store, parent, cluster, input),
                cluster_is_float_like(store, parent, cluster, input),
            ) {
                (Some(true), _) | (_, Some(true)) => {}
                (Some(false), Some(false)) => {
                    let err = un_op_overload_not_found_error(ex, types, site, input);
                    ex.push_error(err);
                    return ResolveOutcome::drop(progress);
                }
                _ => return ResolveOutcome::keep(progress),
            }

            match unify_if_distinct(ex, types, input, out) {
                Ok(changed) => progress |= changed,
                Err(clash) => {
                    ex.push_error(TypeError::ValuesContradict {
                        expectation_reason: "negation requires operand and result types match",
                        site: site.loc,
                        found: site.val,
                        expected_place: site.loc,
                        clash,
                    });
                    return ResolveOutcome::drop(progress);
                }
            }
        }
        BitNot => {
            match cluster_is_int_like(store, parent, cluster, input) {
                Some(true) => {}
                Some(false) => {
                    let err = un_op_overload_not_found_error(ex, types, site, input);
                    ex.push_error(err);
                    return ResolveOutcome::drop(progress);
                }
                None => return ResolveOutcome::keep(progress),
            }

            match unify_if_distinct(ex, types, input, out) {
                Ok(changed) => progress |= changed,
                Err(clash) => {
                    ex.push_error(TypeError::ValuesContradict {
                        expectation_reason: "bitwise not requires operand and result types match",
                        site: site.loc,
                        found: site.val,
                        expected_place: site.loc,
                        clash,
                    });
                    return ResolveOutcome::drop(progress);
                }
            }
        }
    }

    ResolveOutcome::drop(progress)
}

#[inline(always)]
fn resolve_assign_pre_post_site(
    ex: &mut ExternState,
    types: &mut TypeState,
    member_method_type_sites: &mut Vec<PendingMemberMethodType>,
    site: &mut AssignPrePostSite,
) -> ResolveOutcome {
    let mut progress = false;
    let target = types.root(site.target);
    let implicit_rhs = types.root(site.implicit_rhs);

    let target_kind = classify_operand(ex, types, target);
    if let OperandKind::UserStruct(struct_name) = target_kind {
        let Some(struct_name) = struct_name else {
            unreachable!("member overload lookup requires named user struct")
        };

        let method_name = assign_inc_dec_overload_name(site.flavor);
        let method = ex
            .program
            .member_methods
            .get(&struct_name)
            .and_then(|methods| methods.get(&method_name))
            .copied();

        if let Some(method) = method {
            let Some(overload_sig) =
                resolve_member_overload_signature(ex, types, method, struct_name, site.loc)
            else {
                return ResolveOutcome::drop(progress);
            };

            if overload_sig.params.len() != 1 {
                return ResolveOutcome::drop(progress);
            }

            let overload_mismatch: Result<(), TypeClash> = (|| {
                let full_method = overload_sig.full_method;
                let method_closure =
                    make_member_closure(ex, types, target, overload_sig, site.loc)?;
                let expected_fn = types.new_func(FuncInfer {
                    calling_convention: CallingConvention::Unknown,
                    generics: 0,
                    inputs: Vec::new(),
                    output: target,
                });
                progress |= unify_if_distinct(ex, types, method_closure, expected_fn)?;
                member_method_type_sites.push(PendingMemberMethodType {
                    site: site.loc,
                    member: method_name,
                    full_method,
                    receiver: target,
                    receiver_value: site.target_val,
                });
                Ok(())
            })();

            if let Err(clash) = overload_mismatch {
                ex.push_error(TypeError::ValuesContradict {
                    expectation_reason: OP_OVERLOAD_SIGNATURE_MISMATCH,
                    site: site.loc,
                    found: site.target_val,
                    expected_place: site.loc,
                    clash,
                });
            }

            return ResolveOutcome::drop(progress);
        }
    }

    let mut fallback_site = BinOpSite {
        loc: site.loc,
        op: assign_inc_dec_fallback_bin_op(site.flavor),
        lhs_val: site.target_val,
        rhs_val: site.loc,
        lhs: target,
        rhs: implicit_rhs,
        output: target,
    };

    let outcome = resolve_operator_site(ex, types, member_method_type_sites, &mut fallback_site);
    progress |= outcome.progress;
    if outcome.retain {
        site.target = fallback_site.lhs;
        site.implicit_rhs = fallback_site.rhs;
        return ResolveOutcome::keep(progress);
    }
    ResolveOutcome::drop(progress)
}

#[inline(always)]
fn resolve_operator_types(ctx: &mut InferState) -> bool {
    let mut progress = false;
    let ex = &mut ctx.ex;
    let types = &mut ctx.types;
    let member_method_type_sites = &mut ctx.req.member_method_type_sites;
    ctx.req.bin_op_sites.retain_mut(|site| {
        let outcome = resolve_operator_site(ex, types, member_method_type_sites, site);
        progress |= outcome.progress;
        outcome.retain
    });

    ctx.req.un_op_sites.retain_mut(|site| {
        let outcome = resolve_unary_operator_site(ex, types, member_method_type_sites, site);
        progress |= outcome.progress;
        outcome.retain
    });

    ctx.req.assign_pre_post_sites.retain_mut(|site| {
        let outcome = resolve_assign_pre_post_site(ex, types, member_method_type_sites, site);
        progress |= outcome.progress;
        outcome.retain
    });

    progress
}

fn resolve_index_site(
    ex: &mut ExternState,
    types: &mut TypeState,
    site: &mut PendingIndex,
) -> ResolveOutcome {
    let mut progress = false;

    site.base = types.root(site.base);
    site.index = types.root(site.index);

    let mut current = site.base;
    let mut used_implicit_deref_steps = 0usize;
    let max_implicit_deref_steps = 64usize;
    let mut implicit_receivers = Vec::new();

    let element = loop {
        let program = &ex.program;

        match types.core.cluster[current].state {
            ResolveKind::Nothing => {
                site.base = current;
                return ResolveOutcome::keep(progress);
            }
            ResolveKind::Array { element, .. } => break element,
            ResolveKind::Ptr { tgt, .. } => {
                if used_implicit_deref_steps >= max_implicit_deref_steps {
                    ex.push_error(TypeError::Simple {
                        loc: program.value_loc(site.site),
                        message: "index autoderef recursion exceeded safety limit",
                    });
                    return ResolveOutcome::drop(progress);
                }
                implicit_receivers.push(current);
                current = types.root(tgt);
                used_implicit_deref_steps += 1;
            }
            ResolveKind::Solved(t) => match ex.store.type_value(t).clone() {
                TypeValue::Array(element, _) => break types.new_solved(element),
                TypeValue::Ptr { tgt, .. } => {
                    if used_implicit_deref_steps >= max_implicit_deref_steps {
                        ex.push_error(TypeError::Simple {
                            loc: program.value_loc(site.site),
                            message: "index autoderef recursion exceeded safety limit",
                        });
                        return ResolveOutcome::drop(progress);
                    }
                    let next = types.new_solved(tgt);
                    implicit_receivers.push(current);
                    current = types.root(next);
                    used_implicit_deref_steps += 1;
                }
                TypeValue::Struct { id, .. } => {
                    let Some(struct_name) = ex.store.struct_value(id).name else {
                        ex.push_error(TypeError::Simple {
                            loc: program.value_loc(site.site),
                            message: "indexing base must be an array or pointer to array",
                        });
                        return ResolveOutcome::drop(progress);
                    };
                    let Some(target) = resolve_struct_deref_target(
                        ex,
                        types,
                        site.site,
                        site.base_value,
                        current,
                        struct_name,
                    ) else {
                        ex.push_error(TypeError::Simple {
                            loc: ex.program.value_loc(site.site),
                            message: "indexing base must be an array or pointer to array",
                        });
                        return ResolveOutcome::drop(progress);
                    };
                    implicit_receivers.push(current);
                    implicit_receivers.push(target.deref_result_ptr);
                    current = types.root(target.target);
                    used_implicit_deref_steps += 1;
                }
                _ => {
                    ex.push_error(TypeError::Simple {
                        loc: program.value_loc(site.site),
                        message: "indexing base must be an array or pointer to array",
                    });
                    return ResolveOutcome::drop(progress);
                }
            },
            ResolveKind::Struct(rid) => {
                let sid = types.extra.struct_infers[rid.0].sid;
                let Some(struct_name) = ex.store.struct_value(sid).name else {
                    ex.push_error(TypeError::Simple {
                        loc: program.value_loc(site.site),
                        message: "indexing base must be an array or pointer to array",
                    });
                    return ResolveOutcome::drop(progress);
                };
                let Some(target) = resolve_struct_deref_target(
                    ex,
                    types,
                    site.site,
                    site.base_value,
                    current,
                    struct_name,
                ) else {
                    ex.push_error(TypeError::Simple {
                        loc: ex.program.value_loc(site.site),
                        message: "indexing base must be an array or pointer to array",
                    });
                    return ResolveOutcome::drop(progress);
                };
                implicit_receivers.push(current);
                implicit_receivers.push(target.deref_result_ptr);
                current = types.root(target.target);
                used_implicit_deref_steps += 1;
            }
            _ => {
                ex.push_error(TypeError::Simple {
                    loc: program.value_loc(site.site),
                    message: "indexing base must be an array or pointer to array",
                });
                return ResolveOutcome::drop(progress);
            }
        }
    };

    site.base = current;
    site.implicit_receivers = finalize_member_access_implicit_chain(
        implicit_receivers,
        used_implicit_deref_steps,
        current,
    );

    let usize_c = types.new_solved(BuiltinType::Usize.into());
    match unify_if_distinct(ex, types, site.index, usize_c) {
        Ok(changed) => progress |= changed,
        Err(clash) => {
            ex.push_error(TypeError::ValuesContradict {
                expectation_reason: "array indexing requires an index of type usize",
                site: site.site,
                found: site.index_value,
                expected_place: site.site,
                clash,
            });
            return ResolveOutcome::drop(progress);
        }
    }

    match unify_if_distinct(ex, types, element, site.output) {
        Ok(changed) => progress |= changed,
        Err(clash) => {
            ex.push_error(TypeError::ValuesContradict {
                expectation_reason: "index expression result must match indexed element type",
                site: site.site,
                found: site.base_value,
                expected_place: site.site,
                clash,
            });
            return ResolveOutcome::drop(progress);
        }
    }

    ResolveOutcome::drop(progress)
}

#[inline(always)]
fn resolve_deferred_types(ctx: &mut InferState) -> bool {
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
            ResolveKind::Ptr { tgt, raw, mutable } => {
                try_resolve_ptr_type(&mut ctx.ex, &mut ctx.types, tgt, raw, mutable)
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

#[inline(always)]
fn resolve_pointer_likes(ctx: &mut InferState) -> bool {
    let mut progress = false;

    ctx.req.pointer_likes.retain_mut(|pending| {
        let types = &mut ctx.types;
        let ex = &mut ctx.ex;
        let source = types.root(pending.source);
        pending.source = source;

        let result = match types.core.cluster[source].state {
            ResolveKind::Nothing => return true,
            ResolveKind::Ptr { tgt, .. } => Some(tgt),
            ResolveKind::Solved(t) => match ex.store.type_value(t) {
                TypeValue::Ptr { tgt, .. } => Some(types.new_solved(*tgt)),
                TypeValue::Struct { id, .. } => {
                    let struct_name = ex.store.struct_value(*id).name;
                    struct_name.and_then(|struct_name| {
                        resolve_struct_deref_target(
                            ex,
                            types,
                            pending.site,
                            pending.source_value,
                            source,
                            struct_name,
                        )
                        .map(|resolved| resolved.target)
                    })
                }
                _ => None,
            },
            ResolveKind::Struct(rid) => {
                let sid = types.extra.struct_infers[rid.0].sid;
                let struct_name = ex.store.struct_value(sid).name;
                struct_name.and_then(|struct_name| {
                    resolve_struct_deref_target(
                        ex,
                        types,
                        pending.site,
                        pending.source_value,
                        source,
                        struct_name,
                    )
                    .map(|resolved| resolved.target)
                })
            }
            _ => None,
        };

        let Some(result) = result else {
            let source_type = types.bad_type(ex, source);
            ex.push_error(TypeError::CannotDeref {
                site: pending.site,
                operand: pending.source_value,
                operand_type: source_type,
            });
            progress = true;
            return false;
        };

        match unify_if_distinct(ex, types, result, pending.target) {
            Ok(changed) => {
                progress |= changed;
                false
            }
            Err(clash) => {
                ex.push_error(TypeError::ValuesContradict {
                    expectation_reason: "dereference result must match pointee type",
                    site: pending.site,
                    found: pending.source_value,
                    expected_place: pending.site,
                    clash,
                });
                progress = true;
                false
            }
        }
    });

    progress
}

#[inline(always)]
fn resolve_pending_indexes(ctx: &mut InferState) -> bool {
    let mut progress = false;

    let ex = &mut ctx.ex;
    let types = &mut ctx.types;

    ctx.req.pending_indexes.retain_mut(|site| {
        let outcome = resolve_index_site(ex, types, site);
        progress |= outcome.progress;
        if !outcome.retain && !site.implicit_receivers.is_empty() {
            ctx.req
                .index_implicit_deref_sites
                .push(PendingMemberAccessImplicitDeref {
                    site: site.site,
                    receivers: std::mem::take(&mut site.implicit_receivers),
                });
        }
        outcome.retain
    });

    progress
}

#[inline(always)]
fn resolve_pending_member_accesses(ctx: &mut InferState) -> bool {
    let mut progress = false;

    let ex = &mut ctx.ex;
    let types = &mut ctx.types;
    let search = &mut ctx.search;
    let member_method_type_sites = &mut ctx.req.member_method_type_sites;
    let member_access_implicit_deref_sites = &mut ctx.req.member_access_implicit_deref_sites;

    ctx.req.pending_member_accesses.retain_mut(|pending| {
        let source = types.root(pending.source);
        pending.source = source;

        match try_resolve_member_access(
            ex,
            types,
            search,
            member_method_type_sites,
            pending.site,
            pending.base_value,
            source,
            pending.member,
            pending.kind,
        ) {
            MemberAccessResolve::Pending { source } => {
                pending.source = source;
                true
            }
            MemberAccessResolve::Resolved {
                result,
                implicit_receivers,
            } => {
                match unify_if_distinct(ex, types, result, pending.output) {
                    Ok(changed) => progress |= changed,
                    Err(clash) => {
                        ex.push_error(TypeError::ValuesContradict {
                            expectation_reason:
                                "member access result must match its inferred use constraints",
                            site: pending.site,
                            found: pending.site,
                            expected_place: pending.site,
                            clash,
                        });
                        progress = true;
                    }
                }
                if !implicit_receivers.is_empty() {
                    member_access_implicit_deref_sites.push(PendingMemberAccessImplicitDeref {
                        site: pending.site,
                        receivers: implicit_receivers,
                    });
                }
                false
            }
            MemberAccessResolve::Error(err) => {
                ex.push_error(err);
                progress = true;
                false
            }
        }
    });

    progress
}

#[inline(always)]
fn resolve_pending_specializations(ctx: &mut InferState) -> bool {
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

        let found = types.new_struct_instance(sid, p.generics.clone());
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

#[inline(always)]
// #[inline(never)]
// #[unsafe(no_mangle)]
fn finalize(ctx: &mut InferState) {
    let (
        val_cluster,
        pat_cluster,
        member_method_type_sites,
        member_access_implicit_deref_sites,
        index_implicit_deref_sites,
        parent,
        cluster,
        errors,
        ans,
    ) = (
        &ctx.search.val_cluster,
        &ctx.search.pat_cluster,
        &ctx.req.member_method_type_sites,
        &ctx.req.member_access_implicit_deref_sites,
        &ctx.req.index_implicit_deref_sites,
        &mut ctx.types.core.parent,
        &ctx.types.core.cluster,
        &mut ctx.ex.errors,
        &mut ctx.ex.ans,
    );

    // unsafe{perf_begin();}

    let mut reported: IdHashMap<CId, ()> = IdHashMap::default();
    let mut member_method_by_site: IdHashMap<ValId, PendingMemberMethodType> = IdHashMap::default();
    for entry in member_method_type_sites.iter().copied() {
        member_method_by_site.insert(entry.site, entry);
    }
    for (e, c) in ctx.search.typedef_cluster.iter() {
        let root = find_root(parent, *c);
        if let ResolveKind::Solved(t) = cluster[root].state {
            ans.typedef_types.insert(*e, t);
        } else if *c == root {
            errors.push(TypeError::UnresolvedTypeExpr { expr: *e });
            reported.insert(*c, ());
        }
    }

    for sdef in ctx.types.extra.struct_defs.iter() {
        for (i, (_n, c)) in sdef.fields.iter().enumerate() {
            let root = find_root(parent, *c);
            if let ResolveKind::Solved(t) = cluster[root].state {
                ctx.ex.store.structs[sdef.sid.0].fields[i].1 = t;
            } else if *c == root {
                let loc = ctx.ex.program.type_expr_loc(sdef.loc);
                errors.push(TypeError::Simple {
                    loc,
                    message: "could not infer struct field type",
                });
                reported.insert(*c, ());
            }
        }
    }

    for (v, c) in val_cluster.iter() {
        let root = find_root(parent, *c);
        if let ResolveKind::Solved(t) = cluster[root].state {
            ans.set_val(*v, t);
        } else if *c == root && !reported.contains_key(c) {
            errors.push(TypeError::Unresolved { value: *v });
            reported.insert(*c, ());
            if let Some(entry) = member_method_by_site.get(v) {
                let full_root = find_root(parent, entry.full_method);
                reported.insert(full_root, ());
            }
        }
    }

    for (p, c) in pat_cluster.iter() {
        let root = find_root(parent, *c);
        if let ResolveKind::Solved(t) = cluster[root].state {
            ans.set_pat(*p, t);
        } else if *c == root && !reported.contains_key(c) {
            errors.push(TypeError::UnresolvedPattern { pattern: *p });
            reported.insert(*c, ());
        }
    }

    for entry in member_method_type_sites.iter() {
        let root = find_root(parent, entry.full_method);
        if let ResolveKind::Solved(full_type) = cluster[root].state {
            ans.member_method_types.insert(
                entry.site,
                SolvedMemberMethodType {
                    member: entry.member,
                    full_type,
                },
            );
            continue;
        }

        if reported.contains_key(&root) {
            continue;
        }

        //these are tricky to report because there isnt TECHNICALLY a value
        //its an implicit value we added because of a cast.

        //if the output isnt resolved then fundementally this cant be solved so we are good
        //if it CAN be solved but the full signature cant that must be because of &self not being clear
        //in that case we need to report an error but its gona be a bad one...

        let receiver_root = find_root(parent, entry.receiver);
        if !matches!(cluster[receiver_root].state, ResolveKind::Solved(_))
            && !reported.contains_key(&receiver_root)
        {
            errors.push(TypeError::Unresolved {
                value: entry.receiver_value,
            });
            reported.insert(receiver_root, ());
            reported.insert(root, ());
        }
    }

    store_implicit_deref_chains(
        &mut ans.implicit_derefs,
        member_access_implicit_deref_sites,
        parent,
        cluster,
    );
    store_implicit_deref_chains(
        &mut ans.implicit_derefs,
        index_implicit_deref_sites,
        parent,
        cluster,
    );

    // let name = CStr::from_bytes_with_nul(b"finalize\0").unwrap();
    // unsafe { perf_done(name.as_ptr()); }
}

fn store_implicit_deref_chains(
    out: &mut IdHashMap<ValId, Vec<TypeId>>,
    entries: &[PendingMemberAccessImplicitDeref],
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
) {
    for entry in entries.iter() {
        let mut chain = Vec::with_capacity(entry.receivers.len());
        let mut all_solved = true;
        for receiver in entry.receivers.iter() {
            let root = find_root(parent, *receiver);
            match cluster[root].state {
                ResolveKind::Solved(t) => chain.push(t),
                _ => {
                    all_solved = false;
                    break;
                }
            }
        }
        if all_solved {
            out.insert(entry.site, chain);
        }
    }
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
                Defined::Func(v) => Some(v),
                _ => None,
            })
            .expect("expected a function definition")
    }

    fn find_value_by_name(program: &Program, name: &str) -> ValId {
        *program
            .definitions
            .iter()
            .find_map(|(n, def)| match def {
                Defined::Func(v) if program.name_string(*n) == name => Some(v),
                _ => None,
            })
            .unwrap_or_else(|| panic!("value `{}` not found", name))
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
                    .type_of(stmt)
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
                .type_of(call.base)
                .unwrap_or_else(|| panic!("missing type for member access in `{}`", name));
            let result_ty = solved
                .type_of(stmt)
                .unwrap_or_else(|| panic!("missing type for let statement `{}`", name));
            return (call.base, access_ty, result_ty);
        }

        panic!("let binding `{}` not found", name)
    }

    fn implicit_deref_chain_type_strings(
        program: &Program,
        store: &TypeStore,
        solved: &SolvedTypes,
        site: ValId,
    ) -> Option<Vec<String>> {
        solved
            .member_access_implicit_deref_chain(site)
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
        Ok(solved_types.type_of(body).unwrap())
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
        Ok(solved_types.type_of(body).unwrap())
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
        let src = "Box=struct{inner:&[int;2]}; Box.__deref_mut = fn(self:&mut Box)->&mut &[int;2] { &mut self.inner }; f = fn(b:Box)->int { let y:int = b[1:usize]; y };";
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
                "Box₀".to_string(),
                "&mut &[int;2]".to_string(),
                "&[int;2]".to_string(),
                "[int;2]".to_string(),
            ],
            "unexpected implicit deref chain for struct-deref indexing"
        );
    }

    #[test]
    fn generic_box_array_index_chain_includes_box_step() {
        let src = "free = cfn(p:*void); Box = struct[T]{ptr:*T}; Box.__free = fn[T](b:&mut Box[T]){free(b->ptr as *void)} Box.__deref = fn[T](b:&const Box[T])->&T{&*b.ptr} Box.__deref_mut = fn[T](b:&mut Box[T])->&mut T{&*b.ptr} f=fn(b:Box[[int]])->int { let y:int = b[0]; y };";
        let program = gather_program(src);
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
                "Box₀[[int]]".to_string(),
                "&mut [int]".to_string(),
                "[int]".to_string(),
            ],
            "unexpected implicit deref chain for generic Box indexing"
        );
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
            } => Some(*clash),
            _ => None,
        });
        let clash = clash.expect("expected if-condition type mismatch");

        let found = clash.found.expect("missing found type");
        let wanted = clash.wanted.expect("missing expected type");

        assert!(matches!(store.type_value(found.0), TypeValue::Ptr { .. }));
        assert!(matches!(
            store.type_value(wanted.0),
            TypeValue::Builtin(BuiltinType::Bool)
        ));
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
            } => Some(*clash),
            _ => None,
        });
        let clash = clash.expect("expected call-signature mismatch");

        let found = clash.found.expect("missing found type");
        let wanted = clash.wanted.expect("missing expected type");

        assert!(matches!(store.type_value(found.0), TypeValue::Func { .. }));
        assert!(matches!(
            store.type_value(wanted.0),
            TypeValue::Builtin(BuiltinType::Int)
        ));
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
            TypeError::AnnotationMismatch { clash, .. } => Some(*clash),
            _ => None,
        });
        let clash = clash.expect("expected annotation mismatch");

        let found = clash.found.expect("missing found type");
        let wanted = clash.wanted.expect("missing expected type");

        assert!(matches!(store.type_value(found.0), TypeValue::Tuple(_)));
        assert!(matches!(
            store.type_value(wanted.0),
            TypeValue::Builtin(BuiltinType::Int)
        ));
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
                raw: got_raw,
                mutable: got_mut,
            } = *store.type_value(ty)
            else {
                return false;
            };
            got_raw == raw
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
    fn closure_returns_typecheck() {
        let mut store = TypeStore::new();
        let ty = infer_fn(
            "f = fn()->int { let d :float = (fn()->_{if true return 1.0; 2.0})(); 2 }",
            &mut store,
        )
        .unwrap();
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

        let f = find_value_by_name(&program, "f");
        let f_ty = solved_types.type_of(f).expect("missing f type");
        let TypeValue::Func {
            calling_convention,
            generics: _,
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
            TypeError::ValuesContradict { clash, .. } => Some(*clash),
            _ => None,
        });
        let clash = clash.expect("expected mismatch error");

        let mut saw_hot = false;
        let mut saw_c = false;
        for side in [clash.found, clash.wanted] {
            let Some(side) = side else {
                continue;
            };
            let TypeValue::Func {
                calling_convention, ..
            } = store.type_value(side.0)
            else {
                continue;
            };
            saw_hot |= *calling_convention == CallingConvention::Hot;
            saw_c |= *calling_convention == CallingConvention::C;
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
                },
                TypeValue::Struct {
                    id: b_id,
                    generics: b_generics,
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
            TypeValue::Struct { id, generics: _ } => *id,
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
    fn calling_a_closure() {
        let mut store = TypeStore::new();
        infer_fn("f=fn()->int{(fn(x)->_{x})(2)}", &mut store).unwrap();
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
        let errs = infer_fn_body(
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
        .unwrap_err();

        assert!(
            errs.iter()
                .any(|err| matches!(err, TypeError::Unresolved { .. }))
        );
        assert!(!errs.iter().any(|err| matches!(
            err,
            TypeError::ValuesContradict {
                expectation_reason: "if branches must have the same type",
                ..
            }
        )));
    }

    #[test]
    fn if_branch_with_continue_unifies_with_concrete_branch() {
        let mut store = TypeStore::new();
        let errs = infer_fn_body(
            r#"
            f = fn() {
                let keep_going: bool = true;
                let z: int = 0;
                while keep_going {
                    z = z + if keep_going { 1:int } else { continue }
                }
            }
            "#,
            &mut store,
        )
        .unwrap_err();

        assert!(
            errs.iter()
                .any(|err| matches!(err, TypeError::Unresolved { .. }))
        );
        assert!(!errs.iter().any(|err| matches!(
            err,
            TypeError::ValuesContradict {
                expectation_reason: "if branches must have the same type",
                ..
            }
        )));
    }

    #[test]
    fn nested_if_with_never_branch_avoids_branch_mismatch_errors() {
        let mut store = TypeStore::new();
        let errs = infer_fn_body(
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
        .unwrap_err();

        assert!(
            errs.iter()
                .any(|err| matches!(err, TypeError::Unresolved { .. }))
        );
        assert!(!errs.iter().any(|err| matches!(
            err,
            TypeError::ValuesContradict {
                expectation_reason: "if branches must have the same type",
                ..
            }
        )));
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
                TypeError::Unresolved { value } => Some(program.value_loc(*value)),
                TypeError::UnresolvedPattern { pattern } => Some(program.pattern_loc(*pattern)),
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
                TypeError::FunctionOutputAnnotationMismatch { clash, .. } => Some(*clash),
                _ => None,
            })
            .unwrap_or_else(|| panic!("expected type mismatch, got errs={errs:?}"));

        let found = clash
            .found
            .unwrap_or_else(|| panic!("expected found side in clash: {clash:?}"));
        let wanted = clash
            .wanted
            .unwrap_or_else(|| panic!("expected wanted side in clash: {clash:?}"));
        assert_ne!(found.0, UNKNOWN_TYPE, "found side should not be unknown");
        assert_ne!(wanted.0, UNKNOWN_TYPE, "wanted side should not be unknown");

        let found_s = store.get_bad_type_string(&program, found);
        let wanted_s = store.get_bad_type_string(&program, wanted);
        assert!(
            (found_s == "int" && wanted_s.contains("Box"))
                || (wanted_s == "int" && found_s.contains("Box")),
            "expected int-vs-Box mismatch, got found={found_s}, wanted={wanted_s}"
        );
    }

    #[test]
    fn generic_deref_return_mismatch_uses_correct_generic_slot() {
        let src = "Box = struct[T]{ptr:*T}; Box.__deref = fn[T](b:&Box[T])->&T{&*b.ptr}; f = fn[T0,T1,T2](b:Box[Box[T2]])->T1 { *b };";
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
                TypeError::FunctionOutputAnnotationMismatch { clash, .. } => Some(*clash),
                _ => None,
            })
            .unwrap_or_else(|| panic!("expected type mismatch, got errs={errs:?}"));

        let found = clash
            .found
            .unwrap_or_else(|| panic!("expected found side in clash: {clash:?}"));
        let wanted = clash
            .wanted
            .unwrap_or_else(|| panic!("expected wanted side in clash: {clash:?}"));
        assert_ne!(found.0, UNKNOWN_TYPE, "found side should not be unknown");
        assert_ne!(wanted.0, UNKNOWN_TYPE, "wanted side should not be unknown");

        let found_s = store.get_bad_type_string(&program, found);
        let wanted_s = store.get_bad_type_string(&program, wanted);

        let (ret_side, box_side) = if found_s == "T1" {
            (found_s, wanted_s)
        } else if wanted_s == "T1" {
            (wanted_s, found_s)
        } else {
            panic!("expected one side to be T1, got found={found_s}, wanted={wanted_s}");
        };

        assert_eq!(ret_side, "T1");
        assert!(
            box_side.contains("Box") && box_side.contains("T2"),
            "expected box side to mention Box[T2], got {box_side}"
        );
        assert!(
            !box_side.contains("T0") && !box_side.contains("T1"),
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

        let TypeValue::Ptr { tgt, raw, mutable } = *store.type_value(y_ty) else {
            panic!("expected raw pointer result type")
        };
        assert!(raw);
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

        let TypeValue::Ptr { tgt, raw, mutable } = *store.type_value(y_ty) else {
            panic!("expected raw pointer result type")
        };
        assert!(raw);
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
        let x_ty = solved_types
            .pat_type(params.at(0))
            .expect("missing parameter type for x");
        let s_ty = find_typedef_type_by_name(&program, &solved_types, "S");
        assert_eq!(x_ty, s_ty);
        let call_site = find_let_stmt_value(&program, f, "y");
        let (access_site, access_ty, call_ty) =
            find_member_access_and_result_types(&program, &solved_types, f, "y");
        assert_ne!(access_site, call_site);
        assert!(solved_types.member_method_type(call_site).is_none());

        let TypeValue::Func { params, ret, .. } = store.type_value(access_ty) else {
            panic!("expected member access to be curried function")
        };
        assert_eq!(params.len(), 0);
        assert_eq!(*ret, x_ty);
        assert_eq!(call_ty, x_ty);

        let called = solved_types
            .member_method_type(access_site)
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
        let x_ty = solved_types
            .pat_type(params.at(0))
            .expect("missing parameter type for x");
        let s_ty = find_typedef_type_by_name(&program, &solved_types, "S");
        assert_eq!(x_ty, s_ty);
        let call_site = find_let_stmt_value(&program, f, "y");
        let (access_site, access_ty, call_ty) =
            find_member_access_and_result_types(&program, &solved_types, f, "y");
        assert_ne!(access_site, call_site);
        assert!(solved_types.member_method_type(call_site).is_none());

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
            .member_method_type(access_site)
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
        let TypeValue::Ptr { tgt, raw, mutable } = store.type_value(full_params[0]) else {
            panic!("expected tracked full self parameter to stay as pointer")
        };
        assert!(!*raw);
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
            solved_types.member_access_implicit_deref_count(access_site),
            Some(2),
            "plain pointer-like implicit deref should be tracked"
        );
        let chain = implicit_deref_chain_type_strings(&program, &store, &solved_types, access_site)
            .expect("expected implicit deref chain");
        assert_eq!(chain.len(), 2);
        assert!(chain[0].contains("S"));
        assert!(chain[1].contains("S"));
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
            solved_types.member_access_implicit_deref_count(access_site),
            Some(2)
        );
        let chain = implicit_deref_chain_type_strings(&program, &store, &solved_types, access_site)
            .expect("expected implicit deref chain");
        assert_eq!(chain.len(), 2);
        assert!(chain[0].contains("Box"));
        assert!(chain[1].contains("Inner"));
    }

    #[test]
    fn smart_pointer_member_access_exposes_deref_count_for_type_dump() {
        let src = "Inner=struct{x:int}; Box=struct{inner:Inner}; Box.__deref = fn(self:&Box)->&Inner { &self.inner }; f=fn(b:Box){ let y:int = b.x; };";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let _ = infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();

        let access_site = find_let_stmt_value(&program, f, "y");
        assert_eq!(
            solved_types.member_access_implicit_deref_count(access_site),
            Some(2)
        );
        let chain = implicit_deref_chain_type_strings(&program, &store, &solved_types, access_site)
            .expect("expected implicit deref chain");
        assert_eq!(chain.len(), 2);
        assert!(chain[0].contains("Box"));
        assert!(chain[1].contains("Inner"));
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
            solved_types.member_access_implicit_deref_count(access_site),
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
            solved_types.member_access_implicit_deref_count(access_site),
            Some(3)
        );
        let chain = implicit_deref_chain_type_strings(&program, &store, &solved_types, access_site)
            .expect("expected implicit deref chain");
        assert_eq!(chain.len(), 3);
        assert!(chain[0].contains("Wrap"));
        assert!(chain[1].contains("Box"));
        assert!(chain[2].contains("Inner"));
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
            solved_types.member_access_implicit_deref_count(access_site),
            Some(2)
        );
        let chain = implicit_deref_chain_type_strings(&program, &store, &solved_types, access_site)
            .expect("expected implicit deref chain");
        assert_eq!(chain.len(), 2);
        assert!(chain[0].contains("Box"));
        assert!(chain[1].contains("Inner"));
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
            solved_types.member_access_implicit_deref_count(access_site),
            Some(4)
        );

        let chain = implicit_deref_chain_type_strings(&program, &store, &solved_types, access_site)
            .expect("expected implicit deref chain");
        assert_eq!(chain.len(), 4);
        assert!(chain[0].contains("Wrap"));
        assert!(chain.iter().any(|t| t.contains("&")));
        assert!(chain.iter().any(|t| t.contains("Box")));
        assert!(chain[2].contains("Box"));
        assert!(chain[3].contains("Inner"));
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
    fn free_member_requires_mut_ref_self_and_void_output() {
        let errs = infer_global_errs("S=struct{}; S.__free = fn(self:&S)->int { 1 }; f=fn(){};");
        assert_eq!(errs.len(), 1);
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
    fn free_member_non_self_param_reports_single_error() {
        let errs = infer_global_errs("S=struct{}; S.__free = fn(x:int){ }; f=fn(){};");
        assert_eq!(errs.len(), 1);
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

    #[test]
    fn generic_struct_deref_methods_specialize_from_receiver_type() {
        let src = r#"
            Box = struct[T]{p:*T}
            Box.__free = fn[T](b:&mut Box[T]){}
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
            .type_of(body)
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
                    operand_type: Some(BadTypeId(t)),
                    ..
                } if matches!(store.type_value(*t), TypeValue::Builtin(BuiltinType::Int))
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
