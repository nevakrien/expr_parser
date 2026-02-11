//! Type inference sketch
//
// ================================================================
// CONTRACT
// ================================================================
// 1) we parse global type sinature and then internals of functions.
// 2) generics arent normalized and thus are only allowed globaly
//
// ================================================================

use crate::ErrorReporter;
use crate::identity_hasher::IdHashMap;
use crate::ir::AccessKind;
use crate::ir::StructLike;
use crate::ir::VarKind;
use crate::ir::{
    AssignOp, BinOp, Literal, NameId, PatId, Pattern, PatternSpan, TExpId, TypeExpr, UnOp, ValId,
    Value,
};
use crate::parsing::Loc;
use crate::string_intern::{
    ADD_STR, BITAND_STR, BITNOT_STR, BITOR_STR, BITXOR_STR, DIV_STR, EQ_STR, FREE_STR, GE_STR,
    GT_STR, LE_STR, LT_STR, MOD_STR, MUL_STR, NE_STR, NEG_STR, NOT_STR, SHL_STR, SHR_STR, SUB_STR,
    StrId,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeValue {
    Builtin(BuiltinType),
    Tuple(Vec<TypeId>),
    Array(TypeId, usize),
    Func {
        params: Vec<TypeId>,
        ret: TypeId,
    },
    Ptr {
        tgt: TypeId,
        raw: bool,
        mutable: bool,
    },
    WithGenerics {
        count: usize,
        ///note that the body can refer to external generics
        body: TypeId,
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
}

impl StructRep {
    fn new(names: impl Iterator<Item = NameId>, gen_count: usize) -> Self {
        Self {
            //TODO: when solving typedefs in finalize we want to set this value
            //for anonymous structs it wont exist but those are rare
            name: None,
            fields: names.map(|x| (x, UNKNOWN_TYPE)).collect(),
            gen_count,
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

    pub(crate) fn simple_struct(
        &mut self,
        name: Option<NameId>,
        fields: Vec<(NameId, TypeId)>,
    ) -> (StructId, TypeId) {
        let rep = StructRep {
            name,
            fields,
            gen_count: 0,
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
            TypeValue::Func { params, ret } => {
                let params = params
                    .iter()
                    .map(|id| self.get_type_string_nested(program, *id, gen_count))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "fn({}) -> {}",
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

            TypeValue::Array(inner, n) => {
                format!(
                    "[{};{n}]",
                    self.get_type_string_nested(program, *inner, gen_count)
                )
            }
            // TypeValue::Type => "Type".to_string(),
            TypeValue::WithGenerics { count, body } => {
                let new_count = gen_count + count;
                let pars = (gen_count..new_count)
                    .map(|i| format!("T{i}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "for<{pars}> {}",
                    self.get_type_string_nested(program, *body, new_count)
                )
            }
            TypeValue::Generic(g) => format!("T{}", g.0),

            //TODO cover cases where we do know the name
            TypeValue::Struct { id, generics } => {
                self.format_struct_display(program, *id, generics)
            }
        }
    }

    fn format_struct_display(
        &self,
        program: &Program,
        sid: StructId,
        generics: &[TypeId],
    ) -> String {
        let base = match self.struct_value(sid).name {
            Some(name) => program.name_string(name).to_string(),
            None => "UnamedStruct".to_string(),
        };
        let base = format!("{}{}", base, subscript_id(sid.0));
        if !generics.is_empty() {
            //TODO add to base all the generics if there are any base<>
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

///runs the typechecker and reports all errors
///the rhs value is the total number of functions checked
///the lhs value is either the result or the number of errors found
pub fn run_typechecker(
    program: &Program,
    reporter: &mut ErrorReporter,
) -> Result<(Result<(TypeStore, SolvedTypes), usize>, usize), Box<dyn std::error::Error>> {
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

    for (_, def) in program.definitions.iter() {
        let Defined::Func(v) = def else {
            continue;
        };
        function_checked += 1;
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
    let mut ctx = InferState::new(ans, store, program);

    for (n, def) in program.definitions.iter() {
        let Defined::Type(texp) = def else {
            continue;
        };

        let t = match ctx.program.type_expr(*texp) {
            TypeExpr::Struct(def) => compile_struct_type::<true>(&mut ctx, *texp, def),
            _ => compile_type_expr(&mut ctx, *texp),
        };
        if let Some(previous) = ctx.local_types.insert(*n, t) {
            if let Err(clash) = ctx.unify(previous, t) {
                ctx.errors.push(TypeError::TypeClashBeforeMentioned {
                    name: *n,
                    expr: *texp,
                    clash,
                });
            }
        }
        if let ResolveKind::Solved(ty) = ctx.cluster[t].state {
            ctx.ans.typedef_types.insert(*texp, ty);
        } else {
            ctx.typedef_cluster.push((*texp, t));
        }
    }

    main_solver(&mut ctx);
    if !ctx.errors.is_empty() {
        return Err(ctx.errors);
    }

    for (struct_name, methods) in program.member_methods.iter() {
        for (method_name, m) in methods.iter() {
            //each function must solve by itself.
            //since there isnt a body its fine to solve in order
            //note that namespace on generics gurntees this works for the most outer scope
            match ctx.program.value(*m) {
                Value::Func {
                    generics,
                    params,
                    output_type,
                    body: _,
                } => {
                    ctx.clear_local_state();
                    let _ =
                        gather_func_signature::<true>(&mut ctx, *m, generics, params, output_type);
                    main_solver(&mut ctx);
                    check_special_member_method_signature(&mut ctx, *m, *struct_name, *method_name);
                }
                _ => {}
            };
        }
    }

    for (_n, def) in program.definitions.iter() {
        let Defined::Func(v) = def else {
            continue;
        };

        //each function must solve by itself.
        //since there isnt a body its fine to solve in order
        //note that namespace on generics gurntees this works for the most outer scope
        match ctx.program.value(*v) {
            Value::Func {
                generics,
                params,
                output_type,
                body: _,
            } => {
                ctx.clear_local_state();
                let _ = gather_func_signature::<true>(&mut ctx, *v, generics, params, output_type);
                main_solver(&mut ctx);
            }
            _ => {}
        };
    }

    if ctx.errors.is_empty() {
        Ok(ctx.ans)
    } else {
        Err(ctx.errors)
    }
}

pub fn infer_value_internals<'a>(
    program: &'a Program,
    store: &'a mut TypeStore,
    ans: &'a mut SolvedTypes,
    value: ValId,
) -> Result<&'a mut SolvedTypes, Vec<TypeError>> {
    let known = ans.type_of(value);
    let mut ctx = InferState::new(ans, store, program);

    match ctx.program.value(value) {
        Value::Func {
            generics,
            params,
            output_type,
            body,
        } => {
            gather_func_constraints::<true>(&mut ctx, value, generics, params, output_type, body)
            //this case we have a fully known type so no unify.
            //further a unify here is wrong
            //this is because we solved this as Func<> but its actually a WithGenrics
            //this is fine for our local work as we pretend that generics are concrete
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
    if let Some(known) = known {
        debug_assert_eq!(known, ctx.ans.type_of(value).unwrap())
    }

    if ctx.errors.is_empty() {
        Ok(ctx.ans)
    } else {
        Err(ctx.errors)
    }
}

///this is just for tests we PURPOSFULLY ignore the global sig resolution
fn _infer_value_hacky<'a>(
    program: &'a Program,
    store: &'a mut TypeStore,
    ans: &'a mut SolvedTypes,

    value: ValId,
) -> Result<&'a mut SolvedTypes, Vec<TypeError>> {
    let mut ctx = InferState::new(ans, store, program);

    match ctx.program.value(value) {
        Value::Func {
            generics,
            params,
            output_type,
            body,
        } => {
            let _ = gather_func_constraints::<true>(
                &mut ctx,
                value,
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
    if ctx.errors.is_empty() {
        Ok(ctx.ans)
    } else {
        Err(ctx.errors)
    }
}

fn main_solver(ctx: &mut InferState) {
    loop {
        let mut progress = false;
        progress |= resolve_operator_types(ctx);
        progress |= resolve_deferred_types(ctx);
        progress |= resolve_pending_specializations(ctx);
        if !progress {
            break;
        }
    }

    if !ctx.errors.is_empty() {
        return;
    }

    finalize(ctx);
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

struct InferState<'a> {
    store: &'a mut TypeStore,
    program: &'a Program,

    //ir -> cid
    val_cluster: Vec<(ValId, CId)>,
    pat_cluster: Vec<(PatId, CId)>,
    typedef_cluster: Vec<(TExpId, CId)>,
    local_types: IdHashMap<NameId, CId>,
    names: IdHashMap<NameId, CId>,

    // unify-find
    parent: ClusterVec<CId>,
    cluster: ClusterVec<Cluster>,

    //requirments
    bin_op_sites: Vec<BinOpSite>,
    un_op_sites: Vec<UnOpSite>,
    func_defs: Vec<FuncInfer>,
    struct_defs: Vec<StructDef>,
    struct_infers: Vec<StructInfer>,
    generic_func_values: Vec<(ValId, usize)>,
    pending_specializations: Vec<PendingSpecialization>,
    member_method_type_sites: Vec<PendingMemberMethodType>,

    //result
    errors: Vec<TypeError>,
    ans: &'a mut SolvedTypes,
}

#[derive(Debug)]
struct Cluster {
    state: ResolveKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FuncInferId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StructInferId(usize);

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

    Ptr {
        tgt: CId,
        raw: Option<bool>,
        mutable: Option<bool>,
    },
}

#[derive(Debug)]
struct FuncInfer {
    #[allow(dead_code)]
    loc: ValId,
    inputs: Vec<CId>,
    output: CId,
}

#[derive(Debug)]
struct StructInfer {
    sid: StructId,
    generics: Vec<CId>,
}

#[derive(Debug)]
struct StructDef {
    #[allow(dead_code)]
    loc: TExpId,
    fields: Vec<(NameId, CId)>,
    output: CId,
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
struct PendingMemberMethodType {
    site: ValId,
    member: StrId,
    full_method: CId,
    receiver: CId,
    receiver_value: ValId,
}

fn new_cluster(parent: &mut ClusterVec<CId>, cluster: &mut ClusterVec<Cluster>) -> CId {
    let id = CId(parent.len());
    parent.0.push(id);
    cluster.0.push(Cluster {
        state: ResolveKind::Nothing,
    });
    id
}

fn new_solved(parent: &mut ClusterVec<CId>, cluster: &mut ClusterVec<Cluster>, t: TypeId) -> CId {
    //duplicated in Handeler
    let id = new_cluster(parent, cluster);
    cluster[id].state = ResolveKind::Solved(t);
    id
}

fn new_int_like(parent: &mut ClusterVec<CId>, cluster: &mut ClusterVec<Cluster>) -> CId {
    let id = new_cluster(parent, cluster);
    cluster[id].state = ResolveKind::IntLike;
    id
}

fn new_float_like(parent: &mut ClusterVec<CId>, cluster: &mut ClusterVec<Cluster>) -> CId {
    let id = new_cluster(parent, cluster);
    cluster[id].state = ResolveKind::FloatLike;
    id
}

fn new_func(
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    call: FuncInfer,
) -> CId {
    let call_id = FuncInferId(func_defs.len());
    func_defs.push(call);
    let id = new_cluster(parent, cluster);
    cluster[id].state = ResolveKind::Func(call_id);
    id
}

fn new_struct_instance(
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    struct_infers: &mut Vec<StructInfer>,
    sid: StructId,
    generics: Vec<CId>,
) -> CId {
    let call_id = StructInferId(struct_infers.len());
    struct_infers.push(StructInfer { sid, generics });
    let id = new_cluster(parent, cluster);
    cluster[id].state = ResolveKind::Struct(call_id);
    id
}

fn bind_val(val_cluster: &mut Vec<(ValId, CId)>, v: ValId, c: CId) {
    val_cluster.push((v, c));
}

impl<'a> InferState<'a> {
    fn new(ans: &'a mut SolvedTypes, store: &'a mut TypeStore, program: &'a Program) -> Self {
        Self {
            store,
            program,
            val_cluster: Vec::default(),
            pat_cluster: Vec::default(),
            typedef_cluster: Vec::default(),
            local_types: IdHashMap::default(),
            names: IdHashMap::default(),
            parent: ClusterVec::new(),
            cluster: ClusterVec::new(),
            bin_op_sites: Vec::new(),
            un_op_sites: Vec::new(),
            func_defs: Vec::new(),
            struct_defs: Vec::new(),
            struct_infers: Vec::new(),
            generic_func_values: Vec::new(),
            pending_specializations: Vec::new(),
            member_method_type_sites: Vec::new(),
            errors: Vec::new(),
            ans,
        }
    }

    fn clear_local_state(&mut self) {
        let InferState {
            store: _,
            program: _,
            val_cluster,
            pat_cluster,
            typedef_cluster,
            local_types,
            names,
            parent,
            cluster,
            bin_op_sites,
            un_op_sites,
            func_defs,
            struct_defs,
            struct_infers,
            generic_func_values,
            pending_specializations,
            member_method_type_sites,
            errors: _,
            ans: _,
        } = self;

        val_cluster.clear();
        pat_cluster.clear();
        typedef_cluster.clear();
        local_types.clear();
        names.clear();

        *parent = ClusterVec::new();
        *cluster = ClusterVec::new();

        bin_op_sites.clear();
        un_op_sites.clear();
        func_defs.clear();
        struct_defs.clear();
        struct_infers.clear();
        generic_func_values.clear();
        pending_specializations.clear();
        member_method_type_sites.clear();
    }

    fn new_cluster(&mut self) -> CId {
        new_cluster(&mut self.parent, &mut self.cluster)
    }

    fn new_solved(&mut self, t: TypeId) -> CId {
        new_solved(&mut self.parent, &mut self.cluster, t)
    }

    fn new_int_like(&mut self, _v: ValId) -> CId {
        new_int_like(&mut self.parent, &mut self.cluster)
    }

    fn new_float_like(&mut self, _v: ValId) -> CId {
        new_float_like(&mut self.parent, &mut self.cluster)
    }

    fn new_func(&mut self, call: FuncInfer) -> CId {
        new_func(
            &mut self.parent,
            &mut self.cluster,
            &mut self.func_defs,
            call,
        )
    }

    fn new_struct_instance(&mut self, sid: StructId, generics: Vec<CId>) -> CId {
        new_struct_instance(
            &mut self.parent,
            &mut self.cluster,
            &mut self.struct_infers,
            sid,
            generics,
        )
    }

    fn bind_val(&mut self, v: ValId, c: CId) {
        bind_val(&mut self.val_cluster, v, c);
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
            &mut self.func_defs,
            &mut self.struct_infers,
            a,
            b,
        )
    }

    fn force_type(&mut self, a: CId, t: TypeId) -> Result<(), TypeClash> {
        force_type(
            self.store,
            &mut self.parent,
            &mut self.cluster,
            &mut self.func_defs,
            &mut self.struct_infers,
            a,
            t,
        )
    }
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

///tries to combine 2 clusters
///on fail produces a type_clash and keeps the 2 seprate
///infrence can continye with the 2 types seperated for the purpose of gathering more errors (obviously not Unresolved style errors)
fn unify_clusters(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    found: CId,
    wanted: CId,
) -> Result<CId, TypeClash> {
    unify_clusters_inlined(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        found,
        wanted,
    )
}
#[inline(always)]
fn unify_clusters_inlined(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    found: CId,
    wanted: CId,
) -> Result<CId, TypeClash> {
    let rf = find_root(parent, found);
    let rw = find_root(parent, wanted);
    if rf == rw {
        return Ok(rw);
    }

    if matches!(cluster[rw].state, ResolveKind::Never) {
        return Ok(rf);
    }
    if matches!(cluster[rf].state, ResolveKind::Never) {
        return Ok(rw);
    }

    // Try found <- wanted
    if _try_absorb(store, parent, cluster, func_defs, struct_infers, rw, rf)? {
        if rf != parent[rf] {
            todo!()
        }

        parent[rf] = rw;
        return Ok(rw);
    }

    // Otherwise try wanted <- found
    if _try_absorb(store, parent, cluster, func_defs, struct_infers, rf, rw)
        .map_err(TypeClash::swap)?
    {
        if rw != parent[rw] {
            todo!()
        }

        parent[rw] = rf;
        return Ok(rf);
    }

    // Neither direction worked → real contradiction
    Err(TypeClash {
        found: extract_bad_type(store, parent, cluster, func_defs, struct_infers, found),
        wanted: extract_bad_type(store, parent, cluster, func_defs, struct_infers, wanted),
    })
}

#[inline(always)]
fn _try_absorb(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
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
        (Nothing, IntLike) | (Nothing, FloatLike) => {
            cluster[dst].state = cluster[src].state;
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
            if !store.is_int_like(t) {
                return Err(TypeClash {
                    found: Some(BadTypeId(UNKNOWN_INT_SIZE)),
                    wanted: Some(BadTypeId(t)),
                });
            }
            Ok(true)
        }

        (Solved(t), FloatLike) => {
            if !store.is_float_like(t) {
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
            let (dst_len, src_len) = {
                let dst_call = &func_defs[dst_call.0];
                let src_call = &func_defs[src_call.0];
                (dst_call.inputs.len(), src_call.inputs.len())
            };
            if dst_len != src_len {
                return Err(func_call_clash(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    dst_call,
                    src_call,
                ));
            }

            for i in 0..dst_len {
                let (a, b) = {
                    let dst_call = &func_defs[dst_call.0];
                    let src_call = &func_defs[src_call.0];
                    (dst_call.inputs[i], src_call.inputs[i])
                };
                if unify_clusters(store, parent, cluster, func_defs, struct_infers, a, b).is_err() {
                    return Err(func_call_clash(
                        store,
                        parent,
                        cluster,
                        func_defs,
                        struct_infers,
                        dst_call,
                        src_call,
                    ));
                }
            }
            let (dst_out, src_out) = {
                let dst_call = &func_defs[dst_call.0];
                let src_call = &func_defs[src_call.0];
                (dst_call.output, src_call.output)
            };
            if unify_clusters(
                store,
                parent,
                cluster,
                func_defs,
                struct_infers,
                dst_out,
                src_out,
            )
            .is_err()
            {
                return Err(func_call_clash(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    dst_call,
                    src_call,
                ));
            }

            if let Some(t) = try_resolve_func_type(store, parent, cluster, func_defs, dst_call) {
                cluster[dst].state = Solved(t);
            }
            Ok(true)
        }

        (Solved(t), Func(call)) => {
            unify_func_with_type(store, parent, cluster, func_defs, struct_infers, call, t)?;
            Ok(true)
        }

        (Struct(dst_call), Struct(src_call)) => {
            let (dst_sid, src_sid) = {
                let dst_call = &struct_infers[dst_call.0];
                let src_call = &struct_infers[src_call.0];
                (dst_call.sid, src_call.sid)
            };
            if dst_sid != src_sid {
                return Err(struct_call_clash(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    dst_call,
                    src_call,
                ));
            }

            let (dst_len, src_len) = {
                let dst_call = &struct_infers[dst_call.0];
                let src_call = &struct_infers[src_call.0];
                (dst_call.generics.len(), src_call.generics.len())
            };
            if dst_len != src_len {
                return Err(struct_call_clash(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    dst_call,
                    src_call,
                ));
            }

            for i in 0..dst_len {
                let (a, b) = {
                    let dst_call = &struct_infers[dst_call.0];
                    let src_call = &struct_infers[src_call.0];
                    (dst_call.generics[i], src_call.generics[i])
                };
                if unify_clusters(store, parent, cluster, func_defs, struct_infers, a, b).is_err() {
                    return Err(struct_call_clash(
                        store,
                        parent,
                        cluster,
                        func_defs,
                        struct_infers,
                        dst_call,
                        src_call,
                    ));
                }
            }

            if let Some(t) =
                try_resolve_struct_type(store, parent, cluster, struct_infers, dst_call)
            {
                cluster[dst].state = Solved(t);
            }
            Ok(true)
        }

        (Solved(t), Struct(call)) => {
            unify_struct_with_type(store, parent, cluster, func_defs, struct_infers, call, t)?;
            Ok(true)
        }

        (Solved(t), Ptr { tgt, raw, mutable }) => {
            unify_ptr_with_type(
                store,
                parent,
                cluster,
                func_defs,
                struct_infers,
                tgt,
                raw,
                mutable,
                t,
            )?;
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
            let raw = merge_ptr_flag(dst_raw, src_raw).ok_or_else(|| TypeClash {
                found: Some(BadTypeId(make_ptr_mock(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    src_tgt,
                    src_raw,
                    src_mut,
                ))),
                wanted: Some(BadTypeId(make_ptr_mock(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    dst_tgt,
                    dst_raw,
                    dst_mut,
                ))),
            })?;
            let mutable = merge_ptr_flag(dst_mut, src_mut).ok_or_else(|| TypeClash {
                found: Some(BadTypeId(make_ptr_mock(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    src_tgt,
                    src_raw,
                    src_mut,
                ))),
                wanted: Some(BadTypeId(make_ptr_mock(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    dst_tgt,
                    dst_raw,
                    dst_mut,
                ))),
            })?;

            let tgt = unify_clusters(
                store,
                parent,
                cluster,
                func_defs,
                struct_infers,
                dst_tgt,
                src_tgt,
            )?;

            cluster[dst].state = Ptr { tgt, raw, mutable };
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
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
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
                return Err(TypeClash {
                    found: Some(BadTypeId(UNKNOWN_INT_SIZE)),
                    wanted: Some(BadTypeId(ty)),
                });
            }
            cluster[root].state = ResolveKind::Solved(ty);
            Ok(())
        }
        ResolveKind::FloatLike => {
            if !store.is_float_like(ty) {
                return Err(TypeClash {
                    found: Some(BadTypeId(UNKNOWN_FLOAT_SIZE)),
                    wanted: Some(BadTypeId(ty)),
                });
            }
            cluster[root].state = ResolveKind::Solved(ty);
            Ok(())
        }
        ResolveKind::Func(call) => {
            unify_func_with_type(store, parent, cluster, func_defs, struct_infers, call, ty)?;
            cluster[root].state = ResolveKind::Solved(ty);
            Ok(())
        }
        ResolveKind::Struct(call) => {
            unify_struct_with_type(store, parent, cluster, func_defs, struct_infers, call, ty)?;
            cluster[root].state = ResolveKind::Solved(ty);
            Ok(())
        }
        ResolveKind::Ptr { tgt, raw, mutable } => {
            unify_ptr_with_type(
                store,
                parent,
                cluster,
                func_defs,
                struct_infers,
                tgt,
                raw,
                mutable,
                ty,
            )?;
            cluster[root].state = ResolveKind::Solved(ty);
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

fn unify_ptr_with_type(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    tgt: CId,
    raw: Option<bool>,
    mutable: Option<bool>,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let TypeValue::Ptr {
        tgt: ty_tgt,
        raw: ty_raw,
        mutable: ty_mut,
    } = *store.type_value(ty)
    else {
        return Err(TypeClash {
            found: Some(BadTypeId(ty)),
            wanted: Some(BadTypeId(make_ptr_mock(
                store,
                parent,
                cluster,
                func_defs,
                struct_infers,
                tgt,
                raw,
                mutable,
            ))),
        });
    };

    if matches!(raw, Some(x) if x != ty_raw) || matches!(mutable, Some(x) if x != ty_mut) {
        return Err(TypeClash {
            found: Some(BadTypeId(ty)),
            wanted: Some(BadTypeId(make_ptr_mock(
                store,
                parent,
                cluster,
                func_defs,
                struct_infers,
                tgt,
                raw,
                mutable,
            ))),
        });
    }

    force_type(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        tgt,
        ty_tgt,
    )
}

fn unify_func_with_type(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    call: FuncInferId,
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

    let input_len = func_defs[call.0].inputs.len();
    if params.len() != input_len {
        return Err(TypeClash {
            found: Some(BadTypeId(ty)),
            wanted: None,
        });
    }

    for i in 0..input_len {
        let input = func_defs[call.0].inputs[i];

        //TODO (maybe): we constantly take the params again from the spot because borrow checker
        //              technically the Vec params points to never reallocs
        //              so theortically its possible to keep borowing this
        let param_ty = match store.type_value(ty) {
            TypeValue::Func { params, ret: _ } => params[i],
            _ => unreachable!(),
        };
        force_type(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            input,
            param_ty,
        )?;
    }

    let output = func_defs[call.0].output;
    force_type(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        output,
        ret,
    )?;

    Ok(())
}

fn unify_struct_with_type(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    call: StructInferId,
    ty: TypeId,
) -> Result<(), TypeClash> {
    let (sid, glen) = match store.type_value(ty) {
        TypeValue::Struct { id, generics } => (*id, generics.len()),
        _ => {
            return Err(TypeClash {
                found: Some(BadTypeId(ty)),
                wanted: None,
            });
        }
    };

    let call_sid = struct_infers[call.0].sid;
    if call_sid != sid || struct_infers[call.0].generics.len() != glen {
        return Err(TypeClash {
            found: Some(BadTypeId(ty)),
            wanted: None,
        });
    }

    for i in 0..glen {
        let input = struct_infers[call.0].generics[i];
        let TypeValue::Struct { id: _, generics } = store.type_value(ty) else {
            unreachable!();
        };
        let t = generics[i];
        force_type(store, parent, cluster, func_defs, struct_infers, input, t)?;
    }

    Ok(())
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
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    func_defs: &Vec<FuncInfer>,
    struct_infers: &Vec<StructInfer>,
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
        ResolveKind::Func(call) => make_func_mock_inner(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            call,
            visiting,
        ),
        ResolveKind::Struct(call) => make_struct_mock_inner(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            call,
            visiting,
        ),
        ResolveKind::Ptr { tgt, raw, mutable } => make_ptr_mock_inner(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            tgt,
            raw,
            mutable,
            visiting,
        ),
        ResolveKind::Nothing | ResolveKind::Never => UNKNOWN_TYPE,
    };

    visiting.remove(&root);
    ty
}

fn make_func_mock_inner(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    func_defs: &Vec<FuncInfer>,
    struct_infers: &Vec<StructInfer>,
    call: FuncInferId,
    visiting: &mut std::collections::HashSet<CId>,
) -> TypeId {
    let site = &func_defs[call.0];
    let params = site
        .inputs
        .iter()
        .map(|&input| {
            mock_type_from_cluster(
                store,
                parent,
                cluster,
                func_defs,
                struct_infers,
                input,
                visiting,
            )
        })
        .collect::<Vec<_>>();
    let ret = mock_type_from_cluster(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        site.output,
        visiting,
    );

    store.intern(TypeValue::Func { params, ret })
}

fn make_func_mock(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    func_defs: &Vec<FuncInfer>,
    struct_infers: &Vec<StructInfer>,
    call: FuncInferId,
) -> TypeId {
    let mut visiting = std::collections::HashSet::new();
    make_func_mock_inner(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        call,
        &mut visiting,
    )
}

fn make_struct_mock_inner(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    func_defs: &Vec<FuncInfer>,
    struct_infers: &Vec<StructInfer>,
    call: StructInferId,
    visiting: &mut std::collections::HashSet<CId>,
) -> TypeId {
    let site = &struct_infers[call.0];
    let generics = site
        .generics
        .iter()
        .map(|&input| {
            mock_type_from_cluster(
                store,
                parent,
                cluster,
                func_defs,
                struct_infers,
                input,
                visiting,
            )
        })
        .collect::<Vec<_>>();
    store.intern(TypeValue::Struct {
        id: site.sid,
        generics,
    })
}

fn make_struct_mock(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    func_defs: &Vec<FuncInfer>,
    struct_infers: &Vec<StructInfer>,
    call: StructInferId,
) -> TypeId {
    let mut visiting = std::collections::HashSet::new();
    make_struct_mock_inner(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        call,
        &mut visiting,
    )
}

fn make_ptr_mock_inner(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    func_defs: &Vec<FuncInfer>,
    struct_infers: &Vec<StructInfer>,
    tgt: CId,
    raw: Option<bool>,
    mutable: Option<bool>,
    visiting: &mut std::collections::HashSet<CId>,
) -> TypeId {
    let tgt = mock_type_from_cluster(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        tgt,
        visiting,
    );
    store.intern(TypeValue::Ptr {
        tgt,
        raw: raw.unwrap_or(false),
        mutable: mutable.unwrap_or(false),
    })
}

fn make_ptr_mock(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    func_defs: &Vec<FuncInfer>,
    struct_infers: &Vec<StructInfer>,
    tgt: CId,
    raw: Option<bool>,
    mutable: Option<bool>,
) -> TypeId {
    let mut visiting = std::collections::HashSet::new();
    make_ptr_mock_inner(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        tgt,
        raw,
        mutable,
        &mut visiting,
    )
}

fn func_call_clash(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    func_defs: &Vec<FuncInfer>,
    struct_infers: &Vec<StructInfer>,
    dst_call: FuncInferId,
    src_call: FuncInferId,
) -> TypeClash {
    TypeClash {
        found: Some(BadTypeId(make_func_mock(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            src_call,
        ))),
        wanted: Some(BadTypeId(make_func_mock(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            dst_call,
        ))),
    }
}

fn struct_call_clash(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    func_defs: &Vec<FuncInfer>,
    struct_infers: &Vec<StructInfer>,
    dst_call: StructInferId,
    src_call: StructInferId,
) -> TypeClash {
    TypeClash {
        found: Some(BadTypeId(make_struct_mock(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            src_call,
        ))),
        wanted: Some(BadTypeId(make_struct_mock(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            dst_call,
        ))),
    }
}

fn extract_bad_type(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    func_defs: &Vec<FuncInfer>,
    struct_infers: &Vec<StructInfer>,
    cid: CId,
) -> Option<BadTypeId> {
    let root = find_root(parent, cid);
    match cluster[root].state {
        ResolveKind::Solved(t) => Some(BadTypeId(t)),
        ResolveKind::Nothing | ResolveKind::Never => None,
        ResolveKind::Func(call) => Some(BadTypeId(make_func_mock(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            call,
        ))),
        ResolveKind::Struct(call) => Some(BadTypeId(make_struct_mock(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            call,
        ))),
        ResolveKind::Ptr { tgt, raw, mutable } => Some(BadTypeId(make_ptr_mock(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            tgt,
            raw,
            mutable,
        ))),

        ResolveKind::IntLike => Some(BadTypeId(UNKNOWN_INT_SIZE)),
        ResolveKind::FloatLike => Some(BadTypeId(UNKNOWN_FLOAT_SIZE)),
    }
}

fn specialize_type(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    ty: TypeId,
    generics: &[CId],
    loc: ValId,
) -> CId {
    match store.type_value(ty) {
        TypeValue::Generic(id) => generics.get(id.0).copied().unwrap(),
        TypeValue::Func { params, ret } => {
            let ret = *ret;
            let plen = params.len();
            let inputs = (0..plen)
                .map(|i| {
                    let TypeValue::Func { params, .. } = store.type_value(ty) else {
                        unreachable!()
                    };
                    let t = params[i];
                    specialize_type(
                        store,
                        parent,
                        cluster,
                        func_defs,
                        struct_infers,
                        t,
                        generics,
                        loc,
                    )
                })
                .collect::<Vec<_>>();
            let output = specialize_type(
                store,
                parent,
                cluster,
                func_defs,
                struct_infers,
                ret,
                generics,
                loc,
            );
            new_func(
                parent,
                cluster,
                func_defs,
                FuncInfer {
                    inputs,
                    output,
                    loc,
                },
            )
        }
        TypeValue::Struct {
            id,
            generics: parts,
        } => {
            if parts.is_empty() {
                return new_solved(parent, cluster, ty);
            }
            let id = *id;
            let glen = parts.len();

            let resolved = (0..glen)
                .map(|i| {
                    let TypeValue::Struct {
                        id: _,
                        generics: parts,
                    } = store.type_value(ty)
                    else {
                        unreachable!();
                    };
                    let t = parts[i];
                    specialize_type(
                        store,
                        parent,
                        cluster,
                        func_defs,
                        struct_infers,
                        t,
                        generics,
                        loc,
                    )
                })
                .collect::<Vec<_>>();
            new_struct_instance(parent, cluster, struct_infers, id, resolved)
        }
        _ => new_solved(parent, cluster, ty),
    }
}

// ===================================
// Constraint gathering (alias where possible)
// ===================================
fn solved_type_to_specialized_local(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    t: TypeId,
    loc: ValId,
) -> CId {
    if let TypeValue::WithGenerics { count, body } = *store.type_value(t) {
        let gens: Vec<_> = (0..count).map(|_| new_cluster(parent, cluster)).collect();
        return specialize_type(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            body,
            &gens,
            loc,
        );
    }

    new_solved(parent, cluster, t)
}

fn global_to_specialized_local(ctx: &mut InferState, def_val: &ValId, v: ValId) -> CId {
    let Some(t) = ctx.ans.type_of(*def_val) else {
        let loc = ctx.program.value_loc(v);
        let c = ctx.new_cluster();
        ctx.errors.push(TypeError::Simple {
            loc,
            message: "global value has no inferred type",
        });
        ctx.bind_val(v, c);
        return c;
    };

    //TODO this check is actually CURRENTLY non exustive
    //we wana make sure that we add a good way to run this
    //would be done as some normlization function somewhere
    //structs especially are weird with this
    let ans = solved_type_to_specialized_local(
        ctx.store,
        &mut ctx.parent,
        &mut ctx.cluster,
        &mut ctx.func_defs,
        &mut ctx.struct_infers,
        t,
        v,
    );
    bind_val(&mut ctx.val_cluster, v, ans);
    ans
}

fn resolve_member_method_access(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    val_cluster: &mut Vec<(ValId, CId)>,
    member_method_type_sites: &mut Vec<PendingMemberMethodType>,
    errors: &mut Vec<TypeError>,
    program: &Program,
    ans: &SolvedTypes,
    access_site: ValId,
    base_value: ValId,
    base_cluster: CId,
    struct_name: NameId,
    member_name: StrId,
) -> CId {
    let Some(method) = program
        .member_methods
        .get(&struct_name)
        .and_then(|methods| methods.get(&member_name))
        .copied()
    else {
        let unresolved = new_cluster(parent, cluster);
        bind_val(val_cluster, access_site, unresolved);
        errors.push(TypeError::UnknownField {
            field: member_name,
            site: access_site,
        });
        return unresolved;
    };

    let Some(method_ty) = ans.type_of(method) else {
        unreachable!(
            "global member method signatures must be solved before body inference; missing type for member access"
        );
    };

    let method_local = solved_type_to_specialized_local(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        method_ty,
        access_site,
    );

    let Some(self_style) = get_member_self_style(store, method_ty, struct_name) else {
        member_method_type_sites.push(PendingMemberMethodType {
            site: access_site,
            member: member_name,
            full_method: method_local,
            receiver: base_cluster,
            receiver_value: base_value,
        });
        bind_val(val_cluster, access_site, method_local);
        return method_local;
    };

    let Some((params, ret)) =
        function_parts_from_cluster(store, parent, cluster, &*func_defs, method_local)
    else {
        unreachable!("specialized member access method must resolve to a function shape");
    };

    let curried_method = make_member_closure(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
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
            bind_val(val_cluster, access_site, curried);
            curried
        }
        Err(clash) => {
            let unresolved = new_cluster(parent, cluster);
            member_method_type_sites.push(PendingMemberMethodType {
                site: access_site,
                member: member_name,
                full_method: method_local,
                receiver: base_cluster,
                receiver_value: base_value,
            });
            bind_val(val_cluster, access_site, unresolved);
            errors.push(TypeError::ValuesContradict {
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

fn gather_constraints(ctx: &mut InferState, v: ValId, current_output: Option<CId>) -> CId {
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
            if let Some(base) = ctx.names.get_mut(&n) {
                //names might refer to something that us generic in the local scope...
                //so this here is actually wrong for when users define local genric stuff
                let c = find_root(&mut ctx.parent, *base);
                *base = c;
                ctx.bind_val(v, c);
                return c;
            }

            let Some(def) = ctx.program.definitions.get(&n) else {
                unreachable!("name used before binding");
            };

            match def {
                Defined::Type(_t) => {
                    let ans =
                        new_solved(&mut ctx.parent, &mut ctx.cluster, BuiltinType::Type.into());
                    bind_val(&mut ctx.val_cluster, v, ans);
                    ans
                }
                Defined::Func(def_val) => global_to_specialized_local(ctx, def_val, v),
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
            let t = compile_type_expr(ctx, ty);
            ctx.typedef_cluster.push((ty, t));
            if let Some(n) = n {
                ctx.local_types.insert(n, t);
            }
            p
        }

        Value::AddrOf(base, kind) => {
            let tgt = gather_constraints(ctx, base, current_output);
            let mutable = kind.map(|x| matches!(x, VarKind::Mut));
            let ans = ctx.new_cluster();
            ctx.cluster[ans].state = ResolveKind::Ptr {
                tgt,
                raw: None,
                mutable,
            };
            ctx.bind_val(v, ans);
            ans
        }

        Value::Deref(base) => {
            let src = gather_constraints(ctx, base, current_output);
            let tgt = match ctx.cluster[src].state {
                ResolveKind::Ptr { tgt, .. } => tgt,
                ResolveKind::Nothing => {
                    let tgt = ctx.new_cluster();
                    ctx.cluster[src].state = ResolveKind::Ptr {
                        tgt,
                        mutable: None,
                        raw: None,
                    };
                    tgt
                }
                ResolveKind::Struct(_rid) => todo!("might have __smart_pointer"),
                ResolveKind::Solved(t) => match ctx.store.type_value(t) {
                    TypeValue::Ptr { tgt, .. } => ctx.new_solved(*tgt),
                    TypeValue::Struct { id: _, .. } => todo!("might have __smart_pointer"),
                    _ => todo!("report error"),
                },
                _ => todo!(),
            };
            ctx.bind_val(v, tgt);
            tgt
        }

        Value::Assign {
            op: AssignOp::Nothing(value),
            target,
        } => {
            let lhs = gather_constraints(ctx, target, current_output);
            ctx.bind_val(v, lhs);

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

            lhs
        }
        Value::Assign { .. } => todo!(),

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
                let (store, parent, cluster, func_defs, struct_infers, bin_op_sites, errors) = (
                    &mut ctx.store,
                    &mut ctx.parent,
                    &mut ctx.cluster,
                    &mut ctx.func_defs,
                    &mut ctx.struct_infers,
                    &mut ctx.bin_op_sites,
                    &mut ctx.errors,
                );
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
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    &mut ctx.member_method_type_sites,
                    errors,
                    &mut site,
                    ctx.program,
                    &*ctx.ans,
                );
                if outcome.retain {
                    bin_op_sites.push(site);
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
                let (store, parent, cluster, func_defs, struct_infers, un_op_sites, errors) = (
                    &mut ctx.store,
                    &mut ctx.parent,
                    &mut ctx.cluster,
                    &mut ctx.func_defs,
                    &mut ctx.struct_infers,
                    &mut ctx.un_op_sites,
                    &mut ctx.errors,
                );
                let mut site = UnOpSite {
                    loc: v,
                    op,
                    val: value,
                    input,
                    output,
                };
                let outcome = resolve_unary_operator_site(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    &mut ctx.member_method_type_sites,
                    errors,
                    &mut site,
                    ctx.program,
                    &*ctx.ans,
                );
                if outcome.retain {
                    un_op_sites.push(site);
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
            generics,
            params,
            output_type,
            body,
        } => gather_func_constraints::<false>(ctx, v, generics, params, output_type, body),
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
                    loc: v,
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
            let Some(def) = ctx.program.definitions.get(&base_name) else {
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

            let Some(base_type) = ctx.ans.typedef_types.get(texp) else {
                ctx.push_error(TypeError::UnresolvedTypeExpr { expr: *texp });
                for arg in cons.args.ids() {
                    gather_constraints(ctx, arg, current_output);
                }
                let ans = ctx.new_cluster();
                ctx.bind_val(v, ans);
                return ans;
            };
            let base_type = *base_type;

            let sid = match ctx.store.type_value(base_type) {
                TypeValue::Struct { id, generics: _ } => *id,
                // TypeValue::Specialized { base, .. } => {
                //     match ctx.store.type_value(*base) {
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

            // let fields = &ctx.store.struct_value(sid).fields;
            let expected = ctx.store.struct_value(sid).fields.len();
            let provided = cons.args.len();
            if provided > expected {
                ctx.push_error(TypeError::TooManyArguments {
                    site: v,
                    expected,
                    found: provided,
                });
            }

            let TypeValue::Struct { id: _, generics } = ctx.store.type_value(base_type) else {
                unreachable!("verified above");
            };

            let glen = generics.len();

            let mut generic_clusters = Vec::new();
            let mut field_type_clusters = None;
            if glen != 0 {
                generic_clusters = (0..glen).map(|_| ctx.new_cluster()).collect();
                let flen = ctx.store.struct_value(sid).fields.len();

                field_type_clusters = Some(
                    (0..flen)
                        .map(|f| {
                            let (_, t) = ctx.store.struct_value(sid).fields[f];
                            specialize_type(
                                ctx.store,
                                &mut ctx.parent,
                                &mut ctx.cluster,
                                &mut ctx.func_defs,
                                &mut ctx.struct_infers,
                                t,
                                &generic_clusters,
                                v,
                            )
                        })
                        .collect::<Vec<_>>(),
                );
            }

            let missing = CId(usize::MAX);
            let mut args = Vec::with_capacity(expected.max(provided));
            for (i, a) in cons.pos_args().ids().enumerate() {
                let c = gather_constraints(ctx, a, current_output);
                args.push(c);

                let (nid, t) = ctx.store.struct_value(sid).fields[i];
                debug_assert!(t != UNKNOWN_TYPE);
                if let Some(field_types) = &field_type_clusters {
                    let expected = field_types[i];
                    if let Err(clash) = ctx.unify(c, expected) {
                        let name = ctx.program.name_str_id(nid);
                        ctx.push_error(TypeError::FieldTypeMismatch {
                            field: name,
                            value: a,
                            clash,
                        });
                    }
                } else if let Err(clash) = ctx.force_type(c, t) {
                    let name = ctx.program.name_str_id(nid);
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
                let Value::Labeled { name, value } = ctx.program.value(na) else {
                    unreachable!()
                };

                let value_c = gather_constraints(ctx, value, current_output);

                let spot = ctx
                    .store
                    .struct_value(sid)
                    .fields
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_i, (n, _t))| ctx.program.name_str_id(*n) == name);

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

            let fields = &ctx.store.struct_value(sid).fields;
            for ((field, _t), c) in fields.iter().zip(args.iter()) {
                if *c == missing {
                    ctx.errors.push(TypeError::MissingField {
                        field: *field,
                        site: v,
                    });
                }
            }

            if glen == 0 {
                let t = ctx.store.intern(TypeValue::Struct {
                    id: sid,
                    generics: Vec::new(),
                });
                let ans = ctx.new_solved(t);
                ctx.bind_val(v, ans);
                return ans;
            }

            let ans = ctx.new_struct_instance(sid, generic_clusters);
            ctx.bind_val(v, ans);
            return ans;
        }

        Value::Access { base, name, kind } => {
            let b = gather_constraints(ctx, base, current_output);
            let b = find_root(&mut ctx.parent, b);

            //TODO we currently just do structs
            match ctx.cluster[b].state {
                ResolveKind::Nothing => {
                    todo!("put access into some sort of queue which we later check for a solve")
                }
                ResolveKind::Solved(t) => {
                    //TODO do & and &mut coretion to their derefed version
                    let TypeValue::Struct { id, generics } = ctx.store.type_value(t) else {
                        todo!()
                    };
                    let sid = *id;
                    let (field_ty, struct_name) = {
                        let rep = ctx.store.struct_value(sid);
                        let field_ty = rep
                            .fields
                            .iter()
                            .find(|(n, _)| ctx.program.name_str_id(*n) == name)
                            .map(|(_, t)| *t);
                        (field_ty, rep.name)
                    };

                    if let Some(field_ty) = field_ty {
                        match kind {
                            AccessKind::Dot | AccessKind::Ptr => {}
                            AccessKind::Static => todo!("some error on it not making sense"),
                        }

                        let ans = match ctx.store.type_value(field_ty) {
                            TypeValue::Generic(i) => ctx.new_solved(generics[i.0]),
                            _ => ctx.new_solved(field_ty),
                        };
                        ctx.bind_val(v, ans);
                        return ans;
                    }

                    let Some(struct_name) = struct_name else {
                        ctx.push_error(TypeError::UnknownField {
                            field: name,
                            site: v,
                        });
                        let ans = ctx.new_cluster();
                        ctx.bind_val(v, ans);
                        return ans;
                    };

                    return resolve_member_method_access(
                        ctx.store,
                        &mut ctx.parent,
                        &mut ctx.cluster,
                        &mut ctx.func_defs,
                        &mut ctx.struct_infers,
                        &mut ctx.val_cluster,
                        &mut ctx.member_method_type_sites,
                        &mut ctx.errors,
                        ctx.program,
                        &*ctx.ans,
                        v,
                        base,
                        b,
                        struct_name,
                        name,
                    );
                }
                ResolveKind::Struct(rid) => {
                    let sid = ctx.struct_infers[rid.0].sid;
                    let (field_ty, struct_name) = {
                        let rep = ctx.store.struct_value(sid);
                        let field_ty = rep
                            .fields
                            .iter()
                            .find(|(n, _)| ctx.program.name_str_id(*n) == name)
                            .map(|(_, t)| *t);
                        (field_ty, rep.name)
                    };

                    if let Some(field_ty) = field_ty {
                        match kind {
                            AccessKind::Dot | AccessKind::Ptr => {}
                            AccessKind::Static => todo!("some error on it not making sense"),
                        }

                        let ans = match ctx.store.type_value(field_ty) {
                            TypeValue::Generic(i) => ctx.struct_infers[rid.0].generics[i.0],
                            _ => ctx.new_solved(field_ty),
                        };
                        ctx.bind_val(v, ans);
                        return ans;
                    }

                    let Some(struct_name) = struct_name else {
                        ctx.push_error(TypeError::UnknownField {
                            field: name,
                            site: v,
                        });
                        let ans = ctx.new_cluster();
                        ctx.bind_val(v, ans);
                        return ans;
                    };

                    return resolve_member_method_access(
                        ctx.store,
                        &mut ctx.parent,
                        &mut ctx.cluster,
                        &mut ctx.func_defs,
                        &mut ctx.struct_infers,
                        &mut ctx.val_cluster,
                        &mut ctx.member_method_type_sites,
                        &mut ctx.errors,
                        ctx.program,
                        &*ctx.ans,
                        v,
                        base,
                        b,
                        struct_name,
                        name,
                    );
                }
                _ => todo!("emit some sort of error on this making no sense"),
            }
        }
        Value::Break | Value::Continue => {
            let c = ctx.new_cluster();
            ctx.cluster[c].state = ResolveKind::Never;
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
                let loc = ctx.program.value_loc(v);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "return used outside of function body",
                });
            }

            let c = ctx.new_cluster();
            ctx.cluster[c].state = ResolveKind::Never;
            c
        }
        Value::LogicOp { op: _, values: _ } => todo!(),

        Value::Tuple(_) | Value::Array(_) => todo!(),
        Value::Index(_) | Value::Match { .. } => todo!(),

        Value::Labeled { .. } => unreachable!("bug tried compiling labeled normally"),
        Value::MatchArm(_) => unreachable!("bug tried compiling match arm normally"),
    }
}

///this tries to resolve specifically a from a module.
///if what we have is a member of a struct it wont give a name
fn try_get_name(ctx: &mut InferState, v: ValId) -> Option<NameId> {
    match ctx.program.value(v) {
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
fn gather_pattern_constraints_with_generics<const ALLOW_GENERICS: bool>(
    ctx: &mut InferState,
    p: PatId,
) -> CId {
    let (x, _) = gather_pattern_constraints_and_name_with_generics::<ALLOW_GENERICS>(ctx, p);
    x
}

fn gather_pattern_constraints_and_name_with_generics<const ALLOW_GENERICS: bool>(
    ctx: &mut InferState,
    p: PatId,
) -> (CId, Option<NameId>) {
    match ctx.program.pattern(p) {
        Pattern::Wildcard(_) => {
            let c = ctx.new_cluster();
            ctx.bind_pat(p, c);
            (c, None)
        }
        Pattern::Bind(n, _) => {
            let c = ctx.new_cluster();
            ctx.names.insert(n, c);
            ctx.bind_pat(p, c);
            (c, Some(n))
        }

        Pattern::TypeAnnotation { pat, ty } => {
            let (c, n) =
                gather_pattern_constraints_and_name_with_generics::<ALLOW_GENERICS>(ctx, pat);
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

        _ => todo!(),
    }
}

///this method is kinda weird and ill formed
///currently when compiling type expressions we give them a type other the Type::Type
///we dont have a good destinction between the type of THE VALUE ITSELF and the type IT REFERS TO
///and this means that fn[T](){let x=T;} is technically legal and x has type Generic(0).
fn gather_generic_constraints(ctx: &mut InferState, p: PatId, id: GenId) -> CId {
    match ctx.program.pattern(p) {
        Pattern::Bind(n, m) => {
            if m != VarKind::Const {
                let loc = ctx.program.pattern_loc(p);
                ctx.errors.push(TypeError::Simple {
                    loc,
                    message: "generic parameters must be const bindings",
                });
            }
            let t = ctx.store.intern(TypeValue::Generic(id));
            let c = ctx.new_solved(t);
            ctx.names.insert(n, c);
            ctx.local_types.insert(n, c);
            ctx.bind_pat(p, c);
            c
        }

        _ => todo!(),
    }
}

///in order to break recursion this function MUST return a concrete type
///the returned struct is not fully realized yet and its fields are gona be handeled later
fn compile_struct_type<const ALLOW_GENERICS: bool>(
    ctx: &mut InferState,
    texpr: TExpId,
    StructLike { generics, fields }: StructLike,
) -> CId {
    if !ALLOW_GENERICS && !generics.is_empty() {
        let loc = generics
            .ids()
            .next()
            .map(|pat| ctx.program.pattern_loc(pat))
            .unwrap_or_else(|| ctx.program.type_expr_loc(texpr));
        ctx.errors.push(TypeError::Simple {
            loc,
            message: "generic struct types are only allowed at the top level",
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
        match ctx.program.pattern(p) {
            Pattern::Bind(n, _) => {
                let c = ctx.new_cluster();
                field_info.push((n, c));
            }
            Pattern::TypeAnnotation { pat, ty } => {
                let Pattern::Bind(n, _) = ctx.program.pattern(pat) else {
                    let loc = ctx.program.pattern_loc(pat);
                    ctx.errors.push(TypeError::Simple {
                        loc,
                        message: "struct field must be a named binding",
                    });
                    continue;
                };
                let c = compile_type_expr(ctx, ty);
                field_info.push((n, c));
            }
            _ => {
                let loc = ctx.program.pattern_loc(p);
                ctx.errors.push(TypeError::Simple {
                    loc,
                    message: "struct field must be a named binding",
                });
                continue;
            }
        }
    }

    let rep = StructRep::new(field_info.iter().map(|(n, _)| *n), generics.len());
    let sid = ctx.store.new_struct(rep);
    let generics = (0..generics.len())
        .map(|x| ctx.store.intern(TypeValue::Generic(GenId(x))))
        .collect();
    let t = ctx.store.intern(TypeValue::Struct { id: sid, generics });
    let output = ctx.new_solved(t);

    ctx.struct_defs.push(StructDef {
        loc: texpr,
        fields: field_info,
        sid,
        output,
    });
    output
}

fn compile_type_expr(ctx: &mut InferState, texpr: TExpId) -> CId {
    match ctx.program.type_expr(texpr) {
        TypeExpr::NameRef(n) => {
            if let Some(ans) = ctx.local_types.get(&n) {
                return *ans;
            }
            let t = match ctx.program.definitions.get(&n) {
                Some(Defined::BuildinType(b)) => ctx.store.intern(b.clone()),
                Some(Defined::Type(texp)) => {
                    // return ctx.global_types.handle_global(
                    //     n,
                    //     &mut ctx.local_types,
                    //     *texp,
                    //     &mut ctx.parent,
                    //     &mut ctx.cluster,
                    // );
                    let Some(t) = ctx.ans.typedef_types.get(texp) else {
                        let id = ctx.new_cluster();
                        ctx.local_types.insert(n, id);
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

        TypeExpr::Struct(def) => compile_struct_type::<false>(ctx, texpr, def),
        TypeExpr::Ptr { base, raw, mutable } => {
            let tgt = compile_type_expr(ctx, base);
            let ans = ctx.new_cluster();
            ctx.cluster[ans].state = ResolveKind::Ptr {
                tgt,
                raw: Some(raw),
                mutable: Some(mutable),
            };
            ans
        }
        TypeExpr::Index { base, args } => {
            let generics = args
                .ids()
                .map(|arg| compile_type_expr(ctx, arg))
                .collect::<Vec<_>>();

            // let ans = ctx.new_cluster();
            let Some(name) = get_type_name(ctx.program, base) else {
                let loc = ctx.program.type_expr_loc(base);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "type specialization base must be a type name",
                });
                return ctx.new_cluster();
            };

            let Some(def) = ctx.program.definitions.get(&name) else {
                //we dont allow generics on local structs
                //so this is either not a struct at all
                //or a struct with no generics
                let loc = ctx.program.type_expr_loc(base);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "type specialization base must be a global type",
                });
                return ctx.new_cluster();
            };

            let Defined::Type(g) = def else {
                let loc = ctx.program.type_expr_loc(base);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "type specialization expects a type definition",
                });
                return ctx.new_cluster();
            };

            let Some(t) = ctx.ans.typedef_types.get(g) else {
                //this happens only in global context
                //and so it only happens when we specifically solve for global structs
                //because of this to break the recursion we are gona cheat
                //but with a tiny bit of class

                let Some(_cid) = ctx.local_types.get(&name) else {
                    let output = ctx.new_cluster();
                    ctx.pending_specializations.push(PendingSpecialization {
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
                let loc = ctx.program.type_expr_loc(texpr);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "currently we only support specilizing struct definitions directly",
                });

                return ctx.new_cluster();
            };

            let TypeValue::Struct { id: sid, .. } = ctx.store.type_value(*t) else {
                let loc = ctx.program.type_expr_loc(base);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "type specialization expects a struct type",
                });
                return ctx.new_cluster();
            };
            let sid = *sid;

            let expected = ctx.store.struct_value(sid).gen_count;
            if generics.len() != expected {
                let loc = ctx.program.type_expr_loc(texpr);
                ctx.errors.push(TypeError::Simple {
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

fn gather_func_signature<const ALLOW_GENERICS: bool>(
    ctx: &mut InferState,
    v: ValId,
    generics: PatternSpan,
    params: PatternSpan,
    output_type: Option<TExpId>,
) -> (CId, CId) {
    if !ALLOW_GENERICS && !generics.is_empty() {
        let loc = generics
            .ids()
            .next()
            .map(|pat| ctx.program.pattern_loc(pat))
            .unwrap_or_else(|| ctx.program.value_loc(v));
        ctx.errors.push(TypeError::Simple {
            loc,
            message: "generic functions are only allowed at the top level",
        });
    }

    for (i, pat) in generics.ids().enumerate() {
        gather_generic_constraints(ctx, pat, GenId(i));
    }

    if ALLOW_GENERICS && !generics.is_empty() {
        ctx.generic_func_values.push((v, generics.len()));
    }

    let inputs = params
        .ids()
        .map(|pat| gather_pattern_constraints_with_generics::<ALLOW_GENERICS>(ctx, pat))
        .collect::<Vec<_>>();

    let output = if let Some(x) = output_type {
        compile_type_expr(ctx, x)
    } else {
        ctx.new_solved(BuiltinType::Void.into())
    };

    let f = ctx.new_func(FuncInfer {
        inputs,
        output,
        loc: v,
    });
    ctx.bind_val(v, f);
    (f, output)
}

fn gather_func_constraints<const ALLOW_GENERICS: bool>(
    ctx: &mut InferState,
    v: ValId,
    generics: PatternSpan,
    params: PatternSpan,
    output_type: Option<TExpId>,
    body: ValId,
) -> CId {
    let (f, output) =
        gather_func_signature::<ALLOW_GENERICS>(ctx, v, generics, params, output_type);

    let body_cluster = gather_constraints(ctx, body, Some(output));

    if let Err(clash) = ctx.unify(body_cluster, output) {
        let found = match ctx.program.value(body) {
            Value::Block {
                statements: _,
                return_value: Some(x),
            } => x,
            _ => body,
        };
        ctx.push_error(TypeError::ValuesContradict {
            expectation_reason: "function body must match return type",
            site: v,
            found,
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
    matches!(name, NEG_STR | NOT_STR | BITNOT_STR)
}

#[inline(always)]
fn is_known_special_member_method_name(name: StrId) -> bool {
    is_binary_operator_overload_name(name)
        || is_unary_operator_overload_name(name)
        || name == FREE_STR
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
    let fn_ty = match store.type_value(ty) {
        TypeValue::WithGenerics { body, .. } => *body,
        _ => ty,
    };

    match store.type_value(fn_ty) {
        TypeValue::Func { params, ret } => Some((params.as_slice(), *ret)),
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
fn member_self_input_cluster(
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    lhs: CId,
    style: MemberSelfStyle,
) -> CId {
    match style {
        MemberSelfStyle::Value => lhs,
        MemberSelfStyle::Ref { mutable } => {
            let ans = new_cluster(parent, cluster);
            cluster[ans].state = ResolveKind::Ptr {
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

fn check_special_member_method_signature(
    ctx: &mut InferState,
    method_val: ValId,
    struct_name: NameId,
    method_name: StrId,
) {
    let loc = ctx.program.value_loc(method_val);

    if is_reserved_builtin_member_name(ctx.program, method_name)
        && !is_known_special_member_method_name(method_name)
    {
        ctx.push_error(TypeError::UnknownBuiltinMemberMethod {
            site: method_val,
            method: method_name,
        });
        return;
    }

    if !is_known_special_member_method_name(method_name) {
        return;
    }

    let Some(method_ty) = ctx.ans.type_of(method_val) else {
        return;
    };
    let Some((inputs, output)) = method_signature_type_parts(ctx.store, method_ty) else {
        return;
    };

    if method_name == FREE_STR {
        let Some(first_input) = inputs.first().copied() else {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "special member methods must take `self` as the first parameter",
            });
            return;
        };

        if !is_mut_ref_to_named_struct_input_type(ctx.store, first_input, struct_name) {
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

    let Some(first_input) = inputs.first().copied() else {
        ctx.push_error(TypeError::Simple {
            loc,
            message: "special member methods must take `self` as the first parameter",
        });
        return;
    };

    let additional_args = inputs.len() - 1;

    if is_binary_operator_overload_name(method_name) {
        if !is_self_like_member_input_type(ctx.store, first_input, struct_name) {
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
        return;
    }

    if is_unary_operator_overload_name(method_name) {
        if !is_self_like_member_input_type(ctx.store, first_input, struct_name) {
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
        return;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperandKind {
    KnownNonUser,
    UserStruct(Option<NameId>),
    Unknown,
}

#[inline(always)]
fn classify_operand(
    store: &TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    struct_infers: &[StructInfer],
    cid: CId,
) -> OperandKind {
    let root = find_root(parent, cid);
    match cluster[root].state {
        ResolveKind::Solved(t) => match store.type_value(t) {
            TypeValue::Struct { id, generics: _ } => {
                OperandKind::UserStruct(store.struct_value(*id).name)
            }
            _ => OperandKind::KnownNonUser,
        },
        ResolveKind::Struct(call_id) => {
            let sid = struct_infers[call_id.0].sid;
            OperandKind::UserStruct(store.struct_value(sid).name)
        }
        ResolveKind::IntLike
        | ResolveKind::FloatLike
        | ResolveKind::Func(_)
        | ResolveKind::Ptr { .. } => OperandKind::KnownNonUser,
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

/// Unify only if roots differ; report whether a merge happened.
#[inline]
fn unify_if_distinct(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    a: CId,
    b: CId,
) -> Result<bool, TypeClash> {
    let ra = find_root(parent, a);
    let rb = find_root(parent, b);
    if ra == rb {
        return Ok(false);
    }

    if matches!(cluster[ra].state, ResolveKind::Never) {
        return Ok(false);
    }
    if matches!(cluster[rb].state, ResolveKind::Never) {
        return Ok(false);
    }

    unify_clusters_inlined(store, parent, cluster, func_defs, struct_infers, ra, rb)?;
    Ok(true)
}

#[inline(always)]
fn function_parts_from_cluster(
    store: &TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &[FuncInfer],
    cid: CId,
) -> Option<(Vec<CId>, CId)> {
    let root = find_root(parent, cid);
    match cluster[root].state {
        ResolveKind::Func(call) => {
            Some((func_defs[call.0].inputs.clone(), func_defs[call.0].output))
        }
        ResolveKind::Solved(t) => {
            let TypeValue::Func { params, ret } = store.type_value(t) else {
                return None;
            };
            let inputs = params
                .iter()
                .map(|p| new_solved(parent, cluster, *p))
                .collect::<Vec<_>>();
            let output = new_solved(parent, cluster, *ret);
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
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    func_defs: &Vec<FuncInfer>,
    struct_infers: &Vec<StructInfer>,
    site: &BinOpSite,
    lhs: CId,
    rhs: CId,
) -> TypeError {
    TypeError::BinOpOverloadNotFound {
        site: site.loc,
        op: site.op,
        lhs: site.lhs_val,
        rhs: site.rhs_val,
        lhs_type: extract_bad_type(store, parent, cluster, func_defs, struct_infers, lhs),
        rhs_type: extract_bad_type(store, parent, cluster, func_defs, struct_infers, rhs),
    }
}

#[inline(always)]
fn un_op_overload_not_found_error(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &ClusterVec<Cluster>,
    func_defs: &Vec<FuncInfer>,
    struct_infers: &Vec<StructInfer>,
    site: &UnOpSite,
    input: CId,
) -> TypeError {
    TypeError::UnOpOverloadNotFound {
        site: site.loc,
        op: site.op,
        operand: site.val,
        operand_type: extract_bad_type(store, parent, cluster, func_defs, struct_infers, input),
    }
}

#[inline(always)]
fn resolve_member_overload_signature(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    ans: &SolvedTypes,
    method: ValId,
    struct_name: NameId,
    loc: ValId,
) -> Option<ResolvedMemberOverload> {
    let Some(method_ty) = ans.type_of(method) else {
        unreachable!(
            "global member method signatures must be solved before body inference; missing type for operator overload"
        );
    };

    let self_style = get_member_self_style(store, method_ty, struct_name)?;
    let method_local = solved_type_to_specialized_local(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        method_ty,
        loc,
    );
    let Some((params, ret)) =
        function_parts_from_cluster(store, parent, cluster, func_defs, method_local)
    else {
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
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    receiver: CId,
    method: ResolvedMemberOverload,
    loc: ValId,
) -> Result<CId, TypeClash> {
    let ResolvedMemberOverload {
        mut params,
        ret,
        self_style,
        full_method: _,
    } = method;
    debug_assert!(!params.is_empty());

    let self_param = params.remove(0);
    let self_input = member_self_input_cluster(parent, cluster, receiver, self_style);
    unify_if_distinct(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        self_param,
        self_input,
    )?;

    Ok(new_func(
        parent,
        cluster,
        func_defs,
        FuncInfer {
            loc,
            inputs: params,
            output: ret,
        },
    ))
}

#[inline(always)]
fn resolve_operator_site(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    member_method_type_sites: &mut Vec<PendingMemberMethodType>,
    errors: &mut Vec<TypeError>,
    site: &mut BinOpSite,
    program: &Program,
    ans: &SolvedTypes,
) -> ResolveOutcome {
    use BinOp::*;

    let mut progress = false;
    let lhs = find_root(parent, site.lhs);
    let rhs = find_root(parent, site.rhs);
    let out = find_root(parent, site.output);
    let op = site.op;

    let lhs_kind = classify_operand(store, parent, cluster, struct_infers, lhs);
    let rhs_kind = classify_operand(store, parent, cluster, struct_infers, rhs);

    if let OperandKind::UserStruct(struct_name) = lhs_kind {
        let Some(struct_name) = struct_name else {
            unreachable!(
                "operator overload lookup produced a method only when struct_name is present"
            );
        };

        let method = program
            .member_methods
            .get(&struct_name)
            .and_then(|methods| methods.get(&bin_op_overload_name(op)))
            .copied();

        if let Some(method) = method {
            let Some(overload_sig) = resolve_member_overload_signature(
                store,
                parent,
                cluster,
                func_defs,
                struct_infers,
                ans,
                method,
                struct_name,
                site.loc,
            ) else {
                errors.push(bin_op_overload_not_found_error(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    site,
                    lhs,
                    rhs,
                ));
                return ResolveOutcome::drop(false);
            };

            if overload_sig.params.len() != 2 {
                errors.push(bin_op_overload_not_found_error(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    site,
                    lhs,
                    rhs,
                ));
                return ResolveOutcome::drop(false);
            }

            let overload_mismatch: Result<(), TypeClash> = (|| {
                let method_name = bin_op_overload_name(op);
                let full_method = overload_sig.full_method;
                let method_closure = make_member_closure(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    lhs,
                    overload_sig,
                    site.loc,
                )?;
                let expected_fn = new_func(
                    parent,
                    cluster,
                    func_defs,
                    FuncInfer {
                        loc: site.loc,
                        inputs: vec![rhs],
                        output: out,
                    },
                );
                progress |= unify_if_distinct(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    method_closure,
                    expected_fn,
                )?;
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
                errors.push(TypeError::ValuesContradict {
                    expectation_reason: OP_OVERLOAD_SIGNATURE_MISMATCH,
                    site: site.loc,
                    found: site.lhs_val,
                    expected_place: site.rhs_val,
                    clash,
                });
                return ResolveOutcome::drop(false);
            }

            return ResolveOutcome::drop(progress);
        }

        errors.push(bin_op_overload_not_found_error(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            site,
            lhs,
            rhs,
        ));
        return ResolveOutcome::drop(false);
    }

    if matches!(rhs_kind, OperandKind::UserStruct(_)) {
        return ResolveOutcome::keep(false);
    }

    if lhs_kind == OperandKind::Unknown || rhs_kind == OperandKind::Unknown {
        return ResolveOutcome::keep(false);
    }

    // ----------------------------------------------------
    // 1) Early legality rejection (single helper)
    // ----------------------------------------------------

    let lhs_ok = system_types_operator_applicable(store, parent, cluster, op, lhs);
    let rhs_ok = system_types_operator_applicable(store, parent, cluster, op, rhs);

    if lhs_ok == Some(false) || rhs_ok == Some(false) {
        errors.push(bin_op_overload_not_found_error(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            site,
            lhs,
            rhs,
        ));
        return ResolveOutcome::drop(false);
    }

    // ----------------------------------------------------
    // 2) Equality / comparisons
    //
    // NOTE:
    // - operand equality is already enforced in gather
    // - output = bool is already enforced in gather
    // ----------------------------------------------------
    if matches!(op, Eq | Ne | Lt | Le | Gt | Ge) {
        return ResolveOutcome::drop(false);
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
        return ResolveOutcome::keep(false);
    }

    // (a) unify operands
    match unify_if_distinct(store, parent, cluster, func_defs, struct_infers, lhs, rhs) {
        Ok(changed) => progress |= changed,
        Err(clash) => {
            errors.push(TypeError::ValuesContradict {
                expectation_reason: "binary operator requires operands of the same type",
                site: site.loc,
                found: site.lhs_val,
                expected_place: site.rhs_val,
                clash,
            });
            return ResolveOutcome::drop(false);
        }
    }

    let operand = find_root(parent, lhs);

    // (b) unify output with operand
    match unify_if_distinct(
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        out,
        operand,
    ) {
        Ok(changed) => progress |= changed,
        Err(clash) => {
            errors.push(TypeError::ValuesContradict {
                expectation_reason: "operator result type must match operand type",
                site: site.loc,
                found: site.lhs_val,
                expected_place: site.rhs_val,
                clash,
            });
            return ResolveOutcome::drop(false);
        }
    }

    ResolveOutcome::drop(progress)
}

#[inline(always)]
fn resolve_unary_operator_site(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut Vec<FuncInfer>,
    struct_infers: &mut Vec<StructInfer>,
    member_method_type_sites: &mut Vec<PendingMemberMethodType>,
    errors: &mut Vec<TypeError>,
    site: &mut UnOpSite,
    program: &Program,
    ans: &SolvedTypes,
) -> ResolveOutcome {
    use UnOp::*;

    let mut progress = false;
    let input = find_root(parent, site.input);
    let out = find_root(parent, site.output);
    let op = site.op;

    let operand_kind = classify_operand(store, parent, cluster, struct_infers, input);
    if let OperandKind::UserStruct(struct_name) = operand_kind {
        let Some(struct_name) = struct_name else {
            unreachable!(
                "operator overload lookup produced a method only when struct_name is present"
            );
        };

        let method = program
            .member_methods
            .get(&struct_name)
            .and_then(|methods| methods.get(&un_op_overload_name(op)))
            .copied();
        if let Some(method) = method {
            let Some(overload_sig) = resolve_member_overload_signature(
                store,
                parent,
                cluster,
                func_defs,
                struct_infers,
                ans,
                method,
                struct_name,
                site.loc,
            ) else {
                errors.push(un_op_overload_not_found_error(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    site,
                    input,
                ));
                return ResolveOutcome::drop(false);
            };

            if overload_sig.params.len() != 1 {
                errors.push(un_op_overload_not_found_error(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    site,
                    input,
                ));
                return ResolveOutcome::drop(false);
            }

            let overload_mismatch: Result<(), TypeClash> = (|| {
                let method_name = un_op_overload_name(op);
                let full_method = overload_sig.full_method;
                let method_closure = make_member_closure(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    input,
                    overload_sig,
                    site.loc,
                )?;
                let expected_fn = new_func(
                    parent,
                    cluster,
                    func_defs,
                    FuncInfer {
                        loc: site.loc,
                        inputs: Vec::new(),
                        output: out,
                    },
                );
                progress |= unify_if_distinct(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    method_closure,
                    expected_fn,
                )?;
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
                errors.push(TypeError::ValuesContradict {
                    expectation_reason: OP_OVERLOAD_SIGNATURE_MISMATCH,
                    site: site.loc,
                    found: site.val,
                    expected_place: site.loc,
                    clash,
                });
                return ResolveOutcome::drop(false);
            }

            return ResolveOutcome::drop(progress);
        }

        errors.push(un_op_overload_not_found_error(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            site,
            input,
        ));
        return ResolveOutcome::drop(false);
    }

    if operand_kind == OperandKind::Unknown {
        return ResolveOutcome::keep(false);
    }

    match op {
        Not => {
            if let Some(false) = cluster_is_bool(store, parent, cluster, input) {
                errors.push(un_op_overload_not_found_error(
                    store,
                    parent,
                    cluster,
                    func_defs,
                    struct_infers,
                    site,
                    input,
                ));
                return ResolveOutcome::drop(false);
            }
            match unify_if_distinct(store, parent, cluster, func_defs, struct_infers, input, out) {
                Ok(changed) => progress |= changed,
                Err(clash) => {
                    errors.push(TypeError::ValuesContradict {
                        expectation_reason: "logical not requires a bool operand",
                        site: site.loc,
                        found: site.val,
                        expected_place: site.loc,
                        clash,
                    });
                    return ResolveOutcome::drop(false);
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
                    errors.push(un_op_overload_not_found_error(
                        store,
                        parent,
                        cluster,
                        func_defs,
                        struct_infers,
                        site,
                        input,
                    ));
                    return ResolveOutcome::drop(false);
                }
                _ => return ResolveOutcome::keep(false),
            }

            match unify_if_distinct(store, parent, cluster, func_defs, struct_infers, input, out) {
                Ok(changed) => progress |= changed,
                Err(clash) => {
                    errors.push(TypeError::ValuesContradict {
                        expectation_reason: "negation requires operand and result types match",
                        site: site.loc,
                        found: site.val,
                        expected_place: site.loc,
                        clash,
                    });
                    return ResolveOutcome::drop(false);
                }
            }
        }
        BitNot => {
            match cluster_is_int_like(store, parent, cluster, input) {
                Some(true) => {}
                Some(false) => {
                    errors.push(un_op_overload_not_found_error(
                        store,
                        parent,
                        cluster,
                        func_defs,
                        struct_infers,
                        site,
                        input,
                    ));
                    return ResolveOutcome::drop(false);
                }
                None => return ResolveOutcome::keep(false),
            }

            match unify_if_distinct(store, parent, cluster, func_defs, struct_infers, input, out) {
                Ok(changed) => progress |= changed,
                Err(clash) => {
                    errors.push(TypeError::ValuesContradict {
                        expectation_reason: "bitwise not requires operand and result types match",
                        site: site.loc,
                        found: site.val,
                        expected_place: site.loc,
                        clash,
                    });
                    return ResolveOutcome::drop(false);
                }
            }
        }
    }

    ResolveOutcome::drop(progress)
}

#[inline(always)]
fn resolve_operator_types(ctx: &mut InferState) -> bool {
    let mut progress = false;
    let (
        store,
        parent,
        cluster,
        func_defs,
        struct_infers,
        member_method_type_sites,
        bin_op_sites,
        un_op_sites,
        errors,
        ans,
    ) = (
        &mut ctx.store,
        &mut ctx.parent,
        &mut ctx.cluster,
        &mut ctx.func_defs,
        &mut ctx.struct_infers,
        &mut ctx.member_method_type_sites,
        &mut ctx.bin_op_sites,
        &mut ctx.un_op_sites,
        &mut ctx.errors,
        &*ctx.ans,
    );

    bin_op_sites.retain_mut(|site| {
        let outcome = resolve_operator_site(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            member_method_type_sites,
            errors,
            site,
            ctx.program,
            ans,
        );
        progress |= outcome.progress;
        outcome.retain
    });

    un_op_sites.retain_mut(|site| {
        let outcome = resolve_unary_operator_site(
            store,
            parent,
            cluster,
            func_defs,
            struct_infers,
            member_method_type_sites,
            errors,
            site,
            ctx.program,
            ans,
        );
        progress |= outcome.progress;
        outcome.retain
    });

    progress
}

fn try_resolve_func_type(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    func_defs: &mut [FuncInfer],
    call: FuncInferId,
) -> Option<TypeId> {
    let mut params = Vec::with_capacity(func_defs[call.0].inputs.len());
    for i in 0..func_defs[call.0].inputs.len() {
        let input = func_defs[call.0].inputs[i];
        let root = find_root(parent, input);
        func_defs[call.0].inputs[i] = root;
        match cluster[root].state {
            ResolveKind::Solved(t) => params.push(t),
            _ => return None,
        }
    }

    let output = func_defs[call.0].output;
    let root = find_root(parent, output);
    func_defs[call.0].output = root;
    let ret = match cluster[root].state {
        ResolveKind::Solved(t) => t,
        _ => return None,
    };

    Some(store.intern(TypeValue::Func { params, ret }))
}

fn try_resolve_struct_type(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    struct_infers: &mut [StructInfer],
    call: StructInferId,
) -> Option<TypeId> {
    let mut generics = Vec::with_capacity(struct_infers[call.0].generics.len());
    for i in 0..struct_infers[call.0].generics.len() {
        let input = struct_infers[call.0].generics[i];
        let root = find_root(parent, input);
        struct_infers[call.0].generics[i] = root;
        match cluster[root].state {
            ResolveKind::Solved(t) => generics.push(t),
            _ => return None,
        }
    }

    let sid = struct_infers[call.0].sid;
    Some(store.intern(TypeValue::Struct { id: sid, generics }))
}

fn try_resolve_ptr_type(
    store: &mut TypeStore,
    parent: &mut ClusterVec<CId>,
    cluster: &mut ClusterVec<Cluster>,
    tgt: CId,
    raw: Option<bool>,
    mutable: Option<bool>,
) -> Option<TypeId> {
    let raw = raw?;
    let mutable = mutable?;
    let root = find_root(parent, tgt);
    let tgt = match cluster[root].state {
        ResolveKind::Solved(t) => t,
        _ => return None,
    };
    Some(store.intern(TypeValue::Ptr { tgt, raw, mutable }))
}

#[inline(always)]
fn resolve_deferred_types(ctx: &mut InferState) -> bool {
    let mut change = false;
    for cid in (0..ctx.cluster.len()).map(CId) {
        let resolved = match ctx.cluster[cid].state {
            ResolveKind::Func(call) => try_resolve_func_type(
                ctx.store,
                &mut ctx.parent,
                &mut ctx.cluster,
                &mut ctx.func_defs,
                call,
            ),
            ResolveKind::Struct(call) => try_resolve_struct_type(
                ctx.store,
                &mut ctx.parent,
                &mut ctx.cluster,
                &mut ctx.struct_infers,
                call,
            ),
            ResolveKind::Ptr { tgt, raw, mutable } => try_resolve_ptr_type(
                ctx.store,
                &mut ctx.parent,
                &mut ctx.cluster,
                tgt,
                raw,
                mutable,
            ),
            _ => None,
        };

        if let Some(t) = resolved {
            ctx.cluster[cid].state = ResolveKind::Solved(t);
            change = true;
        }
    }
    change
}

#[inline(always)]
fn resolve_pending_specializations(ctx: &mut InferState) -> bool {
    let mut change = false;
    let mut pending = std::mem::take(&mut ctx.pending_specializations);

    pending.retain_mut(|p| {
        let Some(base_type) = ctx.ans.typedef_types.get(&p.global).copied() else {
            return true;
        };

        let sid = match ctx.store.type_value(base_type) {
            TypeValue::Struct { id: sid, .. } => *sid,
            _ => {
                let loc = ctx.program.type_expr_loc(p.global);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "only struct types can be specialized",
                });
                change = true;
                return false;
            }
        };

        let expected = ctx.store.struct_value(sid).gen_count;
        if p.generics.len() != expected {
            let loc = ctx.program.type_expr_loc(p.global);
            ctx.errors.push(TypeError::Simple {
                loc,
                message: "wrong number of generic arguments for struct type",
            });
            change = true;
            return false;
        }

        let found = ctx.new_struct_instance(sid, p.generics.clone());
        if let Err(clash) = ctx.unify(found, p.output) {
            ctx.errors.push(TypeError::TypeClashBeforeMentioned {
                name: p.name,
                expr: p.global,
                clash,
            });
        }

        change = true;
        false
    });

    ctx.pending_specializations = pending;
    change
}

#[inline(always)]
// #[inline(never)]
// #[unsafe(no_mangle)]
fn finalize(ctx: &mut InferState) {
    let (val_cluster, pat_cluster, member_method_type_sites, parent, cluster, errors, ans) = (
        &ctx.val_cluster,
        &ctx.pat_cluster,
        &ctx.member_method_type_sites,
        &mut ctx.parent,
        &ctx.cluster,
        &mut ctx.errors,
        &mut ctx.ans,
    );

    // unsafe{perf_begin();}

    let mut reported: IdHashMap<CId, ()> = IdHashMap::default();
    let mut member_method_by_site: IdHashMap<ValId, PendingMemberMethodType> = IdHashMap::default();
    for entry in member_method_type_sites.iter().copied() {
        member_method_by_site.insert(entry.site, entry);
    }
    for (e, c) in ctx.typedef_cluster.iter() {
        let root = find_root(parent, *c);
        if let ResolveKind::Solved(t) = cluster[root].state {
            ans.typedef_types.insert(*e, t);
        } else if *c == root {
            errors.push(TypeError::UnresolvedTypeExpr { expr: *e });
            reported.insert(*c, ());
        }
    }

    for sdef in ctx.struct_defs.iter() {
        if ctx.store.structs[sdef.sid.0].name.is_none() {
            if let Some((name, _)) = ctx
                .program
                .definitions
                .iter()
                .find(|(_, def)| matches!(def, Defined::Type(texp) if *texp == sdef.loc))
            {
                ctx.store.structs[sdef.sid.0].name = Some(*name);
            }
        }
        for (i, (_n, c)) in sdef.fields.iter().enumerate() {
            let root = find_root(parent, *c);
            if let ResolveKind::Solved(t) = cluster[root].state {
                ctx.store.structs[sdef.sid.0].fields[i].1 = t;
            } else if *c == root {
                let loc = ctx.program.type_expr_loc(sdef.loc);
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
    for (v, count) in ctx.generic_func_values.iter() {
        let Some(spot) = ans.val_types.get_mut(v.0) else {
            unreachable!("bug func value not also a value");
        };
        let body = *spot;
        *spot = ctx.store.intern(TypeValue::WithGenerics {
            body,
            count: *count,
        });
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

    // let name = CStr::from_bytes_with_nul(b"finalize\0").unwrap();
    // unsafe { perf_done(name.as_ptr()); }
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

    /// Run inference on a single function body.
    fn infer_fn(src: &str, store: &mut TypeStore) -> Result<TypeId, Vec<TypeError>> {
        let program = gather_program(src);
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, store, &mut solved_types)?;
        let f = extract_single_fn(&program);
        let body = match program.value(f) {
            Value::Func { body, .. } => body,
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
            Value::Func { body, .. } => body,
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
        let ty = infer_fn("f = fn()->int { let d :float = (fn()->_{if true return 1.0; 2.0})(); 2 }", &mut store).unwrap();
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
    fn generic_function_solves_with_generics() {
        let src = "f = fn[T](x:T)->T { x }";
        let mut store = TypeStore::new();
        let program = gather_program(src);
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();
        let f = extract_single_fn(&program);
        let f_ty = solved_types.type_of(f).unwrap();

        match store.type_value(f_ty) {
            TypeValue::WithGenerics { count, body } => {
                assert_eq!(*count, 1);
                match store.type_value(*body) {
                    TypeValue::Func { params, ret } => {
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
            other => panic!("expected WithGenerics, got {:?}", other),
        }
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
    fn operator_overload_not_found_for_structs() {
        let mut store = TypeStore::new();
        let errs = infer_fn_body("S=struct{}; f=fn(){ S{} + S{}; }", &mut store).unwrap_err();
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
        let TypeValue::Func { params, ret } = store.type_value(called.full_type) else {
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

        let TypeValue::Func { params, ret } = store.type_value(access_ty) else {
            panic!("expected member access to be curried function")
        };
        assert_eq!(params.len(), 0);
        assert_eq!(*ret, x_ty);
        assert_eq!(call_ty, x_ty);

        let called = solved_types
            .member_method_type(access_site)
            .expect("missing solved member method signature for access site");
        assert_eq!(program.str_intern.resolve(called.member), "add_5");
        let TypeValue::Func { params, ret } = store.type_value(called.full_type) else {
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

        let TypeValue::Func { params, ret } = store.type_value(access_ty) else {
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
