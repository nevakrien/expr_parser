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

use crate::ir::{NameId, PatternId, ValueId};
use std::collections::HashMap;

use crate::{
    ir::{BinOp, Literal, Pattern, Value},
    program::{Defined, Program},
};

/* ================================================================
 * Errors (STABLE SHAPE)
 * ================================================================ */

// #[derive(Debug)]
// pub enum TypeError {
//     Unresolved {
//         produced_loc: Loc,
//         message: &'static str,
//     },

//     SimpleMismatch {
//         required_loc: Loc,
//         produced_loc: Loc,
//         expected: TypeId,
//         found: TypeId,
//         note: &'static str,
//     },

//     Unsupported {
//         loc: Loc,
//         message: &'static str,
//     },

//     ExpectedType {
//         loc: Loc,
//         message: &'static str,
//     },

//     InvalidOperator {
//         loc: Loc,
//         op: BinOp,
//         lhs: TypeId,
//         rhs: TypeId,
//         note: &'static str,
//     },
//     InvalidLiteral {
//         loc: Loc,
//         loc_reqired:Loc,
//         literal: Literal,
//         target: TypeId,
//         note: &'static str,
//     },
// }

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
    global_types: HashMap<ValueId, TypeId>,
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
            global_types: HashMap::new(),
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
    pub fn get_global(&self, id: ValueId) -> Option<TypeId> {
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
    types: HashMap<ValueId, TypeId>,
}

impl Default for LocalTypes {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalTypes {
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
        }
    }

    #[inline(always)]
    pub fn type_of(&self, id: ValueId) -> Option<TypeId> {
        self.types.get(&id).copied()
    }
}

// ==============================
// Errors (richer + ValId-based)
// ==============================

#[derive(Debug)]
pub enum TypeError {
    /// Could not infer a concrete type for this value
    Unresolved {
        value: ValueId,
        message: &'static str,
    },

    /// Type expression (the RHS of `:` / `as`) wasn't a valid type
    ExpectedType {
        type_expr: ValueId,
        message: &'static str,
    },

    /// `expr : T` or `pat : T` conflicts with what the value/pattern already implies.
    /// Carries BOTH the annotation node and the constrained node so diagnostics can point at both.
    AnnotationMismatch {
        /// The annotation node (Value::TypeAnnotation / Pattern::TypeAnnotation)
        annotation: ValueId,
        /// The value/pattern being constrained (the `value` inside the annotation)
        constrained: ValueId,
        expected: TypeId,
        found: TypeId,
        note: &'static str,
    },

    /// Pattern annotation mismatch
    PatternAnnotationMismatch {
        annotation: PatternId,
        constrained: PatternId,
        expected: TypeId,
        found: TypeId,
        note: &'static str,
    },

    /// Equality constraint failure at some site (let/match/etc).
    /// Carries a site ValId so you can point at the operator/let/match that demanded equality.
    IncompatibleTypes {
        site: ValueId,
        left: TypeId,
        right: TypeId,
        note: &'static str,
    },

    /// Literal cluster resolved to an incompatible concrete type, or stayed unresolved.
    InvalidLiteral {
        literal: ValueId,
        resolved: Option<TypeId>,
        message: &'static str,
    },

    /// (future) Operator rule failure.
    InvalidOperator {
        site: ValueId,
        op: BinOp,
        lhs: TypeId,
        rhs: TypeId,
        note: &'static str,
    },
}

// ===================================
// Entry point (no allocations)
// ===================================

pub fn infer_value_internals(
    program: &Program,
    store: &mut TypeStore,
    value: ValueId,
) -> Result<LocalTypes, TypeError> {
    let mut ctx = InferState::new(store, program);

    let _root = gather_constraints(&mut ctx, value)?;

    // One linear normalization pass (no extra allocations).
    ctx.normalize_clusters();

    validate_literals(&ctx)?;
    finalize(&mut ctx)?;

    Ok(ctx.ans)
}

// ===================================
// Inference state + union-find clusters
// ===================================

struct InferState<'a> {
    store: &'a mut TypeStore,
    program: &'a Program,

    // ValId -> cluster
    val_cluster: HashMap<ValueId, usize>,
    pat_cluster: HashMap<PatternId, usize>,

    // NameId -> cluster (names already resolved / qualified)
    names: HashMap<NameId, usize>,

    // union-find
    parent: Vec<usize>,
    cluster: Vec<Cluster>,

    // literal bookkeeping: keep ValId for error context
    int_lits: Vec<(ValueId, usize)>,
    float_lits: Vec<(ValueId, usize)>,

    ans: LocalTypes,
}

#[derive(Clone, Debug)]
struct Cluster {
    ty: Option<TypeId>,
    // has_int_lit: bool,
    // has_float_lit: bool,
}

impl<'a> InferState<'a> {
    fn new(store: &'a mut TypeStore, program: &'a Program) -> Self {
        Self {
            store,
            program,
            val_cluster: HashMap::new(),
            pat_cluster: HashMap::new(),
            names: HashMap::new(),
            parent: Vec::new(),
            cluster: Vec::new(),
            int_lits: Vec::new(),
            float_lits: Vec::new(),
            ans: LocalTypes::new(),
        }
    }

    fn new_cluster(&mut self) -> usize {
        let id = self.parent.len();
        self.parent.push(id);
        self.cluster.push(Cluster {
            ty: None,
            // has_int_lit: false,
            // has_float_lit: false,
        });
        id
    }

    fn bind_val(&mut self, v: ValueId, c: usize) {
        self.val_cluster.insert(v, c);
    }

    fn bind_pat(&mut self, p: PatternId, c: usize) {
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
    fn find(&mut self, x: usize) -> usize {
        let p = self.parent[x];
        if p != x {
            let r = self.find(p);
            self.parent[x] = r;
        }
        self.parent[x]
    }

    /// Normalize everything once so later phases can use parent[c] without calling find().
    fn normalize_clusters(&mut self) {
        for i in 0..self.parent.len() {
            let r = self.find(i);
            self.parent[i] = r;
        }
    }

    fn union(&mut self, a: usize, b: usize) -> Result<usize, Clash> {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return Ok(ra);
        }

        let ta = self.cluster[ra].ty;
        let tb = self.cluster[rb].ty;
        if let (Some(a), Some(b)) = (ta, tb) {
            if a != b {
                return Err(Clash { a, b });
            }
        }

        // No rank: simplest correct UF (you can add rank later if you care)
        self.parent[rb] = ra;

        let other_c = self.cluster[rb].clone();
        let root_c = &mut self.cluster[ra];

        root_c.ty = root_c.ty.or(other_c.ty);
        // root_c.has_int_lit |= other_c.has_int_lit;
        // root_c.has_float_lit |= other_c.has_float_lit;

        Ok(ra)
    }

    fn force_type(&mut self, c: usize, ty: TypeId) -> Result<(), Clash> {
        let r = self.find(c);
        match self.cluster[r].ty {
            None => {
                self.cluster[r].ty = Some(ty);
                Ok(())
            }
            Some(t) if t == ty => Ok(()),
            Some(t) => Err(Clash { a: t, b: ty }),
        }
    }

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
fn gather_constraints(ctx: &mut InferState, v: ValueId) -> Result<usize, TypeError> {
    match ctx.program.value(v) {
        Value::Literal(Literal::Num(_)) => {
            let c = ctx.new_cluster();
            // ctx.cluster[c].has_int_lit = true;
            ctx.bind_val(v, c);
            ctx.int_lits.push((v, c));
            Ok(c)
        }

        Value::Literal(Literal::Float(_)) => {
            let c = ctx.new_cluster();
            // ctx.cluster[c].has_float_lit = true;
            ctx.bind_val(v, c);
            ctx.float_lits.push((v, c));
            Ok(c)
        }

        Value::Literal(Literal::Str(_)) => {
            let c = ctx.new_cluster();
            let t = ctx.builtin(BuiltinType::Str);
            ctx.cluster[c].ty = Some(t);
            ctx.bind_val(v, c);
            Ok(c)
        }

        Value::Literal(Literal::Void) => {
            let c = ctx.new_cluster();
            let t = ctx.builtin(BuiltinType::Void);
            ctx.cluster[c].ty = Some(t);
            ctx.bind_val(v, c);
            Ok(c)
        }

        Value::NameRef(n) => {
            if let Some(&c) = ctx.names.get(&n) {
                // immediate alias: this node is the same cluster as the binding
                ctx.bind_val(v, c);
                return Ok(c);
            }

            if ctx.program.definitions.contains_key(&n) {
                todo!("global name resolution / overload sets");
            }

            unreachable!("name used before binding");
        }

        Value::TypeAnnotation { value, ty } => {
            let rhs_cluster = gather_constraints(ctx, value)?;
            let ann_ty = compile_type_expr(ctx, ty)?;

            if let Err(Clash { a, b: _ }) = ctx.force_type(rhs_cluster, ann_ty) {
                return Err(TypeError::AnnotationMismatch {
                    annotation: v,
                    constrained: value,
                    expected: ann_ty,
                    found: a,
                    note: "type annotation does not match value",
                });
            }

            // Annotation does not introduce a new type identity: alias to the value
            ctx.bind_val(v, rhs_cluster);
            Ok(rhs_cluster)
        }

        Value::Cast { value, ty } => {
            let _ = gather_constraints(ctx, value)?;
            // Cast produces a new type identity: the target type
            let c = ctx.new_cluster();
            let t = compile_type_expr(ctx, ty)?;
            ctx.cluster[c].ty = Some(t);
            ctx.bind_val(v, c);
            Ok(c)
        }

        Value::Let {
            pat,
            value,
            else_part,
        } => {
            let rhs = gather_constraints(ctx, value)?;
            let lhs = gather_pattern_constraints(ctx, pat)?;

            if let Err(Clash { a, b }) = ctx.union(lhs, rhs) {
                return Err(TypeError::IncompatibleTypes {
                    site: v,
                    left: a,
                    right: b,
                    note: "let binding types do not match",
                });
            }

            if let Some(e) = else_part {
                let ec = gather_constraints(ctx, e)?;
                if let Err(Clash { a, b }) = ctx.union(lhs, ec) {
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
            Ok(lhs)
        }

        Value::Block {
            statements,
            return_value,
        } => {
            for s in statements.ids() {
                gather_constraints(ctx, s)?;
            }

            // block aliases its return value cluster (or void)
            let c = match return_value {
                Some(r) => gather_constraints(ctx, r)?,
                None => {
                    let c = ctx.new_cluster();
                    let t = ctx.builtin(BuiltinType::Void);
                    ctx.cluster[c].ty = Some(t);
                    c
                }
            };

            ctx.bind_val(v, c);
            Ok(c)
        }

        Value::BinOp { op, values } => {
            let (lhs, rhs) = values;

            //we are assuming no overloading here.
            //TODO: this part probably needs to be pooled into a vector of these constraints
            // in paticular we might want to allow x+y to work for cases like x=i32 and y=u8
            // the main argument aginst is it makes some infrence tricky to do because we cant blindly apply same_as
            // BUT we can apply it for a few extra cases
            // mainly by using the fact literals have 1 and only 1 relation.
            // so its sound to do the following:
            //    if we have {x OP int_lit} we can require the int literal is of the same type as op

            let lc = gather_constraints(ctx, lhs)?;
            let rc = gather_constraints(ctx, rhs)?;

            // Result cluster:
            // - comparisons always produce bool
            // - arithmetic / bitwise produce a value cluster
            match op {
                // ======================
                // Comparisons: bool
                // ======================
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    // operands must be comparable -> same cluster
                    if let Err(Clash { a, b }) = ctx.union(lc, rc) {
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
                    Ok(c)
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
                    let root = match ctx.union(lc, rc) {
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
                    Ok(root)
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
            for pat in params.ids() {
                let _p = gather_pattern_constraints(ctx, pat)?;
            }

            let out_ty = if let Some(x) = output_type {
                compile_type_expr(ctx, x)?
            } else {
                BuiltinType::Void.into()
            };

            let body_cluster = gather_constraints(ctx, body)?;

            if let Err(Clash { a, b: _ }) = ctx.force_type(body_cluster, out_ty) {
                return Err(TypeError::AnnotationMismatch {
                    annotation: v,
                    constrained: body,
                    expected: out_ty,
                    found: a,
                    note: "type annotation does not match function output",
                });
            }

            let f = ctx.new_cluster();
            ctx.bind_val(v, f);
            //TODO limit f on params and out somehow
            //this might need to be done ahead of time globaly for all funcs
            //so that we can have weird type recursions
            //if thats the case this part might be just compiling cluster,(params need to be gathered so we get them in as vars we can use)
            Ok(f)
        }
        _ => panic!("more expressions {:?}", ctx.program.value(v)),
    }
}

fn gather_pattern_constraints(ctx: &mut InferState, p: PatternId) -> Result<usize, TypeError> {
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

            if let Err(Clash { a, b: _ }) = ctx.force_type(c, t) {
                return Err(TypeError::PatternAnnotationMismatch {
                    annotation: p,
                    constrained: pat,
                    expected: t,
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

fn compile_type_expr(ctx: &mut InferState, v: ValueId) -> Result<TypeId, TypeError> {
    match ctx.program.value(v) {
        Value::NameRef(n) => match ctx.program.definitions.get(&n) {
            Some(Defined::BuildinType(b)) => Ok(ctx.store.intern(b.clone())),
            Some(Defined::Type { ty, .. }) => Ok(*ty),
            _ => Err(TypeError::ExpectedType {
                type_expr: v,
                message: "expected type",
            }),
        },
        _ => Err(TypeError::ExpectedType {
            type_expr: v,
            message: "unsupported type expression",
        }),
    }
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
            ctx.ans.types.insert(v, t);
        } else {
            return Err(TypeError::Unresolved {
                value: v,
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
    fn extract_single_fn(program: &Program) -> ValueId {
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

    // #[test]
    // fn infer_empty_function() {
    //     let mut store = TypeStore::new();
    //     infer_fn("f=fn(){}", &mut store).unwrap();

    // }

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
