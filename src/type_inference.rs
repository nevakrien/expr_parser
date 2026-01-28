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

use crate::ir::NameId;
use std::collections::HashMap;

use crate::{
    ir::{BinOp, IPattern, IValue, Literal, Pattern, Value},
    program::{Defined, Program, ValId},
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
    Int = 0,//for now this enum MUST start at 0 and we also need values in order
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
    global_types: HashMap<ValId, TypeId>,
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
    types: HashMap<ValId, TypeId>,
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
    pub fn type_of(&self, id: ValId) -> Option<TypeId> {
        self.types.get(&id).copied()
    }
}

// pub fn infer_value_internals(
//     program: &Program,
//     store: &mut TypeStore,
//     value: &IValue,
// ) -> Result<LocalTypes, TypeError> {
//     let mut state = InferState::new(store,program);

//     gather_constraints(&mut state, value)?;
//     basic_propegation(&mut state)?;
//     finalize(&mut state)?;
//     // todo!()
//     Ok(state.ans)
// }

// /* ================================================================
//  * Constraint model (CORE)
//  * ================================================================ */
// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// struct LocalId(usize);

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// enum InferId {
//     Resolved(TypeId),
//     Local(LocalId),
//     //?more
// }

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// enum BoundReason {
//     Anotated{parrent:ValId,anotation:ValId},
//     MatchArm{full:ValId,other:ValId,me:ValId},
//     ClearDerived{derived_spot:ValId,ty:InferId}
// }

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// struct TypeBound {
//     tgt:InferId,
//     reason:BoundReason,
// }

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// enum NumLit {
//     Float,
//     Int,
// }

// #[derive(Debug, Clone,Copy, PartialEq)]
// enum ResolveState {
//     Unresolved,
//     Good(TypeId),
//     Poison,
// }

// impl ResolveState{
//     fn merge(self,other:ResolveState)->(Self,Option<(TypeId,TypeId)>){
//         match(self,other){
//             (ResolveState::Good(t1), ResolveState::Good(t2)) => (ResolveState::Poison,Some((t1,t2))),
//             (ResolveState::Poison,_)|(_,ResolveState::Poison)=>(ResolveState::Poison,None),
//             (ResolveState::Unresolved,x)|(x,ResolveState::Unresolved)=>(x,None),
//         }
//     }
// }

// #[derive(Debug, Clone, PartialEq)]
// struct InferInfo {
//     bounds:Vec<BoundReason>,
//     litkind:Option<NumLit>,
//     resolved:ResolveState,
// }

// #[derive(Debug, Clone, PartialEq)]
// struct Cluster {
//     resolved:ResolveState,
//     members:LinkedList<LocalId>,
// }

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// enum TypeUse {
//     StatedAs(InferId),
//     InputArg(InferId),

//     ///this was an R value to some assignment
//     ///assign expressions return the var assigned to so
//     WrittenTo,

//     /// this was an L value to an assigment of some type
//     TakenFrom(InferId),

//     ///for SOME reason we infered this should be the same
//     ///this happens when we apply union rules
//     Guessed(InferId),
// }

// type Used = WithId<TypeUse>;//the id is of the expression causing the use

// type Prod = WithId<ProducedBy>;
// struct Reqs {
//     //we union the guesses as much as we can
//     guess:InferId,

//     produced:Prod,

//     //these dont have to be exustive
//     //in many cases we would put them in where guess points to
//     //for exaple {let x = 2: f(x) g(x)} the requirments on NameRef{x} are gona go on x
//     used_as:Vec<Used>,
// }

// impl Reqs {
//     fn add_use(&mut self,x:Used){
//         self.used_as.push(x)
//     }
// }

// struct InferState<'a> {
//     store: &'a mut TypeStore,
//     program:&'a Program,
//     ans: LocalTypes,
//     names: HashMap<NameId, InferId>,
//     reqs_map:HashMap<ValId,LocalId>,
//     ///this is a refcell because natually we want to look at 1 element and use it to modify children
//     ///refcells are actually semantically meaningful here they stop recursion
//     ///using a try_borrow is a more economical way to go about things than a visted set
//     reqs:Vec<RefCell<Reqs>>,
// }

// type RefReqs = RefCell<Reqs>;

// impl<'a> InferState<'a> {
//     fn new(store: &'a mut TypeStore,program:&'a Program) -> Self {
//         Self {
//             store,program,
//             ans: LocalTypes::new(),
//             names: HashMap::new(),
//             reqs:Vec::new(),
//             reqs_map:HashMap::new(),
//         }
//     }

//     fn buildin(&mut self,t:BuiltinType)->TypeId{
//         self.intern(TypeValue::Builtin(t))
//     }

//     fn intern(&mut self,t:TypeValue)->TypeId{
//         self.store.intern(t)
//     }

//     // fn solved(&mut self,id:ValId,t:TypeId) -> Option<TypeId>{
//     //     //TODO if we ever move to multi error
//     //     // this place would be weird
//     //     self.ans.types.insert(id,t)
//     // }

//     fn register_solved(&mut self,id:ValId,t:TypeId) -> &mut Reqs{
//         self.ans.types.insert(id,t);
//         let p = WithId {
//             id,
//             value:ProducedBy::Known(t)
//         };
//         self.register(InferId::Resolved(t),p)
//     }

//     // fn get_local_id()

//     fn register(&mut self,guess:InferId,produced:Prod)->&mut Reqs{
//         self.reqs_map.insert(produced.id,LocalId(self.reqs.len()));
//         self.reqs.push(Reqs{
//             guess,
//             produced,
//             used_as:Vec::new(),

//         }.into());
//         self.reqs.last_mut().unwrap().get_mut()
//     }

//     fn register_unknown(&mut self,produced:Prod)->&mut Reqs{
//         let id = LocalId(self.reqs.len());
//         self.reqs_map.insert(produced.id,id);
//         self.reqs.push(Reqs{
//             guess:InferId::Local(id),
//             produced,
//             used_as:Vec::new(),

//         }.into());
//         self.reqs.last_mut().unwrap().get_mut()
//     }

//     fn register_bind(&mut self,bind:WithId<NameId>)->&mut Reqs{
//         let id = LocalId(self.reqs.len());
//         self.names.insert(bind.value,InferId::Local(id));
//         self.reqs_map.insert(bind.id,id);
//         self.reqs.push(Reqs{
//             guess:InferId::Local(id),
//             produced:bind.map(ProducedBy::Bind),
//             used_as:Vec::new(),

//         }.into());
//         self.reqs.last_mut().unwrap().get_mut()
//     }

// }

// #[derive(Debug, Clone)]
// enum ProducedBy {
//     Explicit {
//         ty: InferId,
//     },

//     Cast {
//         target: InferId,
//     },

//     //for void/string literals blocks that return void or constants that are typed etc.
//     Known(TypeId),

//     IntLit,
//     FloatLit,

//     Let{
//         tgt:ValId,
//         src:ValId,
//         or:Option<ValId>,
//     },
//     NameRef(NameId),
//     Bind(NameId),

//     BinOp {
//         op: BinOp,
//     },
//     UnOp {
//         op: UnOp,
//     },
//     Block {
//         ret:ValId, //if there is no return value its known void so why care
//     },
//     Func {
//         inputs: Vec<(NameId, InferId)>,
//         output_ty: InferId,
//         body: ValId,
//     },
//     Other(&'static str),
// }

// fn gather_constraints<'a>(ctx: &'a mut InferState, v: &IValue) -> Result<&'a mut Reqs, TypeError> {
//     match &v.value {
//         Value::TypeAnnotation { value:other, ty}=>{
//             let ty = compile_type_expr(ctx,ty)?;
//             let other = gather_constraints(ctx,other)?;
//             other.add_use(v.with(TypeUse::StatedAs(ty)));
//             Ok(ctx.register(
//                 ty,
//                 v.with(ProducedBy::Explicit{ty})
//             ))
//         }

//         Value::Cast { value:other, ty}=>{
//             let ty = compile_type_expr(ctx,ty)?;
//             let _ = gather_constraints(ctx,other)?;
//             Ok(ctx.register(
//                 ty,
//                 v.with(ProducedBy::Explicit{ty})
//             ))
//         }

//         Value::Literal(Literal::Str(_))=>{
//             let t = ctx.buildin(BuiltinType::Str);
//             Ok(ctx.register_solved(v.id,t))
//         }

//         Value::Literal(Literal::Void)=>{
//             let t = ctx.buildin(BuiltinType::Void);
//             Ok(ctx.register_solved(v.id,t))
//         }

//         Value::Literal(Literal::Float(_))=>{
//             Ok(ctx.register_unknown(
//                 v.with(ProducedBy::FloatLit)
//             ))
//         }

//         Value::Literal(Literal::Num(_))=>{
//             Ok(ctx.register_unknown(
//                 v.with(ProducedBy::IntLit)
//             ))
//         }

//         Value::NameRef(n)=>{
//             if let Some(guess) = ctx.names.get(n){
//                 let guess = *guess;
//                 let ans = ctx.register(
//                     guess,
//                     v.with(ProducedBy::NameRef(*n))
//                 );
//                 ans.add_use(v.with(TypeUse::TakenFrom(guess)));
//                 return Ok(ans)
//             }
//             if let Some(_x) = ctx.program.definitions.get(n){
//                 todo!("something clever")
//             };

//             unreachable!("BUG: name used before its declared after resolution")
//         }

//         Value::Let { pat, value, else_part }=>{
//             let rhs = gather_constraints(ctx,value)?;
//             rhs.add_use(v.with(TypeUse::WrittenTo));
//             let guess = rhs.guess;

//             let else_guess =  match else_part{
//                 Some(x)=>{
//                     let x = gather_constraints(ctx,x)?;
//                     x.add_use(v.with(TypeUse::WrittenTo));
//                     x.add_use(v.with(TypeUse::Guessed(guess)));
//                     Some(x.guess)
//                 },
//                 None=>None,
//             };

//             let p = gather_pattern_constraints(ctx,pat)?;
//             p.add_use(v.with(TypeUse::TakenFrom(guess)));
//             if let Some(x) = else_guess {
//                 p.add_use(v.with(TypeUse::TakenFrom(x)));
//             }

//             let guess = p.guess;

//             Ok(ctx.register(guess,v.with(ProducedBy::Let {
//                 tgt: pat.id,
//                 src: value.id,
//                 or:else_part.as_ref().map(|x|x.id)
//             })))
//         }

//         Value::Block { statements, return_value} => {
//             for x in statements {
//                 gather_constraints(ctx,x)?;
//             }
//              match return_value {
//                 None=>{
//                     let void = ctx.buildin(BuiltinType::Void);
//                     Ok(ctx.register_solved(
//                         v.id,void
//                     ))
//                 }
//                 Some(x)=>{
//                     let r = gather_constraints(ctx,x)?;
//                     r.add_use(v.with(TypeUse::WrittenTo));
//                     let guess = r.guess;

//                     Ok(ctx.register(
//                         guess,
//                         v.with(ProducedBy::Block { ret: x.id})
//                     ))
//                 }

//             }
//         }

//         Value::Func { generics: _, params, output_type, body}=>{
//             //TODO actually resolve all these and use the generics
//             let inputs = params.iter().map(|_p|{
//                 todo!()
//             }).collect::<Result<_,_>>()?;

//             let output_ty = match output_type {
//                 None=>InferId::Resolved(ctx.buildin(BuiltinType::Void)),
//                 _=>todo!()
//             };

//             let b = gather_constraints(ctx,body)?;
//             b.add_use(v.with(TypeUse::StatedAs(output_ty)));
//             Ok(ctx.register_unknown(
//                 v.with(ProducedBy::Func{
//                     output_ty,body:body.id,
//                     inputs
//                 })

//             ))
//         }

//         _ => todo!("more values"),
//     }
// }

// fn gather_pattern_constraints<'a>(ctx: &'a mut InferState, p: &IPattern) -> Result<&'a mut Reqs, TypeError>{
//     match &p.value {
//         Pattern::TypeAnnotation { pat:other, ty } =>{
//             let ty = compile_type_expr(ctx,ty)?;
//             let other = gather_pattern_constraints(ctx,other)?;
//             other.add_use(p.with(TypeUse::StatedAs(ty)));
//             Ok(ctx.register(
//                 ty,
//                 p.with(ProducedBy::Explicit{ty})
//             ))
//         }

//         Pattern::Bind(n)=>{
//             Ok(ctx.register_bind(p.with(*n)))

//         }
//         _ => todo!(),
//     }
// }

// fn compile_type_expr(ctx: &mut InferState, v: &IValue) -> Result<InferId, TypeError>{
//     match &v.value {
//         Value::NameRef(name) => match ctx.program.definitions.get(name) {
//             Some(Defined::Type { val: _, ty }) => Ok(InferId::Resolved(*ty)),
//             Some(Defined::BuildinType(b)) => Ok(InferId::Resolved(ctx.intern(b.clone()))),

//             _ => Err(TypeError::ExpectedType {
//                 loc: ctx.program.get_loc(v.id),
//                 message: "expected type",
//             }),
//         },
//         _ => Err(TypeError::ExpectedType {
//             loc: ctx.program.get_loc(v.id),
//             message: "unsupported type expr",
//         }),
//     }
// }

// fn basic_propegation(ctx:&mut InferState)->Result<(),TypeError>{
//     let mut seen = Vec::new();//we are using refcell to mark
//     let mut changed = true;

//     while changed{
//         changed = false;
//         seen.clear();

//         for cell in ctx.reqs.iter(){
//             mark_one(&mut changed,ctx,cell,&mut seen)
//         }
//     }

//     #[inline(always)]
//     fn mark_one<'a>(changed:&mut bool,ctx:&'a InferState<'_>,cell:&'a RefCell<Reqs>,seen:&mut Vec<Ref<'a, Reqs>>){
//         let Ok(mut r) = cell.try_borrow_mut() else {
//             return;
//         };

//         let Reqs { guess, produced: _, used_as } = &mut*r;

//         //1. find concrete if possible
//         for u in used_as.iter(){
//             let tgt = match u.value {
//                 TypeUse::StatedAs(x)|TypeUse::TakenFrom(x)=>x,
//                 TypeUse::InputArg(x) | TypeUse::Guessed(x) => x,
//                 TypeUse::WrittenTo=>continue,

//             };

//             match (*guess,tgt) {
//                 (InferId::Local(_),InferId::Resolved(_))=>{
//                     *changed=true;
//                     *guess=tgt;
//                 },
//                 (InferId::Resolved(a),InferId::Resolved(b))=>{
//                     if a!=b{
//                         todo!("error report here")
//                     }
//                 }
//                 _=>{},
//             }

//         }

//         //we are done updating but we wana still push
//         drop(r);
//         let r = cell.borrow();
//         seen.push(cell.borrow());
//         let Reqs { guess, produced: _, used_as } = &*r;

//         //2. push value so all the cluster has our guess
//         for u in used_as.iter(){
//             let tgt = match u.value {
//                 TypeUse::StatedAs(x)|TypeUse::TakenFrom(x)|
//                 TypeUse::InputArg(x) | TypeUse::Guessed(x) => {
//                     match x {
//                         InferId::Local(y)=>y,
//                         _=>continue,
//                     }
//                 },

//                 TypeUse::WrittenTo=>ctx.reqs_map[&u.id],

//             };

//             let Ok(mut other) = ctx.reqs[tgt.0].try_borrow_mut() else {
//                 continue;
//             };
//             match other.guess {
//                 InferId::Resolved(_) => {},
//                 InferId::Local(_) => {
//                     if other.guess!=*guess{
//                         *changed=true;
//                         other.guess = *guess;
//                     }
//                 },
//             }
//             mark_one(changed,ctx, &ctx.reqs[tgt.0],seen);

//         }

//     }

//     Ok(())
// }

// fn finalize(ctx:&mut InferState)->Result<(),TypeError>{
//     for cell in ctx.reqs.iter_mut(){
//         let r = cell.get_mut();
//         match r.guess {
//             InferId::Resolved(t)=>{ctx.ans.types.insert(r.produced.id,t);},
//             _=>{}//todo report an errpr
//         };
//     }
//     Ok(())
// }

// ==============================
// Errors (richer + ValId-based)
// ==============================

#[derive(Debug)]
pub enum TypeError {
    /// Could not infer a concrete type for this value
    Unresolved { value: ValId, message: &'static str },

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
// Entry point (no allocations)
// ===================================

pub fn infer_value_internals(
    program: &Program,
    store: &mut TypeStore,
    value: &IValue,
) -> Result<LocalTypes, TypeError> {
    let mut ctx = InferState::new(store, program);

    let _root = gather_constraints(&mut ctx, value)?;

    // One linear normalization pass (no extra allocations).
    ctx.normalize_clusters();

    validate_literals(&ctx)?;
    finalize(&mut ctx);

    Ok(ctx.ans)
}

// ===================================
// Inference state + union-find clusters
// ===================================

struct InferState<'a> {
    store: &'a mut TypeStore,
    program: &'a Program,

    // ValId -> cluster
    val_cluster: HashMap<ValId, usize>,

    // NameId -> cluster (names already resolved / qualified)
    names: HashMap<NameId, usize>,

    // union-find
    parent: Vec<usize>,
    cluster: Vec<Cluster>,

    // literal bookkeeping: keep ValId for error context
    int_lits: Vec<(ValId, usize)>,
    float_lits: Vec<(ValId, usize)>,

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

    fn bind_val(&mut self, v: ValId, c: usize) {
        self.val_cluster.insert(v, c);
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
fn gather_constraints(ctx: &mut InferState, v: &IValue) -> Result<usize, TypeError> {
    match &v.value {
        Value::Literal(Literal::Num(_)) => {
            let c = ctx.new_cluster();
            // ctx.cluster[c].has_int_lit = true;
            ctx.bind_val(v.id, c);
            ctx.int_lits.push((v.id, c));
            Ok(c)
        }

        Value::Literal(Literal::Float(_)) => {
            let c = ctx.new_cluster();
            // ctx.cluster[c].has_float_lit = true;
            ctx.bind_val(v.id, c);
            ctx.float_lits.push((v.id, c));
            Ok(c)
        }

        Value::Literal(Literal::Str(_)) => {
            let c = ctx.new_cluster();
            let t = ctx.builtin(BuiltinType::Str);
            ctx.cluster[c].ty = Some(t);
            ctx.bind_val(v.id, c);
            Ok(c)
        }

        Value::Literal(Literal::Void) => {
            let c = ctx.new_cluster();
            let t = ctx.builtin(BuiltinType::Void);
            ctx.cluster[c].ty = Some(t);
            ctx.bind_val(v.id, c);
            Ok(c)
        }

        Value::NameRef(n) => {
            if let Some(&c) = ctx.names.get(n) {
                // immediate alias: this node is the same cluster as the binding
                ctx.bind_val(v.id, c);
                return Ok(c);
            }

            if ctx.program.definitions.contains_key(n) {
                todo!("global name resolution / overload sets");
            }

            unreachable!("name used before binding");
        }

        Value::TypeAnnotation { value, ty } => {
            let rhs_cluster = gather_constraints(ctx, value)?;
            let ann_ty = compile_type_expr(ctx, ty)?;

            if let Err(Clash { a, b: _ }) = ctx.force_type(rhs_cluster, ann_ty) {
                return Err(TypeError::AnnotationMismatch {
                    annotation: v.id,
                    constrained: value.id,
                    expected: ann_ty,
                    found: a,
                    note: "type annotation does not match value",
                });
            }

            // Annotation does not introduce a new type identity: alias to the value
            ctx.bind_val(v.id, rhs_cluster);
            Ok(rhs_cluster)
        }

        Value::Cast { value, ty } => {
            let _ = gather_constraints(ctx, value)?;
            // Cast produces a new type identity: the target type
            let c = ctx.new_cluster();
            let t = compile_type_expr(ctx, ty)?;
            ctx.cluster[c].ty = Some(t);
            ctx.bind_val(v.id, c);
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
                    site: v.id,
                    left: a,
                    right: b,
                    note: "let binding types do not match",
                });
            }

            if let Some(e) = else_part {
                let ec = gather_constraints(ctx, e)?;
                if let Err(Clash { a, b }) = ctx.union(lhs, ec) {
                    return Err(TypeError::IncompatibleTypes {
                        site: e.id,
                        left: a,
                        right: b,
                        note: "let-else requires the else value to match the pattern type",
                    });
                }
            }

            // let-expr evaluates to the bound pattern value => alias
            ctx.bind_val(v.id, lhs);
            Ok(lhs)
        }

        Value::Block {
            statements,
            return_value,
        } => {
            for s in statements {
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

            ctx.bind_val(v.id, c);
            Ok(c)
        }

        Value::BinOp { op, values } => {
            let (lhs, rhs) = &**values;

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
                            site: v.id,
                            left: a,
                            right: b,
                            note: "comparison operands must have the same type",
                        });
                    }

                    let c = ctx.new_cluster();
                    let t = ctx.builtin(BuiltinType::Bool);
                    ctx.cluster[c].ty = Some(t);
                    ctx.bind_val(v.id, c);
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
                                site: v.id,
                                left: a,
                                right: b,
                                note: "binary operator requires operands of the same type",
                            });
                        }
                    };

                    // Now: literal handling (currently a no op)
                    // when we add overloading we need to check here that we actualyl merge literals explictly
                    ctx.bind_val(v.id, root);
                    Ok(root)
                }
            }
        }

        _ => panic!("more expressions {:?}", v.value),
    }
}

fn gather_pattern_constraints(ctx: &mut InferState, p: &IPattern) -> Result<usize, TypeError> {
    match &p.value {
        Pattern::Bind(n) => {
            let c = ctx.new_cluster();
            ctx.names.insert(*n, c);
            ctx.bind_val(p.id, c);
            Ok(c)
        }

        Pattern::TypeAnnotation { pat, ty } => {
            let c = gather_pattern_constraints(ctx, pat)?;
            let t = compile_type_expr(ctx, ty)?;

            if let Err(Clash { a, b: _ }) = ctx.force_type(c, t) {
                return Err(TypeError::AnnotationMismatch {
                    annotation: p.id,
                    constrained: pat.id,
                    expected: t,
                    found: a,
                    note: "pattern annotation does not match the value bound here",
                });
            }

            Ok(c)
        }

        _ => todo!(),
    }
}

fn compile_type_expr(ctx: &mut InferState, v: &IValue) -> Result<TypeId, TypeError> {
    match &v.value {
        Value::NameRef(n) => match ctx.program.definitions.get(n) {
            Some(Defined::BuildinType(b)) => Ok(ctx.store.intern(b.clone())),
            Some(Defined::Type { ty, .. }) => Ok(*ty),
            _ => Err(TypeError::ExpectedType {
                type_expr: v.id,
                message: "expected type",
            }),
        },
        _ => Err(TypeError::ExpectedType {
            type_expr: v.id,
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

fn finalize(ctx: &mut InferState) {
    // ctx.parent[] already normalized
    for (&v, &c) in ctx.val_cluster.iter() {
        let r = ctx.parent[c];
        if let Some(t) = ctx.cluster[r].ty {
            ctx.ans.types.insert(v, t);
        }
    }
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
    fn extract_single_fn(program: &Program) -> &IValue {
        program
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
        let body = match &f.value {
            Value::Func { body, .. } => body,
            _ => panic!("expected function value"),
        };

        let types = infer_value_internals(&program, store, f)?;
        Ok(types.type_of(body.id).unwrap())
    }

    //this is a hack for just testing
    fn infer_fn_body(src: &str, store: &mut TypeStore) -> Result<TypeId, TypeError> {
        let program = gather_program(src);
        let f = extract_single_fn(&program);
        let body = match &f.value {
            Value::Func { body, .. } => body,
            _ => panic!("expected function value"),
        };

        let types = infer_value_internals(&program, store, body)?;
        Ok(types.type_of(body.id).unwrap())
    }

    macro_rules! assert_fn_type {
        ($src:expr, $builtin:expr) => {{
            let mut store = TypeStore::new();
            let ty = infer_fn_body($src, &mut store).expect("inference failed");
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
