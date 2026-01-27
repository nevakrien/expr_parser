//! Type inference sketch (constraint-first, local, non-integrated)
//
// ================================================================
// DESIGN GOALS (SHOULD STAY TRUE)
// ================================================================
// 1) Constraints are RECORDS, not guesses.
// 2) Inference is a SEPARATE, mutable process.
// 3) Every value has exactly ONE producer, and 0..N consumers.
// 4) Errors come from constraints (produce + consume sites).
//
// ================================================================

use std::cell::Ref;
use std::cell::RefCell;
use crate::program::WithId;
use crate::ir::NameId;
use std::collections::HashMap;

use crate::{
    ir::{BinOp, IPattern, IValue, Literal, Pattern, UnOp, Value},
    parsing::Loc,
    program::{Defined, Program, ValId},
};

/* ================================================================
 * Errors (STABLE SHAPE)
 * ================================================================ */

#[derive(Debug)]
pub enum TypeError {
    Unresolved {
        produced_loc: Loc,
        message: &'static str,
    },

    SimpleMismatch {
        required_loc: Loc,
        produced_loc: Loc,
        expected: TypeId,
        found: TypeId,
        note: &'static str,
    },

    Unsupported {
        loc: Loc,
        message: &'static str,
    },

    ExpectedType {
        loc: Loc,
        message: &'static str,
    },

    InvalidOperator {
        loc: Loc,
        op: BinOp,
        lhs: TypeId,
        rhs: TypeId,
        note: &'static str,
    },
    // InvalidLiteral {
    //     loc: Loc,
    //     loc_reqired:Loc,
    //     literal: Literal,
    //     target: TypeId,
    //     note: &'static str,
    // },
}

/* ================================================================
 * Core IDs (STABLE)
 * ================================================================ */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinType {
    Int,
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
        Self {
            values: Vec::new(),
            intern: HashMap::new(),
            global_types: HashMap::new(),
        }
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

pub fn infer_value_internals(
    program: &Program,
    store: &mut TypeStore,
    value: &IValue,
) -> Result<LocalTypes, TypeError> {
    let mut state = InferState::new(store,program);

    gather_constraints(&mut state, value)?;
    basic_propegation(&mut state)?;
    finalize(&mut state)?;
    // todo!()
    Ok(state.ans)
}

/* ================================================================
 * Constraint model (CORE)
 * ================================================================ */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LocalId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum InferId {
    Resolved(TypeId),
    Local(LocalId),
    //?more
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TypeUse {
    StatedAs(InferId),
    InputArg(InferId),

    ///this was an R value to some assignment
    ///assign expressions return the var assigned to so
    WrittenTo,
    
    /// this was an L value to an assigment of some type
    TakenFrom(InferId),

    ///for SOME reason we infered this should be the same
    ///this happens when we apply union rules
    Guessed(InferId),
}


type Used = WithId<TypeUse>;//the id is of the expression causing the use

type Prod = WithId<ProducedBy>;
struct Reqs {
    //we union the guesses as much as we can
    guess:InferId,

    produced:Prod,

    //these dont have to be exustive
    //in many cases we would put them in where guess points to
    //for exaple {let x = 2: f(x) g(x)} the requirments on NameRef{x} are gona go on x
    used_as:Vec<Used>,
}

impl Reqs {
    fn add_use(&mut self,x:Used){
        self.used_as.push(x)
    }
}

struct InferState<'a> {
    store: &'a mut TypeStore,
    program:&'a Program,
    ans: LocalTypes,
    names: HashMap<NameId, InferId>,
    reqs_map:HashMap<ValId,LocalId>,
    ///this is a refcell because natually we want to look at 1 element and use it to modify children
    ///refcells are actually semantically meaningful here they stop recursion
    ///using a try_borrow is a more economical way to go about things than a visted set
    reqs:Vec<RefCell<Reqs>>,
}

type RefReqs = RefCell<Reqs>;

impl<'a> InferState<'a> {
    fn new(store: &'a mut TypeStore,program:&'a Program) -> Self {
        Self {
            store,program,
            ans: LocalTypes::new(),
            names: HashMap::new(),
            reqs:Vec::new(),
            reqs_map:HashMap::new(),
        }
    }

    fn buildin(&mut self,t:BuiltinType)->TypeId{
        self.intern(TypeValue::Builtin(t))
    }

    fn intern(&mut self,t:TypeValue)->TypeId{
        self.store.intern(t)
    }

    // fn solved(&mut self,id:ValId,t:TypeId) -> Option<TypeId>{
    //     //TODO if we ever move to multi error
    //     // this place would be weird
    //     self.ans.types.insert(id,t)
    // }

    fn register_solved(&mut self,id:ValId,t:TypeId) -> &mut Reqs{
        self.ans.types.insert(id,t);
        let p = WithId {
            id,
            value:ProducedBy::Known(t)
        };
        self.register(InferId::Resolved(t),p)
    }
    
    // fn get_local_id()    

    fn register(&mut self,guess:InferId,produced:Prod)->&mut Reqs{
        self.reqs_map.insert(produced.id,LocalId(self.reqs.len()));
        self.reqs.push(Reqs{
            guess,
            produced,
            used_as:Vec::new(),

        }.into());
        self.reqs.last_mut().unwrap().get_mut()
    }

    fn register_unknown(&mut self,produced:Prod)->&mut Reqs{
        let id = LocalId(self.reqs.len());
        self.reqs_map.insert(produced.id,id);
        self.reqs.push(Reqs{
            guess:InferId::Local(id),
            produced,
            used_as:Vec::new(),

        }.into());
        self.reqs.last_mut().unwrap().get_mut()
    }

    fn register_bind(&mut self,bind:WithId<NameId>)->&mut Reqs{
        let id = LocalId(self.reqs.len());
        self.names.insert(bind.value,InferId::Local(id));
        self.reqs_map.insert(bind.id,id);
        self.reqs.push(Reqs{
            guess:InferId::Local(id),
            produced:bind.map(ProducedBy::Bind),
            used_as:Vec::new(),

        }.into());
        self.reqs.last_mut().unwrap().get_mut()
    }


}

#[derive(Debug, Clone)]
enum ProducedBy {
    Explicit {
        ty: InferId,
    },

    Cast {
        target: InferId,
    },

    //for void/string literals blocks that return void or constants that are typed etc.
    Known(TypeId),

    IntLit,
    FloatLit,

    Let{
        tgt:ValId,
        src:ValId,
        or:Option<ValId>,
    },
    NameRef(NameId),
    Bind(NameId),

    
    
    BinOp {
        op: BinOp,
    },
    UnOp {
        op: UnOp,
    },
    Block {
        ret:ValId, //if there is no return value its known void so why care
    },
    Func {
        inputs: Vec<(NameId, InferId)>,
        output_ty: InferId,
        body: ValId,
    },
    Other(&'static str),
}





fn gather_constraints<'a>(ctx: &'a mut InferState, v: &IValue) -> Result<&'a mut Reqs, TypeError> {
    match &v.value {
        Value::TypeAnnotation { value:other, ty}=>{
            let ty = compile_type_expr(ctx,ty)?;
            let other = gather_constraints(ctx,other)?;
            other.add_use(v.with(TypeUse::StatedAs(ty)));
            Ok(ctx.register(
                ty,
                v.with(ProducedBy::Explicit{ty})
            ))
        }

        Value::Cast { value:other, ty}=>{
            let ty = compile_type_expr(ctx,ty)?;
            let _ = gather_constraints(ctx,other)?;
            Ok(ctx.register(
                ty,
                v.with(ProducedBy::Explicit{ty})
            ))
        }

        Value::Literal(Literal::Str(_))=>{
            let t = ctx.buildin(BuiltinType::Str);
            Ok(ctx.register_solved(v.id,t))
        }

        Value::Literal(Literal::Void)=>{
            let t = ctx.buildin(BuiltinType::Void);
            Ok(ctx.register_solved(v.id,t))
        }

        Value::Literal(Literal::Float(_))=>{
            Ok(ctx.register_unknown(
                v.with(ProducedBy::FloatLit)
            ))
        }

        Value::Literal(Literal::Num(_))=>{
            Ok(ctx.register_unknown(
                v.with(ProducedBy::IntLit)
            ))
        }

        Value::NameRef(n)=>{
            if let Some(guess) = ctx.names.get(n){
                let guess = *guess;
                let ans = ctx.register(
                    guess,
                    v.with(ProducedBy::NameRef(*n))
                );
                ans.add_use(v.with(TypeUse::TakenFrom(guess)));
                return Ok(ans)
            }
            if let Some(_x) = ctx.program.definitions.get(n){
                todo!("something clever")
            };

            unreachable!("BUG: name used before its declared after resolution")
        }


        Value::Let { pat, value, else_part }=>{
            let rhs = gather_constraints(ctx,value)?;
            rhs.add_use(v.with(TypeUse::WrittenTo));
            let guess = rhs.guess; 

            let else_guess =  match else_part{
                Some(x)=>{
                    let x = gather_constraints(ctx,x)?;
                    x.add_use(v.with(TypeUse::WrittenTo));
                    x.add_use(v.with(TypeUse::Guessed(guess)));
                    Some(x.guess)
                },
                None=>None,
            };

            let p = gather_pattern_constraints(ctx,pat)?;
            p.add_use(v.with(TypeUse::TakenFrom(guess)));
            if let Some(x) = else_guess {
                p.add_use(v.with(TypeUse::TakenFrom(x)));
            }

            let guess = p.guess;

            Ok(ctx.register(guess,v.with(ProducedBy::Let { 
                tgt: pat.id, 
                src: value.id,
                or:else_part.as_ref().map(|x|x.id)
            })))
        }

        Value::Block { statements, return_value} => {
            for x in statements {
                gather_constraints(ctx,x)?;
            }
             match return_value {
                None=>{
                    let void = ctx.buildin(BuiltinType::Void); 
                    Ok(ctx.register_solved(
                        v.id,void
                    ))
                }
                Some(x)=>{
                    let r = gather_constraints(ctx,x)?;
                    r.add_use(v.with(TypeUse::WrittenTo));
                    let guess = r.guess;

                    Ok(ctx.register(
                        guess,
                        v.with(ProducedBy::Block { ret: x.id})
                    ))
                }

            }
        }

        Value::Func { generics: _, params, output_type, body}=>{
            //TODO actually resolve all these and use the generics
            let inputs = params.iter().map(|_p|{
                todo!()
            }).collect::<Result<_,_>>()?;

            let output_ty = match output_type {
                None=>InferId::Resolved(ctx.buildin(BuiltinType::Void)),
                _=>todo!()
            };

            let b = gather_constraints(ctx,body)?;
            b.add_use(v.with(TypeUse::StatedAs(output_ty)));
            Ok(ctx.register_unknown(
                v.with(ProducedBy::Func{
                    output_ty,body:body.id,
                    inputs
                })

            ))
        }

        _ => todo!("more values"),
    }
}

fn gather_pattern_constraints<'a>(ctx: &'a mut InferState, p: &IPattern) -> Result<&'a mut Reqs, TypeError>{
    match &p.value {
        Pattern::TypeAnnotation { pat:other, ty } =>{
            let ty = compile_type_expr(ctx,ty)?;
            let other = gather_pattern_constraints(ctx,other)?;
            other.add_use(p.with(TypeUse::StatedAs(ty)));
            Ok(ctx.register(
                ty,
                p.with(ProducedBy::Explicit{ty})
            ))
        }

        Pattern::Bind(n)=>{
            Ok(ctx.register_bind(p.with(*n)))

        }
        _ => todo!(),
    }
}

fn compile_type_expr(ctx: &mut InferState, v: &IValue) -> Result<InferId, TypeError>{
    match &v.value {
        Value::NameRef(name) => match ctx.program.definitions.get(name) {
            Some(Defined::Type { val: _, ty }) => Ok(InferId::Resolved(*ty)),
            Some(Defined::BuildinType(b)) => Ok(InferId::Resolved(ctx.intern(b.clone()))),

            _ => Err(TypeError::ExpectedType {
                loc: ctx.program.get_loc(v.id),
                message: "expected type",
            }),
        },
        _ => Err(TypeError::ExpectedType {
            loc: ctx.program.get_loc(v.id),
            message: "unsupported type expr",
        }),
    }
}

fn basic_propegation(ctx:&mut InferState)->Result<(),TypeError>{
    let mut seen = Vec::new();//we are using refcell to mark
    let mut changed = true;

    while changed{
        changed = false;
        seen.clear();


        for cell in ctx.reqs.iter(){
            mark_one(&mut changed,ctx,cell,&mut seen)
        }
    }
    


    #[inline(always)]
    fn mark_one<'a>(changed:&mut bool,ctx:&'a InferState<'_>,cell:&'a RefCell<Reqs>,seen:&mut Vec<Ref<'a, Reqs>>){
        let Ok(mut r) = cell.try_borrow_mut() else {
            return;
        };

        let Reqs { guess, produced: _, used_as } = &mut*r;

        //1. find concrete if possible
        for u in used_as.iter(){
            let tgt = match u.value {
                TypeUse::StatedAs(x)|TypeUse::TakenFrom(x)=>x,
                TypeUse::InputArg(x) | TypeUse::Guessed(x) => x,
                TypeUse::WrittenTo=>continue,

            };

            match (*guess,tgt) {
                (InferId::Local(_),InferId::Resolved(_))=>{
                    *changed=true;
                    *guess=tgt;
                },
                (InferId::Resolved(a),InferId::Resolved(b))=>{
                    if a!=b{
                        todo!("error report here")  
                    }
                }
                _=>{},
            }

        }

        //we are done updating but we wana still push
        drop(r);
        let r = cell.borrow();
        seen.push(cell.borrow());
        let Reqs { guess, produced: _, used_as } = &*r;


        //2. push value so all the cluster has our guess
        for u in used_as.iter(){
            let tgt = match u.value {
                TypeUse::StatedAs(x)|TypeUse::TakenFrom(x)|
                TypeUse::InputArg(x) | TypeUse::Guessed(x) => {
                    match x {
                        InferId::Local(y)=>y,
                        _=>continue,
                    }
                },
                
                TypeUse::WrittenTo=>ctx.reqs_map[&u.id],

            };

            let Ok(mut other) = ctx.reqs[tgt.0].try_borrow_mut() else {
                continue;
            };
            match other.guess {
                InferId::Resolved(_) => {},
                InferId::Local(_) => {
                    if other.guess!=*guess{
                        *changed=true;
                        other.guess = *guess;
                    }
                },
            }
            mark_one(changed,ctx, &ctx.reqs[tgt.0],seen);

        }

        
    }

    Ok(())
}

fn finalize(ctx:&mut InferState)->Result<(),TypeError>{
    for cell in ctx.reqs.iter_mut(){
        let r = cell.get_mut();
        match r.guess {
            InferId::Resolved(t)=>{ctx.ans.types.insert(r.produced.id,t);},
            _=>{}//todo report an errpr
        };
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
        assert_fn_type!("f = fn(){ 1 as int }", BuiltinType::Int);
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
        assert_fn_type!("f = fn(){ 1.0 + 2.0 }", BuiltinType::F64);
    }

    /* ------------------------------------------------------------
     * Error cases
     * ------------------------------------------------------------ */

    #[test]
    fn unresolved_variable_errors() {
        let mut store = TypeStore::new();
        let err = infer_fn("f = fn(y){ let x = y; x }", &mut store).unwrap_err();
        match err {
            TypeError::Unresolved { .. } => {}
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
