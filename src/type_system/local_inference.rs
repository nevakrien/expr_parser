use crate::data_structures::identity_hasher::IdHashMap;
use crate::data_structures::index::IndexVec;
use crate::error_reporting::ErrorReporter;
use crate::ir::{
    AssignOp, GenDec, LifeTimeId, Literal, NameId, PatId, Pattern, TExpId, TypeExpr, ValId, Value,
    VarKind,
};
use crate::program::{Defined, FunctionSet, Program};
use crate::type_system::{
    GenId, InnerFunctionTypes, KindId, LifeId, LifeKind, MutId, Nullable, Origin, OriginId,
    OriginKind, OutliveReason, PointerStyle, Projection, SolvedFunctionTypes, SolvedTypes,
    TypeError, TypeKind, TypeUniverse, UniversalLifeId,
};
use std::error::Error;

type Gathered = (KindId, Option<OriginId>);

#[derive(Debug, Clone, Copy)]
struct BindingInfo {
    kind: KindId,
    origin: Option<OriginId>,
}

#[derive(Debug, Clone, Copy)]
struct GlobalFunctionInfo {
    site: ValId,
}

#[derive(Debug, Default)]
struct TypeLoweringContext {
    lifetimes: IdHashMap<LifeTimeId, LifeId>,
    generics: IdHashMap<NameId, KindId>,
}

pub fn run_typechecker_impl(
    program: &Program,
    reporter: &mut ErrorReporter,
) -> Result<(Result<(TypeUniverse, SolvedTypes), usize>, usize), Box<dyn Error>> {
    let mut driver = ConstraintGatherer::new(program);
    driver.gather_global_functions();

    if driver.errors.is_empty() {
        Ok((Ok((driver.types, driver.solved)), driver.checked))
    } else {
        for error in &driver.errors {
            reporter.report_type_error(program, &driver.types, error)?;
        }
        Ok((Err(driver.errors.len()), driver.checked))
    }
}

struct ConstraintGatherer<'a> {
    program: &'a Program,
    types: TypeUniverse,
    solved: SolvedTypes,
    errors: Vec<TypeError>,
    checked: usize,
    global_functions: IdHashMap<NameId, GlobalFunctionInfo>,
}

impl<'a> ConstraintGatherer<'a> {
    fn new(program: &'a Program) -> Self {
        Self {
            program,
            types: TypeUniverse::new(),
            solved: SolvedTypes::new(program),
            errors: Vec::new(),
            checked: 0,
            global_functions: IdHashMap::default(),
        }
    }

    fn gather_global_functions(&mut self) {
        for (&name, defined) in &self.program.definitions {
            if let Defined::Func(functions) = defined {
                self.gather_function_set_signatures(name, functions, None);
            }
        }

        for (&struct_name, methods) in &self.program.member_methods {
            for (&member_name, functions) in methods {
                self.gather_function_set_signatures(struct_name, functions, Some(member_name));
            }
        }

        for defined in self.program.definitions.values() {
            if let Defined::Func(functions) = defined {
                self.gather_function_set_bodies(functions);
            }
        }

        for methods in self.program.member_methods.values() {
            for functions in methods.values() {
                self.gather_function_set_bodies(functions);
            }
        }
    }

    fn gather_function_set_signatures(
        &mut self,
        name: NameId,
        functions: &FunctionSet,
        member_name: Option<crate::data_structures::string_intern::StrId>,
    ) {
        let mut declaration_sites = Vec::new();
        declaration_sites.extend(functions.declarations.iter().copied());
        declaration_sites.extend(functions.implementations.iter().copied());

        for site in functions.values() {
            let Some(signature) = self.gather_function_signature(site, &declaration_sites) else {
                continue;
            };

            if member_name.is_none() {
                // TODO(new-type-system): this is a placeholder for overload/signature
                // selection. Silently keeping the first signature is order-dependent.
                self.global_functions
                    .entry(name)
                    .or_insert(GlobalFunctionInfo { site });
                // TODO(new-type-system): report duplicate/incompatible visible
                // function signatures instead of silently keeping the first one.
                self.solved.function_types.entry(name).or_insert(site);
            } else if let Some(member_name) = member_name {
                // TODO(new-type-system): member overload selection should not be an
                // order-dependent first-signature cache.
                self.solved
                    .member_function_types
                    .entry((name, member_name))
                    .or_insert(site);
            }

            self.solved.function_values.insert(site, signature);
        }
    }

    fn gather_function_set_bodies(&mut self, functions: &FunctionSet) {
        for site in functions.implementations.iter().copied() {
            self.gather_function_body(site);
        }
    }

    fn gather_function_signature(
        &mut self,
        site: ValId,
        declaration_sites: &[ValId],
    ) -> Option<SolvedFunctionTypes> {
        let Value::Func {
            generics,
            params,
            output_type,
            ..
        } = self.program.value(site)
        else {
            self.errors.push(TypeError::Simple {
                loc: self.program.value_loc(site),
                message: "expected function definition",
            });
            return None;
        };

        let mut lifetime_parameters = IndexVec::new();
        let mut lifetime_edges = Vec::new();
        let type_ctx = lower_signature_lifetimes(
            self.program,
            &mut self.types,
            generics,
            &mut self.checked,
            Some(&mut lifetime_parameters),
            Some(&mut lifetime_edges),
        );

        let mut arguments = Vec::with_capacity(params.len());
        let mut param_kinds = Vec::with_capacity(params.len());
        for pat in params.ids() {
            let kind = gather_type_annotation_on_pattern(
                self.program,
                &mut self.types,
                pat,
                &type_ctx,
                &mut self.errors,
                &mut self.checked,
            );
            param_kinds.push(kind);
            arguments.push((pat, binding_name(self.program, pat), kind));
        }

        let ret = output_type
            .map(|ty| {
                gather_type_expr(
                    self.program,
                    &mut self.types,
                    ty,
                    &type_ctx,
                    &mut self.errors,
                    &mut self.checked,
                )
            })
            .unwrap_or(KindId::VOID);

        let params = self.types.intern_kind_span(param_kinds);
        let ty = self.types.intern(TypeKind::Func { params, ret });

        Some(SolvedFunctionTypes {
            ty,
            impl_site: declaration_sites.iter().copied().find(|candidate| {
                matches!(
                    self.program.value(*candidate),
                    Value::Func { body: Some(_), .. }
                )
            }),
            declaration_sites: declaration_sites.to_vec(),
            arguments,
            generic_parameters: generics
                .generics()
                .ids()
                .map(|pat| (pat, binding_name(self.program, pat)))
                .collect(),
            lifetime_parameters,
            lifetime_edges,
            inner: None,
        })
    }

    fn gather_function_body(&mut self, site: ValId) {
        let Value::Func {
            params,
            output_type,
            body,
            ..
        } = self.program.value(site)
        else {
            return;
        };
        let Some(body) = body else {
            return;
        };

        let mut body_gatherer =
            BodyConstraintGatherer::new(self.program, &self.global_functions, site);
        let signature = body_gatherer.specialize_function_signature(site);
        let Some(TypeKind::Func {
            params: signature_params,
            ret,
        }) = body_gatherer.types.get(signature).copied()
        else {
            return;
        };

        for (idx, pat) in params.ids().enumerate() {
            if idx < signature_params.len() {
                body_gatherer.bind_function_parameter(pat, signature_params.at(idx));
            }
        }
        body_gatherer.output_kind = Some(ret);

        let (body_kind, _) = body_gatherer.gather_constraints(body);
        if let Err(clash) = body_gatherer.types.unify(body_kind, ret) {
            let constrained = match self.program.value(body) {
                Value::Block {
                    return_value: Some(ret),
                    ..
                } => ret,
                _ => body,
            };
            body_gatherer
                .errors
                .push(TypeError::FunctionOutputAnnotationMismatch {
                    output_type,
                    constrained,
                    clash,
                });
        }

        self.checked += body_gatherer.checked;
        self.errors.append(&mut body_gatherer.errors);

        if let Some(function) = self.solved.function_values.get_mut(&site) {
            function.inner = Some(body_gatherer.finish());
        }
    }
}

struct BodyConstraintGatherer<'a> {
    program: &'a Program,
    types: TypeUniverse,
    inner: InnerFunctionTypes,
    errors: Vec<TypeError>,
    checked: usize,
    bindings: IdHashMap<NameId, BindingInfo>,
    global_functions: &'a IdHashMap<NameId, GlobalFunctionInfo>,
    function_site: ValId,
    output_kind: Option<KindId>,
}

impl<'a> BodyConstraintGatherer<'a> {
    fn new(
        program: &'a Program,
        global_functions: &'a IdHashMap<NameId, GlobalFunctionInfo>,
        function_site: ValId,
    ) -> Self {
        Self {
            program,
            types: TypeUniverse::new(),
            inner: InnerFunctionTypes::default(),
            errors: Vec::new(),
            checked: 0,
            bindings: IdHashMap::default(),
            global_functions,
            function_site,
            output_kind: None,
        }
    }

    fn finish(mut self) -> InnerFunctionTypes {
        let _ = self.function_site;
        self.inner.my_universe = self.types;
        self.inner
    }

    fn unify_annotation(
        &mut self,
        annotation: ValId,
        constrained: ValId,
        found: KindId,
        wanted: KindId,
    ) {
        if let Err(clash) = self.types.unify(found, wanted) {
            self.errors.push(TypeError::AnnotationMismatch {
                annotation,
                constrained,
                clash,
            });
        }
    }

    fn unify_pattern_annotation(
        &mut self,
        annotation: PatId,
        constrained: PatId,
        found: KindId,
        wanted: KindId,
    ) {
        if let Err(clash) = self.types.unify(found, wanted) {
            self.errors.push(TypeError::PatternAnnotationMismatch {
                annotation,
                constrained,
                clash,
            });
        }
    }

    fn specialize_function_signature(&mut self, site: ValId) -> KindId {
        todo!()
    }

    fn unify_or_todo(&mut self, found: KindId, wanted: KindId, context: &'static str) {
        if let Err(clash) = self.types.unify(found, wanted) {
            todo!("new type system: report {context}: {clash:?}");
        }
    }

    fn bind_function_parameter(&mut self, pat: PatId, kind: KindId) {
        let origin = self.types.add_origin(Some(Origin {
            kind: Some(OriginKind::FuncArg { name: pat }),
            mutability: mutability_of_pattern(self.program, pat),
        }));
        self.bind_pattern(pat, kind, Some(origin));
    }

    fn gather_constraints(&mut self, value: ValId) -> Gathered {
        self.checked += 1;

        // TODO(new-type-system): this walker only gathers obvious equalities and
        // provenance roots today. Overload resolution, member/index lookup, and
        // richer borrow/lifetime obligations still need dedicated logic here.
        let gathered = match self.program.value(value) {
            Value::NameRef(name) => self.lookup_name(name),
            Value::Labeled { value, .. } => self.gather_constraints(value),
            Value::Literal(lit) => self.literal(lit, value),
            Value::Wildcard => (self.types.add_empty(), None),
            Value::LabelDecl(_) | Value::Goto(_) | Value::Break | Value::Continue => {
                (self.types.add_empty(), None)
            }
            Value::Tuple(items) => {
                let item_kinds: Vec<_> = items
                    .ids()
                    .map(|item| self.gather_constraints(item).0)
                    .collect();
                let span = self.types.intern_kind_span(item_kinds);
                (self.types.intern(TypeKind::Tuple(span)), None)
            }
            Value::Array(items) => {
                let inner = self.types.add_empty();
                for item in items.ids() {
                    let (item_kind, _) = self.gather_constraints(item);
                    self.unify_or_todo(item_kind, inner, "array element type mismatch");
                }
                (
                    self.types.intern(TypeKind::Array {
                        inner,
                        size: Some(crate::type_system::ArraySize::Sized(items.len())),
                    }),
                    None,
                )
            }
            Value::BinOp { values, .. } => {
                let _ = self.gather_constraints(values.0);
                let _ = self.gather_constraints(values.1);
                // TODO(new-type-system): route through operator selection instead
                // of returning an unconstrained placeholder kind.
                (self.types.add_empty(), None)
            }
            Value::UnOp { value, .. } => {
                let _ = self.gather_constraints(value);
                // TODO(new-type-system): route through operator selection instead
                // of returning an unconstrained placeholder kind.
                (self.types.add_empty(), None)
            }
            Value::Deref(base) => {
                let (kind, origin) = self.gather_constraints(base);
                // TODO(new-type-system): require pointer-like shape and project the
                // pointee type instead of leaving this unconstrained.
                (
                    self.types.add_empty(),
                    origin.or_else(|| self.ensure_place_origin(base, kind)),
                )
            }
            Value::AddrOf(base, kind) => {
                let (base_kind, base_origin) = self.gather_constraints(base);
                let origin = base_origin.or_else(|| self.ensure_place_origin(base, base_kind));
                (
                    make_pointer_kind(&mut self.types, base_kind, kind, None),
                    origin,
                )
            }
            Value::Construct(call) | Value::Call(call) => {
                let result_kind = self.gather_call_like(value, call.base, call.args.ids());
                (
                    result_kind,
                    Some(self.types.add_origin(Some(Origin {
                        kind: Some(OriginKind::Transient { val: value }),
                        mutability: MutId::FALSE,
                    }))),
                )
            }
            Value::Cast { value: inner, ty } => {
                let (_, origin) = self.gather_constraints(inner);
                let ty = gather_type_expr(
                    self.program,
                    &mut self.types,
                    ty,
                    &TypeLoweringContext::default(),
                    &mut self.errors,
                    &mut self.checked,
                );
                let origin = origin.map(|parent| {
                    self.types.add_origin(Some(Origin {
                        kind: Some(OriginKind::Derived {
                            parent,
                            proj: Projection::Casted,
                        }),
                        mutability: MutId::FALSE,
                    }))
                });
                (ty, origin)
            }
            Value::TypeAnnotation { value: inner, ty } => {
                let (inner_kind, origin) = self.gather_constraints(inner);
                let annotated = gather_type_expr(
                    self.program,
                    &mut self.types,
                    ty,
                    &TypeLoweringContext::default(),
                    &mut self.errors,
                    &mut self.checked,
                );
                self.unify_annotation(value, inner, inner_kind, annotated);
                (annotated, origin)
            }
            Value::TypeDef { pat, ty } => {
                let kind = gather_type_expr(
                    self.program,
                    &mut self.types,
                    ty,
                    &TypeLoweringContext::default(),
                    &mut self.errors,
                    &mut self.checked,
                );
                self.bind_pattern(pat, kind, None);
                (KindId::TYPE, None)
            }
            Value::Assign { op, target } => {
                let (target_kind, _) = self.gather_constraints(target);
                match op {
                    AssignOp::Nothing(value) | AssignOp::Bin(_, value) => {
                        let (value_kind, _) = self.gather_constraints(value);
                        self.unify_or_todo(value_kind, target_kind, "assignment type mismatch");
                    }
                    AssignOp::Pre(_) | AssignOp::Post(_) => {}
                }
                (KindId::VOID, None)
            }
            Value::Index(call) => {
                self.gather_call_like(value, call.base, call.args.ids());
                // TODO(new-type-system): preserve the right place/transient origin
                // and constrain the indexed element type here.
                (self.types.add_empty(), None)
            }
            Value::Access { base, .. } => {
                let (base_kind, base_origin) = self.gather_constraints(base);
                // TODO(new-type-system): perform member lookup and project the real
                // field/member type here.
                (
                    self.types.add_empty(),
                    base_origin.or_else(|| self.ensure_place_origin(base, base_kind)),
                )
            }
            Value::IntAccess { base, id, .. } => {
                let (base_kind, base_origin) = self.gather_constraints(base);
                let origin = base_origin
                    .or_else(|| self.ensure_place_origin(base, base_kind))
                    .map(|parent| {
                        self.types.add_origin(Some(Origin {
                            kind: Some(OriginKind::Derived {
                                parent,
                                proj: Projection::FieldReref(id as u32),
                            }),
                            mutability: MutId::FALSE,
                        }))
                    });
                // TODO(new-type-system): constrain the projected tuple/field type.
                (self.types.add_empty(), origin)
            }
            Value::Let {
                pat,
                value: rhs,
                else_part,
            } => {
                let (rhs_kind, rhs_origin) = self.gather_constraints(rhs);
                self.bind_pattern(pat, rhs_kind, rhs_origin);
                if let Some(else_part) = else_part {
                    let _ = self.gather_constraints(else_part);
                }
                (KindId::VOID, None)
            }
            Value::Block {
                statements,
                return_value,
            } => {
                for stmt in statements.ids() {
                    let _ = self.gather_constraints(stmt);
                }
                return_value
                    .map(|ret| self.gather_constraints(ret))
                    .unwrap_or((KindId::VOID, None))
            }
            Value::LogicOp { values, .. } => {
                let (lhs, _) = self.gather_constraints(values.0);
                let (rhs, _) = self.gather_constraints(values.1);
                self.unify_or_todo(lhs, KindId::BOOL, "logic lhs must be bool");
                self.unify_or_todo(rhs, KindId::BOOL, "logic rhs must be bool");
                (KindId::BOOL, None)
            }
            Value::If { cond, then, els } => {
                let (cond_kind, _) = self.gather_constraints(cond);
                self.unify_or_todo(cond_kind, KindId::BOOL, "if condition must be bool");
                let (then_kind, _) = self.gather_constraints(then);
                if let Some(els) = els {
                    let (else_kind, else_origin) = self.gather_constraints(els);
                    self.unify_or_todo(then_kind, else_kind, "if branch type mismatch");
                    (then_kind, else_origin)
                } else {
                    (KindId::VOID, None)
                }
            }
            Value::While { cond, body } => {
                let (cond_kind, _) = self.gather_constraints(cond);
                self.unify_or_todo(cond_kind, KindId::BOOL, "while condition must be bool");
                let _ = self.gather_constraints(body);
                (KindId::VOID, None)
            }
            Value::Func { .. } => {
                self.errors.push(TypeError::Simple {
                    loc: self.program.value_loc(value),
                    message: "sorry we dont support closures",
                });
                (self.types.add_empty(), None)
            }
            Value::Return(ret) => {
                if let Some(ret) = ret {
                    let (ret_kind, _) = self.gather_constraints(ret);
                    if let Some(expected) = self.output_kind {
                        self.unify_or_todo(ret_kind, expected, "return type mismatch");
                    }
                }
                (KindId::VOID, None)
            }
            Value::Match { value, arms } => {
                let (scrutinee, _) = self.gather_constraints(value);
                let result = self.types.add_empty();
                for arm in arms.ids() {
                    let (arm_kind, _) = self.gather_match_arm(arm, scrutinee);
                    self.unify_or_todo(arm_kind, result, "match arm type mismatch");
                }
                (result, None)
            }
            Value::MatchArm(_) | Value::Poison => (self.types.add_empty(), None),
        };

        self.inner.val_types.insert(value, gathered.0);
        if let Some(origin) = gathered.1 {
            self.inner.value_origins.insert(value, origin);
        }
        gathered
    }

    fn gather_match_arm(&mut self, arm: ValId, scrutinee: KindId) -> Gathered {
        let Value::MatchArm(arm) = self.program.value(arm) else {
            return (self.types.add_empty(), None);
        };
        self.bind_pattern(arm.pat, scrutinee, None);
        self.gather_constraints(arm.body)
    }

    fn bind_pattern(&mut self, pat: PatId, expected: KindId, origin: Option<OriginId>) {
        self.inner.pat_types.insert(pat, expected);
        if let Some(origin) = origin {
            self.inner.pattern_origins.insert(pat, origin);
        }

        match self.program.pattern(pat) {
            Pattern::Bind(name, _) => {
                let origin = origin.or_else(|| {
                    Some(self.types.add_origin(Some(Origin {
                        kind: Some(OriginKind::Local { name: pat }),
                        mutability: mutability_of_pattern(self.program, pat),
                    })))
                });
                self.bindings.insert(
                    name,
                    BindingInfo {
                        kind: expected,
                        origin,
                    },
                );
            }
            Pattern::Wildcard(_) | Pattern::Literal(_) | Pattern::LifeTime(_) | Pattern::Poison => {
            }
            Pattern::Tuple(items) => {
                let item_kinds: Vec<_> = items.ids().map(|_| self.types.add_empty()).collect();
                let tuple_items = self.types.intern_kind_span(item_kinds.iter().copied());
                let tuple_kind = self.types.intern(TypeKind::Tuple(tuple_items));
                self.unify_or_todo(expected, tuple_kind, "tuple pattern type mismatch");
                for (item, item_kind) in items.ids().zip(item_kinds.into_iter()) {
                    self.bind_pattern(item, item_kind, origin);
                }
            }
            Pattern::TypeAnnotation { pat: inner_pat, ty } => {
                let annotated = gather_type_expr(
                    self.program,
                    &mut self.types,
                    ty,
                    &TypeLoweringContext::default(),
                    &mut self.errors,
                    &mut self.checked,
                );
                self.unify_pattern_annotation(pat, inner_pat, expected, annotated);
                self.bind_pattern(inner_pat, annotated, origin);
            }
            Pattern::AddrOf(inner, kind) => {
                let pointed_tgt = self.types.add_empty();
                let pointed = make_pointer_kind(&mut self.types, pointed_tgt, Some(kind), None);
                self.unify_or_todo(expected, pointed, "addr-of pattern type mismatch");
                self.bind_pattern(inner, pointed_tgt, origin);
            }
        }
    }

    fn lookup_name(&mut self, name: NameId) -> Gathered {
        if let Some(binding) = self.bindings.get(&name).copied() {
            return (binding.kind, binding.origin);
        }

        if let Some(global) = self.global_functions.get(&name).copied() {
            let signature = self.specialize_function_signature(global.site);
            let origin = self.types.add_origin(Some(Origin {
                kind: Some(OriginKind::Global { val: global.site }),
                mutability: MutId::FALSE,
            }));
            return (signature, Some(origin));
        }

        (self.types.add_empty(), None)
    }

    fn literal(&mut self, lit: Literal, value: ValId) -> Gathered {
        match lit {
            Literal::Num(_) => (KindId::INT, None),
            Literal::Float(_) => (KindId::FLOAT, None),
            Literal::Bool(_) => (KindId::BOOL, None),
            Literal::Str(_) => (KindId::STR, None),
            Literal::Void => (KindId::VOID, None),
            Literal::Null => {
                let tgt = self.types.add_empty();
                let style = self
                    .types
                    .add_ptr_style(Some(PointerStyle::Raw(Some(Nullable::Yes))));
                let kind = self.types.intern(TypeKind::Ptr {
                    tgt,
                    style,
                    mutable: MutId::FALSE,
                });
                let origin = self.types.add_origin(Some(Origin {
                    kind: Some(OriginKind::Transient { val: value }),
                    mutability: MutId::FALSE,
                }));
                (kind, Some(origin))
            }
        }
    }

    fn gather_call_like(
        &mut self,
        site: ValId,
        base: ValId,
        args: impl IntoIterator<Item = ValId>,
    ) -> KindId {
        let (base_kind, _) = self.gather_constraints(base);
        let arg_kinds: Vec<_> = args
            .into_iter()
            .map(|arg| self.gather_constraints(arg).0)
            .collect();

        let Some(TypeKind::Func { params, ret }) = self.types.get(base_kind).copied() else {
            return self.types.add_empty();
        };

        if arg_kinds.len() != params.len() {
            todo!(
                "new type system: report call argument count mismatch: expected {}, found {}",
                params.len(),
                arg_kinds.len()
            );
        }

        for (idx, arg_kind) in arg_kinds.into_iter().enumerate() {
            if idx >= params.len() {
                unreachable!("argument count checked above");
            }
            self.unify_or_todo(arg_kind, params.at(idx), "call argument type mismatch");
        }

        // TODO(new-type-system): this only covers already-function-shaped callees
        // and positional argument equality. Named args and overload selection still
        // belong here.
        let _ = site;
        ret
    }

    fn ensure_place_origin(&mut self, site: ValId, _kind: KindId) -> Option<OriginId> {
        Some(self.types.add_origin(Some(Origin {
            kind: Some(OriginKind::Transient { val: site }),
            mutability: MutId::FALSE,
        })))
    }
}

fn binding_name(program: &Program, pat: PatId) -> Option<NameId> {
    match program.pattern(pat) {
        Pattern::Bind(name, _) => Some(name),
        Pattern::TypeAnnotation { pat, .. } => binding_name(program, pat),
        Pattern::AddrOf(pat, _) => binding_name(program, pat),
        _ => None,
    }
}

fn mut_from_var_kind(kind: Option<VarKind>) -> MutId {
    match kind {
        Some(VarKind::Mut) => MutId::TRUE,
        Some(VarKind::Const) | None => MutId::FALSE,
    }
}

fn mutability_of_pattern(program: &Program, pat: PatId) -> MutId {
    match program.pattern(pat) {
        Pattern::Bind(_, kind) | Pattern::Wildcard(kind) | Pattern::AddrOf(_, kind) => {
            mut_from_var_kind(Some(kind))
        }
        Pattern::TypeAnnotation { pat, .. } => mutability_of_pattern(program, pat),
        Pattern::Tuple(items) => items
            .ids()
            .map(|item| mutability_of_pattern(program, item))
            .find(|mutability| *mutability == MutId::TRUE)
            .unwrap_or(MutId::FALSE),
        Pattern::Literal(_) | Pattern::LifeTime(_) | Pattern::Poison => MutId::FALSE,
    }
}

fn lower_signature_lifetimes(
    program: &Program,
    types: &mut TypeUniverse,
    generics: GenDec,
    checked: &mut usize,
    mut lifetime_parameters: Option<
        &mut IndexVec<UniversalLifeId, Option<(PatId, Option<LifeTimeId>)>>,
    >,
    mut lifetime_edges: Option<&mut Vec<(UniversalLifeId, UniversalLifeId, OutliveReason)>>,
) -> TypeLoweringContext {
    let mut ctx = TypeLoweringContext::default();
    let mut by_source = IdHashMap::default();
    // This `IndexVec` is keyed by real `UniversalLifeId`s, so for now we reserve
    // slot 0 for `'static` and accept the tiny heap waste. If this becomes hot,
    // we can split the index space so declared parameters stay dense.
    let mut life_ids: IndexVec<UniversalLifeId, LifeId> = IndexVec::new();

    let static_life = types.add_static_life();
    let static_universal = life_ids.push(static_life);
    debug_assert_eq!(static_universal, UniversalLifeId::STATIC);
    if let Some(params) = lifetime_parameters.as_deref_mut() {
        let pushed = params.push(None);
        debug_assert_eq!(pushed, UniversalLifeId::STATIC);
    }

    for pat in generics.lifetimes().ids() {
        let Pattern::LifeTime(life) = program.pattern(pat) else {
            continue;
        };

        let universal = UniversalLifeId(life_ids.len() as u32);
        let pushed = life_ids.push(types.add_life(Some(LifeKind::Univeral(Some(universal)))));
        debug_assert_eq!(pushed, universal);
        if let Some(params) = lifetime_parameters.as_deref_mut() {
            let pushed = params.push(Some((pat, Some(life))));
            debug_assert_eq!(pushed, universal);
        }
        by_source.insert(life, universal);
        ctx.lifetimes.insert(life, life_ids[pushed]);
    }

    for constraint in generics.where_clause().ids() {
        *checked += 1;
        let TypeExpr::Lt { lhs, rhs } = program.type_expr(constraint) else {
            continue;
        };
        let TypeExpr::LifeTime(shorter) = program.type_expr(lhs) else {
            continue;
        };
        let TypeExpr::LifeTime(longer) = program.type_expr(rhs) else {
            continue;
        };
        let Some(&shorter) = by_source.get(&shorter) else {
            continue;
        };
        let Some(&longer) = by_source.get(&longer) else {
            continue;
        };

        if let Some(edges) = lifetime_edges.as_deref_mut() {
            edges.push((longer, shorter, OutliveReason));
        }
        types.require_lifetime_outlives(life_ids[longer], life_ids[shorter], OutliveReason);
    }

    // Bind generic type parameters so that body-level type expressions
    // (e.g. annotations on parameters or within the function body) can
    // refer to generic type names.
    for pat in generics.generics().ids() {
        let gen_id = GenId(ctx.generics.len() as u32);
        let kind_id = types.intern(TypeKind::Generic(gen_id));
        if let Pattern::Bind(name, _) = program.pattern(pat) {
            ctx.generics.insert(name, kind_id);
        }
    }

    ctx
}

fn gather_type_annotation_on_pattern(
    program: &Program,
    types: &mut TypeUniverse,
    pat: PatId,
    ctx: &TypeLoweringContext,
    errors: &mut Vec<TypeError>,
    checked: &mut usize,
) -> KindId {
    match program.pattern(pat) {
        Pattern::TypeAnnotation { ty, .. } => {
            gather_type_expr(program, types, ty, ctx, errors, checked)
        }
        Pattern::AddrOf(inner, kind) => {
            let tgt =
                gather_type_annotation_on_pattern(program, types, inner, ctx, errors, checked);
            make_pointer_kind(types, tgt, Some(kind), None)
        }
        _ => types.add_empty(),
    }
}

fn make_pointer_kind(
    types: &mut TypeUniverse,
    tgt: KindId,
    kind: Option<VarKind>,
    lifetime: Option<LifeId>,
) -> KindId {
    let life = lifetime.unwrap_or_else(|| types.add_life(Some(LifeKind::Local)));
    let style = types.add_ptr_style(Some(PointerStyle::Ref(life)));
    types.intern(TypeKind::Ptr {
        tgt,
        style,
        mutable: mut_from_var_kind(kind),
    })
}

fn gather_type_expr(
    program: &Program,
    types: &mut TypeUniverse,
    expr: TExpId,
    ctx: &TypeLoweringContext,
    errors: &mut Vec<TypeError>,
    checked: &mut usize,
) -> KindId {
    *checked += 1;

    match program.type_expr(expr) {
        TypeExpr::NameRef(name) => {
            // Generic type parameters shadow global definitions
            if let Some(&kind) = ctx.generics.get(&name) {
                return kind;
            }
            match program.definitions.get(&name) {
                Some(Defined::BuildinType(kind)) => *kind,
                Some(Defined::Type(texp)) => {
                    // Recursively lower the alias RHS in the current universe
                    gather_type_expr(program, types, *texp, ctx, errors, checked)
                }
                _ => {
                    errors.push(TypeError::ExpectedTypeExpr { type_expr: expr });
                    types.add_empty()
                }
            }
        }
        TypeExpr::Ptr {
            base,
            lifetime,
            raw,
            mutable,
        } => {
            let tgt = gather_type_expr(program, types, base, ctx, errors, checked);
            let style = if raw {
                types.add_ptr_style(Some(PointerStyle::Raw(Some(Nullable::Yes))))
            } else {
                let life = lifetime
                    .map(|life| lower_lifetime(types, life, ctx))
                    .unwrap_or_else(|| types.add_life(Some(LifeKind::Local)));
                types.add_ptr_style(Some(PointerStyle::Ref(life)))
            };

            types.intern(TypeKind::Ptr {
                tgt,
                style,
                mutable: mut_from_var_kind(mutable.then_some(VarKind::Mut)),
            })
        }
        TypeExpr::Tuple(items) => {
            let kinds: Vec<_> = items
                .ids()
                .map(|item| gather_type_expr(program, types, item, ctx, errors, checked))
                .collect();
            let span = types.intern_kind_span(kinds);
            types.intern(TypeKind::Tuple(span))
        }
        TypeExpr::Array(base, size) => {
            let inner = gather_type_expr(program, types, base, ctx, errors, checked);
            types.intern(TypeKind::Array {
                inner,
                size: size.map(crate::type_system::ArraySize::Sized),
            })
        }
        TypeExpr::Func {
            params,
            output_type,
            ..
        } => {
            let param_kinds: Vec<_> = params
                .ids()
                .map(|param| gather_type_expr(program, types, param, ctx, errors, checked))
                .collect();
            let ret = output_type
                .map(|ret| gather_type_expr(program, types, ret, ctx, errors, checked))
                .unwrap_or(KindId::VOID);
            let span = types.intern_kind_span(param_kinds);
            types.intern(TypeKind::Func { params: span, ret })
        }
        TypeExpr::LifeTime(_) => types.add_empty(),
        TypeExpr::Index { .. }
        | TypeExpr::Lt { .. }
        | TypeExpr::Struct(_)
        | TypeExpr::Enum(_)
        | TypeExpr::Union(_)
        | TypeExpr::Wildcard
        | TypeExpr::Poison => {
            errors.push(TypeError::ExpectedTypeExpr { type_expr: expr });
            types.add_empty()
        }
    }
}

fn lower_lifetime(types: &mut TypeUniverse, life: LifeTimeId, ctx: &TypeLoweringContext) -> LifeId {
    if let Some(&life) = ctx.lifetimes.get(&life) {
        return life;
    }

    match life {
        LifeTimeId::STATIC => types.add_static_life(),
        LifeTimeId::RAW => types.add_life(Some(LifeKind::Local)),
        LifeTimeId::WILDCARD => types.add_life(Some(LifeKind::Local)),
        _ => types.add_life(Some(LifeKind::Local)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_structures::index::Idx;
    use crate::parsing::Parser;
    use crate::type_system::{BuiltinKind, UniversalLifeId};

    fn gather_program(src: &str) -> Program {
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        program.lower_all(&mut parser).unwrap();
        program
    }

    fn run_gatherer(program: &Program) -> ConstraintGatherer<'_> {
        let mut gatherer = ConstraintGatherer::new(program);
        gatherer.gather_global_functions();
        gatherer
    }

    fn find_name(program: &Program, name: &str) -> NameId {
        program
            .definitions
            .keys()
            .copied()
            .find(|id| program.str_intern.resolve(program.name_str_id(*id)) == name)
            .unwrap()
    }

    fn impl_site_for_name(program: &Program, name: &str) -> ValId {
        let name = find_name(program, name);
        let Defined::Func(functions) = &program.definitions[&name] else {
            panic!("expected function definition")
        };
        *functions.implementations.first().unwrap()
    }

    fn expect_builtin(types: &mut TypeUniverse, kind: KindId, expected: BuiltinKind) {
        assert_eq!(types.get(kind), Some(&TypeKind::Builtin(expected)));
    }

    fn expect_tuple_of_builtins(types: &mut TypeUniverse, kind: KindId, expected: &[BuiltinKind]) {
        let Some(TypeKind::Tuple(items)) = types.get(kind).copied() else {
            panic!("expected tuple kind")
        };
        assert_eq!(items.len(), expected.len());
        for (idx, expected) in expected.iter().copied().enumerate() {
            expect_builtin(types, items.at(idx), expected);
        }
    }

    #[test]
    fn lowers_external_function_signature_into_local_call_context() {
        let program = gather_program("id = cfn(x:int) -> int; f = fn() -> int { id(1:int) };");
        let mut gatherer = run_gatherer(&program);
        assert!(
            gatherer.errors.is_empty(),
            "unexpected errors: {:?}",
            gatherer.errors
        );

        let f_site = impl_site_for_name(&program, "f");
        let inner = gatherer
            .solved
            .function_values
            .get_mut(&f_site)
            .unwrap()
            .inner
            .as_mut()
            .unwrap();
        let Value::Func {
            body: Some(body), ..
        } = program.value(f_site)
        else {
            panic!("expected function body")
        };
        let Value::Block {
            return_value: Some(call),
            ..
        } = program.value(body)
        else {
            panic!("expected block return call")
        };
        let Value::Call(call_data) = program.value(call) else {
            panic!("expected call expression")
        };

        let callee_kind = inner.val_types[&call_data.base];
        let Some(TypeKind::Func { params, ret }) = inner.my_universe.get(callee_kind).copied()
        else {
            panic!("expected lowered local function kind")
        };
        assert_eq!(params.len(), 1);
        expect_builtin(
            &mut inner.my_universe,
            params.at(0),
            BuiltinKind::Int(crate::type_system::IntKind {
                size: Some(crate::type_system::IntSize::Int),
                sign: Some(crate::type_system::IntSign::Signed),
            }),
        );
        expect_builtin(
            &mut inner.my_universe,
            ret,
            BuiltinKind::Int(crate::type_system::IntKind {
                size: Some(crate::type_system::IntSize::Int),
                sign: Some(crate::type_system::IntSign::Signed),
            }),
        );
        expect_builtin(
            &mut inner.my_universe,
            inner.val_types[&call],
            BuiltinKind::Int(crate::type_system::IntKind {
                size: Some(crate::type_system::IntSize::Int),
                sign: Some(crate::type_system::IntSign::Signed),
            }),
        );
    }

    #[test]
    fn local_output_annotation_reports_mismatch_after_signature_lowering() {
        let program = gather_program("id = cfn(x:bool) -> bool; f = fn() -> int { id(false) };");
        let gatherer = run_gatherer(&program);

        assert_eq!(
            gatherer.errors.len(),
            1,
            "unexpected errors: {:?}",
            gatherer.errors
        );
        assert!(matches!(
            &gatherer.errors[0],
            TypeError::FunctionOutputAnnotationMismatch {
                clash,
                ..
            } if clash.found() == Some("bool") && clash.wanted() == Some("int")
        ));
    }

    #[test]
    fn lowering_specialization_does_not_reuse_global_tuple_kind_ids() {
        let program = gather_program("pair = cfn() -> (int, int);");
        let mut gatherer = run_gatherer(&program);
        assert!(
            gatherer.errors.is_empty(),
            "unexpected errors: {:?}",
            gatherer.errors
        );

        let pair_name = find_name(&program, "pair");
        let pair_signature = gatherer.solved.function_types_by_name(pair_name).unwrap();
        let Some(TypeKind::Func {
            ret: global_ret, ..
        }) = gatherer.types.get(pair_signature.ty).copied()
        else {
            panic!("expected global pair signature")
        };
        expect_tuple_of_builtins(
            &mut gatherer.types,
            global_ret,
            &[
                BuiltinKind::Int(crate::type_system::IntKind {
                    size: Some(crate::type_system::IntSize::Int),
                    sign: Some(crate::type_system::IntSign::Signed),
                }),
                BuiltinKind::Int(crate::type_system::IntKind {
                    size: Some(crate::type_system::IntSize::Int),
                    sign: Some(crate::type_system::IntSign::Signed),
                }),
            ],
        );

        let pair_site = gatherer.global_functions[&pair_name].site;
        let mut body = BodyConstraintGatherer::new(&program, &gatherer.global_functions, pair_site);
        while body.types.storage.types.storage.len() < global_ret.index() {
            body.types.add_empty();
        }
        let local_items = body.types.intern_kind_span([KindId::BOOL, KindId::STR]);
        let local_tuple = body.types.intern(TypeKind::Tuple(local_items));
        assert_eq!(local_tuple.index(), global_ret.index());
        expect_tuple_of_builtins(
            &mut body.types,
            local_tuple,
            &[BuiltinKind::Bool, BuiltinKind::Str],
        );

        let specialized = body.specialize_function_signature(pair_site);
        let Some(TypeKind::Func {
            ret: lowered_ret, ..
        }) = body.types.get(specialized).copied()
        else {
            panic!("expected specialized local function kind")
        };
        assert_ne!(lowered_ret, local_tuple);
        expect_tuple_of_builtins(
            &mut body.types,
            lowered_ret,
            &[
                BuiltinKind::Int(crate::type_system::IntKind {
                    size: Some(crate::type_system::IntSize::Int),
                    sign: Some(crate::type_system::IntSign::Signed),
                }),
                BuiltinKind::Int(crate::type_system::IntKind {
                    size: Some(crate::type_system::IntSize::Int),
                    sign: Some(crate::type_system::IntSign::Signed),
                }),
            ],
        );
    }

    #[test]
    fn signature_lifetime_universals_reserve_zero_for_static() {
        let program = gather_program("id = fn['a, 'b, where 'a < 'b](x: &'a int) -> &'b int;");
        let gatherer = run_gatherer(&program);
        assert!(
            gatherer.errors.is_empty(),
            "unexpected errors: {:?}",
            gatherer.errors
        );

        let signature = gatherer
            .solved
            .function_types_by_name(find_name(&program, "id"))
            .unwrap();

        assert_eq!(signature.lifetime_parameters.len(), 3);
        assert_eq!(signature.lifetime_parameters[UniversalLifeId::STATIC], None);
        assert!(signature.lifetime_parameters[UniversalLifeId(1)].is_some());
        assert!(signature.lifetime_parameters[UniversalLifeId(2)].is_some());
        assert_eq!(
            signature.lifetime_edges,
            vec![(UniversalLifeId(2), UniversalLifeId(1), OutliveReason)]
        );
    }

    #[test]
    fn local_pattern_annotation_reports_addr_of_pointee_mismatch() {
        let program = gather_program("f = fn(x:bool) { let y:&int = &x; };");
        let gatherer = run_gatherer(&program);

        assert!(
            gatherer
                .errors
                .iter()
                .any(|error| matches!(error, TypeError::PatternAnnotationMismatch { .. }))
        );
    }

    #[test]
    fn tuple_pattern_propagates_bound_item_types() {
        let program = gather_program("f = fn() -> int { let (x, y) = (1:int, false); y };");
        let gatherer = run_gatherer(&program);

        assert!(matches!(
            gatherer.errors.as_slice(),
            [TypeError::FunctionOutputAnnotationMismatch { clash, .. }]
                if clash.found() == Some("bool") && clash.wanted() == Some("int")
        ));
    }

    #[test]
    fn addr_of_pattern_propagates_inner_type() {
        let program = gather_program("f = fn() -> int { let &x = &false; x };");
        let gatherer = run_gatherer(&program);

        assert!(matches!(
            gatherer.errors.as_slice(),
            [TypeError::FunctionOutputAnnotationMismatch { clash, .. }]
                if clash.found() == Some("bool") && clash.wanted() == Some("int")
        ));
    }

    #[test]
    fn invalid_type_expr_in_signature_reports_type_error() {
        let program = gather_program("f = fn(x:_) {};");
        let gatherer = run_gatherer(&program);

        assert!(
            gatherer
                .errors
                .iter()
                .any(|error| matches!(error, TypeError::ExpectedTypeExpr { .. }))
        );
    }
}
