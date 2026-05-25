use crate::identity_hasher::IdHashMap;
use crate::ir::CallingConvention;
use crate::ir::LifeTimeId;
use crate::ir::StructLayoutSpec;
use crate::ir::StructLike;
use crate::ir::VarKind;
use crate::ir::{GenDec, NameId, PatId, Pattern, PatternSpan, TExpId, TypeExpr, ValId, Value};
use crate::lifetime_graph::{
    LifetimeGraphId, collect_decl_lifetime_orderings, solve_where_clause_lifetime_scc,
};
use crate::string_intern::{
    ADD_STR, ALIGN_OF_STR, BITAND_STR, BITNOT_STR, BITOR_STR, BITXOR_STR, DEREF_MUT_STR, DEREF_STR,
    DIV_STR, DSIZED_STR, EQ_STR, FORGET_STR, FREE_STR, GE_STR, GT_STR, LE_STR, LT_STR, MOD_STR,
    MUL_STR, NE_STR, NEG_STR, NOT_STR, POST_DEC_STR, POST_INC_STR, PRE_DEC_STR, PRE_INC_STR,
    SHL_STR, SHR_STR, SIZE_OF_STR, SUB_STR, StrId, USER_FREE_STR,
};

use crate::program::{Defined, FunctionSet, Program};
///this function gathers global typedefs/structs
use crate::type_inference::*;

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

        // Global typedef/struct resolution intentionally uses generated fallback names.
        // We currently do not rely on rich clash rendering here because this flow mostly
        // reports unresolved/simple diagnostics instead of function-context type clashes.
        ctx.ex.name_render = GenLifeNameRender::Generate;

        //structs have to all resolve in the same scope so they see eachother
        //but we need to preserve them to have their own lifetime...
        //this is 100% a hack but because structs are so simple in terms of lifetimes it should work
        ctx.types.lifetimes.next_undeclared_lifetime = 0;
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

    global_solver(&mut ctx);

    for (_n, def) in program.definitions.iter() {
        let Defined::Type(texp) = def else {
            continue;
        };
        check_unused_struct_signature_generics_and_lifetimes(&mut ctx, *texp);
    }

    if !ctx.ex.errors.is_empty() {
        return Err(ctx.ex.errors);
    }

    for (struct_name, methods) in program.member_methods.iter() {
        for (_method_name, method_set) in methods.iter() {
            for m in method_set.values() {
                //each function must solve by itself.
                //since there isnt a body its fine to solve in order
                //note that namespace on generics gurntees this works for the most outer scope
                if let Value::Func {
                    calling_convention,
                    generics,
                    params,
                    output_type,
                    body: _,
                } = ctx.ex.program.value(m)
                {
                    ctx.clear_local_state();
                    type_check_func_signature(
                        &mut ctx,
                        m,
                        calling_convention,
                        generics,
                        params,
                        output_type,
                    );
                    check_unused_function_signature_generics_and_lifetimes(&mut ctx, m);
                };
            }
        }

        let mut overloads = StructOverloadInfo::default();
        let mut pending_deref_methods = PendingStructDerefMethods::default();
        for (method_name, method_set) in methods.iter() {
            let Some((reference_type, reference_site)) = check_and_record_function_set_types(
                &mut ctx,
                *struct_name,
                Some(*method_name),
                method_set,
            ) else {
                continue;
            };

            validate_and_insert_member_overload(
                &mut ctx,
                &mut overloads,
                &mut pending_deref_methods,
                *struct_name,
                *method_name,
                reference_site,
                reference_type,
            );
        }

        if overloads.has_any() {
            ctx.ex
                .store
                .struct_overloads
                .insert(*struct_name, overloads);
        }
    }

    for (name, def) in program.definitions.iter() {
        let Defined::Func(funcs) = def else {
            continue;
        };

        for v in funcs.values() {
            //each function must solve by itself.
            //since there isnt a body its fine to solve in order
            //note that namespace on generics gurntees this works for the most outer scope
            if let Value::Func {
                calling_convention,
                generics,
                params,
                output_type,
                body: _,
            } = ctx.ex.program.value(v)
            {
                ctx.clear_local_state();
                type_check_func_signature(
                    &mut ctx,
                    v,
                    calling_convention,
                    generics,
                    params,
                    output_type,
                );
                check_unused_function_signature_generics_and_lifetimes(&mut ctx, v);
            };
        }

        check_and_record_function_set_types(&mut ctx, *name, None, funcs);
    }

    if ctx.ex.errors.is_empty() {
        Ok(ctx.ex.ans)
    } else {
        Err(ctx.ex.errors)
    }
}

fn global_solver(ctx: &mut InferState) {
    loop {
        let mut progress = false;
        // progress |= resolve_deferred_types(ctx);
        progress |= resolve_pending_specializations(ctx);

        if !progress {
            break;
        }
    }

    full_resolve_deferred_types(ctx);
    let _ = resolve_pending_sized_requirements(ctx);

    if !ctx.ex.errors.is_empty() {
        return;
    }

    finalize_global(ctx);
    validate_global_where_clause_requirements(ctx);
}

fn validate_global_where_clause_requirements(ctx: &mut InferState) {
    validate_struct_storage_where_clause_requirements(ctx);
    validate_function_argument_where_clause_requirements(ctx);
}

fn validate_struct_storage_where_clause_requirements(ctx: &mut InferState) {
    for (_name, def) in ctx.ex.program.definitions.iter() {
        let Defined::Type(type_expr) = def else {
            continue;
        };
        let TypeExpr::Struct(def) = ctx.ex.program.type_expr(*type_expr) else {
            continue;
        };
        let Some(ty) = ctx.ex.ans.typedef_types.get(type_expr).copied() else {
            continue;
        };
        let TypeValue::Struct { id, .. } = *ctx.ex.store.type_value(ty) else {
            continue;
        };

        let previous_name_render = std::mem::replace(
            &mut ctx.ex.name_render,
            GenLifeNameRender::from_decl(ctx.ex.program, def.generics),
        );

        let (life_count, field_len, solve, reachability) = {
            let struct_rep = ctx.ex.store.struct_value(id);
            let life_count = struct_rep.life_count;
            let field_len = struct_rep.fields.len();
            let (_graph, solve, reachability) =
                solve_decl_where_clause_graph(life_count, &struct_rep.lifetime_orderings);
            (life_count, field_len, solve, reachability)
        };

        for field_index in 0..field_len {
            let field_pat = def.fields.at(field_index);
            let field_ty = ctx.ex.store.struct_value(id).fields[field_index].1;
            let field_loc = match ctx.ex.program.pattern(field_pat) {
                Pattern::TypeAnnotation { ty, .. } => ctx.ex.program.type_expr_loc(ty),
                _ => ctx.ex.program.pattern_loc(field_pat),
            };
            validate_type_uses_allowed_lifetime_orderings(
                ctx,
                &field_loc,
                field_ty,
                &solve.component_of,
                &reachability,
                life_count,
                "struct field type",
            );
        }

        ctx.ex.name_render = previous_name_render;
    }
}

fn validate_function_argument_where_clause_requirements(ctx: &mut InferState) {
    for (_name, def) in ctx.ex.program.definitions.iter() {
        let Defined::Func(funcs) = def else {
            continue;
        };
        for function in funcs.values() {
            validate_single_function_argument_where_clause_requirements(ctx, function);
        }
    }

    for (_struct_name, methods) in ctx.ex.program.member_methods.iter() {
        for (_method_name, method_set) in methods.iter() {
            for function in method_set.values() {
                validate_single_function_argument_where_clause_requirements(ctx, function);
            }
        }
    }
}

fn validate_single_function_argument_where_clause_requirements(
    ctx: &mut InferState,
    function: ValId,
) {
    let Value::Func {
        generics, params, ..
    } = ctx.ex.program.value(function)
    else {
        return;
    };

    let Some(ty) = ctx.ex.ans.function_types_by_value(function).map(|f| f.ty) else {
        return;
    };
    let (lifetime_count, param_len, solve, reachability) = {
        let TypeValue::Func {
            lifetimes,
            lifetime_orderings,
            params: solved_params,
            ..
        } = ctx.ex.store.type_value(ty)
        else {
            return;
        };
        let (_graph, solve, reachability) =
            solve_decl_where_clause_graph(*lifetimes, lifetime_orderings);
        (*lifetimes, solved_params.len(), solve, reachability)
    };

    let previous_name_render = std::mem::replace(
        &mut ctx.ex.name_render,
        GenLifeNameRender::from_decl(ctx.ex.program, generics),
    );

    for param_index in 0..param_len {
        let param_pat = params.at(param_index);
        let param_ty = match ctx.ex.store.type_value(ty) {
            TypeValue::Func { params, .. } => params[param_index],
            _ => unreachable!(),
        };
        let param_loc = ctx.ex.program.pattern_loc(param_pat);
        validate_type_uses_allowed_lifetime_orderings(
            ctx,
            &param_loc,
            param_ty,
            &solve.component_of,
            &reachability,
            lifetime_count,
            "function parameter type",
        );
    }

    ctx.ex.name_render = previous_name_render;
}

fn solve_decl_where_clause_graph(
    lifetime_count: usize,
    lifetime_orderings: &[LifetimeOrderingEdge],
) -> (
    crate::lifetime_graph::LifetimeOrderingGraph,
    crate::lifetime_graph::LifetimeSccSolve,
    Vec<Vec<bool>>,
) {
    let graph = collect_decl_lifetime_orderings(lifetime_count, lifetime_orderings);
    let solve = solve_where_clause_lifetime_scc(&graph);
    let component_count = solve.components.len();
    let mut reachability = vec![vec![false; component_count]; component_count];

    for component_index in 0..component_count {
        reachability[component_index][component_index] = true;
    }

    for edge in graph.where_clause_edges() {
        let shorter_component = solve.component_of[edge.shorter.0];
        let longer_component = solve.component_of[edge.longer.0];
        reachability[shorter_component][longer_component] = true;
    }

    for mid in 0..component_count {
        for src in 0..component_count {
            if !reachability[src][mid] {
                continue;
            }
            for dst in 0..component_count {
                reachability[src][dst] |= reachability[mid][dst];
            }
        }
    }

    (graph, solve, reachability)
}

fn validate_type_uses_allowed_lifetime_orderings(
    ctx: &mut InferState,
    loc: &crate::parsing::Loc,
    ty: TypeId,
    component_of: &[usize],
    reachability: &[Vec<bool>],
    owner_lifetime_count: usize,
    site_kind: &'static str,
) {
    match ctx.ex.store.type_value(ty) {
        TypeValue::Struct {
            id,
            generics,
            lifetimes,
        } => {
            let generic_len = generics.len();
            {
                let struct_rep = ctx.ex.store.struct_value(*id);
                let mut missing_edges = Vec::with_capacity(struct_rep.lifetime_orderings.len());
                for edge in &struct_rep.lifetime_orderings {
                    let Some(shorter) = lifetimes.get(edge.shorter.0).copied() else {
                        continue;
                    };
                    let Some(longer) = lifetimes.get(edge.longer.0).copied() else {
                        continue;
                    };
                    if declaration_allows_lifetime_ordering(
                        shorter,
                        longer,
                        component_of,
                        reachability,
                        owner_lifetime_count,
                    ) {
                        continue;
                    }
                    missing_edges.push((shorter, longer));
                }

                for (shorter, longer) in missing_edges {
                    let shorter_name = lifetime_name_for_decl_error(ctx, shorter);
                    let longer_name = lifetime_name_for_decl_error(ctx, longer);
                    let found_ty = ctx.ex.store.get_type_string(ctx.ex.program, ty);
                    ctx.push_error(TypeError::LifetimeError {
                        loc: loc.clone(),
                        message: format!(
                            "missing where-clause requirement '{shorter_name} < '{longer_name} for {site_kind} `{found_ty}`"
                        ),
                        label: format!(
                            "`{found_ty}` requires '{shorter_name} < '{longer_name}"
                        ),
                        related: None,
                        related_label: None,
                    });
                }
            }

            for generic_index in 0..generic_len {
                let generic = match ctx.ex.store.type_value(ty) {
                    TypeValue::Struct { generics, .. } => generics[generic_index],
                    _ => unreachable!(),
                };
                validate_type_uses_allowed_lifetime_orderings(
                    ctx,
                    loc,
                    generic,
                    component_of,
                    reachability,
                    owner_lifetime_count,
                    site_kind,
                );
            }
        }
        TypeValue::Tuple(items) => {
            let item_len = items.len();
            for item_index in 0..item_len {
                let item = match ctx.ex.store.type_value(ty) {
                    TypeValue::Tuple(items) => items[item_index],
                    _ => unreachable!(),
                };
                validate_type_uses_allowed_lifetime_orderings(
                    ctx,
                    loc,
                    item,
                    component_of,
                    reachability,
                    owner_lifetime_count,
                    site_kind,
                );
            }
        }
        TypeValue::Array(element, _) => {
            let element = *element;
            validate_type_uses_allowed_lifetime_orderings(
                ctx,
                loc,
                element,
                component_of,
                reachability,
                owner_lifetime_count,
                site_kind,
            )
        }
        TypeValue::Ptr { tgt, .. } => {
            let tgt = *tgt;
            validate_type_uses_allowed_lifetime_orderings(
                ctx,
                loc,
                tgt,
                component_of,
                reachability,
                owner_lifetime_count,
                site_kind,
            )
        }
        TypeValue::Func { params, ret, .. } => {
            let ret = *ret;
            let param_len = params.len();
            for param_index in 0..param_len {
                let param = match ctx.ex.store.type_value(ty) {
                    TypeValue::Func { params, .. } => params[param_index],
                    _ => unreachable!(),
                };
                validate_type_uses_allowed_lifetime_orderings(
                    ctx,
                    loc,
                    param,
                    component_of,
                    reachability,
                    owner_lifetime_count,
                    site_kind,
                );
            }
            validate_type_uses_allowed_lifetime_orderings(
                ctx,
                loc,
                ret,
                component_of,
                reachability,
                owner_lifetime_count,
                site_kind,
            );
        }
        TypeValue::Builtin(_) | TypeValue::Generic(_, _) => {}
    }
}

fn declaration_allows_lifetime_ordering(
    shorter: LifeTime,
    longer: LifeTime,
    component_of: &[usize],
    reachability: &[Vec<bool>],
    owner_lifetime_count: usize,
) -> bool {
    if shorter == longer {
        return true;
    }

    if matches!(
        shorter.partial_cmp(&longer),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    ) {
        return true;
    }

    let (LifeTime::External(shorter), LifeTime::External(longer)) = (shorter, longer) else {
        return false;
    };

    let shorter = shorter as usize;
    let longer = longer as usize;
    if shorter >= owner_lifetime_count || longer >= owner_lifetime_count {
        return false;
    }

    let shorter_component = component_of[LifetimeGraphId(shorter).0];
    let longer_component = component_of[LifetimeGraphId(longer).0];
    reachability[shorter_component][longer_component]
}

fn lifetime_name_for_decl_error(ctx: &InferState, lt: LifeTime) -> String {
    lifetime_for_display(&ctx.ex, lt)
}

#[inline(always)]
// #[inline(never)]
// #[unsafe(no_mangle)]
fn finalize_global(ctx: &mut InferState) {
    let InferState {
        search, types, ex, ..
    } = ctx;

    let mut reported: IdHashMap<CId, ()> = IdHashMap::default();

    for (e, c) in &search.typedef_cluster {
        let root = types.root(*c);
        if let ResolveKind::Solved(t) = types.cluster_state(root) {
            ex.ans.typedef_types.insert(*e, t);
        } else if *c == root {
            let found = types.bad_type(ex, root);
            ex.errors
                .push(TypeError::UnresolvedTypeExpr { expr: *e, found });
            reported.insert(root, ());
        }
    }

    let struct_count = types.extra.struct_defs.len();
    for sid_i in 0..struct_count {
        let (loc_expr, field_len) = {
            let s = &types.extra.struct_defs[sid_i];
            (s.loc, s.fields.len())
        };

        for i in 0..field_len {
            let c = types.extra.struct_defs[sid_i].fields[i].1;
            let root = types.root(c);

            if let ResolveKind::Solved(t) = types.cluster_state(root) {
                ex.store.structs[StructId(sid_i)].fields[i].1 = t;
            } else if c == root {
                let loc = ex.program.type_expr_loc(loc_expr);
                ex.errors.push(TypeError::Simple {
                    loc,
                    message: "could not infer struct field type",
                });
                reported.insert(root, ());
            }
        }
    }

    let mut pat_type_by_id: IdHashMap<PatId, TypeId> = IdHashMap::default();
    for (p, c) in &search.pat_cluster {
        let root = types.root(*c);
        if let ResolveKind::Solved(t) = types.cluster_state(root) {
            pat_type_by_id.insert(*p, t);
        } else if *c == root && !reported.contains_key(c) {
            let found = types.bad_type(ex, root);
            ex.errors
                .push(TypeError::UnresolvedPattern { pattern: *p, found });
            reported.insert(root, ());
        }
    }

    for (v, c) in &search.val_cluster {
        let root = types.root(*c);
        if let ResolveKind::Solved(t) = types.cluster_state(root) {
            let Value::Func {
                generics,
                params,
                body,
                ..
            } = ex.program.value(*v)
            else {
                continue;
            };

            let arguments = params
                .ids()
                .map(|pat| {
                    (
                        pat,
                        pattern_bind_name(ex.program, pat),
                        pat_type_by_id.get(&pat).copied().unwrap_or(UNKNOWN_TYPE),
                    )
                })
                .collect::<Vec<_>>();

            let generic_parameters = generics
                .generics()
                .ids()
                .map(|pat| (pat, pattern_bind_name(ex.program, pat)))
                .collect::<Vec<_>>();

            let lifetime_parameters = generics
                .lifetimes()
                .ids()
                .map(|pat| {
                    (
                        pat,
                        match ex.program.pattern(pat) {
                            Pattern::LifeTime(lt) => Some(lt),
                            _ => None,
                        },
                    )
                })
                .collect::<Vec<_>>();

            ex.ans.set_function_signature(
                *v,
                SolvedFunctionTypes {
                    ty: t,
                    impl_site: body.map(|_| *v),
                    declaration_sites: body.is_none().then_some(*v).into_iter().collect(),
                    arguments,
                    generic_parameters,
                    lifetime_parameters,
                    inner: None,
                },
            );
        } else if *c == root && !reported.contains_key(c) {
            let found = types.bad_type(ex, root);
            ex.errors.push(TypeError::Unresolved { value: *v, found });
            reported.insert(root, ());
        }
    }
}

// ----------------------------------------------------------
// Function Set Recording
// ----------------------------------------------------------
fn check_and_record_function_set_types(
    ctx: &mut InferState,
    name: NameId,
    method_str: Option<StrId>,
    functions: &FunctionSet,
) -> Option<(TypeId, ValId)> {
    let first_decl = functions.declarations.first().copied();
    let first_impl = functions.implementations.first().copied();

    let (reference_site, reference_type, first_decl_site) = if let Some(decl) = first_decl {
        let Some(reference_type) = ctx.ex.ans.function_types_by_value(decl).map(|f| f.ty) else {
            return Some((UNKNOWN_TYPE, decl));
        };
        (decl, reference_type, Some(decl))
    } else if let Some(imp) = first_impl {
        let Some(reference_type) = ctx.ex.ans.function_types_by_value(imp).map(|f| f.ty) else {
            return Some((UNKNOWN_TYPE, imp));
        };
        (imp, reference_type, None)
    } else {
        return None;
    };

    for &decl in &functions.declarations {
        let Some(ty) = ctx.ex.ans.function_types_by_value(decl).map(|f| f.ty) else {
            return Some((UNKNOWN_TYPE, reference_site));
        };
        if ty != reference_type {
            ctx.push_error(TypeError::ValuesContradict {
                expectation_reason:
                    "all declarations must exactly match the first declaration signature",
                site: decl,
                found: decl,
                expected_place: reference_site,
                clash: simple_type_clash(&ctx.ex, ty, reference_type),
            });
        }
    }

    if let Some(first_impl) = first_impl {
        for extra_impl in functions.implementations.iter().copied().skip(1) {
            ctx.push_error(TypeError::DuplicateFunctionImplementation {
                first_implementation: first_impl,
                duplicate_implementation: extra_impl,
            });
        }

        let Some(impl_type) = ctx.ex.ans.function_types_by_value(first_impl).map(|f| f.ty) else {
            return Some((UNKNOWN_TYPE, reference_site));
        };
        if impl_type != reference_type {
            let expected_place = first_decl_site.unwrap_or(reference_site);
            ctx.push_error(TypeError::ValuesContradict {
                expectation_reason:
                    "function implementation must exactly match the declared signature",
                site: first_impl,
                found: first_impl,
                expected_place,
                clash: simple_type_clash(&ctx.ex, impl_type, reference_type),
            });
        }
    }
    if let Some(reference) = ctx.ex.ans.function_types_by_value_mut(reference_site) {
        reference.impl_site = first_impl;
        reference.declaration_sites.clear();
        reference
            .declaration_sites
            .extend(functions.declarations.iter().copied());
    }

    if let Some(s) = method_str {
        ctx.ex
            .ans
            .member_function_types
            .insert((name, s), reference_site);
    } else {
        ctx.ex.ans.function_types.insert(name, reference_site);
    }

    Some((reference_type, reference_site))
}

///this method is kinda weird and ill formed
///currently when compiling type expressions we give them a type other the Type::Type
///we dont have a good destinction between the type of THE VALUE ITSELF and the type IT REFERS TO
///and this means that fn[T](){let x=T;} is technically legal and x has type Generic(0).
fn gather_generic_constraints(ctx: &mut InferState, p: PatId, id: GenId) -> (CId, TraitInfo) {
    match ctx.ex.program.pattern(p) {
        Pattern::Bind(n, m) => {
            if m != VarKind::Const {
                let loc = ctx.ex.program.pattern_loc(p);
                ctx.ex.push_error(TypeError::Simple {
                    loc,
                    message: "generic parameters must be const bindings",
                });
            }
            let info = TraitInfo { sized: true };
            let t = ctx.ex.store.intern(TypeValue::Generic(id, info));
            let c = ctx.new_solved(t);
            ctx.search.insert_name(n, c, NameBindingKind::Generic, None);
            ctx.search.local_types.insert(n, c);
            ctx.bind_pat(p, c);
            (c, info)
        }

        Pattern::TypeAnnotation { pat, ty } => {
            let info = match ctx.ex.program.type_expr(ty) {
                TypeExpr::NameRef(name)
                    if matches!(
                        ctx.ex.program.definitions.get(&name),
                        Some(Defined::BuildinInterface(interface_name)) if *interface_name == DSIZED_STR
                    ) =>
                {
                    TraitInfo { sized: false }
                }
                _ => {
                    let loc = ctx.ex.program.type_expr_loc(ty);
                    ctx.ex.push_error(TypeError::Simple {
                        loc,
                        message: "unsupported generic trait bound; only `dsize` is supported",
                    });
                    TraitInfo { sized: true }
                }
            };

            let Pattern::Bind(n, m) = ctx.ex.program.pattern(pat) else {
                let loc = ctx.ex.program.pattern_loc(pat);
                ctx.ex.push_error(TypeError::Simple {
                    loc,
                    message: "generic parameters must be named const bindings",
                });
                return (ctx.new_cluster(), info);
            };

            if m != VarKind::Const {
                let loc = ctx.ex.program.pattern_loc(p);
                ctx.ex.push_error(TypeError::Simple {
                    loc,
                    message: "generic parameters must be const bindings",
                });
            }

            let t = ctx.ex.store.intern(TypeValue::Generic(id, info));
            let c = ctx.new_solved(t);
            ctx.search.insert_name(n, c, NameBindingKind::Generic, None);
            ctx.search.local_types.insert(n, c);
            ctx.bind_pat(p, c);
            (c, info)
        }

        _ => todo!(),
    }
}

fn bind_lifetime_generics(ctx: &mut InferState, generics: PatternSpan) {
    for lifetime_pat in generics.ids() {
        let Pattern::LifeTime(id) = ctx.ex.program.pattern(lifetime_pat) else {
            let loc = ctx.ex.program.pattern_loc(lifetime_pat);
            ctx.ex.push_error(TypeError::Simple {
                loc,
                message: "function lifetime parameters must be lifetime names",
            });
            continue;
        };
        let fresh = ctx.types.mint_undeclared_signature_lifetime();
        let lid = ctx.types.new_lid_known(fresh);
        ctx.search.local_lifetimes.insert(id, (fresh, lid));
    }
}

fn collect_decl_where_requirements(
    ctx: &mut InferState,
    generics: GenDec,
) -> (Vec<LifetimeOrderingEdge>, Vec<GenericLifetimeRequirement>) {
    let mut lifetime_indexes: IdHashMap<LifeTimeId, usize> = IdHashMap::default();
    for (index, lifetime_pat) in generics.lifetimes().ids().enumerate() {
        if let Pattern::LifeTime(id) = ctx.ex.program.pattern(lifetime_pat) {
            lifetime_indexes.insert(id, index);
        }
    }

    let mut generic_indexes: IdHashMap<NameId, usize> = IdHashMap::default();
    for (index, generic_pat) in generics.generics().ids().enumerate() {
        if let Pattern::Bind(name, _) = ctx.ex.program.pattern(generic_pat) {
            generic_indexes.insert(name, index);
        }
    }

    let mut orderings = Vec::with_capacity(generics.where_clause().len());
    let mut generic_lifetime_requirements = Vec::with_capacity(generics.where_clause().len());
    for constraint in generics.where_clause().ids() {
        let TypeExpr::Lt { lhs, rhs } = ctx.ex.program.type_expr(constraint) else {
            continue;
        };

        let resolve_lifetime = |expr: TExpId| match ctx.ex.program.type_expr(expr) {
            TypeExpr::LifeTime(id) => lifetime_indexes.get(&id).copied().map(LifetimeGraphId),
            TypeExpr::NameRef(name) => {
                generics
                    .lifetimes()
                    .ids()
                    .enumerate()
                    .find_map(|(index, pat)| match ctx.ex.program.pattern(pat) {
                        Pattern::LifeTime(id)
                            if ctx.ex.program.lifetime_string(id)
                                == ctx.ex.program.name_string(name) =>
                        {
                            Some(LifetimeGraphId(index))
                        }
                        _ => None,
                    })
            }
            _ => None,
        };

        let resolve_generic = |expr: TExpId| match ctx.ex.program.type_expr(expr) {
            TypeExpr::NameRef(name) => generic_indexes.get(&name).copied(),
            _ => None,
        };

        if let (Some(shorter), Some(longer)) = (resolve_lifetime(lhs), resolve_lifetime(rhs)) {
            orderings.push(LifetimeOrderingEdge { shorter, longer });
            continue;
        }

        if let (Some(generic), Some(lifetime)) = (resolve_generic(lhs), resolve_lifetime(rhs)) {
            generic_lifetime_requirements.push(GenericLifetimeRequirement { generic, lifetime });
        }
    }

    canonicalize_lifetime_orderings(&mut orderings);
    canonicalize_generic_lifetime_requirements(&mut generic_lifetime_requirements);
    (orderings, generic_lifetime_requirements)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TypeExprCompileMode {
    Signature,
    Struct,
    Local,
}

fn compile_lifetime_specialization_arg(
    ctx: &mut InferState,
    arg: TExpId,
    _mode: TypeExprCompileMode,
) -> LId {
    match ctx.ex.program.type_expr(arg) {
        TypeExpr::Wildcard => ctx.types.new_lid(),
        TypeExpr::LifeTime(lid) => {
            // if lid==LifeTimeId::STATIC{
            //     return ctx.types.new_lid_known(LifeTime::Static);
            // }
            ctx.search
                .local_lifetimes
                .get(&lid)
                .copied()
                .map(|(_, x)| x)
                .unwrap_or_else(|| {
                    debug_assert_eq!(lid, LifeTimeId::WILDCARD);
                    ctx.types.new_lid()
                })
        }

        _ => {
            let loc = ctx.ex.program.type_expr_loc(arg);
            ctx.ex.push_error(TypeError::Simple {
                loc,
                message: "expected a lifetime argument",
            });
            ctx.types.new_lid()
        }
    }
}

fn infer_elided_output_lifetime(
    ctx: &mut InferState,
    output_type: Option<TExpId>,
    undeclared_before_inputs: u32,
    undeclared_after_inputs: u32,
) -> Option<LifeTime> {
    let out_expr = output_type?;
    if !matches!(
        ctx.ex.program.type_expr(out_expr),
        TypeExpr::Ptr {
            raw: false,
            lifetime: None,
            ..
        }
    ) {
        return None;
    }

    if undeclared_after_inputs - undeclared_before_inputs == 1 {
        Some(LifeTime::External(undeclared_before_inputs))
    } else {
        let loc = ctx.ex.program.type_expr_loc(out_expr);
        ctx.ex.push_error(TypeError::Simple {
            loc,
            message: "elided output lifetime requires exactly one elided input lifetime",
        });
        None
    }
}

fn apply_signature_elided_output_lifetime_rule(
    ctx: &mut InferState,
    output_type: Option<TExpId>,
    implicit_input_lifetimes: &[LifeTime],
    lids_before_output: usize,
    lids_after_output: usize,
) {
    let Some(output_type) = output_type else {
        return;
    };

    let mut seen_output_roots = vec![false; ctx.types.lifetimes.life_parent.len()];
    let mut output_elided_roots = Vec::with_capacity(lids_after_output - lids_before_output);
    for lid in lids_before_output..lids_after_output {
        let root = ctx.types.find_lid_root(LId(lid));
        if seen_output_roots[root.0] {
            continue;
        }
        seen_output_roots[root.0] = true;
        if ctx.types.lifetimes.life_known[root].is_none() {
            output_elided_roots.push(root);
        }
    }

    if output_elided_roots.is_empty() {
        return;
    }

    if implicit_input_lifetimes.len() != 1 {
        let loc = ctx.ex.program.type_expr_loc(output_type);
        ctx.ex.push_error(TypeError::Simple {
            loc,
            message: "elided output lifetime requires exactly one elided input lifetime",
        });
        return;
    }

    let target_lifetime = implicit_input_lifetimes[0];
    for output_root in output_elided_roots {
        let _ = bind_struct_lid_to_lifetime(&mut ctx.types, output_root, target_lifetime);
    }
}

fn assign_signature_implicit_input_lifetimes(
    ctx: &mut InferState,
    lids_before_inputs: usize,
    lids_after_inputs: usize,
) -> Vec<LifeTime> {
    let mut seen_input_roots = vec![false; ctx.types.lifetimes.life_parent.len()];
    let mut implicit_input_roots = Vec::with_capacity(lids_after_inputs - lids_before_inputs);
    for lid in lids_before_inputs..lids_after_inputs {
        let root = ctx.types.find_lid_root(LId(lid));
        if seen_input_roots[root.0] {
            continue;
        }
        seen_input_roots[root.0] = true;
        if ctx.types.lifetimes.life_known[root].is_none() {
            implicit_input_roots.push(root);
        }
    }

    let mut implicit_input_lifetimes = Vec::with_capacity(implicit_input_roots.len());
    for root in implicit_input_roots {
        let fresh = ctx.types.mint_undeclared_signature_lifetime();
        let _ = bind_struct_lid_to_lifetime(&mut ctx.types, root, fresh);
        implicit_input_lifetimes.push(fresh);
    }
    implicit_input_lifetimes
}

fn compile_type_expr_with_forced_output_lifetime(
    ctx: &mut InferState,
    texpr: TExpId,
    forced_output_lifetime: Option<LifeTime>,
    mode: TypeExprCompileMode,
) -> CId {
    if let Some(lifetime) = forced_output_lifetime
        && let TypeExpr::Ptr {
            base,
            raw: false,
            mutable,
            lifetime: None,
        } = ctx.ex.program.type_expr(texpr)
    {
        let tgt = compile_type_expr_with_mode(ctx, base, mode);
        let ans = ctx.new_cluster();
        ctx.types.core.cluster[ans].state = ResolveKind::Ptr {
            tgt,
            kind: PtrKind::Solved(PointerStyle::Ref(lifetime)),
            mutable: Some(mutable),
        };
        return ans;
    }

    compile_type_expr_with_mode(ctx, texpr, mode)
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
    let decl_generics = generics;
    let lifetimes = generics.lifetimes();
    let generics = generics.generics();
    // Reject struct definitions in local scope.
    // The type inference is monomorphic (rank-1, no higher-ranked types)
    // and performs type inference by unification, which fundamentally cannot
    // handle generic type parameters inside function bodies - we would need
    // higher-rank polymorphism (rank-2+) or a more expressive constraint system.
    // Generic types are only allowed at the top-level where they are explicitly
    // declared and can be monomorphized at instantiation sites.
    if !GLOBAL_SCOPE {
        let loc = ctx.ex.program.type_expr_loc(texpr);
        ctx.ex.push_error(TypeError::Simple {
            loc,
            message: "struct types are only allowed at the top level",
        });
    }
    bind_lifetime_generics(ctx, lifetimes);

    let mut trait_info = Vec::with_capacity(generics.len());
    for (i, g) in generics.ids().enumerate() {
        let gid = GenId(i);
        let (_c, x) = gather_generic_constraints(ctx, g, gid);
        trait_info.push(x);
        // todo!()
        //TODO: we probably wana do something with generics that are ints here if we have them
    }

    let undeclared_before_fields = ctx.types.lifetimes.next_undeclared_lifetime;

    let mut field_info = Vec::with_capacity(fields.len());
    for p in fields.ids() {
        match ctx.ex.program.pattern(p) {
            Pattern::Bind(n, _) => {
                let c = ctx.new_cluster();
                let field_loc = ctx.ex.program.pattern_loc(p);
                require_sized_or_enqueue(
                    &mut ctx.ex,
                    &mut ctx.types,
                    &mut ctx.req.pending_sized_requirements,
                    field_loc,
                    c,
                    "struct field types must be sized",
                );
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
                let c = compile_type_expr_with_mode(ctx, ty, TypeExprCompileMode::Struct);
                let field_loc = ctx.ex.program.type_expr_loc(ty);
                require_sized_or_enqueue(
                    &mut ctx.ex,
                    &mut ctx.types,
                    &mut ctx.req.pending_sized_requirements,
                    field_loc,
                    c,
                    "struct field types must be sized",
                );
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

    let undeclared_after_fields = ctx.types.lifetimes.next_undeclared_lifetime;
    if undeclared_after_fields != undeclared_before_fields {
        let loc = ctx.ex.program.type_expr_loc(texpr);
        ctx.ex.push_error(TypeError::Simple {
            loc,
            message: "struct fields cannot use elided lifetimes; declare them in struct lifetime parameters",
        });
    }

    let generics = (0..generics.len())
        .map(|x| {
            ctx.ex
                .store
                .intern(TypeValue::Generic(GenId(x), trait_info[x]))
        })
        .collect();
    let lifetimes: Vec<_> = (0..lifetimes.len())
        .map(|x| LifeTime::External(x as u32))
        .collect();

    let (lifetime_orderings, generic_lifetime_requirements) =
        collect_decl_where_requirements(ctx, decl_generics);
    let rep = StructRep::new(
        field_info.iter().map(|(n, _)| *n),
        trait_info,
        lifetimes.len(),
        lifetime_orderings,
        generic_lifetime_requirements,
        layout,
    );

    let sid = ctx.ex.store.new_struct(rep);
    let t = ctx.ex.store.intern(TypeValue::Struct {
        id: sid,
        generics,
        lifetimes,
    });
    let output = ctx.new_solved(t);

    ctx.types.extra.struct_defs.push(StructDef {
        loc: texpr,
        fields: field_info,
        sid,
    });
    output
}

pub(crate) fn do_typedef<const ALLOW_STRUCT_GENERICS: bool>(
    ctx: &mut InferState,
    typedef_name: NameId,
    texpr: TExpId,
) -> CId {
    match ctx.ex.program.type_expr(texpr) {
        TypeExpr::Struct(def) => {
            let cid = compile_struct_type::<ALLOW_STRUCT_GENERICS>(ctx, texpr, def);
            let sid = match ctx.types.core.cluster[cid].state {
                ResolveKind::Struct(rid) => ctx.types.extra.struct_infers[rid].sid,
                ResolveKind::Solved(t) => match ctx.ex.store.type_value(t) {
                    TypeValue::Struct { id, .. } => *id,
                    _ => unreachable!("struct def didnt return struct"),
                },
                _ => unreachable!("struct def didnt return struct"),
            };

            debug_assert_eq!(ctx.ex.store.structs[sid].name, None);
            ctx.ex.store.structs[sid].name = Some(typedef_name);

            cid
        }
        _ => compile_type_expr_with_mode(
            ctx,
            texpr,
            if ALLOW_STRUCT_GENERICS {
                TypeExprCompileMode::Signature
            } else {
                TypeExprCompileMode::Local
            },
        ),
    }
}

pub(crate) fn compile_type_expr_with_mode(
    ctx: &mut InferState,
    texpr: TExpId,
    mode: TypeExprCompileMode,
) -> CId {
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
                .map(|item| compile_type_expr_with_mode(ctx, item, mode))
                .collect::<Vec<_>>();
            for (item_expr, item_cluster) in items.ids().zip(item_clusters.iter().copied()) {
                let item_loc = ctx.ex.program.type_expr_loc(item_expr);
                require_sized_or_enqueue(
                    &mut ctx.ex,
                    &mut ctx.types,
                    &mut ctx.req.pending_sized_requirements,
                    item_loc,
                    item_cluster,
                    "tuple element types must be sized",
                );
            }
            ctx.new_tuple_instance(item_clusters)
        }

        TypeExpr::Struct(def) => compile_struct_type::<false>(ctx, texpr, def),
        TypeExpr::Ptr {
            base,
            raw,
            mutable,
            lifetime,
        } => {
            let kind = if raw {
                PtrKind::Solved(PointerStyle::Raw(Nullable::Yes))
            // } else if lifetime == Some(LifeTimeId::STATIC) {
            //     PtrKind::Solved(PointerStyle::Ref(LifeTime::Static))
            } else if lifetime == Some(LifeTimeId::RAW) {
                PtrKind::Solved(PointerStyle::Raw(Nullable::No))
            } else if lifetime == Some(LifeTimeId::WILDCARD) {
                match mode {
                    TypeExprCompileMode::Struct => {
                        let loc = ctx.ex.program.type_expr_loc(texpr);
                        ctx.ex.push_error(TypeError::Simple {
                            loc,
                            message: "struct fields cannot use elided lifetimes; declare them in struct lifetime parameters",
                        });
                        PtrKind::RefInfer(ctx.types.new_lid())
                    }
                    _ => PtrKind::RefInfer(ctx.types.new_lid()),
                }
            } else if let Some(lid) = lifetime {
                let (lt, _) = ctx
                    .search
                    .local_lifetimes
                    .get(&lid)
                    .copied()
                    .expect("lifetime used before mentioned");
                PtrKind::Solved(PointerStyle::Ref(lt))
            } else {
                match mode {
                    TypeExprCompileMode::Struct => {
                        let loc = ctx.ex.program.type_expr_loc(texpr);
                        ctx.ex.push_error(TypeError::Simple {
                            loc,
                            message: "struct fields cannot use elided lifetimes; declare them in struct lifetime parameters",
                        });
                        PtrKind::RefInfer(ctx.types.new_lid())
                    }
                    _ => PtrKind::RefInfer(ctx.types.new_lid()),
                }
            };

            let tgt = compile_type_expr_with_mode(ctx, base, mode);
            let ans = ctx.new_cluster();
            ctx.types.core.cluster[ans].state = ResolveKind::Ptr {
                tgt,
                kind,
                mutable: Some(mutable),
            };
            ans
        }
        TypeExpr::Func {
            calling_convention,
            params,
            output_type,
        } => {
            let undeclared_before_inputs = ctx.types.lifetimes.next_undeclared_lifetime;
            let inputs = params
                .ids()
                .map(|arg| compile_type_expr_with_mode(ctx, arg, mode))
                .collect::<Vec<_>>();
            let output = match mode {
                TypeExprCompileMode::Signature => {
                    let undeclared_after_inputs = ctx.types.lifetimes.next_undeclared_lifetime;
                    let output_lifetime = infer_elided_output_lifetime(
                        ctx,
                        output_type,
                        undeclared_before_inputs,
                        undeclared_after_inputs,
                    );
                    output_type
                        .map(|o| {
                            compile_type_expr_with_forced_output_lifetime(
                                ctx,
                                o,
                                output_lifetime,
                                mode,
                            )
                        })
                        .unwrap_or_else(|| ctx.new_solved(BuiltinType::Void.into()))
                }
                TypeExprCompileMode::Local => output_type
                    .map(|o| compile_type_expr_with_mode(ctx, o, mode))
                    .unwrap_or_else(|| ctx.new_solved(BuiltinType::Void.into())),
                TypeExprCompileMode::Struct => output_type
                    .map(|o| compile_type_expr_with_mode(ctx, o, mode))
                    .unwrap_or_else(|| ctx.new_solved(BuiltinType::Void.into())),
            };

            ctx.new_func(FuncInfer {
                calling_convention,
                generics: Vec::new(),
                lifetimes: 0,
                lifetime_orderings: Vec::new(),
                generic_lifetime_requirements: Vec::new(),
                inputs,
                output,
            })
        }
        TypeExpr::Array(element, len) => {
            let element = compile_type_expr_with_mode(ctx, element, mode);
            let size = len.map_or(ArrayType::Unsized, ArrayType::Sized);
            ctx.new_array_instance(element, size)
        }
        TypeExpr::Index { base, args } => {
            let mut lifetimes = args
                .lifetimes()
                .ids()
                .map(|arg| compile_lifetime_specialization_arg(ctx, arg, mode))
                .collect::<Vec<_>>();
            let args = args.generics();
            let generic_args = args
                .ids()
                .map(|arg| (arg, compile_type_expr_with_mode(ctx, arg, mode)))
                .collect::<Vec<_>>();
            let generics = generic_args.iter().map(|(_, c)| *c).collect::<Vec<_>>();

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
                // Reject type specialization (e.g., `MyStruct[int]`) on local types.
                // The type inference is monomorphic (rank-1) - we cannot track
                // generic type parameters inside function bodies. Only global types
                // can be specialized since they are defined at the top level where
                // we can properly monomorphize them at use sites.
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
                        lifetimes,
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

            let TypeValue::Struct {
                id: sid,
                lifetimes: expected_lifetimes,
                ..
            } = ctx.ex.store.type_value(*t)
            else {
                let loc = ctx.ex.program.type_expr_loc(base);
                ctx.push_error(TypeError::Simple {
                    loc,
                    message: "type specialization expects a struct type",
                });
                return ctx.new_cluster();
            };
            let sid = *sid;
            let expected_lifetimes = expected_lifetimes.to_vec();

            let expected = ctx.ex.store.struct_value(sid).gen_info.len();
            if generics.len() != expected {
                let loc = ctx.ex.program.type_expr_loc(texpr);
                ctx.ex.push_error(TypeError::Simple {
                    loc,
                    message: "wrong number of generic arguments for struct type",
                });
                return ctx.new_cluster();
            }

            for i in 0..expected {
                if !ctx.ex.store.struct_value(sid).gen_info[i].sized {
                    continue;
                }
                let arg_loc = ctx.ex.program.type_expr_loc(generic_args[i].0);
                require_sized_or_enqueue(
                    &mut ctx.ex,
                    &mut ctx.types,
                    &mut ctx.req.pending_sized_requirements,
                    arg_loc,
                    generics[i],
                    "generic argument for this parameter must be sized",
                );
            }

            if lifetimes.is_empty() && !expected_lifetimes.is_empty() {
                lifetimes = expected_lifetimes
                    .iter()
                    .map(|_| ctx.types.new_lid())
                    .collect();
            }

            if lifetimes.len() != expected_lifetimes.len() {
                let loc = ctx.ex.program.type_expr_loc(texpr);
                ctx.ex.push_error(TypeError::Simple {
                    loc,
                    message: "wrong number of lifetime arguments for struct type",
                });
                return ctx.new_cluster();
            }

            ctx.new_struct_instance(sid, generics, lifetimes)
        }
        _ => {
            let c = ctx.new_cluster();
            ctx.push_error(TypeError::ExpectedTypeExpr { type_expr: texpr });
            c
        }
    }
}

fn type_check_func_signature(
    ctx: &mut InferState,
    v: ValId,
    calling_convention: CallingConvention,
    generics: GenDec,
    params: PatternSpan,
    output_type: Option<TExpId>,
) {
    let previous_name_render = std::mem::replace(
        &mut ctx.ex.name_render,
        GenLifeNameRender::from_decl(ctx.ex.program, generics),
    );

    let (f, _) =
        gather_func_signature::<true>(ctx, v, calling_convention, generics, params, output_type);
    ctx.bind_val(v, f);
    global_solver(ctx);

    ctx.ex.name_render = previous_name_render;
}

fn check_unused_function_signature_generics_and_lifetimes(ctx: &mut InferState, function: ValId) {
    let Some(ty) = ctx.ex.ans.function_types_by_value(function).map(|f| f.ty) else {
        return;
    };

    let unused_indexes = ctx.ex.store.unused_function_generic_indexes(ty).to_vec();
    for generic_index in unused_indexes {
        ctx.push_error(TypeError::UnusedFunctionGeneric {
            function,
            generic_index,
        });
    }

    let lifetime_count = match ctx.ex.program.value(function) {
        Value::Func { generics, .. } => generics.lifetimes().len(),
        _ => 0,
    };
    let unused_lifetimes = ctx
        .ex
        .store
        .unused_function_lifetime_indexes(ty, lifetime_count);
    for lifetime_index in unused_lifetimes {
        ctx.push_error(TypeError::UnusedFunctionLifetime {
            function,
            lifetime_index,
        });
    }
}

#[allow(dead_code)]
fn mark_used_generics_and_lifetimes_from_type(
    store: &TypeStore,
    ty: TypeId,
    generic_count: usize,
    lifetime_count: usize,
    used_generics: &mut [bool],
    used_lifetimes: &mut [bool],
) {
    match store.type_value(ty) {
        TypeValue::Builtin(_) => {}
        TypeValue::Generic(gid, _) => {
            if gid.0 < generic_count {
                used_generics[gid.0] = true;
            }
        }
        TypeValue::Tuple(items) => {
            for &item in items {
                mark_used_generics_and_lifetimes_from_type(
                    store,
                    item,
                    generic_count,
                    lifetime_count,
                    used_generics,
                    used_lifetimes,
                );
            }
        }
        TypeValue::Array(inner, _) => {
            mark_used_generics_and_lifetimes_from_type(
                store,
                *inner,
                generic_count,
                lifetime_count,
                used_generics,
                used_lifetimes,
            );
        }
        TypeValue::Func {
            params,
            ret,
            lifetime_orderings,
            generic_lifetime_requirements,
            ..
        } => {
            for &param in params {
                mark_used_generics_and_lifetimes_from_type(
                    store,
                    param,
                    generic_count,
                    lifetime_count,
                    used_generics,
                    used_lifetimes,
                );
            }
            mark_used_generics_and_lifetimes_from_type(
                store,
                *ret,
                generic_count,
                lifetime_count,
                used_generics,
                used_lifetimes,
            );
            for edge in lifetime_orderings {
                if edge.shorter.0 < lifetime_count {
                    used_lifetimes[edge.shorter.0] = true;
                }
                if edge.longer.0 < lifetime_count {
                    used_lifetimes[edge.longer.0] = true;
                }
            }
            for requirement in generic_lifetime_requirements {
                if requirement.generic < generic_count {
                    used_generics[requirement.generic] = true;
                }
                if requirement.lifetime.0 < lifetime_count {
                    used_lifetimes[requirement.lifetime.0] = true;
                }
            }
        }
        TypeValue::Ptr { tgt, style, .. } => {
            if let PointerStyle::Ref(LifeTime::External(i)) = style
                && (*i as usize) < lifetime_count
            {
                used_lifetimes[*i as usize] = true;
            }
            mark_used_generics_and_lifetimes_from_type(
                store,
                *tgt,
                generic_count,
                lifetime_count,
                used_generics,
                used_lifetimes,
            );
        }
        TypeValue::Struct {
            generics,
            lifetimes,
            ..
        } => {
            for &generic in generics {
                mark_used_generics_and_lifetimes_from_type(
                    store,
                    generic,
                    generic_count,
                    lifetime_count,
                    used_generics,
                    used_lifetimes,
                );
            }
            for lt in lifetimes {
                if let LifeTime::External(i) = lt
                    && (*i as usize) < lifetime_count
                {
                    used_lifetimes[*i as usize] = true;
                }
            }
        }
    }
}

fn mark_used_struct_signature_from_type(
    store: &TypeStore,
    ty: TypeId,
    generic_count: usize,
    used_generics: &mut [bool],
    used_lifetimes: &mut [bool],
) {
    match store.type_value(ty) {
        TypeValue::Builtin(_) => {}
        TypeValue::Generic(gid, _) => {
            if gid.0 < generic_count {
                used_generics[gid.0] = true;
            }
        }
        TypeValue::Tuple(items) => {
            for &item in items {
                mark_used_struct_signature_from_type(
                    store,
                    item,
                    generic_count,
                    used_generics,
                    used_lifetimes,
                );
            }
        }
        TypeValue::Array(inner, _) => {
            mark_used_struct_signature_from_type(
                store,
                *inner,
                generic_count,
                used_generics,
                used_lifetimes,
            );
        }
        TypeValue::Func {
            params,
            ret,
            lifetime_orderings,
            generic_lifetime_requirements,
            ..
        } => {
            for &param in params {
                mark_used_struct_signature_from_type(
                    store,
                    param,
                    generic_count,
                    used_generics,
                    used_lifetimes,
                );
            }
            mark_used_struct_signature_from_type(
                store,
                *ret,
                generic_count,
                used_generics,
                used_lifetimes,
            );
            for edge in lifetime_orderings {
                if edge.shorter.0 < used_lifetimes.len() {
                    used_lifetimes[edge.shorter.0] = true;
                }
                if edge.longer.0 < used_lifetimes.len() {
                    used_lifetimes[edge.longer.0] = true;
                }
            }
            for requirement in generic_lifetime_requirements {
                if requirement.generic < generic_count {
                    used_generics[requirement.generic] = true;
                }
                if requirement.lifetime.0 < used_lifetimes.len() {
                    used_lifetimes[requirement.lifetime.0] = true;
                }
            }
        }
        TypeValue::Ptr { tgt, style, .. } => {
            if let PointerStyle::Ref(LifeTime::External(i)) = style
                && (*i as usize) < used_lifetimes.len()
            {
                used_lifetimes[*i as usize] = true;
            }
            mark_used_struct_signature_from_type(
                store,
                *tgt,
                generic_count,
                used_generics,
                used_lifetimes,
            );
        }
        TypeValue::Struct {
            generics,
            lifetimes,
            ..
        } => {
            for &generic in generics {
                mark_used_struct_signature_from_type(
                    store,
                    generic,
                    generic_count,
                    used_generics,
                    used_lifetimes,
                );
            }
            for lt in lifetimes {
                if let LifeTime::External(i) = lt {
                    used_lifetimes[*i as usize] = true;
                }
            }
        }
    }
}

fn check_unused_struct_signature_generics_and_lifetimes(ctx: &mut InferState, type_expr: TExpId) {
    let TypeExpr::Struct(def) = ctx.ex.program.type_expr(type_expr) else {
        return;
    };

    let generic_count = def.generics.generics().len();
    let lifetime_count = def.generics.lifetimes().len();
    if generic_count == 0 && lifetime_count == 0 {
        return;
    }

    let Some(ty) = ctx.ex.ans.typedef_types.get(&type_expr).copied() else {
        return;
    };
    let TypeValue::Struct { id: sid, .. } = *ctx.ex.store.type_value(ty) else {
        return;
    };

    let struct_rep = ctx.ex.store.struct_value(sid);
    // if struct_rep.gen_count != generic_count || struct_rep.life_count != lifetime_count {
    //     return;
    // }

    let mut used_generics = vec![false; generic_count];
    let mut used_lifetimes = vec![false; lifetime_count];

    for (_, field_ty) in struct_rep.fields.iter() {
        mark_used_struct_signature_from_type(
            ctx.ex.store,
            *field_ty,
            generic_count,
            &mut used_generics,
            &mut used_lifetimes,
        );
    }

    for edge in &struct_rep.lifetime_orderings {
        if edge.shorter.0 < used_lifetimes.len() {
            used_lifetimes[edge.shorter.0] = true;
        }
        if edge.longer.0 < used_lifetimes.len() {
            used_lifetimes[edge.longer.0] = true;
        }
    }
    for requirement in &struct_rep.generic_lifetime_requirements {
        if requirement.generic < used_generics.len() {
            used_generics[requirement.generic] = true;
        }
        if requirement.lifetime.0 < used_lifetimes.len() {
            used_lifetimes[requirement.lifetime.0] = true;
        }
    }

    for (generic_index, used) in used_generics.into_iter().enumerate() {
        if !used {
            ctx.push_error(TypeError::UnusedStructGeneric {
                type_expr,
                generic_index,
            });
        }
    }

    for (lifetime_index, used) in used_lifetimes.into_iter().enumerate() {
        if !used {
            ctx.push_error(TypeError::UnusedStructLifetime {
                type_expr,
                lifetime_index,
            });
        }
    }
}

pub(crate) fn gather_func_signature<const GLOBAL_SCOPE: bool>(
    ctx: &mut InferState,
    v: ValId,
    calling_convention: CallingConvention,
    generics: GenDec,
    params: PatternSpan,
    output_type: Option<TExpId>,
) -> (CId, CId) {
    let decl_generics = generics;
    let lifetime_before_signature = ctx.types.lifetimes.life_parent.len();
    let lifetime_generics = generics.lifetimes();
    let generics = generics.generics();
    // Reject generic functions in local scope.
    // The type inference is monomorphic (rank-1, no higher-ranked types)
    // and performs type inference by unification, which fundamentally cannot
    // handle generic type parameters inside function bodies - we would need
    // higher-rank polymorphism (rank-2+) or a more expressive constraint system.
    // Generic functions are only allowed at the top-level where they can be
    // monomorphized at each call site.
    if !GLOBAL_SCOPE && (!generics.is_empty() || !lifetime_generics.is_empty()) {
        let loc = generics
            .ids()
            .next()
            .or_else(|| lifetime_generics.ids().next())
            .map(|pat| ctx.ex.program.pattern_loc(pat))
            .unwrap_or_else(|| ctx.ex.program.value_loc(v));
        ctx.push_error(TypeError::Simple {
            loc,
            message: "generic functions are only allowed at the top level",
        });
    }

    let lids_before_inputs = ctx.types.lifetimes.life_parent.len();
    bind_lifetime_generics(ctx, lifetime_generics);

    let mut generic_info = Vec::with_capacity(generics.len());
    for (i, pat) in generics.ids().enumerate() {
        let (_, info) = gather_generic_constraints(ctx, pat, GenId(i));
        generic_info.push(info);
    }

    let inputs = params
        .ids()
        .map(|pat| gather_pattern_constraints_with_generics::<GLOBAL_SCOPE>(ctx, pat, None))
        .collect::<Vec<_>>();
    let lids_after_inputs = ctx.types.lifetimes.life_parent.len();
    let implicit_input_lifetimes =
        assign_signature_implicit_input_lifetimes(ctx, lids_before_inputs, lids_after_inputs);
    let lids_before_output = ctx.types.lifetimes.life_parent.len();
    let output = if let Some(x) = output_type {
        compile_type_expr_with_mode(ctx, x, TypeExprCompileMode::Signature)
    } else {
        ctx.new_solved(BuiltinType::Void.into())
    };

    for (param_pat, input_ty) in params.ids().zip(inputs.iter().copied()) {
        let param_loc = ctx.ex.program.pattern_loc(param_pat);
        require_sized_or_enqueue(
            &mut ctx.ex,
            &mut ctx.types,
            &mut ctx.req.pending_sized_requirements,
            param_loc,
            input_ty,
            "function parameter types must be sized",
        );
    }

    let output_loc = output_type
        .map(|texpr| ctx.ex.program.type_expr_loc(texpr))
        .unwrap_or_else(|| ctx.ex.program.value_loc(v));
    require_sized_or_enqueue(
        &mut ctx.ex,
        &mut ctx.types,
        &mut ctx.req.pending_sized_requirements,
        output_loc,
        output,
        "function return type must be sized",
    );

    let lids_after_output = ctx.types.lifetimes.life_parent.len();
    apply_signature_elided_output_lifetime_rule(
        ctx,
        output_type,
        &implicit_input_lifetimes,
        lids_before_output,
        lids_after_output,
    );

    let lifetime_count = lids_before_output - lifetime_before_signature;
    let (lifetime_orderings, generic_lifetime_requirements) = if GLOBAL_SCOPE {
        collect_decl_where_requirements(ctx, decl_generics)
    } else {
        (Vec::new(), Vec::new())
    };

    let f = ctx.new_func(FuncInfer {
        calling_convention,
        generics: if GLOBAL_SCOPE {
            generic_info
        } else {
            Vec::new()
        },
        lifetimes: if GLOBAL_SCOPE { lifetime_count } else { 0 },
        lifetime_orderings,
        generic_lifetime_requirements,
        inputs,
        output,
    });

    if !GLOBAL_SCOPE {
        ctx.bind_val(v, f);
    }
    (f, output)
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
        || name == FORGET_STR
        || name == FREE_STR
        || name == USER_FREE_STR
        || name == SIZE_OF_STR
        || name == ALIGN_OF_STR
        || name == DEREF_STR
        || name == DEREF_MUT_STR
}

#[inline(always)]
pub(crate) fn is_any_type_builtin_member_name(name: StrId) -> bool {
    matches!(
        name,
        FORGET_STR | FREE_STR | USER_FREE_STR | SIZE_OF_STR | ALIGN_OF_STR
    )
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
fn is_named_struct_type_with_all_generics_free(
    store: &TypeStore,
    ty: TypeId,
    struct_name: NameId,
) -> bool {
    let TypeValue::Struct {
        id,
        generics,
        lifetimes,
    } = store.type_value(ty)
    else {
        return false;
    };

    let rep = store.struct_value(*id);
    if rep.name != Some(struct_name) || generics.len() != rep.gen_info.len() {
        return false;
    }

    generics.iter().enumerate().all(|(i, generic_ty)| {
        matches!(store.type_value(*generic_ty), TypeValue::Generic(gid,_) if *gid == GenId(i))
    })
    &&
    lifetimes.iter().enumerate().all(|(i, life)| {
        matches!(life, LifeTime::External(l) if *l == i as u32)
    })
}

#[inline(always)]
fn method_signature_type_parts(store: &TypeStore, ty: TypeId) -> Option<(&[TypeId], TypeId)> {
    if ty.0 >= store.values.len() {
        return None;
    }
    match store.type_value(ty) {
        TypeValue::Func { params, ret, .. } => Some((params.as_slice(), *ret)),
        _ => None,
    }
}

#[inline(always)]
fn get_member_self_pointer_style(
    store: &TypeStore,
    method_ty: TypeId,
    struct_name: NameId,
) -> Option<Option<PointerStyle>> {
    let (inputs, _) = method_signature_type_parts(store, method_ty)?;
    let first_input = *inputs.first()?;
    match store.type_value(first_input) {
        TypeValue::Struct { .. } if is_named_struct_type(store, first_input, struct_name) => {
            Some(None)
        }
        TypeValue::Ptr {
            tgt,
            style,
            mutable,
        } if style.is_fancy() && is_named_struct_type(store, *tgt, struct_name) => {
            let _ = mutable;
            Some(Some(*style))
        }
        _ => None,
    }
}

#[inline(always)]
pub(crate) fn receiver_cluster_for_self_param(
    ex: &mut ExternState,
    types: &mut TypeState,
    receiver: CId,
    self_param: CId,
) -> Option<CId> {
    let self_root = types.root(self_param);
    match types.cluster_state(self_root) {
        ResolveKind::Struct(_) => Some(receiver),
        ResolveKind::Solved(t) => match ex.store.type_value(t) {
            TypeValue::Struct { .. } => Some(receiver),
            TypeValue::Ptr { style, mutable, .. } => {
                let adapted = types.new_cluster();
                types.set_cluster_state(
                    adapted,
                    ResolveKind::Ptr {
                        tgt: receiver,
                        kind: PtrKind::Solved(*style),
                        mutable: Some(*mutable),
                    },
                );
                Some(adapted)
            }
            _ => Some(receiver),
        },
        ResolveKind::Ptr { kind, mutable, .. } => {
            let adapted = types.new_cluster();
            types.set_cluster_state(
                adapted,
                ResolveKind::Ptr {
                    tgt: receiver,
                    kind,
                    mutable,
                },
            );
            Some(adapted)
        }
        _ => Some(receiver),
    }
}

#[inline(always)]
fn is_self_like_member_input_type(store: &TypeStore, input: TypeId, struct_name: NameId) -> bool {
    match store.type_value(input) {
        TypeValue::Struct { .. } => is_named_struct_type(store, input, struct_name),
        TypeValue::Ptr {
            tgt,
            style,
            mutable: _,
        } => style.is_fancy() && is_named_struct_type(store, *tgt, struct_name),
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
            style,
            mutable: is_mut,
        } => {
            style.is_fancy() && *is_mut == mutable && is_named_struct_type(store, *tgt, struct_name)
        }
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
        TypeValue::Ptr {
            tgt,
            style,
            mutable,
        } => {
            style.is_fancy()
                && *mutable
                && is_named_struct_type_with_all_generics_free(store, *tgt, struct_name)
        }
        _ => false,
    }
}

#[inline(always)]
fn get_ref_target_type_if_kind(store: &TypeStore, ty: TypeId, mutable: bool) -> Option<TypeId> {
    match store.type_value(ty) {
        TypeValue::Ptr {
            tgt,
            style,
            mutable: is_mut,
        } if style.is_fancy() && *is_mut == mutable => Some(*tgt),
        _ => None,
    }
}

#[inline(always)]
#[allow(dead_code)]
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

#[derive(Debug, Default, Clone, Copy)]
struct PendingStructDerefMethods {
    deref: Option<(TypeId, ValId)>,
    deref_mut: Option<(TypeId, ValId)>,
}

fn validate_and_insert_member_overload(
    ctx: &mut InferState,
    info: &mut StructOverloadInfo,
    pending_deref_methods: &mut PendingStructDerefMethods,
    struct_name: NameId,
    method_name: StrId,
    method_site: ValId,
    method_ty: TypeId,
) {
    let loc = ctx.ex.program.value_loc(method_site);

    if is_reserved_builtin_member_name(ctx.ex.program, method_name)
        && !is_known_special_member_method_name(method_name)
    {
        //this is technically a bug the site is the name itself but eh
        ctx.push_error(TypeError::UnknownBuiltinMemberMethod {
            site: method_site,
            method: method_name,
        });
    }

    if !is_known_special_member_method_name(method_name) {
        return;
    }

    let Some((inputs, output)) = method_signature_type_parts(ctx.ex.store, method_ty) else {
        return;
    };
    let inputs = inputs.to_vec();
    let Some(first_input) = inputs.first().copied() else {
        ctx.push_error(TypeError::Simple {
            loc,
            message: "special member methods must take `self` as the first parameter",
        });
        return;
    };

    if method_name == FORGET_STR {
        ctx.push_error(TypeError::Simple {
            loc,
            message: "users may not implement forget",
        });
        return;
    }

    if matches!(method_name, FREE_STR | USER_FREE_STR) {
        if !is_mut_ref_to_named_struct_input_type(ctx.ex.store, first_input, struct_name) {
            ctx.push_error(TypeError::Simple {
                loc: loc.clone(),
                message: "`__free` must take `&mut self` as the first parameter",
            });
        }

        if inputs.len() != 1 {
            ctx.push_error(TypeError::Simple {
                loc: loc.clone(),
                message: "`__free` must not take parameters after `self`",
            });
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
        let mut valid = true;

        if !is_ref_to_named_struct_input_type(ctx.ex.store, first_input, struct_name, false) {
            ctx.push_error(TypeError::Simple {
                loc: loc.clone(),
                message: "`__deref` must take `&self` as the first parameter",
            });
            valid = false;
        }

        if inputs.len() != 1 {
            ctx.push_error(TypeError::Simple {
                loc: loc.clone(),
                message: "`__deref` must not take parameters after `self`",
            });
            valid = false;
        }

        if get_ref_target_type_if_kind(ctx.ex.store, output, false).is_none() {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "`__deref` must return a non-raw shared reference `&T`",
            });
            valid = false;
        }

        if valid {
            pending_deref_methods.deref = Some((method_ty, method_site));
            update_cached_deref_site_if_present(ctx, info, method_ty, method_site, false);
            check_inserted_deref_pair_compatible(ctx, pending_deref_methods, method_site);
        }
        return;
    }

    if method_name == DEREF_MUT_STR {
        let mut valid = true;

        if !is_ref_to_named_struct_input_type(ctx.ex.store, first_input, struct_name, true) {
            ctx.push_error(TypeError::Simple {
                loc: loc.clone(),
                message: "`__deref_mut` must take `&mut self` as the first parameter",
            });
            valid = false;
        }

        if inputs.len() != 1 {
            ctx.push_error(TypeError::Simple {
                loc: loc.clone(),
                message: "`__deref_mut` must not take parameters after `self`",
            });
            valid = false;
        }

        if get_ref_target_type_if_kind(ctx.ex.store, output, true).is_none() {
            ctx.push_error(TypeError::Simple {
                loc,
                message: "`__deref_mut` must return a non-raw mutable reference `&mut T`",
            });
            valid = false;
        }

        if valid {
            pending_deref_methods.deref_mut = Some((method_ty, method_site));
            update_cached_deref_site_if_present(ctx, info, method_ty, method_site, true);
            check_inserted_deref_pair_compatible(ctx, pending_deref_methods, method_site);
        }
        return;
    }

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

        if is_self_like_member_input_type(ctx.ex.store, first_input, struct_name)
            && additional_args == 1
        {
            let self_pointer_style =
                get_member_self_pointer_style(ctx.ex.store, method_ty, struct_name).expect(
                    "validated binary operator overload must have self-like first parameter",
                );
            info.operators.insert(
                method_name,
                StructOperatorOverload {
                    method_type: method_ty,
                    method_site,
                    self_pointer_style,
                },
            );
        }

        return;
    }

    if is_unary_operator_overload_name(method_name) {
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

        if is_self_like_member_input_type(ctx.ex.store, first_input, struct_name)
            && additional_args == 0
        {
            let self_pointer_style =
                get_member_self_pointer_style(ctx.ex.store, method_ty, struct_name).expect(
                    "validated unary operator overload must have self-like first parameter",
                );
            info.operators.insert(
                method_name,
                StructOperatorOverload {
                    method_type: method_ty,
                    method_site,
                    self_pointer_style,
                },
            );
        }

        return;
    }

    ctx.push_error(TypeError::IlegalToImplMethod {
        method_site,
        method_name,
    });
}

#[inline(always)]
fn update_cached_deref_site_if_present(
    ctx: &mut InferState,
    info: &mut StructOverloadInfo,
    method_ty: TypeId,
    method_site: ValId,
    mutable: bool,
) {
    let entry = info.deref_style.get_or_insert_with(|| {
        resolved_deref_style_for_method(ctx, method_ty, method_site, mutable)
    });

    let site_slot = if mutable {
        &mut entry.deref_mut_site
    } else {
        &mut entry.deref_site
    };

    if let Some(existing_site) = *site_slot {
        assert_eq!(
            existing_site, method_site,
            "global deref overload entry unexpectedly changed for same struct"
        );
    } else {
        *site_slot = Some(method_site);
    }

    entry.mutable = match (entry.deref_site.is_some(), entry.deref_mut_site.is_some()) {
        (true, true) => None,
        (true, false) => Some(false),
        (false, true) => Some(true),
        (false, false) => None,
    };
}

#[inline(always)]
fn resolved_deref_style_for_method(
    ctx: &mut InferState,
    method_ty: TypeId,
    method_site: ValId,
    mutable: bool,
) -> ResolvedStructDerefMethod {
    let (inputs, output) = method_signature_type_parts(ctx.ex.store, method_ty)
        .expect("validated deref method must have function signature type");
    let self_ty = *inputs
        .first()
        .expect("validated deref method must have self parameter");

    let (self_param_ty, self_style) = match ctx.ex.store.type_value(self_ty) {
        TypeValue::Ptr { tgt, style, .. } => (*tgt, *style),
        _ => unreachable!("validated deref method self must be pointer-like"),
    };
    let (target_ty, ret_style) = match ctx.ex.store.type_value(output) {
        TypeValue::Ptr { tgt, style, .. } => (*tgt, *style),
        _ => unreachable!("validated deref method return must be pointer-like"),
    };

    ResolvedStructDerefMethod {
        deref_site: (!mutable).then_some(method_site),
        deref_mut_site: mutable.then_some(method_site),
        mutable: Some(mutable),
        self_ptr: ctx.new_solved(self_ty),
        self_param: ctx.new_solved(self_param_ty),
        self_kind: PtrKind::Solved(self_style),
        target_ptr: ctx.new_solved(output),
        target: ctx.new_solved(target_ty),
        ret_kind: PtrKind::Solved(ret_style),
    }
}

fn check_inserted_deref_pair_compatible(
    ctx: &mut InferState,
    pending_deref_methods: &PendingStructDerefMethods,
    mismatch_site: ValId,
) {
    let (Some((deref_ty, _)), Some((deref_mut_ty, _))) =
        (pending_deref_methods.deref, pending_deref_methods.deref_mut)
    else {
        return;
    };

    let Some((deref_inputs, deref_output)) = method_signature_type_parts(ctx.ex.store, deref_ty)
    else {
        return;
    };
    let Some((deref_mut_inputs, deref_mut_output)) =
        method_signature_type_parts(ctx.ex.store, deref_mut_ty)
    else {
        return;
    };
    if deref_inputs.len() != 1 || deref_mut_inputs.len() != 1 {
        return;
    }

    let (deref_self_style, deref_self_mut) = match ctx.ex.store.type_value(deref_inputs[0]) {
        TypeValue::Ptr { style, mutable, .. } => (*style, *mutable),
        _ => return,
    };
    let (deref_mut_self_style, deref_mut_self_mut) =
        match ctx.ex.store.type_value(deref_mut_inputs[0]) {
            TypeValue::Ptr { style, mutable, .. } => (*style, *mutable),
            _ => return,
        };
    let (deref_out_style, deref_out_mut, deref_target) = match ctx.ex.store.type_value(deref_output)
    {
        TypeValue::Ptr {
            style,
            mutable,
            tgt,
        } => (*style, *mutable, *tgt),
        _ => return,
    };
    let (deref_mut_out_style, deref_mut_out_mut, deref_mut_target) =
        match ctx.ex.store.type_value(deref_mut_output) {
            TypeValue::Ptr {
                style,
                mutable,
                tgt,
            } => (*style, *mutable, *tgt),
            _ => return,
        };

    if deref_target != deref_mut_target
        || deref_self_style != deref_mut_self_style
        || deref_out_style != deref_mut_out_style
        || deref_self_mut
        || !deref_mut_self_mut
        || deref_out_mut
        || !deref_mut_out_mut
    {
        ctx.push_error(TypeError::Simple {
            loc: ctx.ex.program.value_loc(mismatch_site),
            message: "`__deref` and `__deref_mut` must dereference to the same target type",
        });
    }
}

impl StructRep {
    fn new(
        names: impl Iterator<Item = NameId>,
        gen_info: Vec<TraitInfo>,
        life_count: usize,
        lifetime_orderings: Vec<LifetimeOrderingEdge>,
        generic_lifetime_requirements: Vec<GenericLifetimeRequirement>,
        layout: StructLayoutSpec,
    ) -> Self {
        Self {
            //TODO: when solving typedefs in finalize we want to set this value
            //for anonymous structs it wont exist but those are rare
            name: None,
            fields: names.map(|x| (x, UNKNOWN_TYPE)).collect(),
            gen_info,
            life_count,
            lifetime_orderings,
            generic_lifetime_requirements,
            layout,
        }
    }

    // pub(crate) fn with_fields(name: Option<NameId>, fields: Vec<(NameId, TypeId)>) -> Self {
    //     Self { name, fields,gen_count:0 }
    // }
}
