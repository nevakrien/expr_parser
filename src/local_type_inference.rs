use crate::global_type_inference::{
    do_typedef, gather_func_signature, is_any_type_builtin_member_name,
    receiver_cluster_for_self_param,
};
use crate::identity_hasher::IdHashMap;
use crate::ir::AccessKind;
use crate::ir::CallingConvention;
use crate::ir::GenDec;
use crate::ir::PatternSpan;
use crate::ir::TExpId;
use crate::ir::VarKind;
use crate::ir::{AssignOp, BinOp, Dir, Literal, NameId, UnOp, ValId, Value};
use crate::program::{Defined, Program};
use crate::string_intern::{
    ADD_STR, ALIGN_OF_STR, BITAND_STR, BITNOT_STR, BITOR_STR, BITXOR_STR, DIV_STR, EQ_STR,
    FREE_STR, GE_STR, GT_STR, LE_STR, LT_STR, MOD_STR, MUL_STR, NE_STR, NEG_STR, NOT_STR,
    POST_DEC_STR, POST_INC_STR, PRE_DEC_STR, PRE_INC_STR, SHL_STR, SHR_STR, SIZE_OF_STR, SUB_STR,
    StrId,
};
use crate::type_inference::*;

pub fn infer_value_internals<'a>(
    program: &'a Program,
    store: &'a mut TypeStore,
    ans: &'a mut SolvedTypes,
    value: ValId,
) -> Result<&'a mut SolvedTypes, Vec<TypeError>> {
    let known_function_exists = ans.function_types_by_value(value).is_some();
    let known_function_ty = ans.function_types_by_value(value).map(|known| known.ty);
    let mut ctx = InferState::new(store, program, ans);
    ctx.req.owner = Some(value);
    let mut restore_name_render: Option<GenLifeNameRender<'a>> = None;

    match ctx.ex.program.value(value) {
        Value::Func {
            calling_convention,
            generics,
            params,
            output_type,
            body,
        } => {
            let previous_name_render = std::mem::replace(
                &mut ctx.ex.name_render,
                GenLifeNameRender::from_decl(ctx.ex.program, generics),
            );
            restore_name_render = Some(previous_name_render);

            let output = if known_function_exists {
                load_known_function_signature_for_value(&mut ctx, value)
            } else {
                let (_found_sig, output) = gather_func_signature::<true>(
                    &mut ctx,
                    value,
                    calling_convention,
                    generics,
                    params,
                    output_type,
                );
                output
            };

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

            ctx.new_solved(known_function_ty.unwrap_or(UNKNOWN_TYPE))
        }
        _ => {
            let found = gather_constraints(&mut ctx, value, None);

            found
        }
    };

    local_solver(&mut ctx);

    if let Some(previous_name_render) = restore_name_render {
        ctx.ex.name_render = previous_name_render;
    }

    //this debug assert is mostly meaningless
    //it shouldnt even be SET by us in the firstplace
    //we specifically do NOT bind_val and finalize cant handle generics
    //so this trigers as soon as we fuckup and bind_val ourselvs on anything with generics
    if let Some(known_ty) = known_function_ty {
        debug_assert_eq!(known_ty, ctx.ex.ans.type_of(value).unwrap())
    }

    if ctx.ex.errors.is_empty() {
        Ok(ctx.ex.ans)
    } else {
        Err(ctx.ex.errors)
    }
}

pub fn local_solver(ctx: &mut InferState) {
    //this loop only exists once ALL requirments have checked and didnt complain
    //on the state we are gona release. since there was no change
    //this is SUPER important because they are not just progressions
    let mut unknown_count = 0;

    loop {
        let mut progress = false;
        progress |= resolve_operator_types(ctx);
        progress |= resolve_deferred_types(ctx);
        progress |= resolve_pending_indexes(ctx);
        progress |= resolve_pending_member_accesses(ctx);
        progress |= resolve_pending_int_accesses(ctx);
        progress |= resolve_pending_specializations(ctx);

        if progress {
            continue;
        }

        if resolve_pending_derefs(ctx) {
            continue;
        }

        //ORDER SENSATIVE semi hacks
        //these are all assuming defualts on the type system
        //so they are mostly last resorts for that exact reason

        if finalize_unresolved_lifetimes_as_unknown(ctx, &mut unknown_count) {
            continue;
        }

        if force_unresolved_refs_to_safe(ctx, &mut unknown_count) {
            continue;
        }

        if force_unresolved_ptr_mutability_to_immut(ctx) {
            continue;
        }

        break;
    }

    if !ctx.ex.errors.is_empty() {
        return;
    }

    finalize_local(ctx);
}

#[inline(always)]
fn force_unresolved_refs_to_safe(ctx: &mut InferState, unknown_count: &mut u32) -> bool {
    let mut progress = false;

    for i in 0..ctx.types.core.cluster.len() {
        let cid = CId(i);

        if ctx.types.core.parent[cid] != cid {
            continue;
        }

        let ResolveKind::Ptr { tgt, kind, mutable } = ctx.types.cluster_state(cid) else {
            continue;
        };

        let should_force = matches!(kind, PtrKind::SafeRef | PtrKind::SomeRef | PtrKind::Unknown);
        if !should_force {
            continue;
        }

        // mint an unknown lifetime for display/model completion
        let lt = LifeTime::Unknown(LifeId(*unknown_count));
        *unknown_count += 1;

        ctx.types.set_cluster_state(
            cid,
            ResolveKind::Ptr {
                tgt,
                kind: PtrKind::Solved(PointerStyle::Ref(lt)),
                mutable,
            },
        );

        progress = true;
    }

    progress
}

#[inline(always)]
fn force_unresolved_ptr_mutability_to_immut(ctx: &mut InferState) -> bool {
    let mut progress = false;

    for i in 0..ctx.types.core.cluster.len() {
        let cid = CId(i);

        // only roots
        if ctx.types.core.parent[cid] != cid {
            continue;
        }

        let ResolveKind::Ptr { tgt, kind, mutable } = ctx.types.cluster_state(cid) else {
            continue;
        };

        if mutable.is_some() {
            continue;
        }

        ctx.types.set_cluster_state(
            cid,
            ResolveKind::Ptr {
                tgt,
                kind,
                mutable: Some(false),
            },
        );

        progress = true;
    }

    progress
}

#[inline(always)]
fn finalize_unresolved_lifetimes_as_unknown(ctx: &mut InferState, unknown_count: &mut u32) -> bool {
    let mut progress = false;
    //should properly increment

    for lid in ctx.types.life_parent.0.iter() {
        if *lid != ctx.types.life_parent[*lid] {
            continue;
        }

        if ctx.types.life_known[*lid].is_none() {
            let hack = LifeId(*unknown_count);
            ctx.types.life_known[*lid] = Some(LifeTime::Unknown(hack));
            *unknown_count += 1;
            progress = true;
        }
    }

    progress
}

fn finalize_local(ctx: &mut InferState) {
    let InferState {
        search,
        req,
        types,
        ex,
        ..
    } = ctx;

    let val_cluster = &search.val_cluster;
    let pat_cluster = &search.pat_cluster;
    let member_method_type_sites = &req.member_method_type_sites;
    let member_access_implicit_deref_sites = &req.member_access_implicit_deref_sites;
    let index_implicit_deref_sites = &req.index_implicit_deref_sites;

    // unsafe{perf_begin();}

    let mut reported: IdHashMap<CId, ()> = IdHashMap::default();
    let mut member_method_by_site: IdHashMap<ValId, PendingMemberMethodType> = IdHashMap::default();
    for &entry in member_method_type_sites {
        member_method_by_site.insert(entry.site, entry);
    }

    let mut inner = InnerFunctionTypes::default();

    for (v, c) in val_cluster {
        let root = types.root(*c);
        if let Some(t) = types.cluster_solved_type(root) {
            inner.val_types.insert(*v, t);
        } else if *c == root && !reported.contains_key(&c) {
            let found = types.bad_type(ex, root);
            ex.errors.push(TypeError::Unresolved { value: *v, found });
            reported.insert(root, ());
            if let Some(entry) = member_method_by_site.get(&v) {
                let full_root = types.root(entry.full_method);
                reported.insert(full_root, ());
            }
        }
    }

    for (p, c) in pat_cluster {
        let root = types.root(*c);
        if let Some(t) = types.cluster_solved_type(root) {
            inner.pat_types.insert(*p, t);
        } else if *c == root && !reported.contains_key(&c) {
            let found = types.bad_type(ex, root);
            ex.errors
                .push(TypeError::UnresolvedPattern { pattern: *p, found });
            reported.insert(root, ());
        }
    }

    for entry in member_method_type_sites {
        let root = types.root(entry.full_method);
        if let Some(full_type) = types.cluster_solved_type(root) {
            inner.member_method_types.insert(
                entry.site,
                SolvedMemberMethodAccessType {
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

        let receiver_root = types.root(entry.receiver);
        if types.cluster_solved_type(receiver_root).is_none()
            && !reported.contains_key(&receiver_root)
        {
            let found = types.bad_type(ex, receiver_root);
            ex.errors.push(TypeError::Unresolved {
                value: entry.receiver_value,
                found,
            });
            reported.insert(receiver_root, ());
            reported.insert(root, ());
        }
    }

    store_implicit_deref_chains(
        &mut inner.implicit_derefs,
        member_access_implicit_deref_sites,
        &mut types.core.parent,
        &types.core.cluster,
    );
    store_implicit_deref_chains(
        &mut inner.implicit_derefs,
        index_implicit_deref_sites,
        &mut types.core.parent,
        &types.core.cluster,
    );

    let owner = req.owner.or_else(|| {
        val_cluster
            .iter()
            .find_map(|(v, _)| matches!(ex.program.value(*v), Value::Func { .. }).then_some(*v))
            .or_else(|| val_cluster.first().map(|(v, _)| *v))
    });

    if let Some(owner) = owner {
        ex.ans.set_function_inner(owner, inner);
    }

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
            if let Some(t) = cluster[root].solved_ty {
                chain.push(t);
            } else {
                all_solved = false;
                break;
            }
        }
        if all_solved {
            out.insert(entry.site, chain);
        }
    }
}

pub(crate) fn gather_func_constraints<const GLOBAL_SCOPE: bool>(
    ctx: &mut InferState,
    v: ValId,
    calling_convention: CallingConvention,
    generics: GenDec,
    params: PatternSpan,
    output_type: Option<TExpId>,
    body: Option<ValId>,
) -> CId {
    let previous_name_render = std::mem::replace(
        &mut ctx.ex.name_render,
        GenLifeNameRender::from_decl(ctx.ex.program, generics),
    );

    let (f, output) = gather_func_signature::<GLOBAL_SCOPE>(
        ctx,
        v,
        calling_convention,
        generics,
        params,
        output_type,
    );

    let Some(body) = body else {
        ctx.ex.name_render = previous_name_render;
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
    ctx.ex.name_render = previous_name_render;
    f
}
pub(crate) fn gather_constraints(
    ctx: &mut InferState,
    v: ValId,
    current_output: Option<CId>,
) -> CId {
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

        Value::Poison => {
            let c = ctx.new_solved(UNKNOWN_TYPE);
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

            if let Some(f) = ctx.ex.ans.function_types_by_name(n) {
                let t = f.ty;
                return global_to_specialized_local(
                    &mut ctx.ex,
                    &mut ctx.search,
                    &mut ctx.types,
                    t,
                    v,
                );
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
                Defined::Func(_funcs) => {
                    unreachable!("we checked for it earlier")
                }
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
                kind: PtrKind::Unknown,
                mutable,
            };
            ctx.bind_val(v, ans);
            ans
        }

        Value::Deref(base) => {
            let output = ctx.new_cluster();
            ctx.bind_val(v, output);

            let src = gather_constraints(ctx, base, current_output);
            let mut pending = PendingDeref::new(&mut ctx.types, v, base, src, output);

            let outcome = pending.step(&mut ctx.ex, &mut ctx.types);
            if outcome.retain {
                ctx.req.pending_derefs.push(pending);
            }

            output
        }

        Value::Access { base, name, kind } => {
            //special case static members like we do functions
            if kind == AccessKind::Static {
                let Value::NameRef(sname) = ctx.ex.program.value(base) else {
                    let loc = ctx.ex.program.value_loc(v);
                    ctx.push_error(TypeError::Simple {
                        loc,
                        message: "static methods require a struct name",
                    });
                    return ctx.new_cluster();
                };

                let Some(types) = ctx.ex.ans.member_function_types_by_name(sname, name) else {
                    let loc = ctx.ex.program.value_loc(v);
                    ctx.push_error(TypeError::Simple {
                        loc,
                        message: "static methods require a struct name",
                    });
                    return ctx.new_cluster();
                };
                let t = types.ty;
                return global_to_specialized_local(
                    &mut ctx.ex,
                    &mut ctx.search,
                    &mut ctx.types,
                    t,
                    v,
                );
            }

            let source = gather_constraints(ctx, base, current_output);

            if is_any_type_builtin_member_name(name) {
                let result = resolve_any_type_builtin_member_access(
                    &mut ctx.ex,
                    &mut ctx.types,
                    &mut ctx.search,
                    &mut ctx.req.member_method_type_sites,
                    v,
                    base,
                    source,
                    name,
                );
                ctx.bind_val(v, result);
                return result;
            }

            let out = ctx.new_cluster();
            let mut pending =
                PendingMemberAccess::new(&mut ctx.types, v, base, source, out, name, kind);

            match pending.step(
                &mut ctx.ex,
                &mut ctx.types,
                &mut ctx.search,
                &mut ctx.req.member_method_type_sites,
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

                MemberAccessResolve::Pending { .. } => {
                    let result = pending.output;
                    ctx.bind_val(v, result);

                    // IMPORTANT: push the entire frame
                    ctx.req.pending_member_accesses.push(pending);

                    result
                }

                MemberAccessResolve::Error(err) => {
                    ctx.push_error(err);

                    let result = pending.output;
                    ctx.bind_val(v, result);

                    result
                }
            }
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
        } => {
            ctx.push_error(TypeError::Simple {
                loc: ctx.ex.program.value_loc(v),
                message: CLOSURES_UNSUPPORTED_MSG,
            });
            gather_func_constraints::<false>(
                ctx,
                v,
                calling_convention,
                generics,
                params,
                output_type,
                body,
            )
        }
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
                    lifetimes: 0,
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
                ctx.push_error(TypeError::UnresolvedTypeExpr {
                    expr: *texp,
                    found: None,
                });
                for arg in cons.args.ids() {
                    gather_constraints(ctx, arg, current_output);
                }
                let ans = ctx.new_cluster();
                ctx.bind_val(v, ans);
                return ans;
            };
            let base_type = *base_type;

            let sid = match ctx.ex.store.type_value(base_type) {
                TypeValue::Struct {
                    id, generics: _, ..
                } => *id,
                // TypeValue::Specialized { base, .. } => {
                //     match ctx.ex.store.type_value(*base) {
                //         TypeValue::Struct(sid) => *sid,
                //         _ => {
                //             ctx.push_error(TypeError::ConstructorBaseNotStruct {
                //                 site: cons.base,
                //                 found: Some(ctx.ex.store.get_type_string(ctx.ex.program, *base)),
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
                        found: Some(ctx.ex.store.get_type_string(ctx.ex.program, base_type)),
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

            let (glen, lifetime_generics) = match ctx.ex.store.type_value(base_type) {
                TypeValue::Struct {
                    id: _,
                    generics,
                    lifetimes,
                } => (generics.len(), lifetimes),
                _ => unreachable!("verified above"),
            };
            let llen = lifetime_generics.len();

            let generic_clusters = (0..glen).map(|_| ctx.new_cluster()).collect::<Vec<_>>();
            let lifetime_clusters = (0..llen).map(|_| ctx.types.new_lid()).collect::<Vec<_>>();

            let mut field_type_clusters = None;
            if glen != 0 || llen != 0 {
                let flen = ctx.ex.store.struct_value(sid).fields.len();

                field_type_clusters = Some(
                    (0..flen)
                        .map(|f| {
                            let (_, t) = ctx.ex.store.struct_value(sid).fields[f];
                            specialize_type(
                                &mut ctx.ex,
                                &mut ctx.types,
                                t,
                                &generic_clusters,
                                &lifetime_clusters,
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

            if glen == 0 && llen == 0 {
                let t = ctx.ex.store.intern(TypeValue::Struct {
                    id: sid,
                    generics: Vec::new(),
                    lifetimes: Vec::new(),
                });
                let ans = ctx.new_solved(t);
                ctx.bind_val(v, ans);
                return ans;
            }

            let ans = ctx.new_struct_instance(sid, generic_clusters, lifetime_clusters);
            ctx.bind_val(v, ans);
            ans
        }

        Value::IntAccess { base, id, kind } => {
            let source = gather_constraints(ctx, base, current_output);
            match try_resolve_tuple_int_access(&mut ctx.ex, &mut ctx.types, v, source, id, kind) {
                IntAccessResolve::Resolved {
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
                IntAccessResolve::Pending { source } => {
                    let result = ctx.new_cluster();
                    ctx.bind_val(v, result);
                    ctx.req.pending_int_accesses.push(PendingIntAccess {
                        site: v,
                        source,
                        output: result,
                        id,
                        kind,
                    });
                    result
                }
                IntAccessResolve::Error(err) => {
                    ctx.push_error(err);
                    let result = ctx.new_cluster();
                    ctx.bind_val(v, result);
                    result
                }
            }
        }
        Value::Goto(_) | Value::Break | Value::Continue | Value::LabelDecl(_) => ctx.new_cluster(),
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

            ctx.new_cluster()
        }
        Value::LogicOp { op: _, values } => {
            let out = ctx.new_solved(BuiltinType::Bool.into());
            let a = gather_constraints(ctx, values.0, current_output);
            if let Err(clash) = ctx.unify(a, out) {
                ctx.push_error(TypeError::ValuesContradict {
                    site: v,
                    expected_place: v,
                    found: values.0,
                    expectation_reason: "boolean logic can only be done on bools",
                    clash,
                })
            }
            let b = gather_constraints(ctx, values.1, current_output);
            if let Err(clash) = ctx.unify(b, out) {
                ctx.push_error(TypeError::ValuesContradict {
                    site: v,
                    expected_place: v,
                    found: values.1,
                    expectation_reason: "boolean logic can only be done on bools",
                    clash,
                })
            }
            out
        }

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

            let mut site = PendingIndex::new(
                &mut ctx.types,
                v,
                call.base,
                pos_args[0],
                base,
                pos_arg_clusters[0],
                output,
            );

            let outcome = site.step(&mut ctx.ex, &mut ctx.types);

            if outcome.retain {
                ctx.req.pending_indexes.push(site);
            } else if !site.implicit_deref.implicit_receivers.is_empty() {
                ctx.req
                    .index_implicit_deref_sites
                    .push(PendingMemberAccessImplicitDeref {
                        site: v,
                        receivers: site.implicit_deref.implicit_receivers,
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

fn load_known_function_signature_for_value(ctx: &mut InferState, value: ValId) -> CId {
    let Some(known_ty) = ctx
        .ex
        .ans
        .function_types_by_value(value)
        .map(|known| known.ty)
    else {
        return ctx.new_cluster();
    };

    let ret_ty = match ctx.ex.store.type_value(known_ty) {
        TypeValue::Func { ret, .. } => *ret,
        _ => return ctx.new_cluster(),
    };

    let argument_count = ctx
        .ex
        .ans
        .function_types_by_value(value)
        .map_or(0, |known| known.arguments.len());

    for i in 0..argument_count {
        let Some((pat, maybe_name, ty)) = ctx
            .ex
            .ans
            .function_types_by_value(value)
            .and_then(|known| known.arguments.get(i))
            .copied()
        else {
            continue;
        };

        let c = if ty == UNKNOWN_TYPE {
            ctx.new_cluster()
        } else {
            ctx.new_solved(ty)
        };
        ctx.bind_pat(pat, c);
        if let Some(name) = maybe_name {
            ctx.search.names.insert(name, c);
        }
    }

    let generic_param_count = ctx
        .ex
        .ans
        .function_types_by_value(value)
        .map_or(0, |known| known.generic_parameters.len());

    for i in 0..generic_param_count {
        let maybe_name = ctx
            .ex
            .ans
            .function_types_by_value(value)
            .and_then(|known| known.generic_parameters.get(i))
            .and_then(|(_pat, maybe_name)| *maybe_name);

        let generic_ty = ctx.ex.store.intern(TypeValue::Generic(GenId(i)));
        let generic_cid = ctx.new_solved(generic_ty);
        if let Some(name) = maybe_name {
            ctx.search.local_types.insert(name, generic_cid);
            ctx.search.names.insert(name, generic_cid);
        }
    }

    let lifetime_param_count = ctx
        .ex
        .ans
        .function_types_by_value(value)
        .map_or(0, |known| known.lifetime_parameters.len());

    for i in 0..lifetime_param_count {
        let lid = ctx.types.new_lid_known(LifeTime::External(i as u32));
        if let Some((pat, maybe_lt_name)) = ctx
            .ex
            .ans
            .function_types_by_value(value)
            .and_then(|known| known.lifetime_parameters.get(i))
            .copied()
            && let Some(lt_name) = maybe_lt_name
        {
            ctx.search
                .local_lifetimes
                .insert(lt_name, (LifeTime::External(i as u32), lid));
            let _ = pat;
        }
    }

    ctx.new_solved(ret_ty)
}

#[inline(always)]
fn try_resolve_tuple_int_access(
    ex: &mut ExternState,
    types: &mut TypeState,
    site: ValId,
    source: CId,
    id: usize,
    kind: AccessKind,
) -> IntAccessResolve {
    let mut current = types.root(source);
    let mut implicit_receivers = Vec::new();
    let max_implicit_deref_steps = match kind {
        AccessKind::Dot => 1usize,
        AccessKind::Ptr => 64usize,
        AccessKind::Static => 0usize,
    };
    let implicit_deref_limit_message = match kind {
        AccessKind::Dot => "`.` tuple access performs at most one implicit dereference",
        AccessKind::Ptr => "tuple access autoderef recursion exceeded safety limit",
        AccessKind::Static => "static tuple access does not support implicit dereference",
    };
    let mut used_implicit_deref_steps = 0usize;

    loop {
        if let Some(t) = types.cluster_solved_type(current) {
            match ex.store.type_value(t) {
                TypeValue::Ptr { tgt, .. } => {
                    if used_implicit_deref_steps >= max_implicit_deref_steps {
                        return IntAccessResolve::Error(TypeError::Simple {
                            loc: ex.program.value_loc(site),
                            message: implicit_deref_limit_message,
                        });
                    }
                    let next = types.new_solved(ex.store, *tgt);
                    let next = types.root(next);
                    implicit_receivers.push(current);
                    used_implicit_deref_steps += 1;
                    current = next;
                    continue;
                }
                TypeValue::Tuple(items) => {
                    if kind == AccessKind::Static {
                        return IntAccessResolve::Error(TypeError::Simple {
                            loc: ex.program.value_loc(site),
                            message: "tuple element access does not support `::`",
                        });
                    }
                    let Some(item) = items.get(id).copied() else {
                        return IntAccessResolve::Error(TypeError::Simple {
                            loc: ex.program.value_loc(site),
                            message: "tuple element index is out of bounds for this tuple",
                        });
                    };
                    let result = types.new_solved(ex.store, item);
                    return IntAccessResolve::Resolved {
                        result,
                        implicit_receivers: finalize_member_access_implicit_chain(
                            implicit_receivers,
                            used_implicit_deref_steps,
                            current,
                        ),
                    };
                }
                _ => {
                    return IntAccessResolve::Error(TypeError::Simple {
                        loc: ex.program.value_loc(site),
                        message: "tuple element access requires a tuple or pointer-like base",
                    });
                }
            }
        }

        match types.core.cluster[current].state {
            ResolveKind::Nothing => return IntAccessResolve::Pending { source: current },
            ResolveKind::Ptr { tgt, .. } => {
                if used_implicit_deref_steps >= max_implicit_deref_steps {
                    return IntAccessResolve::Error(TypeError::Simple {
                        loc: ex.program.value_loc(site),
                        message: implicit_deref_limit_message,
                    });
                }
                let next = types.root(tgt);
                implicit_receivers.push(current);
                used_implicit_deref_steps += 1;
                current = next;
            }
            ResolveKind::Tuple(tuple_id) => {
                if kind == AccessKind::Static {
                    return IntAccessResolve::Error(TypeError::Simple {
                        loc: ex.program.value_loc(site),
                        message: "tuple element access does not support `::`",
                    });
                }
                let Some(result) = types.extra.tuple_infers[tuple_id.0].items.get(id).copied()
                else {
                    return IntAccessResolve::Error(TypeError::Simple {
                        loc: ex.program.value_loc(site),
                        message: "tuple element index is out of bounds for this tuple",
                    });
                };
                return IntAccessResolve::Resolved {
                    result,
                    implicit_receivers: finalize_member_access_implicit_chain(
                        implicit_receivers,
                        used_implicit_deref_steps,
                        current,
                    ),
                };
            }
            _ => {
                return IntAccessResolve::Error(TypeError::Simple {
                    loc: ex.program.value_loc(site),
                    message: "tuple element access requires a tuple or pointer-like base",
                });
            }
        }
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

fn solved_type_to_specialized_local(
    ex: &mut ExternState,
    types: &mut TypeState,
    t: TypeId,
    loc: ValId,
) -> CId {
    //BUG
    if let TypeValue::Func {
        generics,
        lifetimes,
        ..
    } = *ex.store.type_value(t)
    {
        let gens: Vec<_> = (0..generics).map(|_| types.new_cluster()).collect();
        let lifes: Vec<_> = (0..lifetimes).map(|_| types.new_lid()).collect();
        return specialize_type(ex, types, t, &gens, &lifes, loc);
    }

    types.new_solved(ex.store, t)
}

fn global_to_specialized_local(
    ex: &mut ExternState,
    search: &mut SearchState,
    types: &mut TypeState,
    reference_type: TypeId,
    v: ValId,
) -> CId {
    if reference_type == UNKNOWN_TYPE {
        let loc = ex.program.value_loc(v);
        let c = types.new_cluster();
        ex.push_error(TypeError::Simple {
            loc,
            message: "global function has no inferred signature",
        });
        search.bind_val(v, c);
        return c;
    }

    //TODO this check is actually CURRENTLY non exustive
    //we wana make sure that we add a good way to run this
    //would be done as some normlization function somewhere
    //structs especially are weird with this
    let ans = solved_type_to_specialized_local(ex, types, reference_type, v);
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
    member_name: StrId,
    method_ty: TypeId,
) -> CId {
    let method_local = solved_type_to_specialized_local(ex, types, method_ty, access_site);

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

fn resolve_any_type_builtin_member_access(
    ex: &mut ExternState,
    types: &mut TypeState,
    search: &mut SearchState,
    member_method_type_sites: &mut Vec<PendingMemberMethodType>,
    access_site: ValId,
    base_value: ValId,
    base_cluster: CId,
    member_name: StrId,
) -> CId {
    //this assumes u can full solve here
    //u cant we dont know the lifetime at all
    //we need to do a downcast of it which is a bit anoying
    let (self_param, output) = if member_name == FREE_STR {
        let generic_self = types.new_cluster();
        let self_param = types.new_cluster();
        types.core.cluster[self_param].state = ResolveKind::Ptr {
            tgt: generic_self,
            kind: PtrKind::SafeRef,
            mutable: Some(true),
        };
        (
            self_param,
            types.new_solved(ex.store, BuiltinType::Void.into()),
        )
    } else if matches!(member_name, SIZE_OF_STR | ALIGN_OF_STR) {
        let self_param = types.new_cluster();
        types.core.cluster[self_param].state = ResolveKind::Ptr {
            tgt: base_cluster,
            kind: PtrKind::Solved(PointerStyle::Raw(Nullable::No)),
            mutable: Some(false),
        };
        (
            self_param,
            types.new_solved(ex.store, BuiltinType::Usize.into()),
        )
    } else {
        ex.push_error(TypeError::IlegalMethod {
            member_name,
            access_site,
        });
        return types.new_cluster();
    };

    let full_method = types.new_func(FuncInfer {
        calling_convention: CallingConvention::Unknown,
        generics: 0,
        lifetimes: 0,
        inputs: vec![self_param],
        output,
    });
    let overload = ResolvedMemberOverload {
        params: vec![self_param],
        ret: output,
        full_method,
    };

    match make_member_closure(ex, types, base_cluster, overload, access_site) {
        Ok(curried) => {
            member_method_type_sites.push(PendingMemberMethodType {
                site: access_site,
                member: member_name,
                full_method,
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
                full_method,
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
fn resolve_struct_deref_method(
    ex: &mut ExternState,
    types: &mut TypeState,
    site: ValId,
    method_ty: TypeId,
) -> Option<ResolvedStructDerefMethod> {
    let method_local = solved_type_to_specialized_local(ex, types, method_ty, site);

    let Some((params, ret)) = function_parts_from_cluster(ex, types, method_local) else {
        unreachable!("specialized deref method must resolve to a function shape");
    };

    let self_param = params.first().copied()?;
    debug_assert_eq!(params.len(), 1);

    let (_, self_kind, self_mutable) = ptr_parts_from_cluster(ex, types, self_param)?;
    if matches!(self_kind.is_fancy(), Some(false)) {
        return None;
    }

    let (target, ret_kind, ret_mutable) = ptr_parts_from_cluster(ex, types, ret)?;
    if matches!(ret_kind.is_fancy(), Some(false)) {
        return None;
    }

    Some(ResolvedStructDerefMethod {
        self_param,
        self_kind,
        self_mutable,
        target,
        ret_kind,
        ret_mutable,
    })
}

fn resolve_struct_deref_target(
    ex: &mut ExternState,
    types: &mut TypeState,
    site: ValId,
    base_value: ValId,
    base_cluster: CId,
    struct_name: NameId,
    shared_lid: &mut Option<LId>,
    chain_mutability: &mut Option<bool>,
) -> Option<ResolvedStructDerefTarget> {
    let (deref, deref_mut) = ex
        .store
        .struct_overload_info(struct_name)
        .map(|info| (info.deref, info.deref_mut))
        .unwrap_or((None, None));

    let deref_resolved =
        deref.and_then(|method| resolve_struct_deref_method(ex, types, site, method));
    let deref_mut_resolved =
        deref_mut.and_then(|method| resolve_struct_deref_method(ex, types, site, method));

    let resolved = match (deref_resolved, deref_mut_resolved) {
        (Some(x), Some(_y)) => x,
        (Some(x), None) => x,
        (None, Some(y)) => y,
        (None, None) => return None,
    };

    let self_kind = resolved.self_kind;
    let ret_kind = resolved.ret_kind;
    let self_mutable = resolved.self_mutable;
    let ret_mutable = resolved.ret_mutable;
    let _ = shared_lid;
    let _ = chain_mutability;

    let receiver_input = types.new_cluster();
    types.set_cluster_state(
        receiver_input,
        ResolveKind::Ptr {
            tgt: base_cluster,
            kind: self_kind,
            mutable: self_mutable,
        },
    );

    if let Err(clash) = unify_if_distinct(ex, types, resolved.self_param, receiver_input) {
        ex.push_error(TypeError::ValuesContradict {
            expectation_reason: "deref receiver must match special deref method self parameter",
            site,
            found: base_value,
            expected_place: site,
            clash,
        });
        return None;
    }

    let deref_result_ptr = types.new_cluster();
    types.set_cluster_state(
        deref_result_ptr,
        ResolveKind::Ptr {
            tgt: resolved.target,
            kind: ret_kind,
            mutable: ret_mutable,
        },
    );

    Some(ResolvedStructDerefTarget {
        target: resolved.target,
        deref_receiver_ptr: receiver_input,
        deref_result_ptr,
    })
}

impl PendingImplicitDeref {
    #[inline(always)]
    fn sync_roots(&mut self, types: &mut TypeState) -> CId {
        self.source = types.root(self.source);
        self.current = types.root(self.current);
        self.current
    }

    #[inline(always)]
    fn step(
        &mut self,
        ex: &mut ExternState,
        types: &mut TypeState,
        max_implicit_deref_steps: usize,
        implicit_deref_limit_message: &'static str,
    ) -> Result<ImplicitDerefStep, TypeError> {
        let current = self.sync_roots(types);

        if let Some(t) = types.cluster_solved_type(current) {
            return match ex.store.type_value(t) {
                TypeValue::Ptr { tgt, .. } => {
                    if self.implicit_receivers.len() >= max_implicit_deref_steps {
                        return Err(TypeError::Simple {
                            loc: ex.program.value_loc(self.site),
                            message: implicit_deref_limit_message,
                        });
                    }

                    self.implicit_receivers.push(current);
                    let c = types.new_solved(ex.store, *tgt);
                    self.current = types.root(c);
                    Ok(ImplicitDerefStep::Stepped)
                }

                TypeValue::Struct { id, .. } => {
                    if self.implicit_receivers.len() >= max_implicit_deref_steps {
                        return Err(TypeError::Simple {
                            loc: ex.program.value_loc(self.site),
                            message: implicit_deref_limit_message,
                        });
                    }

                    let Some(struct_name) = ex.store.struct_value(*id).name else {
                        return Ok(ImplicitDerefStep::Done);
                    };

                    let Some(target) = resolve_struct_deref_target(
                        ex,
                        types,
                        self.site,
                        self.base_value,
                        current,
                        struct_name,
                        &mut self.deref_chain_lid,
                        &mut self.deref_chain_mutability,
                    ) else {
                        return Ok(ImplicitDerefStep::Done);
                    };

                    self.implicit_receivers.push(current);
                    self.implicit_receivers.push(target.deref_receiver_ptr);
                    self.implicit_receivers.push(target.deref_result_ptr);
                    self.current = types.root(target.target);
                    Ok(ImplicitDerefStep::Stepped)
                }

                _ => Ok(ImplicitDerefStep::Done),
            };
        }

        match types.core.cluster[current].state {
            ResolveKind::Nothing => Ok(ImplicitDerefStep::Pending),

            ResolveKind::Ptr { tgt, .. } => {
                if self.implicit_receivers.len() >= max_implicit_deref_steps {
                    return Err(TypeError::Simple {
                        loc: ex.program.value_loc(self.site),
                        message: implicit_deref_limit_message,
                    });
                }

                self.implicit_receivers.push(current);
                self.current = types.root(tgt);
                Ok(ImplicitDerefStep::Stepped)
            }

            ResolveKind::Struct(rid) => {
                if self.implicit_receivers.len() >= max_implicit_deref_steps {
                    return Err(TypeError::Simple {
                        loc: ex.program.value_loc(self.site),
                        message: implicit_deref_limit_message,
                    });
                }

                let sid = types.extra.struct_infers[rid.0].sid;
                let Some(struct_name) = ex.store.struct_value(sid).name else {
                    return Ok(ImplicitDerefStep::Done);
                };

                let Some(target) = resolve_struct_deref_target(
                    ex,
                    types,
                    self.site,
                    self.base_value,
                    current,
                    struct_name,
                    &mut self.deref_chain_lid,
                    &mut self.deref_chain_mutability,
                ) else {
                    return Ok(ImplicitDerefStep::Done);
                };

                self.implicit_receivers.push(current);
                self.implicit_receivers.push(target.deref_receiver_ptr);
                self.implicit_receivers.push(target.deref_result_ptr);
                self.current = types.root(target.target);
                Ok(ImplicitDerefStep::Stepped)
            }

            _ => Ok(ImplicitDerefStep::Done),
        }
    }

    #[inline(always)]
    fn finalize_chain(&mut self, resolved_base: CId) -> Vec<CId> {
        let used = self.implicit_receivers.len();
        let receivers = std::mem::take(&mut self.implicit_receivers);
        finalize_member_access_implicit_chain(receivers, used, resolved_base)
    }
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

fn specialize_struct_field_type(
    ex: &mut ExternState,
    types: &mut TypeState,
    site: ValId,
    _sid: StructId,
    field_ty: TypeId,
    generics: &[CId],
    lifetimes: &[LId],
) -> CId {
    specialize_type(ex, types, field_ty, generics, lifetimes, site)
}

impl PendingMemberAccess {
    #[inline(always)]
    fn step(
        &mut self,
        ex: &mut ExternState,
        types: &mut TypeState,
        search: &mut SearchState,
        member_method_type_sites: &mut Vec<PendingMemberMethodType>,
    ) -> MemberAccessResolve {
        let max_implicit_deref_steps = match self.kind {
            AccessKind::Dot => 1usize,
            AccessKind::Ptr => 64usize,
            AccessKind::Static => 0usize,
        };

        let implicit_deref_limit_message = match self.kind {
            AccessKind::Dot => "`.` member access performs at most one implicit dereference",
            AccessKind::Ptr => "member access autoderef recursion exceeded safety limit",
            AccessKind::Static => "static member access does not support implicit dereference",
        };

        loop {
            let current = self.implicit_deref.sync_roots(types);
            let solved = types.cluster_solved_type(current);

            match (solved, types.core.cluster[current].state) {
                (Some(t), _) => {
                    if let TypeValue::Struct {
                        id: sid,
                        generics,
                        lifetimes,
                    } = ex.store.type_value(t)
                    {
                        let (field_ty, struct_name) = {
                            let rep = ex.store.struct_value(*sid);
                            let field_ty = rep
                                .fields
                                .iter()
                                .find(|(n, _)| ex.program.name_str_id(*n) == self.member)
                                .map(|(_, t)| *t);
                            (field_ty, rep.name)
                        };

                        if let Some(field_ty) = field_ty {
                            if matches!(self.kind, AccessKind::Static) {
                                return MemberAccessResolve::Error(TypeError::Simple {
                                    loc: ex.program.value_loc(self.site),
                                    message: "static access cannot target instance field",
                                });
                            }

                            let generic_inputs = generics
                                .iter()
                                .map(|&t| types.new_solved(ex.store, t))
                                .collect::<Vec<_>>();

                            let lifetime_inputs = lifetimes
                                .iter()
                                .map(|&lt| types.new_lid_known(lt))
                                .collect::<Vec<_>>();

                            let result = specialize_struct_field_type(
                                ex,
                                types,
                                self.site,
                                *sid,
                                field_ty,
                                &generic_inputs,
                                &lifetime_inputs,
                            );

                            return MemberAccessResolve::Resolved {
                                result,
                                implicit_receivers: self.implicit_deref.finalize_chain(current),
                            };
                        }

                        if let Some(struct_name) = struct_name
                            && let Some(member_method) = ex
                                .ans
                                .member_function_types_by_name(struct_name, self.member)
                        {
                            let result = resolve_member_method_access(
                                ex,
                                types,
                                search,
                                member_method_type_sites,
                                self.site,
                                self.implicit_deref.base_value,
                                current,
                                self.member,
                                member_method.ty,
                            );

                            return MemberAccessResolve::Resolved {
                                result,
                                implicit_receivers: self.implicit_deref.finalize_chain(current),
                            };
                        }

                        if self.implicit_deref.implicit_receivers.len() < max_implicit_deref_steps {
                            match self.implicit_deref.step(
                                ex,
                                types,
                                max_implicit_deref_steps,
                                implicit_deref_limit_message,
                            ) {
                                Ok(ImplicitDerefStep::Stepped) => continue,
                                Ok(ImplicitDerefStep::Pending) => {
                                    return MemberAccessResolve::Pending {
                                        source: self.implicit_deref.source,
                                    };
                                }
                                Ok(ImplicitDerefStep::Done) => {
                                    return MemberAccessResolve::Error(TypeError::UnknownField {
                                        field: self.member,
                                        site: self.site,
                                    });
                                }
                                Err(err) => return MemberAccessResolve::Error(err),
                            }
                        }

                        return MemberAccessResolve::Error(TypeError::UnknownField {
                            field: self.member,
                            site: self.site,
                        });
                    }
                }

                (None, ResolveKind::Struct(rid)) => {
                    let sid = types.extra.struct_infers[rid.0].sid;

                    let (field_ty, struct_name) = {
                        let rep = ex.store.struct_value(sid);
                        let field_ty = rep
                            .fields
                            .iter()
                            .find(|(n, _)| ex.program.name_str_id(*n) == self.member)
                            .map(|(_, t)| *t);
                        (field_ty, rep.name)
                    };

                    if let Some(field_ty) = field_ty {
                        if matches!(self.kind, AccessKind::Static) {
                            return MemberAccessResolve::Error(TypeError::Simple {
                                loc: ex.program.value_loc(self.site),
                                message: "static access cannot target instance field",
                            });
                        }

                        let infer = &types.extra.struct_infers[rid.0];

                        //unfortunatly yes this does require a clone at the moment
                        //the reason is that we have to borrow struct_infers inside specilize as mut
                        //there is some tricks we can do here with unsafe as those SHOULD... never be changed during specilize
                        let gens = infer.generics.clone();
                        let lifes = infer.lifetimes.clone();
                        let result = specialize_struct_field_type(
                            ex, types, self.site, sid, field_ty, &gens, &lifes,
                        );

                        return MemberAccessResolve::Resolved {
                            result,
                            implicit_receivers: self.implicit_deref.finalize_chain(current),
                        };
                    }

                    if let Some(struct_name) = struct_name
                        && let Some(member_method) = ex
                            .ans
                            .member_function_types_by_name(struct_name, self.member)
                    {
                        let result = resolve_member_method_access(
                            ex,
                            types,
                            search,
                            member_method_type_sites,
                            self.site,
                            self.implicit_deref.base_value,
                            current,
                            self.member,
                            member_method.ty,
                        );

                        return MemberAccessResolve::Resolved {
                            result,
                            implicit_receivers: self.implicit_deref.finalize_chain(current),
                        };
                    }

                    if self.implicit_deref.implicit_receivers.len() < max_implicit_deref_steps {
                        match self.implicit_deref.step(
                            ex,
                            types,
                            max_implicit_deref_steps,
                            implicit_deref_limit_message,
                        ) {
                            Ok(ImplicitDerefStep::Stepped) => continue,
                            Ok(ImplicitDerefStep::Pending) => {
                                return MemberAccessResolve::Pending {
                                    source: self.implicit_deref.source,
                                };
                            }
                            Ok(ImplicitDerefStep::Done) => {
                                return MemberAccessResolve::Error(TypeError::UnknownField {
                                    field: self.member,
                                    site: self.site,
                                });
                            }
                            Err(err) => return MemberAccessResolve::Error(err),
                        }
                    }

                    return MemberAccessResolve::Error(TypeError::UnknownField {
                        field: self.member,
                        site: self.site,
                    });
                }

                _ => {}
            }

            match self.implicit_deref.step(
                ex,
                types,
                max_implicit_deref_steps,
                implicit_deref_limit_message,
            ) {
                Ok(ImplicitDerefStep::Stepped) => continue,
                Ok(ImplicitDerefStep::Pending) => {
                    return MemberAccessResolve::Pending {
                        source: self.implicit_deref.source,
                    };
                }
                Ok(ImplicitDerefStep::Done) => {
                    return MemberAccessResolve::Error(TypeError::Simple {
                        loc: ex.program.value_loc(self.site),
                        message: "member access requires a struct or pointer-like base",
                    });
                }
                Err(err) => return MemberAccessResolve::Error(err),
            }
        }
    }
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
    method_ty: TypeId,
    loc: ValId,
) -> Option<ResolvedMemberOverload> {
    let method_local = solved_type_to_specialized_local(ex, types, method_ty, loc);
    let Some((params, ret)) = function_parts_from_cluster(ex, types, method_local) else {
        unreachable!("specialized operator overload method must resolve to a function shape");
    };

    Some(ResolvedMemberOverload {
        params,
        ret,
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
        full_method: _,
    } = method;
    debug_assert!(!params.is_empty());

    let self_param = params.remove(0);
    let self_input = receiver_cluster_for_self_param(ex, types, receiver, self_param)
        .ok_or_else(|| types.clash(ex, self_param, receiver))?;
    unify_if_distinct(ex, types, self_param, self_input)?;

    Ok(types.new_func(FuncInfer {
        calling_convention: CallingConvention::Unknown,
        generics: 0,
        lifetimes: 0,
        inputs: params,
        output: ret,
    }))
}

#[inline(always)]
fn function_parts_from_cluster(
    ex: &ExternState,
    types: &mut TypeState,
    cid: CId,
) -> Option<(Vec<CId>, CId)> {
    let root = types.root(cid);

    if let Some(t) = types.cluster_solved_type(root) {
        let TypeValue::Func { params, ret, .. } = ex.store.type_value(t) else {
            return None;
        };

        // Reify solved function type into fresh local clusters
        let inputs = params
            .iter()
            .map(|p| types.new_solved(ex.store, *p))
            .collect::<Vec<_>>();

        let output = types.new_solved(ex.store, *ret);
        return Some((inputs, output));
    }

    match types.cluster_state(root) {
        ResolveKind::Func(call) => {
            // IMPORTANT:
            // clone inputs because unify may mutate graph later
            let inputs = types.func(call).inputs.clone();
            let output = types.func(call).output;
            Some((inputs, output))
        }

        _ => None,
    }
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

        let method_name = bin_op_overload_name(op);
        let method = ex
            .store
            .struct_overload_info(struct_name)
            .and_then(|info| info.operators.get(&method_name))
            .copied();

        if let Some(method) = method {
            let Some(overload_sig) =
                resolve_member_overload_signature(ex, types, method.method_type, site.loc)
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
                let full_method = overload_sig.full_method;
                let method_closure = make_member_closure(ex, types, lhs, overload_sig, site.loc)?;
                let expected_fn = types.new_func(FuncInfer {
                    calling_convention: CallingConvention::Unknown,
                    generics: 0,
                    lifetimes: 0,
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
            && let ResolveKind::Ptr { ref mut kind, .. } = types.core.cluster[lhs].state
            && kind.is_fancy().is_none()
        {
            progress = true;
            *kind = PtrKind::SafeRef;
        }

        // else if matches!(kind.is_fancy(), Some(true)) {
        //     todo!("error because pointer arithmetic is for nullables")
        // }

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

        let method_name = un_op_overload_name(op);
        let method = ex
            .store
            .struct_overload_info(struct_name)
            .and_then(|info| info.operators.get(&method_name))
            .copied();
        if let Some(method) = method {
            let Some(overload_sig) =
                resolve_member_overload_signature(ex, types, method.method_type, site.loc)
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
                let full_method = overload_sig.full_method;
                let method_closure = make_member_closure(ex, types, input, overload_sig, site.loc)?;
                let expected_fn = types.new_func(FuncInfer {
                    calling_convention: CallingConvention::Unknown,
                    generics: 0,
                    lifetimes: 0,
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
            .store
            .struct_overload_info(struct_name)
            .and_then(|info| info.operators.get(&method_name))
            .copied();

        if let Some(method) = method {
            let Some(overload_sig) =
                resolve_member_overload_signature(ex, types, method.method_type, site.loc)
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
                    lifetimes: 0,
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
pub(crate) fn resolve_operator_types(ctx: &mut InferState) -> bool {
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

impl PendingIndex {
    #[inline(always)]
    fn step(&mut self, ex: &mut ExternState, types: &mut TypeState) -> ResolveOutcome {
        let mut progress = false;

        self.index = types.root(self.index);

        let max_implicit_deref_steps = 64usize;
        let implicit_deref_limit_message = "index autoderef recursion exceeded safety limit";

        let element: CId = loop {
            let current = self.implicit_deref.sync_roots(types);

            if let Some(t) = types.cluster_solved_type(current) {
                match ex.store.type_value(t) {
                    TypeValue::Array(element, _) => break types.new_solved(ex.store, *element),

                    _ => match self.implicit_deref.step(
                        ex,
                        types,
                        max_implicit_deref_steps,
                        implicit_deref_limit_message,
                    ) {
                        Ok(ImplicitDerefStep::Stepped) => continue,
                        Ok(ImplicitDerefStep::Pending) => return ResolveOutcome::keep(progress),
                        Ok(ImplicitDerefStep::Done) => {
                            ex.push_error(TypeError::Simple {
                                loc: ex.program.value_loc(self.site),
                                message: "indexing base must be an array or pointer to array",
                            });
                            return ResolveOutcome::drop(progress);
                        }
                        Err(err) => {
                            ex.push_error(err);
                            return ResolveOutcome::drop(progress);
                        }
                    },
                }
            }

            match types.core.cluster[current].state {
                ResolveKind::Nothing => {
                    return ResolveOutcome::keep(progress);
                }

                ResolveKind::Array { element, .. } => break element,

                _ => {
                    match self.implicit_deref.step(
                        ex,
                        types,
                        max_implicit_deref_steps,
                        implicit_deref_limit_message,
                    ) {
                        Ok(ImplicitDerefStep::Stepped) => continue,
                        Ok(ImplicitDerefStep::Pending) => return ResolveOutcome::keep(progress),
                        Ok(ImplicitDerefStep::Done) => {
                            ex.push_error(TypeError::Simple {
                                loc: ex.program.value_loc(self.site),
                                message: "indexing base must be an array or pointer to array",
                            });
                            return ResolveOutcome::drop(progress);
                        }
                        Err(err) => {
                            ex.push_error(err);
                            return ResolveOutcome::drop(progress);
                        }
                    }
                }
            }
        };

        let current = self.implicit_deref.current;
        self.implicit_deref.implicit_receivers = self.implicit_deref.finalize_chain(current);

        // index must be usize
        let usize_c = types.new_solved(ex.store, BuiltinType::Usize.into());
        match unify_if_distinct(ex, types, self.index, usize_c) {
            Ok(changed) => progress |= changed,
            Err(clash) => {
                ex.push_error(TypeError::ValuesContradict {
                    expectation_reason: "array indexing requires an index of type usize",
                    site: self.site,
                    found: self.index_value,
                    expected_place: self.site,
                    clash,
                });
                return ResolveOutcome::drop(progress);
            }
        }

        // output element type
        match unify_if_distinct(ex, types, element, self.output) {
            Ok(changed) => progress |= changed,
            Err(clash) => {
                ex.push_error(TypeError::ValuesContradict {
                    expectation_reason: "index expression result must match indexed element type",
                    site: self.site,
                    found: self.base_value,
                    expected_place: self.site,
                    clash,
                });
                return ResolveOutcome::drop(progress);
            }
        }

        ResolveOutcome::drop(progress)
    }
}

impl PendingDeref {
    #[inline(always)]
    fn step(&mut self, ex: &mut ExternState, types: &mut TypeState) -> ResolveOutcome {
        let mut progress = false;

        let source = types.root(self.source);
        self.source = source;

        let mut deref_chain_lid = None;
        let mut deref_chain_mutability = None;

        if let Some(t) = types.cluster_solved_type(source) {
            let result = match ex.store.type_value(t) {
                TypeValue::Ptr { tgt, .. } => Some(types.new_solved(ex.store, *tgt)),

                TypeValue::Struct { id, .. } => {
                    let struct_name = ex.store.struct_value(*id).name;

                    struct_name.and_then(|struct_name| {
                        resolve_struct_deref_target(
                            ex,
                            types,
                            self.site,
                            self.source_value,
                            source,
                            struct_name,
                            &mut deref_chain_lid,
                            &mut deref_chain_mutability,
                        )
                        .map(|resolved| resolved.target)
                    })
                }

                _ => None,
            };

            let Some(result) = result else {
                let source_type = types.bad_type(ex, source);

                ex.push_error(TypeError::CannotDeref {
                    site: self.site,
                    operand: self.source_value,
                    operand_type: source_type,
                });

                return ResolveOutcome::drop(true);
            };

            return match unify_if_distinct(ex, types, result, self.target) {
                Ok(changed) => {
                    progress |= changed;
                    ResolveOutcome::drop(progress)
                }
                Err(clash) => {
                    ex.push_error(TypeError::ValuesContradict {
                        expectation_reason: "dereference result must match pointee type",
                        site: self.site,
                        found: self.source_value,
                        expected_place: self.site,
                        clash,
                    });

                    ResolveOutcome::drop(true)
                }
            };
        }

        let result = match types.core.cluster[source].state {
            // unknown — wait
            ResolveKind::Nothing => {
                return ResolveOutcome::keep(false);
            }

            // raw pointer cluster
            ResolveKind::Ptr { tgt, .. } => Some(tgt),

            // inferred struct
            ResolveKind::Struct(rid) => {
                let sid = types.extra.struct_infers[rid.0].sid;
                let struct_name = ex.store.struct_value(sid).name;

                struct_name.and_then(|struct_name| {
                    resolve_struct_deref_target(
                        ex,
                        types,
                        self.site,
                        self.source_value,
                        source,
                        struct_name,
                        &mut deref_chain_lid,
                        &mut deref_chain_mutability,
                    )
                    .map(|resolved| resolved.target)
                })
            }

            _ => None,
        };

        let Some(result) = result else {
            let source_type = types.bad_type(ex, source);

            ex.push_error(TypeError::CannotDeref {
                site: self.site,
                operand: self.source_value,
                operand_type: source_type,
            });

            return ResolveOutcome::drop(true);
        };

        match unify_if_distinct(ex, types, result, self.target) {
            Ok(changed) => {
                progress |= changed;
                ResolveOutcome::drop(progress)
            }
            Err(clash) => {
                ex.push_error(TypeError::ValuesContradict {
                    expectation_reason: "dereference result must match pointee type",
                    site: self.site,
                    found: self.source_value,
                    expected_place: self.site,
                    clash,
                });

                ResolveOutcome::drop(true)
            }
        }
    }
}

#[inline(always)]
pub(crate) fn resolve_pending_derefs(ctx: &mut InferState) -> bool {
    let mut progress = false;

    let ex = &mut ctx.ex;
    let types = &mut ctx.types;

    ctx.req.pending_derefs.retain_mut(|pending| {
        let outcome = pending.step(ex, types);
        progress |= outcome.progress;
        outcome.retain
    });

    progress
}

#[inline(always)]
pub(crate) fn resolve_pending_indexes(ctx: &mut InferState) -> bool {
    let mut progress = false;

    let ex = &mut ctx.ex;
    let types = &mut ctx.types;

    ctx.req.pending_indexes.retain_mut(|site| {
        site.implicit_deref.sync_roots(types);
        let outcome = site.step(ex, types);
        progress |= outcome.progress;
        if !outcome.retain && !site.implicit_deref.implicit_receivers.is_empty() {
            ctx.req
                .index_implicit_deref_sites
                .push(PendingMemberAccessImplicitDeref {
                    site: site.site,
                    receivers: std::mem::take(&mut site.implicit_deref.implicit_receivers),
                });
        }
        outcome.retain
    });

    progress
}

#[inline(always)]
pub(crate) fn resolve_pending_member_accesses(ctx: &mut InferState) -> bool {
    let mut progress = false;

    let ex = &mut ctx.ex;
    let types = &mut ctx.types;
    let search = &mut ctx.search;
    let member_method_type_sites = &mut ctx.req.member_method_type_sites;
    let member_access_implicit_deref_sites = &mut ctx.req.member_access_implicit_deref_sites;

    ctx.req.pending_member_accesses.retain_mut(|pending| {
        pending.implicit_deref.sync_roots(types);

        match pending.step(ex, types, search, member_method_type_sites) {
            MemberAccessResolve::Pending { source } => {
                pending.implicit_deref.source = source;
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
pub(crate) fn resolve_pending_int_accesses(ctx: &mut InferState) -> bool {
    let mut progress = false;

    let ex = &mut ctx.ex;
    let types = &mut ctx.types;
    let member_access_implicit_deref_sites = &mut ctx.req.member_access_implicit_deref_sites;

    ctx.req.pending_int_accesses.retain_mut(|pending| {
        let source = types.root(pending.source);
        pending.source = source;

        match try_resolve_tuple_int_access(
            ex,
            types,
            pending.site,
            source,
            pending.id,
            pending.kind,
        ) {
            IntAccessResolve::Pending { source } => {
                pending.source = source;
                true
            }
            IntAccessResolve::Resolved {
                result,
                implicit_receivers,
            } => {
                match unify_if_distinct(ex, types, result, pending.output) {
                    Ok(changed) => progress |= changed,
                    Err(clash) => {
                        ex.push_error(TypeError::ValuesContradict {
                            expectation_reason:
                                "tuple element access result must match its inferred use constraints",
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
            IntAccessResolve::Error(err) => {
                ex.push_error(err);
                progress = true;
                false
            }
        }
    });

    progress
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
pub(crate) enum OperandKind {
    KnownNonUser,
    UserStruct(Option<NameId>),
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RawPointerOperandKind {
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
    if let Some(t) = core.cluster[root].solved_ty {
        return match ex.store.type_value(t) {
            TypeValue::Ptr {
                style: PointerStyle::Raw(Nullable::Yes),
                ..
            } => RawPointerOperandKind::RawPointer(root),
            TypeValue::Ptr { .. } => RawPointerOperandKind::NonRawPointer,
            _ => RawPointerOperandKind::NotPointer,
        };
    }

    match core.cluster[root].state {
        ResolveKind::Ptr { kind, .. } => match kind.is_fancy() {
            Some(false) => RawPointerOperandKind::RawPointer(root),
            Some(true) => RawPointerOperandKind::NonRawPointer,
            None => RawPointerOperandKind::UnknownRawPointer(root),
        },

        ResolveKind::Nothing => RawPointerOperandKind::Unknown,
        _ => RawPointerOperandKind::NotPointer,
    }
}

#[inline(always)]
fn classify_operand(ex: &mut ExternState, types: &mut TypeState, cid: CId) -> OperandKind {
    let root = types.root(cid);
    if let Some(t) = types.core.cluster[root].solved_ty {
        return match ex.store.type_value(t) {
            TypeValue::Struct { id, .. } => {
                OperandKind::UserStruct(ex.store.struct_value(*id).name)
            }
            _ => OperandKind::KnownNonUser,
        };
    }

    match types.core.cluster[root].state {
        ResolveKind::IntLike
        | ResolveKind::FloatLike
        | ResolveKind::Func(_)
        | ResolveKind::Array { .. }
        | ResolveKind::Tuple(_) => OperandKind::KnownNonUser,
        ResolveKind::Struct(call_id) => {
            let sid = types.extra.struct_infers[call_id.0].sid;
            OperandKind::UserStruct(ex.store.struct_value(sid).name)
        }
        ResolveKind::Ptr { tgt, kind, .. } => match kind.is_fancy() {
            Some(false) => OperandKind::KnownNonUser,
            Some(true) => classify_operand(ex, types, tgt),
            None => match classify_operand(ex, types, tgt) {
                OperandKind::KnownNonUser => OperandKind::KnownNonUser,
                _ => OperandKind::Unknown,
            },
        },

        ResolveKind::Nothing => OperandKind::Unknown,
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

fn ptr_parts_from_cluster(
    ex: &mut ExternState,
    types: &mut TypeState,
    cid: CId,
) -> Option<(CId, PtrKind, Option<bool>)> {
    let root = types.root(cid);
    if let Some(ty) = types.cluster_solved_type(root) {
        return match ex.store.type_value(ty) {
            TypeValue::Ptr {
                tgt,
                style,
                mutable,
            } => Some((
                types.new_solved(ex.store, *tgt),
                PtrKind::Solved(*style),
                Some(*mutable),
            )),
            _ => None,
        };
    }

    match types.cluster_state(root) {
        ResolveKind::Ptr { tgt, kind, mutable } => Some((tgt, kind, mutable)),
        _ => None,
    }
}
