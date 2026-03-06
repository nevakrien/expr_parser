use crate::type_inference::{CId, LId, LifeVec, OriginId, OriginKind, OriginNode, OriginVec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LifetimeOrderingReason {
    Reborrow,
    Deref,
    MemberProjection,
    IndexProjection,
    CastProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LifetimeOrdering {
    pub(crate) shorter: LId,
    pub(crate) longer: LId,
    pub(crate) source_origin: OriginId,
    pub(crate) target_origin: OriginId,
    pub(crate) reason: LifetimeOrderingReason,
}

pub(crate) struct LifetimeOrderingGraph {
    edges: Vec<LifetimeOrdering>,
    outgoing: LifeVec<Vec<usize>>,
}

impl LifetimeOrderingGraph {
    pub(crate) fn new(lid_count: usize) -> Self {
        let mut outgoing = LifeVec(Vec::with_capacity(lid_count));
        outgoing.0.resize_with(lid_count, Vec::new);
        Self {
            edges: Vec::new(),
            outgoing,
        }
    }

    pub(crate) fn edges(&self) -> &[LifetimeOrdering] {
        &self.edges
    }

    #[allow(dead_code)]
    pub(crate) fn outgoing(&self, lid: LId) -> &[usize] {
        &self.outgoing[lid]
    }

    fn push_edge(&mut self, edge: LifetimeOrdering) {
        if self
            .edges
            .iter()
            .any(|existing| existing.shorter == edge.shorter && existing.longer == edge.longer)
        {
            return;
        }

        let index = self.edges.len();
        self.edges.push(edge);
        self.outgoing[edge.shorter].push(index);
    }
}

pub(crate) fn collect_origin_lifetime_orderings<F>(
    origins: &OriginVec<OriginNode>,
    lid_count: usize,
    mut lid_for_pointer: F,
) -> LifetimeOrderingGraph
where
    F: FnMut(CId) -> Option<LId>,
{
    let mut graph = LifetimeOrderingGraph::new(lid_count);

    for raw in 0..origins.len() {
        let origin = OriginId(raw as u32);
        let Some(node) = origins.get(origin) else {
            continue;
        };

        let Some(reason) = ordering_reason(node.kind) else {
            continue;
        };

        let Some(shorter) = lifetime_for_origin(node, &mut lid_for_pointer) else {
            continue;
        };

        let Some(parent) = source_origin_for_ordering(origins, node.parent, &mut lid_for_pointer)
        else {
            continue;
        };

        if shorter == parent.lid {
            continue;
        }

        graph.push_edge(LifetimeOrdering {
            shorter,
            longer: parent.lid,
            source_origin: origin,
            target_origin: parent.origin,
            reason,
        });
    }

    graph
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OriginLifetime {
    origin: OriginId,
    lid: LId,
}

fn ordering_reason(kind: OriginKind) -> Option<LifetimeOrderingReason> {
    match kind {
        OriginKind::Reborrow(_) => Some(LifetimeOrderingReason::Reborrow),
        OriginKind::Deref(_) => Some(LifetimeOrderingReason::Deref),
        OriginKind::MemberProjection => Some(LifetimeOrderingReason::MemberProjection),
        OriginKind::IndexProjection => Some(LifetimeOrderingReason::IndexProjection),
        OriginKind::CastProjection(_) => Some(LifetimeOrderingReason::CastProjection),
        OriginKind::BindingRoot
        | OriginKind::ArgumentRoot(_)
        | OriginKind::CallReturnRoot(_)
        | OriginKind::RawRoot(_) => None,
    }
}

fn lifetime_for_origin<F>(node: &OriginNode, lid_for_pointer: &mut F) -> Option<LId>
where
    F: FnMut(CId) -> Option<LId>,
{
    node.lifetime_seed
        .or_else(|| node.kind.associated_pointer().and_then(lid_for_pointer))
}

fn source_origin_for_ordering<F>(
    origins: &OriginVec<OriginNode>,
    mut current: Option<OriginId>,
    lid_for_pointer: &mut F,
) -> Option<OriginLifetime>
where
    F: FnMut(CId) -> Option<LId>,
{
    while let Some(origin) = current {
        let node = origins.get(origin)?;

        match node.kind {
            OriginKind::BindingRoot if node.parent.is_some() => {
                current = node.parent;
            }
            OriginKind::MemberProjection | OriginKind::IndexProjection => {
                if let Some(lid) = lifetime_for_origin(node, lid_for_pointer) {
                    return Some(OriginLifetime { origin, lid });
                }
                current = node.parent;
            }
            OriginKind::RawRoot(_) => return None,
            _ => {
                let lid = lifetime_for_origin(node, lid_for_pointer)?;
                return Some(OriginLifetime { origin, lid });
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{LifetimeOrdering, LifetimeOrderingReason, collect_origin_lifetime_orderings};
    use crate::global_type_inference::infer_global_types;
    use crate::ir::{ValId, Value};
    use crate::local_type_inference::{gather_func_constraints, local_solver};
    use crate::parsing::Parser;
    use crate::program::{Defined, Program};
    use crate::type_inference::{
        CId, InferState, LId, LifeTime, PtrKind, ResolveKind, SolvedTypes, TypeStore, TypeValue,
        find_lid_root,
    };
    use std::collections::HashMap;

    fn gather_program(src: &str) -> Program {
        let mut program = Program::new();
        let mut parser = Parser::new(src, 0);
        program
            .lower_all(&mut parser)
            .unwrap_or_else(|errs| panic!("lowering errors: {errs:?}"));
        program
    }

    fn find_value_by_name(program: &Program, name: &str) -> ValId {
        program
            .definitions
            .iter()
            .find_map(|(n, def)| match def {
                Defined::Func(funcs) if program.name_string(*n) == name => {
                    funcs.implementations.first().copied()
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("implementation `{name}` not found"))
    }

    fn collect_from_function(src: &str, name: &str) -> Vec<LifetimeOrdering> {
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let function = find_value_by_name(&program, name);
        let mut ctx = InferState::new(&mut store, &program, &mut solved_types);
        ctx.req.owner = Some(function);

        let Value::Func {
            calling_convention,
            generics,
            params,
            output_type,
            body,
        } = ctx.ex.program.value(function).clone()
        else {
            panic!("expected function value")
        };

        let _ = gather_func_constraints::<true>(
            &mut ctx,
            function,
            calling_convention,
            generics,
            params,
            output_type,
            body,
        );
        local_solver(&mut ctx);
        assert!(
            ctx.ex.errors.is_empty(),
            "unexpected type errors: {:?}",
            ctx.ex.errors
        );

        let origins = ctx
            .ex
            .ans
            .inner_types_of_function(function)
            .expect("expected finalized inner function types")
            .origins
            .clone();
        let base_lid_count = ctx.types.lifetimes.life_parent.0.len();
        let mut synthetic_lids = HashMap::new();
        let mut next_synthetic = base_lid_count;

        collect_origin_lifetime_orderings(&origins, base_lid_count + origins.len(), |cid| {
            resolve_pointer_lid(&mut ctx, cid, &mut synthetic_lids, &mut next_synthetic)
        })
        .edges()
        .to_vec()
    }

    fn resolve_pointer_lid(
        ctx: &mut InferState<'_>,
        cid: CId,
        synthetic_lids: &mut HashMap<CId, LId>,
        next_synthetic: &mut usize,
    ) -> Option<LId> {
        let cid = ctx.types.root(cid);
        match ctx.types.cluster_state(cid) {
            ResolveKind::Ptr { kind, .. } => match kind {
                PtrKind::RefInfer(lid) => {
                    Some(find_lid_root(&mut ctx.types.lifetimes.life_parent, lid))
                }
                PtrKind::Solved(crate::type_inference::PointerStyle::Ref(lt)) => {
                    find_lid_for_lifetime(ctx, cid, lt, synthetic_lids, next_synthetic)
                }
                _ => None,
            },
            ResolveKind::Solved(ty) => match *ctx.ex.store.type_value(ty) {
                TypeValue::Ptr {
                    style: crate::type_inference::PointerStyle::Ref(lt),
                    ..
                } => find_lid_for_lifetime(ctx, cid, lt, synthetic_lids, next_synthetic),
                _ => None,
            },
            _ => None,
        }
    }

    fn find_lid_for_lifetime(
        ctx: &mut InferState<'_>,
        cid: CId,
        lt: LifeTime,
        synthetic_lids: &mut HashMap<CId, LId>,
        next_synthetic: &mut usize,
    ) -> Option<LId> {
        for raw in 0..ctx.types.lifetimes.life_parent.0.len() {
            let lid = LId(raw);
            let root = find_lid_root(&mut ctx.types.lifetimes.life_parent, lid);
            if ctx.types.lifetimes.life_known[root] == Some(lt) {
                return Some(root);
            }
        }

        match lt {
            LifeTime::Unknown(_) => {
                if let Some(&lid) = synthetic_lids.get(&cid) {
                    return Some(lid);
                }
                let lid = LId(*next_synthetic);
                *next_synthetic += 1;
                synthetic_lids.insert(cid, lid);
                Some(lid)
            }
            _ => None,
        }
    }

    #[test]
    fn source_reborrow_emits_single_ordering_edge() {
        let edges = collect_from_function("f=fn(x:&int){ let y = &*x; };", "f");

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].reason, LifetimeOrderingReason::Reborrow);
    }

    #[test]
    fn source_member_projection_chain_with_raw_root_emits_no_ordering() {
        let src = "S=struct{x:int}; f=fn(s:&S){ let y = &s.x; };";
        let edges = collect_from_function(src, "f");

        assert!(edges.is_empty());
    }

    #[test]
    fn source_call_return_root_does_not_emit_ordering() {
        let src = "id=fn(x:&int)->&int{x}; f=fn(x:&int)->&int{ id(x) };";
        let edges = collect_from_function(src, "f");

        assert!(edges.is_empty());
    }
}
