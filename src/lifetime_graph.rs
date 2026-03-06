use crate::type_inference::{CId, LId, LifeTime, OriginId, OriginKind, OriginNode, OriginVec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LifetimeGraphId(pub(crate) usize);

impl LifetimeGraphId {
    fn from_lid(lid: LId, node_count: usize) -> Option<Self> {
        (lid.0 < node_count).then_some(Self(lid.0))
    }
}

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
    pub(crate) shorter: LifetimeGraphId,
    pub(crate) longer: LifetimeGraphId,
    pub(crate) source_origin: OriginId,
    pub(crate) target_origin: OriginId,
    pub(crate) reason: LifetimeOrderingReason,
}

pub(crate) struct LifetimeOrderingGraph {
    edges: Vec<LifetimeOrdering>,
    outgoing: Vec<Vec<usize>>,
}

impl LifetimeOrderingGraph {
    pub(crate) fn new(lid_count: usize) -> Self {
        let mut outgoing = Vec::with_capacity(lid_count);
        outgoing.resize_with(lid_count, Vec::new);
        Self {
            edges: Vec::new(),
            outgoing,
        }
    }

    pub(crate) fn edges(&self) -> &[LifetimeOrdering] {
        &self.edges
    }

    pub(crate) fn lid_count(&self) -> usize {
        self.outgoing.len()
    }

    #[allow(dead_code)]
    pub(crate) fn outgoing(&self, lid: LifetimeGraphId) -> &[usize] {
        &self.outgoing[lid.0]
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
        self.outgoing[edge.shorter.0].push(index);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PointerLifetimeInfo {
    pub(crate) lid: LId,
    pub(crate) from_raw: bool,
}

pub(crate) fn collect_origin_lifetime_orderings<F>(
    origins: &OriginVec<OriginNode>,
    lid_count: usize,
    mut pointer_lifetime: F,
) -> LifetimeOrderingGraph
where
    F: FnMut(CId) -> Option<PointerLifetimeInfo>,
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

        let Some(shorter) = lifetime_for_origin(node, &mut pointer_lifetime) else {
            continue;
        };

        let Some(parent) = source_origin_for_ordering(origins, node.parent, &mut pointer_lifetime)
        else {
            continue;
        };

        if shorter.from_raw || parent.from_raw {
            continue;
        }

        let Some(shorter_id) = LifetimeGraphId::from_lid(shorter.lid, lid_count) else {
            continue;
        };
        let Some(longer_id) = LifetimeGraphId::from_lid(parent.lid, lid_count) else {
            continue;
        };

        if shorter_id == longer_id {
            continue;
        }

        graph.push_edge(LifetimeOrdering {
            shorter: shorter_id,
            longer: longer_id,
            source_origin: origin,
            target_origin: parent.origin,
            reason,
        });
    }

    graph
}

#[derive(Debug, Clone)]
pub(crate) struct LifetimeSccComponent {
    pub(crate) nodes: Vec<LifetimeGraphId>,
    pub(crate) lifetime: LifeTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LifetimeEdgeRejectionReason {
    IncompatibleKnownEndpoints,
    IncompatibleComponent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LifetimeRejectedEdge {
    pub(crate) edge_index: usize,
    pub(crate) reason: LifetimeEdgeRejectionReason,
}

pub(crate) struct LifetimeSccSolve {
    pub(crate) component_of: Vec<usize>,
    pub(crate) components: Vec<LifetimeSccComponent>,
    pub(crate) rejected_edges: Vec<LifetimeRejectedEdge>,
}

pub(crate) fn solve_lifetime_scc<F>(
    graph: &LifetimeOrderingGraph,
    mut node_lifetime: F,
) -> LifetimeSccSolve
where
    F: FnMut(LifetimeGraphId) -> LifeTime,
{
    let lid_count = graph.lid_count();
    let node_lifetimes: Vec<LifeTime> = (0..lid_count)
        .map(|raw| node_lifetime(LifetimeGraphId(raw)))
        .collect();
    let state = LifetimeSccState::new(graph, node_lifetimes);
    state.solve()
}

struct LifetimeSccState<'a> {
    graph: &'a LifetimeOrderingGraph,
    node_lifetimes: Vec<LifeTime>,

    // lifetime known for the tentative SCC representative (stack node)
    component_known: Vec<Option<LifeTime>>,

    active_edges: Vec<bool>,
    rejected_edges: Vec<LifetimeRejectedEdge>,

    next_index: usize,
    index: Vec<Option<usize>>,
    low: Vec<usize>,

    on_stack: Vec<bool>,
    stack: Vec<LifetimeGraphId>,
    stack_pos: Vec<usize>,

    component_of: Vec<usize>,
    components: Vec<LifetimeSccComponent>,
}

impl<'a> LifetimeSccState<'a> {
    fn new(graph: &'a LifetimeOrderingGraph, node_lifetimes: Vec<LifeTime>) -> Self {
        let lid_count = graph.lid_count();

        let component_known = node_lifetimes
            .iter()
            .map(|&lt| match lt {
                LifeTime::Unknown(_) => None,
                other => Some(other),
            })
            .collect();

        Self {
            graph,
            node_lifetimes,
            component_known,
            active_edges: vec![true; graph.edges.len()],
            rejected_edges: Vec::new(),
            next_index: 0,
            index: vec![None; lid_count],
            low: vec![0; lid_count],
            on_stack: vec![false; lid_count],
            stack: Vec::new(),
            stack_pos: vec![usize::MAX; lid_count],
            component_of: vec![usize::MAX; lid_count],
            components: Vec::new(),
        }
    }

    fn solve(mut self) -> LifetimeSccSolve {
        for raw in 0..self.graph.lid_count() {
            let lid = LifetimeGraphId(raw);
            if self.index[lid.0].is_none() {
                self.visit(lid);
            }
        }

        LifetimeSccSolve {
            component_of: self.component_of,
            components: self.components,
            rejected_edges: self.rejected_edges,
        }
    }

    fn visit(&mut self, v: LifetimeGraphId) {
        let v_index = self.next_index;
        self.next_index += 1;

        self.index[v.0] = Some(v_index);
        self.low[v.0] = v_index;

        self.push_stack(v);

        let out_len = self.graph.outgoing(v).len();

        for out_i in 0..out_len {
            let edge_index = self.graph.outgoing(v)[out_i];

            if !self.active_edges[edge_index] {
                continue;
            }

            let edge = self.graph.edges[edge_index];
            let w = edge.longer;

            if self.index[w.0].is_none() {
                self.visit(w);

                if !self.active_edges[edge_index] || !self.on_stack[w.0] {
                    continue;
                }

                if self.low[w.0] <= v_index {
                    if !self.edge_merge_allowed(v, w) {
                        self.reject_edge(edge_index);
                        continue;
                    }

                    self.merge_component_state(v, w);
                }

                self.low[v.0] = self.low[v.0].min(self.low[w.0]);
                continue;
            }

            if !self.on_stack[w.0] {
                continue;
            }

            let Some(w_index) = self.index[w.0] else {
                continue;
            };

            if w_index <= v_index {
                if !self.edge_merge_allowed(v, w) {
                    self.reject_edge(edge_index);
                    continue;
                }

                self.merge_component_state(v, w);
            }

            self.low[v.0] = self.low[v.0].min(w_index);
        }

        if self.low[v.0] == v_index {
            self.finish_component(v);
        }
    }

    fn edge_merge_allowed(&self, from: LifetimeGraphId, to: LifetimeGraphId) -> bool {
        match (self.component_known[from.0], self.component_known[to.0]) {
            (None, _) | (_, None) => true,
            (Some(a), Some(b)) => a == b,
        }
    }

    fn merge_component_state(&mut self, a: LifetimeGraphId, b: LifetimeGraphId) {
        let merged = match (self.component_known[a.0], self.component_known[b.0]) {
            (None, x) | (x, None) => x,
            (Some(a), Some(_b)) => Some(a), // safe because equality already checked
        };

        self.component_known[a.0] = merged;
    }

    fn reject_edge(&mut self, edge_index: usize) {
        self.active_edges[edge_index] = false;

        let edge = self.graph.edges[edge_index];
        let a = self.node_lifetimes[edge.shorter.0];
        let b = self.node_lifetimes[edge.longer.0];

        let reason =
            if !matches!(a, LifeTime::Unknown(_)) && !matches!(b, LifeTime::Unknown(_)) && a != b {
                LifetimeEdgeRejectionReason::IncompatibleKnownEndpoints
            } else {
                LifetimeEdgeRejectionReason::IncompatibleComponent
            };

        self.rejected_edges
            .push(LifetimeRejectedEdge { edge_index, reason });
    }

    fn push_stack(&mut self, lid: LifetimeGraphId) {
        self.stack_pos[lid.0] = self.stack.len();
        self.stack.push(lid);
        self.on_stack[lid.0] = true;
    }

    fn finish_component(&mut self, root: LifetimeGraphId) {
        let mut nodes = Vec::new();

        loop {
            let lid = self
                .stack
                .pop()
                .expect("tarjan stack must contain current root");

            self.stack_pos[lid.0] = usize::MAX;
            self.on_stack[lid.0] = false;

            nodes.push(lid);

            if lid == root {
                break;
            }
        }

        let lifetime = self.component_known[root.0]
            .unwrap_or(self.node_lifetimes[root.0]);

        let component_id = self.components.len();

        for lid in &nodes {
            self.component_of[lid.0] = component_id;
        }

        self.components
            .push(LifetimeSccComponent { nodes, lifetime });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OriginLifetime {
    origin: OriginId,
    lid: LId,
    from_raw: bool,
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
        | OriginKind::PlaceRoot(_)
        | OriginKind::RawRoot(_) => None,
    }
}

fn lifetime_for_origin<F>(
    node: &OriginNode,
    pointer_lifetime: &mut F,
) -> Option<PointerLifetimeInfo>
where
    F: FnMut(CId) -> Option<PointerLifetimeInfo>,
{
    let pointer_info = node.kind.associated_pointer().and_then(pointer_lifetime);
    let lid = node.lifetime_seed.or(pointer_info.map(|info| info.lid))?;
    Some(PointerLifetimeInfo {
        lid,
        from_raw: pointer_info.map(|info| info.from_raw).unwrap_or(false),
    })
}

fn source_origin_for_ordering<F>(
    origins: &OriginVec<OriginNode>,
    mut current: Option<OriginId>,
    pointer_lifetime: &mut F,
) -> Option<OriginLifetime>
where
    F: FnMut(CId) -> Option<PointerLifetimeInfo>,
{
    while let Some(origin) = current {
        let node = origins.get(origin)?;

        match node.kind {
            OriginKind::BindingRoot if node.parent.is_some() => {
                current = node.parent;
            }
            OriginKind::MemberProjection | OriginKind::IndexProjection => {
                if let Some(info) = lifetime_for_origin(node, pointer_lifetime) {
                    return Some(OriginLifetime {
                        origin,
                        lid: info.lid,
                        from_raw: info.from_raw,
                    });
                }
                current = node.parent;
            }
            OriginKind::RawRoot(_) => return None,
            _ => {
                let info = lifetime_for_origin(node, pointer_lifetime)?;
                return Some(OriginLifetime {
                    origin,
                    lid: info.lid,
                    from_raw: info.from_raw,
                });
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        LifetimeGraphId, LifetimeOrdering, LifetimeOrderingReason, PointerLifetimeInfo,
        collect_origin_lifetime_orderings, solve_lifetime_scc,
    };
    use crate::global_type_inference::infer_global_types;
    use crate::ir::{ValId, Value};
    use crate::local_type_inference::{gather_func_constraints, local_solver};
    use crate::parsing::Parser;
    use crate::program::{Defined, Program};
    use crate::type_inference::{
        CId, InferState, LId, LifeId, LifeTime, OriginId, OriginKind, OriginNode, OriginVec,
        PtrKind, ResolveKind, SolvedTypes, TypeStore, TypeValue, find_lid_root,
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
            resolve_pointer_lifetime(&mut ctx, cid, &mut synthetic_lids, &mut next_synthetic)
        })
        .edges()
        .to_vec()
    }

    fn collect_origins_from_function(src: &str, name: &str) -> OriginVec<OriginNode> {
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

        ctx.ex
            .ans
            .inner_types_of_function(function)
            .expect("expected finalized inner function types")
            .origins
            .clone()
    }

    fn resolve_pointer_lifetime(
        ctx: &mut InferState<'_>,
        cid: CId,
        synthetic_lids: &mut HashMap<CId, LId>,
        next_synthetic: &mut usize,
    ) -> Option<PointerLifetimeInfo> {
        let cid = ctx.types.root(cid);
        match ctx.types.cluster_state(cid) {
            ResolveKind::Ptr { kind, .. } => match kind {
                PtrKind::RefInfer(lid) => Some(PointerLifetimeInfo {
                    lid: find_lid_root(&mut ctx.types.lifetimes.life_parent, lid),
                    from_raw: false,
                }),
                PtrKind::Solved(crate::type_inference::PointerStyle::Ref(lt)) => {
                    let lid = find_lid_for_lifetime(ctx, cid, lt, synthetic_lids, next_synthetic)?;
                    Some(PointerLifetimeInfo {
                        lid,
                        from_raw: false,
                    })
                }
                PtrKind::Solved(crate::type_inference::PointerStyle::Raw(_)) => {
                    Some(PointerLifetimeInfo {
                        lid: LId(0),
                        from_raw: true,
                    })
                }
                _ => None,
            },
            ResolveKind::Solved(ty) => match *ctx.ex.store.type_value(ty) {
                TypeValue::Ptr {
                    style: crate::type_inference::PointerStyle::Ref(lt),
                    ..
                } => {
                    let lid = find_lid_for_lifetime(ctx, cid, lt, synthetic_lids, next_synthetic)?;
                    Some(PointerLifetimeInfo {
                        lid,
                        from_raw: false,
                    })
                }
                TypeValue::Ptr {
                    style: crate::type_inference::PointerStyle::Raw(_),
                    ..
                } => Some(PointerLifetimeInfo {
                    lid: LId(0),
                    from_raw: true,
                }),
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
    fn source_member_projection_chain_through_place_root_emits_ordering() {
        let src = "S=struct{x:int}; f=fn(s:&S){ let y = &s.x; };";
        let edges = collect_from_function(src, "f");

        assert!(!edges.is_empty(), "expected ordering edges, got {edges:?}");
    }

    #[test]
    fn source_member_projection_chain_with_place_root_emits_ordering() {
        let mut origins = OriginVec::new();
        let place_root = OriginId(origins.len() as u32);
        origins.push(OriginNode {
            kind: OriginKind::PlaceRoot(CId(0)),
            parent: None,
            decl_site: None,
            declared_mutability: None,
            effective_mutability: None,
            lifetime_seed: Some(LId(1)),
        });
        origins.push(OriginNode {
            kind: OriginKind::MemberProjection,
            parent: Some(place_root),
            decl_site: None,
            declared_mutability: None,
            effective_mutability: None,
            lifetime_seed: Some(LId(0)),
        });

        let edges = collect_origin_lifetime_orderings(&origins, 2, |_| None)
            .edges()
            .to_vec();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].reason, LifetimeOrderingReason::MemberProjection);
        assert_eq!(edges[0].shorter, LifetimeGraphId(0));
        assert_eq!(edges[0].longer, LifetimeGraphId(1));
    }

    #[test]
    fn source_member_projection_chain_with_raw_root_still_emits_no_ordering() {
        let mut origins = OriginVec::new();
        let raw_root = OriginId(origins.len() as u32);
        origins.push(OriginNode {
            kind: OriginKind::RawRoot(CId(0)),
            parent: None,
            decl_site: None,
            declared_mutability: None,
            effective_mutability: None,
            lifetime_seed: Some(LId(1)),
        });
        origins.push(OriginNode {
            kind: OriginKind::MemberProjection,
            parent: Some(raw_root),
            decl_site: None,
            declared_mutability: None,
            effective_mutability: None,
            lifetime_seed: Some(LId(0)),
        });

        let edges = collect_origin_lifetime_orderings(&origins, 2, |_| None)
            .edges()
            .to_vec();
        assert!(edges.is_empty());
    }

    #[test]
    fn source_raw_endpoint_does_not_emit_lifetime_ordering() {
        let mut origins = OriginVec::new();
        let parent = OriginId(origins.len() as u32);
        origins.push(OriginNode {
            kind: OriginKind::Reborrow(CId(1)),
            parent: None,
            decl_site: None,
            declared_mutability: None,
            effective_mutability: None,
            lifetime_seed: None,
        });
        origins.push(OriginNode {
            kind: OriginKind::Reborrow(CId(0)),
            parent: Some(parent),
            decl_site: None,
            declared_mutability: None,
            effective_mutability: None,
            lifetime_seed: None,
        });

        let edges = collect_origin_lifetime_orderings(&origins, 2, |cid| match cid {
            CId(0) => Some(PointerLifetimeInfo {
                lid: LId(0),
                from_raw: true,
            }),
            CId(1) => Some(PointerLifetimeInfo {
                lid: LId(1),
                from_raw: false,
            }),
            _ => None,
        })
        .edges()
        .to_vec();
        assert!(edges.is_empty());
    }

    #[test]
    fn target_raw_endpoint_does_not_emit_lifetime_ordering() {
        let mut origins = OriginVec::new();
        let parent = OriginId(origins.len() as u32);
        origins.push(OriginNode {
            kind: OriginKind::Reborrow(CId(1)),
            parent: None,
            decl_site: None,
            declared_mutability: None,
            effective_mutability: None,
            lifetime_seed: None,
        });
        origins.push(OriginNode {
            kind: OriginKind::Reborrow(CId(0)),
            parent: Some(parent),
            decl_site: None,
            declared_mutability: None,
            effective_mutability: None,
            lifetime_seed: None,
        });

        let edges = collect_origin_lifetime_orderings(&origins, 2, |cid| match cid {
            CId(0) => Some(PointerLifetimeInfo {
                lid: LId(0),
                from_raw: false,
            }),
            CId(1) => Some(PointerLifetimeInfo {
                lid: LId(1),
                from_raw: true,
            }),
            _ => None,
        })
        .edges()
        .to_vec();
        assert!(edges.is_empty());
    }

    #[test]
    fn source_call_return_root_does_not_emit_ordering() {
        let src = "id=fn(x:&int)->&int{x}; f=fn(x:&int)->&int{ id(x) };";
        let edges = collect_from_function(src, "f");

        assert!(edges.is_empty());
    }

    #[test]
    fn simple_let_addr_of_uses_place_root_not_raw_root() {
        let origins =
            collect_origins_from_function("S=struct{x:int}; f=fn(s:&S){ let y = &s.x; };", "f");
        assert!(
            origins
                .0
                .iter()
                .any(|node| matches!(node.kind, OriginKind::PlaceRoot(_))),
            "expected at least one PlaceRoot origin in simple let addr-of case"
        );
        assert!(
            !origins
                .0
                .iter()
                .any(|node| matches!(node.kind, OriginKind::RawRoot(_))),
            "did not expect RawRoot origin in simple let addr-of case"
        );
    }

    #[test]
    fn scc_unknown_upgrades_to_local() {
        let mut graph = super::LifetimeOrderingGraph::new(2);
        graph.push_edge(LifetimeOrdering {
            shorter: LifetimeGraphId(0),
            longer: LifetimeGraphId(1),
            source_origin: OriginId(0),
            target_origin: OriginId(0),
            reason: LifetimeOrderingReason::Reborrow,
        });
        graph.push_edge(LifetimeOrdering {
            shorter: LifetimeGraphId(1),
            longer: LifetimeGraphId(0),
            source_origin: OriginId(0),
            target_origin: OriginId(0),
            reason: LifetimeOrderingReason::Reborrow,
        });

        let solve = solve_lifetime_scc(&graph, |lid| match lid {
            LifetimeGraphId(0) => LifeTime::Unknown(LifeId(0)),
            LifetimeGraphId(1) => LifeTime::Local(LifeId(1)),
            _ => unreachable!(),
        });

        assert_eq!(solve.components.len(), 1);
        assert_eq!(solve.components[0].lifetime, LifeTime::Local(LifeId(1)));
        assert!(solve.rejected_edges.is_empty());
    }

    #[test]
    fn scc_rejects_local_external_cycle_edges() {
        let mut graph = super::LifetimeOrderingGraph::new(2);
        graph.push_edge(LifetimeOrdering {
            shorter: LifetimeGraphId(0),
            longer: LifetimeGraphId(1),
            source_origin: OriginId(0),
            target_origin: OriginId(0),
            reason: LifetimeOrderingReason::Reborrow,
        });
        graph.push_edge(LifetimeOrdering {
            shorter: LifetimeGraphId(1),
            longer: LifetimeGraphId(0),
            source_origin: OriginId(0),
            target_origin: OriginId(0),
            reason: LifetimeOrderingReason::Reborrow,
        });

        let solve = solve_lifetime_scc(&graph, |lid| match lid {
            LifetimeGraphId(0) => LifeTime::Local(LifeId(0)),
            LifetimeGraphId(1) => LifeTime::External(0),
            _ => unreachable!(),
        });

        assert_eq!(solve.components.len(), 2);
        assert!(!solve.rejected_edges.is_empty());
    }
}
