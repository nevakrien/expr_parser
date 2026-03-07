use crate::type_inference::{
    CId, InferState, LId, LifeId, LifeTime, OriginDeclSite, OriginId, OriginKind, OriginNode,
    OriginVec, PointerStyle, ResolveKind, TypeError, TypeValue, find_lid_root, unify_struct_lids,
};
use std::cmp::Ordering;

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

pub(crate) fn collect_origin_lifetime_orderings(
    origins: &OriginVec<OriginNode>,
    lid_count: usize,
) -> LifetimeOrderingGraph {
    let mut graph = LifetimeOrderingGraph::new(lid_count);

    for raw in 0..origins.len() {
        let origin = OriginId(raw as u32);
        let Some(node) = origins.get(origin) else {
            continue;
        };

        let Some(reason) = ordering_reason(node.kind) else {
            continue;
        };

        let Some(shorter) = lifetime_for_origin(origin, node) else {
            continue;
        };

        let Some(parent) = source_origin_for_ordering(origins, node.parent) else {
            continue;
        };

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
}

pub(crate) struct LifetimeSccSolve {
    pub(crate) component_of: Vec<usize>,
    pub(crate) components: Vec<LifetimeSccComponent>,
}

pub(crate) fn solve_lifetime_scc(graph: &LifetimeOrderingGraph) -> LifetimeSccSolve {
    LifetimeSccState::new(graph).solve()
}

struct LifetimeSccState<'a> {
    graph: &'a LifetimeOrderingGraph,

    next_index: usize,
    index: Vec<Option<usize>>,
    low: Vec<usize>,

    on_stack: Vec<bool>,
    stack: Vec<LifetimeGraphId>,

    component_of: Vec<usize>,
    components: Vec<LifetimeSccComponent>,
}

impl<'a> LifetimeSccState<'a> {
    fn new(graph: &'a LifetimeOrderingGraph) -> Self {
        let lid_count = graph.lid_count();

        Self {
            graph,
            next_index: 0,
            index: vec![None; lid_count],
            low: vec![0; lid_count],
            on_stack: vec![false; lid_count],
            stack: Vec::new(),
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
            let edge = self.graph.edges[edge_index];
            let w = edge.longer;

            if self.index[w.0].is_none() {
                self.visit(w);

                if self.on_stack[w.0] {
                    self.low[v.0] = self.low[v.0].min(self.low[w.0]);
                }
                continue;
            }

            if !self.on_stack[w.0] {
                continue;
            }

            let Some(w_index) = self.index[w.0] else {
                continue;
            };

            self.low[v.0] = self.low[v.0].min(w_index);
        }

        if self.low[v.0] == v_index {
            self.finish_component(v);
        }
    }

    fn push_stack(&mut self, lid: LifetimeGraphId) {
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

            self.on_stack[lid.0] = false;

            nodes.push(lid);

            if lid == root {
                break;
            }
        }

        let component_id = self.components.len();

        for lid in &nodes {
            self.component_of[lid.0] = component_id;
        }

        self.components.push(LifetimeSccComponent { nodes });
    }
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
        | OriginKind::PlaceRoot(_)
        | OriginKind::RawRoot(_) => None,
    }
}

fn lifetime_for_origin(origin: OriginId, node: &OriginNode) -> Option<OriginLifetime> {
    let lid = node.lifetime_seed?;
    Some(OriginLifetime { origin, lid })
}

fn source_origin_for_ordering(
    origins: &OriginVec<OriginNode>,
    mut current: Option<OriginId>,
) -> Option<OriginLifetime> {
    while let Some(origin) = current {
        let node = origins.get(origin)?;

        match node.kind {
            OriginKind::BindingRoot if node.parent.is_some() => {
                current = node.parent;
            }
            OriginKind::MemberProjection | OriginKind::IndexProjection => {
                if let Some(info) = lifetime_for_origin(origin, node) {
                    return Some(info);
                }
                current = node.parent;
            }
            OriginKind::RawRoot(_) => return None,
            _ => {
                return lifetime_for_origin(origin, node);
            }
        }
    }

    None
}

pub(crate) fn solve_local_lifetimes_by_graph(ctx: &mut InferState) {
    seed_origin_lifetimes_for_graph(ctx);

    let lid_count = ctx.types.lifetimes.life_parent.0.len();
    let graph = collect_origin_lifetime_orderings(&ctx.types.lifetimes.origins, lid_count);
    let solve = solve_lifetime_scc(&graph);
    let anchors = collect_lid_origin_anchors(&ctx.types.lifetimes.origins, lid_count);

    for component in &solve.components {
        if component.nodes.len() < 2 {
            continue;
        }

        let leader = LId(component.nodes[0].0);
        for &node in component.nodes.iter().skip(1) {
            let lid = LId(node.0);
            if unify_struct_lids(&mut ctx.types, leader, lid) {
                continue;
            }

            match (anchors[leader.0], anchors[lid.0]) {
                (Some(source), Some(target)) => {
                    report_lifetime_cycle_conflict(ctx, source, target);
                }
                _ => {
                    if let Some(loc) = ctx.req.owner.map(|owner| ctx.ex.program.value_loc(owner)) {
                        ctx.push_error(TypeError::Simple {
                            loc,
                            message: "lifetime cycle requires incompatible lifetime equality",
                        });
                    }
                }
            }
        }
    }

    validate_known_lifetime_orderings(ctx, &graph);

    assign_remaining_unresolved_lifetimes_as_unknown(ctx, &graph);
}

fn validate_known_lifetime_orderings(ctx: &mut InferState, graph: &LifetimeOrderingGraph) {
    let mut seen_root_pairs: Vec<(usize, usize)> = Vec::new();

    for edge in graph.edges() {
        let shorter_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(edge.shorter.0));
        let longer_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(edge.longer.0));

        if shorter_root == longer_root {
            continue;
        }

        let shorter_lt = ctx.types.lifetimes.life_known[shorter_root];
        let longer_lt = ctx.types.lifetimes.life_known[longer_root];
        let (Some(shorter_lt), Some(longer_lt)) = (shorter_lt, longer_lt) else {
            continue;
        };

        if !matches!(shorter_lt.partial_cmp(&longer_lt), Some(Ordering::Greater)) {
            continue;
        }

        let pair = (shorter_root.0, longer_root.0);
        if seen_root_pairs.contains(&pair) {
            continue;
        }
        seen_root_pairs.push(pair);

        report_lifetime_ordering_conflict(ctx, edge.source_origin, edge.target_origin);
    }
}

fn seed_origin_lifetimes_for_graph(ctx: &mut InferState) {
    let mut next_local = next_local_lifetime_id(ctx);
    let origin_count = ctx.types.lifetimes.origins.len();
    if origin_count == 0 {
        return;
    }

    for raw in 0..origin_count {
        let origin = OriginId(raw as u32);
        let (kind, parent, current_seed) = match ctx.types.origin(origin) {
            Some(node) => (node.kind, node.parent, node.lifetime_seed),
            None => continue,
        };

        let mut seed_lid =
            current_seed.map(|lid| find_lid_root(&mut ctx.types.lifetimes.life_parent, lid));

        if let Some(cid) = kind.associated_pointer()
            && let Some(lid) = resolve_pointer_lifetime_for_graph(ctx, cid)
        {
            let pointer_lid = find_lid_root(&mut ctx.types.lifetimes.life_parent, lid);
            seed_lid = Some(match seed_lid {
                Some(existing) => {
                    if existing != pointer_lid {
                        if !unify_struct_lids(&mut ctx.types, existing, pointer_lid) {
                            pointer_lid
                        } else {
                            find_lid_root(&mut ctx.types.lifetimes.life_parent, existing)
                        }
                    } else {
                        existing
                    }
                }
                None => pointer_lid,
            });
        }

        if seed_lid.is_none()
            && let Some(parent_origin) = parent
            && let Some(parent_node) = ctx.types.origin(parent_origin)
            && let Some(parent_lid) = parent_node.lifetime_seed
        {
            seed_lid = Some(find_lid_root(
                &mut ctx.types.lifetimes.life_parent,
                parent_lid,
            ));
        }

        if seed_lid.is_none() {
            seed_lid = Some(if matches!(kind, OriginKind::BindingRoot) {
                let lid = ctx.types.new_lid_known(LifeTime::Local(LifeId(next_local)));
                next_local += 1;
                lid
            } else {
                ctx.types.new_lid()
            });
        }

        let root_seed =
            seed_lid.map(|lid| find_lid_root(&mut ctx.types.lifetimes.life_parent, lid));
        if let Some(node) = ctx.types.origin_mut(origin) {
            node.lifetime_seed = root_seed;
        }
    }

    canonicalize_origin_lifetime_seed_roots(ctx);
}

fn resolve_pointer_lifetime_for_graph(ctx: &mut InferState, cid: CId) -> Option<LId> {
    let cid = ctx.types.root(cid);
    match ctx.types.cluster_state(cid) {
        ResolveKind::Ptr { kind, .. } => match kind {
            crate::type_inference::PtrKind::RefInfer(lid) => {
                Some(find_lid_root(&mut ctx.types.lifetimes.life_parent, lid))
            }
            crate::type_inference::PtrKind::Solved(PointerStyle::Raw(_)) => {
                Some(ctx.types.new_lid())
            }
            _ => None,
        },
        ResolveKind::Solved(ty) => match *ctx.ex.store.type_value(ty) {
            TypeValue::Ptr {
                style: PointerStyle::Ref(lt),
                ..
            } => find_or_create_lid_for_lifetime(ctx, lt),
            TypeValue::Ptr {
                style: PointerStyle::Raw(_),
                ..
            } => Some(ctx.types.new_lid()),
            _ => None,
        },
        _ => None,
    }
}

fn find_or_create_lid_for_lifetime(ctx: &mut InferState, lt: LifeTime) -> Option<LId> {
    for raw in 0..ctx.types.lifetimes.life_parent.0.len() {
        let lid = LId(raw);
        let root = find_lid_root(&mut ctx.types.lifetimes.life_parent, lid);
        if ctx.types.lifetimes.life_known[root] == Some(lt) {
            return Some(root);
        }
    }

    Some(ctx.types.new_lid_known(lt))
}

fn assign_remaining_unresolved_lifetimes_as_unknown(
    ctx: &mut InferState,
    graph: &LifetimeOrderingGraph,
) {
    let root_count = ctx.types.lifetimes.life_parent.0.len();
    let mut predecessors: Vec<Vec<LId>> = vec![Vec::new(); root_count];

    for edge in graph.edges() {
        let shorter_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(edge.shorter.0));
        let longer_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(edge.longer.0));
        if shorter_root == longer_root {
            continue;
        }
        predecessors[longer_root.0].push(shorter_root);
    }

    let mut promote_to_local = vec![false; root_count];
    let mut stack = Vec::new();
    for root_raw in 0..root_count {
        let root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(root_raw));
        if matches!(
            ctx.types.lifetimes.life_known[root],
            Some(LifeTime::Local(_))
        ) {
            stack.push(root);
        }
    }

    while let Some(longer_root) = stack.pop() {
        for &shorter_root in &predecessors[longer_root.0] {
            if ctx.types.lifetimes.life_known[shorter_root].is_some()
                || promote_to_local[shorter_root.0]
            {
                continue;
            }

            promote_to_local[shorter_root.0] = true;
            stack.push(shorter_root);
        }
    }

    let mut next_local = next_local_lifetime_id(ctx);
    let mut next_unknown = next_unknown_lifetime_id(ctx);
    let lids = ctx.types.lifetimes.life_parent.0.clone();

    for lid in lids {
        let root = find_lid_root(&mut ctx.types.lifetimes.life_parent, lid);
        if ctx.types.lifetimes.life_known[root].is_some() {
            continue;
        }

        if promote_to_local[root.0] {
            ctx.types.lifetimes.life_known[root] = Some(LifeTime::Local(LifeId(next_local)));
            next_local += 1;
        } else {
            ctx.types.lifetimes.life_known[root] = Some(LifeTime::Unknown(LifeId(next_unknown)));
            next_unknown += 1;
        }
    }
}

fn next_local_lifetime_id(ctx: &InferState) -> u32 {
    let mut next = 0;
    for known in ctx.types.lifetimes.life_known.0.iter().copied().flatten() {
        if let LifeTime::Local(id) = known {
            next = next.max(id.0.saturating_add(1));
        }
    }
    next
}

fn canonicalize_origin_lifetime_seed_roots(ctx: &mut InferState) {
    let origin_count = ctx.types.lifetimes.origins.len();
    for raw in 0..origin_count {
        let origin = OriginId(raw as u32);
        let root = ctx
            .types
            .origin(origin)
            .and_then(|node| node.lifetime_seed)
            .map(|lid| find_lid_root(&mut ctx.types.lifetimes.life_parent, lid));
        if let Some(node) = ctx.types.origin_mut(origin) {
            node.lifetime_seed = root;
        }
    }
}

fn collect_lid_origin_anchors(
    origins: &OriginVec<OriginNode>,
    lid_count: usize,
) -> Vec<Option<OriginId>> {
    let mut anchors = vec![None; lid_count];

    for raw in 0..origins.len() {
        let origin = OriginId(raw as u32);
        let Some(node) = origins.get(origin) else {
            continue;
        };
        let Some(seed) = node.lifetime_seed else {
            continue;
        };
        if seed.0 < lid_count && anchors[seed.0].is_none() {
            anchors[seed.0] = Some(origin);
        }
    }

    anchors
}

fn next_unknown_lifetime_id(ctx: &InferState) -> u32 {
    let mut next = 0;
    for known in ctx.types.lifetimes.life_known.0.iter().copied().flatten() {
        if let LifeTime::Unknown(id) = known {
            next = next.max(id.0.saturating_add(1));
        }
    }
    next
}

fn report_lifetime_cycle_conflict(
    ctx: &mut InferState,
    source_origin: OriginId,
    target_origin: OriginId,
) {
    let source_loc = ctx
        .types
        .origin(source_origin)
        .and_then(|node| node.decl_site)
        .map(|site| match site {
            OriginDeclSite::Pattern(pattern) => ctx.ex.program.pattern_loc(pattern),
            OriginDeclSite::Value(value) => ctx.ex.program.value_loc(value),
        });
    let target_loc = ctx
        .types
        .origin(target_origin)
        .and_then(|node| node.decl_site)
        .map(|site| match site {
            OriginDeclSite::Pattern(pattern) => ctx.ex.program.pattern_loc(pattern),
            OriginDeclSite::Value(value) => ctx.ex.program.value_loc(value),
        });

    if let (Some(loc), Some(related)) = (&source_loc, &target_loc) {
        ctx.push_error(TypeError::SimpleRelated {
            loc: loc.clone(),
            message: "lifetime cycle requires incompatible lifetime equality",
            related: related.clone(),
            related_message: "conflicting lifetime source",
        });
        return;
    }

    if let Some(loc) = source_loc.or(target_loc) {
        ctx.push_error(TypeError::Simple {
            loc,
            message: "lifetime cycle requires incompatible lifetime equality",
        });
    }
}

fn report_lifetime_ordering_conflict(
    ctx: &mut InferState,
    source_origin: OriginId,
    target_origin: OriginId,
) {
    let source_loc = ctx
        .types
        .origin(source_origin)
        .and_then(|node| node.decl_site)
        .map(|site| match site {
            OriginDeclSite::Pattern(pattern) => ctx.ex.program.pattern_loc(pattern),
            OriginDeclSite::Value(value) => ctx.ex.program.value_loc(value),
        });
    let target_loc = ctx
        .types
        .origin(target_origin)
        .and_then(|node| node.decl_site)
        .map(|site| match site {
            OriginDeclSite::Pattern(pattern) => ctx.ex.program.pattern_loc(pattern),
            OriginDeclSite::Value(value) => ctx.ex.program.value_loc(value),
        });

    if let (Some(loc), Some(related)) = (&source_loc, &target_loc) {
        ctx.push_error(TypeError::SimpleRelated {
            loc: loc.clone(),
            message: "borrowed value does not live long enough for required lifetime",
            related: related.clone(),
            related_message: "borrow source lifetime is shorter than required",
        });
        return;
    }

    if let Some(loc) = source_loc.or(target_loc) {
        ctx.push_error(TypeError::Simple {
            loc,
            message: "borrowed value does not live long enough for required lifetime",
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LifetimeOrdering, LifetimeOrderingGraph, LifetimeOrderingReason,
        assign_remaining_unresolved_lifetimes_as_unknown, collect_origin_lifetime_orderings,
        seed_origin_lifetimes_for_graph, solve_lifetime_scc,
    };
    use crate::ErrorReporter;
    use crate::global_type_inference::infer_global_types;
    use crate::ir::{ValId, Value};
    use crate::local_type_inference::{gather_func_constraints, local_solver};
    use crate::parsing::Parser;
    use crate::program::{Defined, Program};
    use crate::type_inference::run_typecheck_scan;
    use crate::type_inference::{
        InferState, LifeTime, OriginId, OriginKind, OriginNode, OriginVec, PtrKind, ResolveKind,
        SolvedTypes, TypeError, TypeStore, find_lid_root,
    };

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

    fn with_inferred_function_ctx<R>(
        src: &str,
        name: &str,
        action: impl FnOnce(&mut InferState) -> R,
    ) -> R {
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

        action(&mut ctx)
    }

    fn collect_graph_from_function(src: &str, name: &str) -> LifetimeOrderingGraph {
        with_inferred_function_ctx(src, name, |ctx| {
            let origins = ctx
                .ex
                .ans
                .inner_types_of_function(ctx.req.owner.expect("owner should be set"))
                .map(|inner| inner.origins.clone())
                .unwrap_or_else(|| ctx.types.lifetimes.origins.clone());
            let lid_count = ctx.types.lifetimes.life_parent.0.len();
            collect_origin_lifetime_orderings(&origins, lid_count)
        })
    }

    fn collect_from_function(src: &str, name: &str) -> Vec<LifetimeOrdering> {
        collect_graph_from_function(src, name).edges().to_vec()
    }

    fn collect_origins_from_function(src: &str, name: &str) -> OriginVec<OriginNode> {
        with_inferred_function_ctx(src, name, |ctx| {
            ctx.ex
                .ans
                .inner_types_of_function(ctx.req.owner.expect("owner should be set"))
                .map(|inner| inner.origins.clone())
                .unwrap_or_else(|| ctx.types.lifetimes.origins.clone())
        })
    }

    #[test]
    fn source_reborrow_emits_reborrow_ordering_edge() {
        let edges = collect_from_function("f=fn(x:&int){ let y = &*x; };", "f");

        assert!(!edges.is_empty(), "expected ordering edges, got {edges:?}");
        assert!(
            edges
                .iter()
                .any(|edge| edge.reason == LifetimeOrderingReason::Reborrow),
            "expected a reborrow edge, got {edges:?}"
        );
    }

    #[test]
    fn source_member_projection_chain_through_place_root_emits_ordering() {
        let src = "S=struct{x:int}; f=fn(s:&S){ let y = &s.x; };";
        let edges = collect_from_function(src, "f");

        assert!(
            edges
                .iter()
                .any(|edge| edge.reason == LifetimeOrderingReason::MemberProjection),
            "expected a member-projection ordering edge, got {edges:?}"
        );
    }

    #[test]
    fn source_member_projection_records_direction_from_projected_to_base() {
        let src = "S=struct{x:int}; f=fn(s:&S){ let y = &s.x; };";
        let edges = collect_from_function(src, "f");
        let edge = edges
            .iter()
            .find(|edge| edge.reason == LifetimeOrderingReason::MemberProjection)
            .unwrap_or_else(|| panic!("expected member projection edge, got {edges:?}"));

        assert_ne!(edge.shorter, edge.longer);
    }

    #[test]
    fn source_member_projection_chain_with_raw_root_still_emits_no_ordering() {
        let src = "S=struct{x:int}; f=fn(s:S){ let p:*S = &s; let y = &(*p).x; };";
        let edges = collect_from_function(src, "f");
        assert!(
            !edges
                .iter()
                .any(|edge| edge.reason == LifetimeOrderingReason::MemberProjection),
            "did not expect member-projection ordering through raw root, got {edges:?}"
        );
    }

    #[test]
    fn source_call_return_root_does_not_emit_ordering() {
        let src = "id=fn(x:&int)->&int{x}; f=fn(x:&int)->&int{ id(x) };";
        let edges = collect_from_function(src, "f");

        assert!(edges.is_empty());
    }

    #[test]
    fn returning_reference_to_local_reports_lifetime_error() {
        let src = "f=fn(x:&int)->&int{ let x = 2; &x };";
        with_inferred_function_ctx(src, "f", |ctx| {
            assert!(
                ctx.ex.errors.iter().any(|err| {
                    matches!(
                        err,
                        TypeError::Simple { message, .. }
                            | TypeError::SimpleRelated { message, .. }
                            if *message
                                == "borrowed value does not live long enough for required lifetime"
                    )
                }),
                "expected lifetime ordering error, got {:?}",
                ctx.ex.errors
            );
        });
    }

    #[test]
    fn cve_rs_style_code_is_rejected() {
        let src = r#"
        get_static = fn()->&'static &'static void;
        weird_func = fn['a,'b,T](r:&'a &'b void,y:&'b T)->&'a T{
            y
        }

        cheat = fn['a,'b,T](x:&'a T)->&'b T {
            weird_func(get_static(),x)
        }
        "#;
        let program = gather_program(src);
        let (result, _checked) =
            run_typecheck_scan(&program, |_,_,_| Ok(())).expect("typechecker should run");

        assert!(
            matches!(result, Err(err_count) if err_count > 0),
            "expected cve-rs style input to produce at least one type error"
        );
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
    fn scc_cycle_collapses_nodes() {
        let mut graph = collect_graph_from_function("f=fn(x:&int){ let y = &*x; };", "f");
        let forward = *graph
            .edges()
            .first()
            .unwrap_or_else(|| panic!("expected at least one source-derived edge"));
        graph.push_edge(LifetimeOrdering {
            shorter: forward.longer,
            longer: forward.shorter,
            source_origin: forward.target_origin,
            target_origin: forward.source_origin,
            reason: LifetimeOrderingReason::Reborrow,
        });

        let solve = solve_lifetime_scc(&graph);

        assert_eq!(
            solve.component_of[forward.shorter.0],
            solve.component_of[forward.longer.0]
        );
    }

    #[test]
    fn scc_acyclic_nodes_stay_separate() {
        let graph = collect_graph_from_function("f=fn(x:&int){ let y = &*x; };", "f");
        let forward = *graph
            .edges()
            .first()
            .unwrap_or_else(|| panic!("expected at least one source-derived edge"));
        let solve = solve_lifetime_scc(&graph);

        assert_ne!(
            solve.component_of[forward.shorter.0],
            solve.component_of[forward.longer.0]
        );
    }

    #[test]
    fn seeding_binding_roots_assigns_unique_local_lifetimes() {
        with_inferred_function_ctx("f=fn(){ let a = 1; let b = 2; };", "f", |ctx| {
            let function = ctx.req.owner.expect("owner should be set");
            let origins = ctx
                .ex
                .ans
                .inner_types_of_function(function)
                .map(|inner| inner.origins.clone())
                .unwrap_or_else(|| ctx.types.lifetimes.origins.clone());
            ctx.types.lifetimes.origins = origins;
            seed_origin_lifetimes_for_graph(ctx);

            let mut local_binding_lifetimes = Vec::new();
            for node in &ctx.types.lifetimes.origins.0 {
                if !matches!(node.kind, OriginKind::BindingRoot) {
                    continue;
                }
                let Some(seed) = node.lifetime_seed else {
                    continue;
                };
                let root = find_lid_root(&mut ctx.types.lifetimes.life_parent, seed);
                let Some(lt) = ctx.types.lifetimes.life_known[root] else {
                    continue;
                };
                if let LifeTime::Local(id) = lt {
                    local_binding_lifetimes.push(id);
                }
            }

            assert!(
                local_binding_lifetimes.len() >= 2,
                "expected at least two local binding lifetimes, got {local_binding_lifetimes:?}"
            );
            assert_ne!(local_binding_lifetimes[0], local_binding_lifetimes[1]);
        });
    }

    #[test]
    fn seeding_uses_associated_pointer_lifetimes_from_source_program() {
        with_inferred_function_ctx("f=fn(x:&int){ let y = &*x; };", "f", |ctx| {
            let function = ctx.req.owner.expect("owner should be set");
            let origins = ctx
                .ex
                .ans
                .inner_types_of_function(function)
                .map(|inner| inner.origins.clone())
                .unwrap_or_else(|| ctx.types.lifetimes.origins.clone());
            ctx.types.lifetimes.origins = origins;
            seed_origin_lifetimes_for_graph(ctx);

            let origin_count = ctx.types.lifetimes.origins.len();
            for raw in 0..origin_count {
                let (kind, seed_lid) = match ctx.types.origin(OriginId(raw as u32)) {
                    Some(node) => (node.kind, node.lifetime_seed),
                    None => continue,
                };

                let Some(cid) = kind.associated_pointer() else {
                    continue;
                };

                let Some(seed_lid) = seed_lid else {
                    continue;
                };
                let seed_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, seed_lid);
                let Some(seed_lt) = ctx.types.lifetimes.life_known[seed_root] else {
                    continue;
                };

                if matches!(seed_lt, LifeTime::External(_)) {
                    return;
                }

                let cid_root = ctx.types.root(cid);
                if let ResolveKind::Ptr {
                    kind: PtrKind::RefInfer(ptr_lid),
                    ..
                } = ctx.types.cluster_state(cid_root)
                {
                    let pointer_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, ptr_lid);
                    assert_eq!(
                        seed_root, pointer_root,
                        "origin seed should reuse associated pointer lifetime"
                    );
                    return;
                }
            }

            panic!(
                "expected at least one associated-pointer origin seeded to an external lifetime"
            );
        });
    }

    #[test]
    fn unresolved_under_local_ordering_is_promoted_to_local() {
        let src = "f=fn(){ let x = 1; let r = &x; let y = &*r; let p:*int = &x; let z = &*p; };";
        with_inferred_function_ctx(src, "f", |ctx| {
            let function = ctx.req.owner.expect("owner should be set");
            let origins = ctx
                .ex
                .ans
                .inner_types_of_function(function)
                .map(|inner| inner.origins.clone())
                .unwrap_or_else(|| ctx.types.lifetimes.origins.clone());
            ctx.types.lifetimes.origins = origins;
            seed_origin_lifetimes_for_graph(ctx);
            let graph = collect_origin_lifetime_orderings(
                &ctx.types.lifetimes.origins,
                ctx.types.lifetimes.life_parent.0.len(),
            );
            assign_remaining_unresolved_lifetimes_as_unknown(ctx, &graph);

            let mut saw_local_non_binding = false;
            let mut saw_raw_derived_origin = false;
            let mut raw_derived_all_known = true;

            let origin_count = ctx.types.lifetimes.origins.len();
            for raw in 0..origin_count {
                let origin = OriginId(raw as u32);
                let (kind, seed, parent) = match ctx.types.origin(origin) {
                    Some(node) => (node.kind, node.lifetime_seed, node.parent),
                    None => continue,
                };

                let Some(seed) = seed else {
                    continue;
                };
                let root = find_lid_root(&mut ctx.types.lifetimes.life_parent, seed);
                let known = ctx.types.lifetimes.life_known[root];

                if !matches!(kind, OriginKind::BindingRoot)
                    && matches!(known, Some(LifeTime::Local(_)))
                {
                    saw_local_non_binding = true;
                }

                let mut has_raw_ancestor = false;
                let mut current = parent;
                while let Some(parent_origin) = current {
                    let Some(parent_node) = ctx.types.origin(parent_origin) else {
                        break;
                    };
                    if matches!(parent_node.kind, OriginKind::RawRoot(_)) {
                        has_raw_ancestor = true;
                        break;
                    }
                    current = parent_node.parent;
                }

                if has_raw_ancestor {
                    saw_raw_derived_origin = true;
                    raw_derived_all_known &= known.is_some();
                }
            }

            assert!(
                saw_local_non_binding,
                "expected at least one non-binding origin lifetime to be local"
            );
            assert!(
                !saw_raw_derived_origin || raw_derived_all_known,
                "expected raw-derived origins to have assigned lifetimes after resolution pass"
            );
        });
    }
}
