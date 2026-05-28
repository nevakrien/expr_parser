use crate::data_structures::bit_sets::{SparseBits, propegate_constraints};
use crate::data_structures::graph::{CompId, SCCS, VecGraph, tarjan};
use crate::data_structures::index::{Idx, IndexVec, UnionFind};
use crate::type_inference::{
    CId, ImportedLifetimeOrdering, InferState, LId, LifeId, LifeTime, LifetimeOrderingEdge,
    OriginDeclSite, OriginId, OriginKind, OriginNode, PointerStyle, ResolveKind, TypeError,
    TypeValue, find_lid_root, lifetime_for_display, unify_struct_lids,
};
use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LifetimeGraphId(pub usize);

impl LifetimeGraphId {
    fn from_lid(lid: LId, node_count: usize) -> Option<Self> {
        (lid.0 < node_count).then_some(Self(lid.0))
    }
}

impl Idx for LifetimeGraphId {
    fn new(idx: usize) -> Self {
        LifetimeGraphId(idx)
    }
    fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct UniversalLifetimeId(usize);

impl Idx for UniversalLifetimeId {
    fn new(idx: usize) -> Self {
        UniversalLifetimeId(idx)
    }
    fn index(self) -> usize {
        self.0
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WhereClauseLifetimeOrdering {
    pub(crate) shorter: LifetimeGraphId,
    pub(crate) longer: LifetimeGraphId,
}

pub(crate) struct LifetimeOrderingGraph {
    origin_edges: Vec<LifetimeOrdering>,
    where_clause_edges: Vec<WhereClauseLifetimeOrdering>,
    outgoing: Vec<Vec<usize>>,
    where_clause_outgoing: Vec<Vec<usize>>,
    seen_edges: HashSet<(usize, usize)>,
    seen_where_clause_edges: HashSet<(usize, usize)>,
}

impl LifetimeOrderingGraph {
    pub(crate) fn new(lid_count: usize) -> Self {
        let mut outgoing = Vec::with_capacity(lid_count);
        outgoing.resize_with(lid_count, Vec::new);
        let mut where_clause_outgoing = Vec::with_capacity(lid_count);
        where_clause_outgoing.resize_with(lid_count, Vec::new);
        Self {
            origin_edges: Vec::new(),
            where_clause_edges: Vec::new(),
            outgoing,
            where_clause_outgoing,
            seen_edges: HashSet::new(),
            seen_where_clause_edges: HashSet::new(),
        }
    }

    pub(crate) fn origin_edges(&self) -> &[LifetimeOrdering] {
        &self.origin_edges
    }

    pub(crate) fn lid_count(&self) -> usize {
        self.outgoing.len()
    }

    #[allow(dead_code)]
    pub(crate) fn outgoing(&self, lid: LifetimeGraphId) -> &[usize] {
        &self.outgoing[lid.0]
    }

    pub(crate) fn where_clause_edges(&self) -> &[WhereClauseLifetimeOrdering] {
        &self.where_clause_edges
    }

    pub(crate) fn where_clause_outgoing(&self, lid: LifetimeGraphId) -> &[usize] {
        &self.where_clause_outgoing[lid.0]
    }

    fn push_edge(&mut self, edge: LifetimeOrdering) {
        if !self.seen_edges.insert((edge.shorter.0, edge.longer.0)) {
            return;
        }

        let index = self.origin_edges.len();
        self.origin_edges.push(edge);
        self.outgoing[edge.shorter.0].push(index);
    }

    pub(crate) fn push_where_clause_edge(
        &mut self,
        shorter: LifetimeGraphId,
        longer: LifetimeGraphId,
    ) {
        if shorter == longer || shorter.0 >= self.lid_count() || longer.0 >= self.lid_count() {
            return;
        }

        if !self.seen_where_clause_edges.insert((shorter.0, longer.0)) {
            return;
        }

        let index = self.where_clause_edges.len();
        self.where_clause_edges
            .push(WhereClauseLifetimeOrdering { shorter, longer });
        self.where_clause_outgoing[shorter.0].push(index);
    }
}

pub(crate) fn collect_decl_lifetime_orderings(
    lid_count: usize,
    edges: &[LifetimeOrderingEdge],
) -> LifetimeOrderingGraph {
    let mut graph = LifetimeOrderingGraph::new(lid_count);
    for edge in edges {
        graph.push_where_clause_edge(edge.shorter, edge.longer);
    }
    graph
}

pub(crate) fn collect_origin_lifetime_orderings(
    origins: &IndexVec<OriginId, OriginNode>,
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
    solve_lifetime_scc_with_modes(graph, true, true)
}

pub(crate) fn solve_where_clause_lifetime_scc(graph: &LifetimeOrderingGraph) -> LifetimeSccSolve {
    solve_lifetime_scc_with_modes(graph, false, true)
}

fn solve_lifetime_scc_with_modes(
    graph: &LifetimeOrderingGraph,
    include_origin_edges: bool,
    include_where_clause_edges: bool,
) -> LifetimeSccSolve {
    let sccs = lifetime_sccs_with_modes(graph, include_origin_edges, include_where_clause_edges);

    let lid_count = sccs.map.len();
    let mut component_of = vec![0; lid_count];
    for (i, cid) in sccs.map.iter().enumerate() {
        component_of[i] = cid.index();
    }
    let components = sccs
        .comps
        .into_raw()
        .into_iter()
        .map(|nodes| LifetimeSccComponent { nodes })
        .collect();

    LifetimeSccSolve {
        component_of,
        components,
    }
}

fn lifetime_sccs_with_modes(
    graph: &LifetimeOrderingGraph,
    include_origin_edges: bool,
    include_where_clause_edges: bool,
) -> SCCS<LifetimeGraphId> {
    let node_count = graph.lid_count();
    let mut edges: Vec<LifetimeGraphId> = Vec::new();
    let mut nodes: IndexVec<LifetimeGraphId, std::ops::Range<usize>> =
        IndexVec::with_capacity(node_count);

    for n in 0..node_count {
        let start = edges.len();
        if include_origin_edges {
            for &idx in &graph.outgoing[n] {
                edges.push(graph.origin_edges[idx].longer);
            }
        }
        if include_where_clause_edges {
            for &idx in &graph.where_clause_outgoing[n] {
                edges.push(graph.where_clause_edges[idx].longer);
            }
        }
        nodes.push(start..edges.len());
    }

    tarjan(&VecGraph::from_raw(nodes, edges))
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

fn ordering_reason_text(reason: LifetimeOrderingReason) -> &'static str {
    match reason {
        LifetimeOrderingReason::Reborrow => "reborrow",
        LifetimeOrderingReason::Deref => "dereference",
        LifetimeOrderingReason::MemberProjection => "member access",
        LifetimeOrderingReason::IndexProjection => "index access",
        LifetimeOrderingReason::CastProjection => "cast",
    }
}

fn lifetime_for_origin(origin: OriginId, node: &OriginNode) -> Option<OriginLifetime> {
    let lid = node.lifetime_seed?;
    Some(OriginLifetime { origin, lid })
}

fn source_origin_for_ordering(
    origins: &IndexVec<OriginId, OriginNode>,
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

    let lid_count = ctx.types.lifetimes.life_parent.len();
    let mut graph = LifetimeOrderingGraph::new(lid_count);
    collect_imported_lifetime_orderings(
        &mut graph,
        &mut ctx.types.lifetimes.life_parent,
        &ctx.types.lifetimes.imported_orderings,
        lid_count,
    );
    let origin_graph = collect_origin_lifetime_orderings(&ctx.types.lifetimes.origins, lid_count);
    for edge in origin_graph.origin_edges() {
        graph.push_edge(*edge);
    }
    let solve = solve_lifetime_scc(&graph);
    let invalid_components = collect_invalid_lifetime_ordering_components(ctx, &graph, &solve);
    for (component_index, component) in solve.components.iter().enumerate() {
        if component.nodes.len() < 2 {
            continue;
        }

        if invalid_components
            .get(component_index)
            .copied()
            .unwrap_or(false)
        {
            continue;
        }

        let leader = LId(component.nodes[0].0);
        for &node in component.nodes.iter().skip(1) {
            let lid = LId(node.0);
            let _ = unify_struct_lids(&mut ctx.types, leader, lid);
        }
    }

    assign_remaining_unresolved_lifetimes_as_unknown(ctx, &graph);
    validate_discovered_global_lifetime_paths(ctx, &graph);
    validate_known_lifetime_orderings(ctx, &graph);
    validate_imported_known_lifetime_orderings(ctx);
}

fn collect_invalid_lifetime_ordering_components(
    ctx: &mut InferState,
    graph: &LifetimeOrderingGraph,
    solve: &LifetimeSccSolve,
) -> Vec<bool> {
    let mut invalid_components = vec![false; solve.components.len()];

    for edge in graph.origin_edges() {
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

        if !requires_explicit_where_ordering(shorter_lt, longer_lt)
            && !matches!(shorter_lt.partial_cmp(&longer_lt), Some(Ordering::Greater))
        {
            continue;
        }

        let Some(component_id) = solve.component_of.get(edge.shorter.0).copied() else {
            continue;
        };
        if solve.component_of.get(edge.longer.0).copied() != Some(component_id) {
            continue;
        }
        invalid_components[component_id] = true;
    }

    invalid_components
}

fn validate_known_lifetime_orderings(ctx: &mut InferState, graph: &LifetimeOrderingGraph) {
    let mut seen_root_pairs: HashSet<(usize, usize)> =
        HashSet::with_capacity(graph.origin_edges().len());

    for edge in graph.origin_edges() {
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

        let has_impossible_known_order =
            matches!(shorter_lt.partial_cmp(&longer_lt), Some(Ordering::Greater));
        if !has_impossible_known_order {
            continue;
        }

        let pair = (shorter_root.0, longer_root.0);
        if !seen_root_pairs.insert(pair) {
            continue;
        }

        report_lifetime_ordering_conflict(ctx, edge);
    }
}

fn validate_imported_known_lifetime_orderings(ctx: &mut InferState) {
    let mut seen_root_pairs: HashSet<(usize, usize, crate::parsing::Loc)> =
        HashSet::with_capacity(ctx.types.lifetimes.imported_orderings.len());

    for idx in 0..ctx.types.lifetimes.imported_orderings.len() {
        let edge = ctx.types.lifetimes.imported_orderings[idx].clone();
        let shorter_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, edge.shorter);
        let longer_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, edge.longer);

        if shorter_root == longer_root {
            continue;
        }

        let shorter_lt = ctx.types.lifetimes.life_known[shorter_root];
        let longer_lt = ctx.types.lifetimes.life_known[longer_root];
        let (Some(shorter_lt), Some(longer_lt)) = (shorter_lt, longer_lt) else {
            continue;
        };

        let has_impossible_known_order =
            matches!(shorter_lt.partial_cmp(&longer_lt), Some(Ordering::Greater));
        if !has_impossible_known_order {
            continue;
        }

        let pair = (shorter_root.0, longer_root.0, edge.site.clone());
        if !seen_root_pairs.insert(pair) {
            continue;
        }

        let loc = edge.site.clone();
        let shorter = format_lid_name(ctx, shorter_root);
        let longer = format_lid_name(ctx, longer_root);
        ctx.push_error(TypeError::LifetimeOrderingConflict {
            loc,
            operation: "imported where-clause bound",
            shorter,
            longer,
            related: None,
        });
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
    for raw in 0..ctx.types.lifetimes.life_parent.len() {
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
    let root_count = ctx.types.lifetimes.life_parent.len();
    let mut predecessors: Vec<Vec<LId>> = vec![Vec::new(); root_count];

    for edge in graph.origin_edges() {
        let shorter_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(edge.shorter.0));
        let longer_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(edge.longer.0));
        if shorter_root == longer_root {
            continue;
        }
        predecessors[longer_root.0].push(shorter_root);
    }

    for edge in graph.where_clause_edges() {
        let shorter_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(edge.shorter.0));
        let longer_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(edge.longer.0));
        if shorter_root == longer_root {
            continue;
        }
        predecessors[longer_root.0].push(shorter_root);
    }

    let mut promote_to_local = vec![false; root_count];
    let mut stack = Vec::with_capacity(root_count);
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

    for lid_index in 0..root_count {
        let root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(lid_index));
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
    for known in ctx.types.lifetimes.life_known.iter().copied().flatten() {
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

fn next_unknown_lifetime_id(ctx: &InferState) -> u32 {
    let mut next = 0;
    for known in ctx.types.lifetimes.life_known.iter().copied().flatten() {
        if let LifeTime::Unknown(id) = known {
            next = next.max(id.0.saturating_add(1));
        }
    }
    next
}

fn collect_imported_lifetime_orderings(
    graph: &mut LifetimeOrderingGraph,
    life_parent: &mut UnionFind<LId>,
    imported_orderings: &[ImportedLifetimeOrdering],
    lid_count: usize,
) {
    for edge in imported_orderings {
        let shorter = life_parent.find_root(edge.shorter);
        let longer = life_parent.find_root(edge.longer);
        let Some(shorter) = LifetimeGraphId::from_lid(shorter, lid_count) else {
            continue;
        };
        let Some(longer) = LifetimeGraphId::from_lid(longer, lid_count) else {
            continue;
        };
        graph.push_where_clause_edge(shorter, longer);
    }
}

fn validate_discovered_global_lifetime_paths(ctx: &mut InferState, graph: &LifetimeOrderingGraph) {
    let found_sccs = lifetime_sccs_with_modes(graph, true, true);
    let found_reachable = propagated_component_reachability(&found_sccs);
    let allowed_external_reachable = collect_allowed_external_lifetime_reachability(ctx);

    let mut known_lifetimes = Vec::with_capacity(graph.lid_count());
    for raw in 0..graph.lid_count() {
        let root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(raw));
        known_lifetimes.push(ctx.types.lifetimes.life_known[root]);
    }

    let mut global_nodes = Vec::new();
    for (raw, lt) in known_lifetimes.iter().copied().enumerate() {
        if matches!(lt, Some(LifeTime::External(_) | LifeTime::Static)) {
            global_nodes.push(LifetimeGraphId(raw));
        }
    }

    let mut seen = HashSet::new();
    for &shorter in &global_nodes {
        for &longer in &global_nodes {
            if shorter == longer {
                continue;
            }

            let shorter_lt = known_lifetimes[shorter.0].expect("global node must be known");
            let longer_lt = known_lifetimes[longer.0].expect("global node must be known");
            if !disallowed_global_ordering_needs_report(shorter_lt, longer_lt) {
                continue;
            }

            let found_from = found_sccs.map[shorter];
            let found_to = found_sccs.map[longer];
            if !found_reachable.contains(found_from, found_to) {
                continue;
            }

            if external_lifetime_ordering_allowed(
                &allowed_external_reachable,
                shorter_lt,
                longer_lt,
            ) {
                continue;
            }

            let shorter_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(shorter.0));
            let longer_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(longer.0));
            if !seen.insert((shorter_root.0, longer_root.0)) {
                continue;
            }

            if let Some(path) = shortest_lifetime_path(graph, shorter, longer)
                && report_illegal_global_lifetime_path(ctx, &path)
            {
                continue;
            }
            report_illegal_imported_lifetime_ordering(ctx, shorter, longer);
        }

        for raw_longer in 0..known_lifetimes.len() {
            let longer = LifetimeGraphId(raw_longer);
            let Some(longer_lt) = known_lifetimes[longer.0] else {
                continue;
            };
            if !matches!(longer_lt, LifeTime::Local(_)) {
                continue;
            }

            let found_from = found_sccs.map[shorter];
            let found_to = found_sccs.map[longer];
            if !found_reachable.contains(found_from, found_to) {
                continue;
            }

            let shorter_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(shorter.0));
            let longer_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(longer.0));
            if !seen.insert((shorter_root.0, longer_root.0)) {
                continue;
            }

            if let Some(path) = shortest_lifetime_path(graph, shorter, longer)
                && report_illegal_global_lifetime_path(ctx, &path)
            {
                continue;
            }
            report_illegal_imported_lifetime_ordering(ctx, shorter, longer);
        }
    }

    report_global_components_containing_locals(ctx, graph, &found_sccs, &known_lifetimes);
}

fn collect_allowed_external_lifetime_reachability(
    ctx: &mut InferState,
) -> SparseBits<UniversalLifetimeId, UniversalLifetimeId> {
    let mut edges = Vec::with_capacity(ctx.types.lifetimes.declared_allowed_orderings.len());
    let lifetime_count = external_lifetime_count(ctx);

    for idx in 0..ctx.types.lifetimes.declared_allowed_orderings.len() {
        let edge = ctx.types.lifetimes.declared_allowed_orderings[idx];
        let shorter = find_lid_root(&mut ctx.types.lifetimes.life_parent, edge.shorter);
        let longer = find_lid_root(&mut ctx.types.lifetimes.life_parent, edge.longer);
        let Some(LifeTime::External(shorter)) = ctx.types.lifetimes.life_known[shorter] else {
            continue;
        };
        let Some(LifeTime::External(longer)) = ctx.types.lifetimes.life_known[longer] else {
            continue;
        };

        let shorter = UniversalLifetimeId(shorter as usize);
        let longer = UniversalLifetimeId(longer as usize);
        edges.push((shorter, longer));
    }

    let mut outgoing: IndexVec<UniversalLifetimeId, Vec<UniversalLifetimeId>> =
        (0..lifetime_count).map(|_| Vec::new()).collect();
    for (shorter, longer) in edges {
        if shorter != longer {
            outgoing[shorter].push(longer);
        }
    }

    let sccs = tarjan(&ExternalLifetimeGraph { outgoing });
    let component_reachable = propagated_external_component_reachability(&sccs);
    let mut reachable = SparseBits::new(UniversalLifetimeId(lifetime_count), lifetime_count);
    for shorter in 0..lifetime_count {
        for longer in 0..lifetime_count {
            let shorter = UniversalLifetimeId(shorter);
            let longer = UniversalLifetimeId(longer);
            if component_reachable.contains(sccs.map[shorter], sccs.map[longer]) {
                reachable.insert(shorter, longer);
            }
        }
    }
    reachable
}

fn external_lifetime_count(ctx: &InferState) -> usize {
    ctx.types
        .lifetimes
        .life_known
        .iter()
        .copied()
        .flatten()
        .filter_map(|lt| match lt {
            LifeTime::External(id) => Some(id as usize + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

struct ExternalLifetimeGraph {
    outgoing: IndexVec<UniversalLifetimeId, Vec<UniversalLifetimeId>>,
}

impl crate::data_structures::graph::DirectedGraph for ExternalLifetimeGraph {
    type Node = UniversalLifetimeId;

    fn num_nodes(&self) -> usize {
        self.outgoing.len()
    }

    fn edges(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.outgoing[node].iter().copied()
    }
}

fn propagated_external_component_reachability(
    sccs: &SCCS<UniversalLifetimeId>,
) -> SparseBits<CompId<UniversalLifetimeId>, CompId<UniversalLifetimeId>> {
    let component_count = sccs.comps.len();
    let mut reachable = SparseBits::new(CompId::new(component_count), component_count);
    for raw in 0..component_count {
        let component = CompId::new(raw);
        reachable.insert(component, component);
    }
    propegate_constraints(&mut reachable, sccs);
    reachable
}

fn external_lifetime_ordering_allowed(
    allowed: &SparseBits<UniversalLifetimeId, UniversalLifetimeId>,
    shorter: LifeTime,
    longer: LifeTime,
) -> bool {
    let (LifeTime::External(shorter), LifeTime::External(longer)) = (shorter, longer) else {
        return false;
    };
    allowed.contains(
        UniversalLifetimeId(shorter as usize),
        UniversalLifetimeId(longer as usize),
    )
}

fn report_illegal_imported_lifetime_ordering(
    ctx: &mut InferState,
    shorter: LifetimeGraphId,
    longer: LifetimeGraphId,
) {
    let shorter_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(shorter.0));
    let longer_root = find_lid_root(&mut ctx.types.lifetimes.life_parent, LId(longer.0));

    for idx in 0..ctx.types.lifetimes.imported_orderings.len() {
        let edge = ctx.types.lifetimes.imported_orderings[idx].clone();
        let imported_shorter = find_lid_root(&mut ctx.types.lifetimes.life_parent, edge.shorter);
        let imported_longer = find_lid_root(&mut ctx.types.lifetimes.life_parent, edge.longer);
        if imported_shorter != shorter_root || imported_longer != longer_root {
            continue;
        }

        let loc = edge.site.clone();
        let shorter = format_lid_name(ctx, shorter_root);
        let longer = format_lid_name(ctx, longer_root);
        ctx.push_error(TypeError::IllegalGlobalLifetimeOrdering {
            loc,
            operation: "imported where-clause bound",
            shorter,
            longer,
            path: None,
            related: None,
        });
        return;
    }
}

fn propagated_component_reachability(
    sccs: &SCCS<LifetimeGraphId>,
) -> SparseBits<CompId<LifetimeGraphId>, CompId<LifetimeGraphId>> {
    let component_count = sccs.comps.len();
    let mut reachable = SparseBits::new(CompId::new(component_count), component_count);
    for raw in 0..component_count {
        let component = CompId::new(raw);
        reachable.insert(component, component);
    }
    propegate_constraints(&mut reachable, sccs);
    reachable
}

fn requires_explicit_where_ordering(shorter: LifeTime, longer: LifeTime) -> bool {
    matches!((shorter, longer), (LifeTime::External(a), LifeTime::External(b)) if a != b)
}

fn illegal_global_to_local_ordering(shorter: LifeTime, longer: LifeTime) -> bool {
    matches!(shorter, LifeTime::External(_) | LifeTime::Static)
        && matches!(longer, LifeTime::Local(_))
}

fn disallowed_global_ordering_needs_report(shorter: LifeTime, longer: LifeTime) -> bool {
    requires_explicit_where_ordering(shorter, longer)
        || illegal_global_to_local_ordering(shorter, longer)
}

#[derive(Debug, Clone, Copy)]
enum LifetimePathStep {
    Origin(LifetimeOrdering),
    Where(WhereClauseLifetimeOrdering),
}

impl LifetimePathStep {
    fn shorter(self) -> LifetimeGraphId {
        match self {
            LifetimePathStep::Origin(edge) => edge.shorter,
            LifetimePathStep::Where(edge) => edge.shorter,
        }
    }

    fn longer(self) -> LifetimeGraphId {
        match self {
            LifetimePathStep::Origin(edge) => edge.longer,
            LifetimePathStep::Where(edge) => edge.longer,
        }
    }
}

fn shortest_lifetime_path(
    graph: &LifetimeOrderingGraph,
    start: LifetimeGraphId,
    target: LifetimeGraphId,
) -> Option<Vec<LifetimePathStep>> {
    let mut visited = vec![false; graph.lid_count()];
    let mut previous: Vec<Option<(LifetimeGraphId, LifetimePathStep)>> =
        vec![None; graph.lid_count()];
    let mut queue = VecDeque::new();
    visited[start.0] = true;
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        if current == target {
            break;
        }

        for step in lifetime_path_steps_from(graph, current) {
            let next = step.longer();
            if visited[next.0] {
                continue;
            }
            visited[next.0] = true;
            previous[next.0] = Some((current, step));
            queue.push_back(next);
        }
    }

    if !visited[target.0] {
        return None;
    }

    let mut path = Vec::new();
    let mut current = target;
    while current != start {
        let (prev, step) = previous[current.0]?;
        path.push(step);
        current = prev;
    }
    path.reverse();
    Some(path)
}

fn lifetime_path_steps_from(
    graph: &LifetimeOrderingGraph,
    node: LifetimeGraphId,
) -> impl Iterator<Item = LifetimePathStep> + '_ {
    graph
        .outgoing(node)
        .iter()
        .map(|&idx| LifetimePathStep::Origin(graph.origin_edges()[idx]))
        .chain(
            graph
                .where_clause_outgoing(node)
                .iter()
                .map(|&idx| LifetimePathStep::Where(graph.where_clause_edges()[idx])),
        )
}

fn report_illegal_global_lifetime_path(ctx: &mut InferState, path: &[LifetimePathStep]) -> bool {
    if path.is_empty() {
        return false;
    }
    let path_text = format_lifetime_path(ctx, path);

    if let Some(origin_edge) = path.iter().find_map(|step| match step {
        LifetimePathStep::Origin(edge) => Some(*edge),
        LifetimePathStep::Where(_) => None,
    }) {
        report_illegal_global_lifetime_ordering_with_path(ctx, &origin_edge, Some(path_text));
        return true;
    }

    false
}

fn format_lifetime_path(ctx: &mut InferState, path: &[LifetimePathStep]) -> String {
    let mut parts = Vec::with_capacity(path.len() + 1);
    if let Some(first) = path.first() {
        parts.push(format_lid_name(ctx, LId(first.shorter().0)));
    }
    for step in path {
        parts.push(format_lid_name(ctx, LId(step.longer().0)));
    }
    parts.join(" -> ")
}

fn report_global_components_containing_locals(
    ctx: &mut InferState,
    graph: &LifetimeOrderingGraph,
    sccs: &SCCS<LifetimeGraphId>,
    known_lifetimes: &[Option<LifeTime>],
) {
    let mut seen_components = HashSet::new();
    for (component_id, component) in sccs.comps.iter_enumerated() {
        let global = component.iter().copied().find(|node| {
            matches!(
                known_lifetimes[node.0],
                Some(LifeTime::External(_) | LifeTime::Static)
            )
        });
        let local = component
            .iter()
            .copied()
            .find(|node| matches!(known_lifetimes[node.0], Some(LifeTime::Local(_))));
        let (Some(global), Some(local)) = (global, local) else {
            continue;
        };
        if !seen_components.insert(component_id.index()) {
            continue;
        }
        if let Some(path) = shortest_lifetime_path(graph, global, local)
            .or_else(|| shortest_lifetime_path(graph, local, global))
        {
            report_lifetime_path_conflict(ctx, &path);
        }
    }
}

fn report_lifetime_path_conflict(ctx: &mut InferState, path: &[LifetimePathStep]) {
    let Some(origin_edge) = path.iter().find_map(|step| match step {
        LifetimePathStep::Origin(edge) => Some(*edge),
        LifetimePathStep::Where(_) => None,
    }) else {
        return;
    };
    report_lifetime_ordering_conflict(ctx, &origin_edge);
}

fn format_lid_name(ctx: &mut InferState, lid: LId) -> String {
    let root = find_lid_root(&mut ctx.types.lifetimes.life_parent, lid);
    match ctx.types.lifetimes.life_known[root] {
        Some(lt) => format!("'{}", lifetime_for_display(&ctx.ex, lt)),
        None => format!("lifetime#{}", root.0),
    }
}

fn origin_loc(ctx: &InferState, origin: OriginId) -> Option<crate::parsing::Loc> {
    ctx.types
        .origin(origin)
        .and_then(|node| node.decl_site)
        .map(|site| match site {
            OriginDeclSite::Pattern(pattern) => ctx.ex.program.pattern_loc(pattern),
            OriginDeclSite::Value(value) => ctx.ex.program.value_loc(value),
        })
}

fn report_lifetime_ordering_conflict(ctx: &mut InferState, edge: &LifetimeOrdering) {
    let source_loc = origin_loc(ctx, edge.source_origin);
    let target_loc = origin_loc(ctx, edge.target_origin);
    let shorter_name = format_lid_name(ctx, LId(edge.shorter.0));
    let longer_name = format_lid_name(ctx, LId(edge.longer.0));
    let operation = ordering_reason_text(edge.reason);

    if let (Some(loc), Some(related)) = (source_loc.clone(), target_loc.clone()) {
        ctx.push_error(TypeError::LifetimeOrderingConflict {
            loc,
            operation,
            shorter: shorter_name,
            longer: longer_name,
            related: Some(related),
        });
        return;
    }

    if let Some(loc) = source_loc.or(target_loc) {
        ctx.push_error(TypeError::LifetimeOrderingConflict {
            loc,
            operation,
            shorter: shorter_name,
            longer: longer_name,
            related: None,
        });
    }
}

fn report_illegal_global_lifetime_ordering_with_path(
    ctx: &mut InferState,
    edge: &LifetimeOrdering,
    path: Option<String>,
) {
    let source_loc = origin_loc(ctx, edge.source_origin);
    let target_loc = origin_loc(ctx, edge.target_origin);
    let shorter_name = format_lid_name(ctx, LId(edge.shorter.0));
    let longer_name = format_lid_name(ctx, LId(edge.longer.0));
    let operation = ordering_reason_text(edge.reason);

    if let (Some(loc), Some(related)) = (source_loc.clone(), target_loc.clone()) {
        ctx.push_error(TypeError::IllegalGlobalLifetimeOrdering {
            loc,
            operation,
            shorter: shorter_name,
            longer: longer_name,
            path,
            related: Some(related),
        });
        return;
    }

    if let Some(loc) = source_loc.or(target_loc) {
        ctx.push_error(TypeError::IllegalGlobalLifetimeOrdering {
            loc,
            operation,
            shorter: shorter_name,
            longer: longer_name,
            path,
            related: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LifetimeOrdering, LifetimeOrderingGraph, LifetimeOrderingReason,
        assign_remaining_unresolved_lifetimes_as_unknown, collect_origin_lifetime_orderings,
        lifetime_sccs_with_modes, propagated_component_reachability,
        seed_origin_lifetimes_for_graph, shortest_lifetime_path, solve_lifetime_scc,
    };
    use crate::data_structures::index::IndexVec;
    use crate::global_type_inference::infer_global_types;
    use crate::ir::{ValId, Value};
    use crate::local_type_inference::{
        gather_func_constraints, infer_value_internals, local_solver,
    };
    use crate::parsing::Parser;
    use crate::program::{Defined, Program};
    use crate::type_inference::run_typecheck_scan;
    use crate::type_inference::{
        InferState, LifeTime, OriginId, OriginKind, OriginNode, PtrKind, ResolveKind, SolvedTypes,
        TypeError, TypeStore, find_lid_root,
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

        let previous_name_render = std::mem::replace(
            &mut ctx.ex.name_render,
            crate::type_inference::GenLifeNameRender::from_decl(ctx.ex.program, generics),
        );

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
        ctx.ex.name_render = previous_name_render;

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
            let lid_count = ctx.types.lifetimes.life_parent.len();
            collect_origin_lifetime_orderings(&origins, lid_count)
        })
    }

    fn collect_from_function(src: &str, name: &str) -> Vec<LifetimeOrdering> {
        collect_graph_from_function(src, name)
            .origin_edges()
            .to_vec()
    }

    fn collect_origins_from_function(src: &str, name: &str) -> IndexVec<OriginId, OriginNode> {
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
                        TypeError::LifetimeOrderingConflict { shorter, longer, .. }
                            if shorter == "'a0" && longer == "'l0"
                    )
                }),
                "expected lifetime ordering error, got {:?}",
                ctx.ex.errors
            );
        });
    }

    #[test]
    fn lifetime_cycle_error_mentions_named_lifetimes() {
        let src = "f=fn['a,'b](r1:&'a &'a int,r2:&'a &'b int)->&'a &'a int { & & * * r2 }";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let errs = infer_value_internals(&program, &mut store, &mut solved_types, f)
            .err()
            .expect("expected lifetime cycle to fail");

        assert!(
            errs.iter()
                .filter(|err| {
                    matches!(
                        err,
                        TypeError::IllegalGlobalLifetimeOrdering { shorter, longer, .. }
                            if shorter.contains("'a") && longer.contains("'b")
                                || shorter.contains("'b") && longer.contains("'a")
                    )
                })
                .count()
                == 2,
            "expected exactly two illegal global lifetime ordering errors, got {:?}",
            errs
        );
    }

    #[test]
    fn distinct_global_lifetime_ordering_is_rejected_without_cycle() {
        let src = "f=fn['a,'b](r1:&'a &'a int)->&'b &'a int { & & * * r1 }";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        let errs = infer_value_internals(&program, &mut store, &mut solved_types, f)
            .err()
            .expect("expected distinct global lifetime ordering to fail");

        assert!(
            errs.iter().any(|err| {
                matches!(
                    err,
                    TypeError::IllegalGlobalLifetimeOrdering { shorter, longer, .. }
                        if (shorter.contains("'a") || shorter.contains("'b"))
                            && (longer.contains("'a") || longer.contains("'b"))
                )
            }),
            "expected illegal global lifetime ordering error, got {:?}",
            errs
        );
    }

    #[test]
    fn declared_where_clause_allows_discovered_global_lifetime_ordering() {
        let src = "f=fn['a,'b where 'b < 'a](r1:&'a &'a int)->&'b &'a int { & & * * r1 }";
        let program = gather_program(src);
        let mut store = TypeStore::new();
        let mut solved_types = SolvedTypes::new(&program);
        infer_global_types(&program, &mut store, &mut solved_types).unwrap();

        let f = find_value_by_name(&program, "f");
        infer_value_internals(&program, &mut store, &mut solved_types, f).unwrap();
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
            run_typecheck_scan(&program, |_, _, _| Ok(())).expect("typechecker should run");

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
                .iter()
                .any(|node| matches!(node.kind, OriginKind::PlaceRoot(_))),
            "expected at least one PlaceRoot origin in simple let addr-of case"
        );
        assert!(
            !origins
                .iter()
                .any(|node| matches!(node.kind, OriginKind::RawRoot(_))),
            "did not expect RawRoot origin in simple let addr-of case"
        );
    }

    #[test]
    fn scc_cycle_collapses_nodes() {
        let mut graph = collect_graph_from_function("f=fn(x:&int){ let y = &*x; };", "f");
        let forward = *graph
            .origin_edges()
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
            .origin_edges()
            .first()
            .unwrap_or_else(|| panic!("expected at least one source-derived edge"));
        let solve = solve_lifetime_scc(&graph);

        assert_ne!(
            solve.component_of[forward.shorter.0],
            solve.component_of[forward.longer.0]
        );
    }

    #[test]
    fn where_clause_scc_reachability_includes_transitive_ordering() {
        let mut graph = LifetimeOrderingGraph::new(3);
        graph.push_where_clause_edge(super::LifetimeGraphId(0), super::LifetimeGraphId(1));
        graph.push_where_clause_edge(super::LifetimeGraphId(1), super::LifetimeGraphId(2));

        let sccs = lifetime_sccs_with_modes(&graph, false, true);
        let reachable = propagated_component_reachability(&sccs);

        assert!(reachable.contains(
            sccs.map[super::LifetimeGraphId(0)],
            sccs.map[super::LifetimeGraphId(2)]
        ));
    }

    #[test]
    fn shortest_lifetime_path_uses_bfs_path() {
        let mut graph = LifetimeOrderingGraph::new(4);
        graph.push_edge(LifetimeOrdering {
            shorter: super::LifetimeGraphId(0),
            longer: super::LifetimeGraphId(1),
            source_origin: OriginId(0),
            target_origin: OriginId(1),
            reason: LifetimeOrderingReason::Reborrow,
        });
        graph.push_edge(LifetimeOrdering {
            shorter: super::LifetimeGraphId(1),
            longer: super::LifetimeGraphId(3),
            source_origin: OriginId(1),
            target_origin: OriginId(3),
            reason: LifetimeOrderingReason::Reborrow,
        });
        graph.push_edge(LifetimeOrdering {
            shorter: super::LifetimeGraphId(0),
            longer: super::LifetimeGraphId(2),
            source_origin: OriginId(0),
            target_origin: OriginId(2),
            reason: LifetimeOrderingReason::Deref,
        });
        graph.push_edge(LifetimeOrdering {
            shorter: super::LifetimeGraphId(2),
            longer: super::LifetimeGraphId(1),
            source_origin: OriginId(2),
            target_origin: OriginId(1),
            reason: LifetimeOrderingReason::Deref,
        });

        let path =
            shortest_lifetime_path(&graph, super::LifetimeGraphId(0), super::LifetimeGraphId(3))
                .expect("expected path");

        assert_eq!(path.len(), 2);
        assert_eq!(path[0].longer(), super::LifetimeGraphId(1));
        assert_eq!(path[1].longer(), super::LifetimeGraphId(3));
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
            for node in ctx.types.lifetimes.origins.iter() {
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
                ctx.types.lifetimes.life_parent.len(),
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
