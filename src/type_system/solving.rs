use super::Projection;
use super::TypeClash;
use super::{
    ArraySize, BuiltinKind, FloatKind, HARD_CODED_BUILTIN_KINDS, IntKind, KindId, KindSpan, LifeId,
    LifeKind, LifeSpan, MutId, Nullable, Origin, OriginId, OriginVec, PointerStyle, PtrId,
    StructId, TypeKind,
};
use crate::data_structures::graph::BasicOrder;
use crate::data_structures::identity_hasher::IdHashMap;
use crate::data_structures::index::{Idx, IndexVec, UnionFind};
use crate::data_structures::string_intern::StrId;
use crate::ir::{LifeTimeId, NameId, PatId, TExpId, ValId};
use crate::program::Program;
use std::collections::{BTreeSet, HashMap};
use std::ops::{Index, IndexMut};

#[derive(Debug)]
pub struct TypeUniverse {
    pub look: KindLookUp,
    pub storage: KindStorage,
}

#[derive(Debug)]
pub struct TypeIntern {
    map: IdHashMap<TypeKind, KindId>,
    pub storage: IndexVec<KindId, Option<TypeKind>>,
}

impl TypeIntern {
    pub fn new() -> Self {
        Self {
            map: IdHashMap::default(),
            storage: IndexVec::new(),
        }
    }

    pub fn intern(&mut self, uf: &mut UnionFind<KindId>, ty: TypeKind) -> KindId {
        if let Some(id) = self.map.get(&ty).copied() {
            return id;
        }

        let id = self.storage.push(Some(ty));
        self.map.insert(ty, id);
        let uf_id = uf.push_singleton();
        debug_assert_eq!(id, uf_id);

        id
    }

    pub fn add_empty(&mut self, uf: &mut UnionFind<KindId>) -> KindId {
        uf.push_singleton();
        self.storage.push(None)
    }
}

impl Default for TypeIntern {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<KindId> for TypeIntern {
    type Output = Option<TypeKind>;

    #[inline]
    fn index(&self, i: KindId) -> &Self::Output {
        &self.storage[i]
    }
}

impl IndexMut<KindId> for TypeIntern {
    #[inline]
    fn index_mut(&mut self, i: KindId) -> &mut Self::Output {
        &mut self.storage[i]
    }
}

///placeholder, explains why A derives B mut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MutReason;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutReasonPath {
    pub reason: MutReason,
    pub parent: Option<MutId>,
    pub depth: u32,
}

impl MutReasonPath {
    pub fn direct(reason: MutReason) -> Self {
        Self {
            reason,
            parent: None,
            depth: 0,
        }
    }

    fn implied(parent: MutId, parent_reason: &MutReasonPath) -> Self {
        Self {
            reason: parent_reason.reason,
            parent: Some(parent),
            depth: parent_reason.depth.saturating_add(1),
        }
    }

    fn is_better_than(&self, other: &Self) -> bool {
        self.depth < other.depth
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutConflict {
    pub mut_reason: MutReasonPath,
    pub const_reason: MutReasonPath,
}

///this struct handles mutability tracking for all code
///it heavily abuses the fact mutability is a true/false fact
///and that its all a mut => b mut connections
///thus we dont need an O(N^2) iteration process or SCC to solve mutability
#[derive(Debug)]
pub struct MutInfo {
    nodes: IndexVec<MutId, MutInner>,
}

#[derive(Debug)]
struct MutInner {
    parent: MutId,
    notify: BTreeSet<MutId>,
    reason: Option<MutReasonPath>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutGuessMode {
    UnknownAsConst,
    UnknownAsUnknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutGuess {
    Const,
    Mut,
    Unknown,
}

impl MutId {
    pub const FALSE: MutId = MutId(0);
    pub const TRUE: MutId = MutId(1);

    fn try_answer(&self) -> Option<bool> {
        match *self {
            MutId::FALSE => Some(false),
            MutId::TRUE => Some(true),
            _ => None,
        }
    }
}

impl MutInfo {
    pub fn new() -> Self {
        let mut nodes = IndexVec::new();
        nodes.push(MutInner {
            parent: MutId::FALSE,
            notify: BTreeSet::new(),
            reason: Some(MutReasonPath::direct(MutReason)),
        });
        nodes.push(MutInner {
            parent: MutId::TRUE,
            notify: BTreeSet::new(),
            reason: Some(MutReasonPath::direct(MutReason)),
        });

        Self { nodes }
    }

    pub fn add_unknown(&mut self) -> MutId {
        let id = self.nodes.push(MutInner {
            parent: MutId::new(self.nodes.len()),
            notify: BTreeSet::new(),
            reason: None,
        });
        debug_assert_eq!(id, self.nodes[id].parent);
        id
    }

    pub fn must_mut(&mut self, idx: MutId) -> bool {
        self.get_repr(idx).try_answer().unwrap_or(false)
    }

    ///this method sometimes returns None instead of Some(false)
    ///this happens for longer cycles we dont want to solve for in O(N^2)
    ///if all requirments are inserted than None
    ///can just be viewed as a Some(false) to default to imutable
    ///this is anyway the behivior we want
    pub fn get_guess(&mut self, idx: MutId) -> Option<bool> {
        self.get_repr(idx).try_answer()
    }

    pub fn guess(&self, idx: MutId, mode: MutGuessMode) -> MutGuess {
        match self.find_repr(idx).try_answer() {
            Some(false) => MutGuess::Const,
            Some(true) => MutGuess::Mut,
            None => match mode {
                MutGuessMode::UnknownAsConst => MutGuess::Const,
                MutGuessMode::UnknownAsUnknown => MutGuess::Unknown,
            },
        }
    }

    fn find_repr(&self, idx: MutId) -> MutId {
        let mut cur = idx;
        loop {
            let parent = self.nodes[cur].parent;
            if parent == cur {
                return cur;
            }
            cur = parent;
        }
    }

    pub fn get_repr(&mut self, idx: MutId) -> MutId {
        let p = self.nodes[idx].parent;
        if p == idx {
            return idx;
        }
        let root = self.get_repr(p);
        self.merge_into(idx, root);

        root
    }

    fn merge_into(&mut self, idx: MutId, root: MutId) {
        //make sure we never overide TRUE and FALSE
        if idx.try_answer().is_some() {
            if root.try_answer().is_none() {
                self.nodes[root].parent = idx;
            }

            return;
        }

        //update the union find
        self.nodes[idx].parent = root;

        if let Some(reason) = self.nodes[idx].reason.take() {
            self.record_reason(root, reason);
        }

        //clear the exostomg notify set
        let mut my_map = std::mem::take(&mut self.nodes[idx].notify);
        if root.try_answer().is_none() {
            self.nodes[root].notify.append(&mut my_map);
        };
    }

    fn record_reason(&mut self, idx: MutId, reason: MutReasonPath) -> bool {
        let slot = &mut self.nodes[idx].reason;
        let replace = slot
            .as_ref()
            .is_none_or(|old_reason| reason.is_better_than(old_reason));
        if replace {
            *slot = Some(reason);
        }
        replace
    }

    fn reason_for(&self, idx: MutId) -> Option<MutReasonPath> {
        self.nodes[idx]
            .reason
            .clone()
            .or_else(|| self.nodes[self.find_repr(idx)].reason.clone())
    }

    pub fn try_unify(&mut self, src: MutId, dst: MutId) -> Option<(bool, bool)> {
        let src = self.get_repr(src);
        let dst = self.get_repr(dst);

        let ans = (|| {
            let s = src.try_answer()?;
            let d = dst.try_answer()?;

            if s == d {
                return None;
            }

            Some((s, d))
        })();

        if ans.is_none() {
            self.merge_into(src, dst);
        };

        ans
    }

    pub fn add_edge(&mut self, src: MutId, dst: MutId) -> Option<MutConflict> {
        let original_dst = dst;
        let src = self.get_repr(src);
        let dst = self.get_repr(dst);

        if matches!(src.try_answer(), Some(false)) || matches!(dst.try_answer(), Some(true)) {
            return None;
        }

        if matches!(src.try_answer(), Some(true)) {
            let source_reason = self
                .reason_for(src)
                .expect("known mutable node should have a mutability reason");
            let implied_reason = MutReasonPath::implied(src, &source_reason);
            return match self.set_true(original_dst, implied_reason) {
                MutSetRes::Conflict(conflict) => Some(conflict),
                MutSetRes::Ok(_) => None,
            };
        }

        if !self.nodes[src].notify.insert(dst) {
            return None;
        }

        if !self.nodes[dst].notify.contains(&src) {
            return None;
        }

        self.nodes[dst].notify.remove(&src);
        self.nodes[src].notify.remove(&dst);

        if let Some((src_answer, _dst_answer)) = self.try_unify(src, dst) {
            let (mut_idx, const_idx) = if src_answer { (src, dst) } else { (dst, src) };
            return Some(MutConflict {
                mut_reason: self
                    .reason_for(mut_idx)
                    .expect("known mutable node should have a mutability reason"),
                const_reason: self
                    .reason_for(const_idx)
                    .expect("known const node should have a mutability reason"),
            });
        }

        None
    }

    pub fn set_true(&mut self, idx: MutId, reason: MutReasonPath) -> MutSetRes {
        let original = idx;
        let idx = self.get_repr(idx);
        match idx.try_answer() {
            Some(false) => {
                return MutSetRes::Conflict(MutConflict {
                    mut_reason: reason,
                    const_reason: self
                        .reason_for(original)
                        .expect("known const node should have a mutability reason"),
                });
            }
            Some(true) => {
                self.record_reason(original, reason);
                return MutSetRes::Ok(false);
            }
            _ => {
                self.record_reason(idx, reason.clone());
                self.nodes[idx].parent = MutId::TRUE;
            }
        };

        let my_map = std::mem::take(&mut self.nodes[idx].notify);
        for dst in my_map {
            let implied_reason = MutReasonPath::implied(idx, &reason);
            if let MutSetRes::Conflict(conflict) = self.set_true(dst, implied_reason) {
                return MutSetRes::Conflict(conflict);
            }
        }

        MutSetRes::Ok(true)
    }

    pub fn set_false(&mut self, idx: MutId, reason: MutReasonPath) -> Result<bool, MutConflict> {
        let original = idx;
        let idx = self.get_repr(idx);
        match idx.try_answer() {
            Some(false) => {
                self.record_reason(original, reason);
                Ok(false)
            }
            Some(true) => Err(MutConflict {
                mut_reason: self
                    .reason_for(original)
                    .expect("known mutable node should have a mutability reason"),
                const_reason: reason,
            }),
            _ => {
                let _old_map = std::mem::take(&mut self.nodes[idx].notify);
                self.record_reason(idx, reason);
                self.nodes[idx].parent = MutId::FALSE;
                Ok(true)
            }
        }
    }
}

impl Default for MutInfo {
    fn default() -> Self {
        Self::new()
    }
}

#[must_use]
pub enum MutSetRes {
    Conflict(MutConflict),
    Ok(bool),
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutliveReason;

#[derive(Debug)]
pub struct KindLookUp {
    kinds: UnionFind<KindId>,
    ptr: UnionFind<PtrId>,
    pub mutable: MutInfo,
    origin: UnionFind<OriginId>,

    life_order: BasicOrder<LifeId,OutliveReason>,
}

impl KindLookUp {
    pub fn new() -> Self {
        Self {
            kinds: UnionFind::new(),
            ptr: UnionFind::new(),
            origin: UnionFind::new(),
            mutable: MutInfo::new(),
            life_order: BasicOrder::new(),
        }
    }
}

impl KindLookUp {
    pub fn kind_root(&mut self, id: KindId) -> KindId {
        self.kinds.find_root(id)
    }

}

impl Default for KindLookUp {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct KindStorage {
    pub types: TypeIntern,
    pub structs: IndexVec<StructId, StructInfo>,
    ptr: IndexVec<PtrId, Option<PointerStyle>>,
    origin: OriginVec,
    pub life: IndexVec<LifeId, Option<LifeKind>>,
    kind_arg_spans: HashMap<Vec<KindId>, KindSpan>,
    life_arg_spans: HashMap<Vec<LifeId>, LifeSpan>,
    // pub func_style:IndexVec<FKId,Option<>>,
}

#[derive(Debug, Clone)]
pub struct StructInfo {
    pub name: Option<NameId>,
}

impl KindStorage {
    pub fn new() -> Self {
        Self {
            types: TypeIntern::new(),
            structs: IndexVec::new(),
            ptr: IndexVec::new(),
            origin: OriginVec::new(),
            life: IndexVec::new(),
            kind_arg_spans: HashMap::new(),
            life_arg_spans: HashMap::new(),
        }
    }

    pub fn new_struct(&mut self, info: StructInfo) -> StructId {
        self.structs.push(info)
    }

    pub fn struct_info(&self, id: StructId) -> Option<&StructInfo> {
        self.structs.get(id)
    }

    pub fn get<'a>(&'a self, look: &mut KindLookUp, id: KindId) -> Option<&'a TypeKind> {
        let root = look.kind_root(id);
        self.__get_root(root)
    }

    /// arg `id` is required to be its own root.
    fn __get_root(&self, id: KindId) -> Option<&TypeKind> {
        self.types[id].as_ref()
    }

    pub fn get_origin<'a>(&'a self, look: &mut KindLookUp, id: OriginId) -> Option<&'a Origin> {
        let root = look.origin.find_root(id);
        self.__get_origin(root)
    }

    /// arg `id` is required to be its own root.
    fn __get_origin(&self, id: OriginId) -> Option<&Origin> {
        self.origin[id].as_ref()
    }
}

impl Default for KindStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeUniverse {
    pub fn new() -> Self {
        let mut ans = Self {
            look: KindLookUp::new(),
            storage: KindStorage::new(),
        };
        ans.insert_hard_coded_builtin_kinds();
        ans
    }

    fn insert_hard_coded_builtin_kinds(&mut self) {
        for (idx, builtin) in HARD_CODED_BUILTIN_KINDS.iter().copied().enumerate() {
            let id = self.intern_builtin(builtin);
            debug_assert_eq!(id, KindId::new(idx));
        }
    }

    pub fn intern_builtin(&mut self, builtin: BuiltinKind) -> KindId {
        self.intern(TypeKind::Builtin(builtin))
    }

    pub fn intern(&mut self, ty: TypeKind) -> KindId {
        // Interning is paired with monotone union-find refinement. Partial shapes
        // are safe to share only while later solver steps make facts more precise.
        self.storage.types.intern(&mut self.look.kinds, ty)
    }

    pub fn add_empty(&mut self) -> KindId {
        self.storage.types.add_empty(&mut self.look.kinds)
    }

    pub fn add_life(&mut self, life: Option<LifeKind>) -> LifeId {
        let id = self.storage.life.push(life);
        let order_id = self.look.life_order.add_node();
        debug_assert_eq!(id, order_id);
        id
    }

    pub fn add_ptr_style(&mut self, style: Option<PointerStyle>) -> PtrId {
        let id = self.storage.ptr.push(style);
        let root_id = self.look.ptr.push_singleton();
        debug_assert_eq!(id, root_id);
        id
    }

    pub fn add_origin(&mut self, origin: Option<Origin>) -> OriginId {
        let id = self.storage.origin.push(origin);
        let root_id = self.look.origin.push_singleton();
        debug_assert_eq!(id, root_id);
        id
    }

    fn add_kind_alias(&mut self, target: KindId) {
        let alias = self.add_empty();
        self.look.kinds[alias] = target;
    }

    fn add_life_alias(&mut self, target: LifeId) {
        let alias = self.add_life(None);
        self.merge_life(alias, target)
            .expect("fresh lifetime alias should merge with target");
    }

    pub fn intern_kind_span(&mut self, items: impl IntoIterator<Item = KindId>) -> KindSpan {
        let mut items = items.into_iter();
        let Some(start) = items.next() else {
            return KindSpan::new(KindId::new(0), 0);
        };

        let mut len = 1;
        let mut cur = start;
        loop {
            let Some(next) = items.next() else {
                return KindSpan::new(start, len);
            };

            if next == cur.plus(1) {
                cur = next;
                len += 1;
                continue;
            }

            let (remaining, _) = items.size_hint();
            let mut row = Vec::with_capacity(len + 1 + remaining);
            row.extend((0..len).map(|offset| start.plus(offset)));
            row.push(next);
            row.extend(items);

            if let Some(span) = self.storage.kind_arg_spans.get(&row).copied() {
                return span;
            }

            let alias_start = self.storage.types.storage.next_index();
            for item in row.iter().copied() {
                self.add_kind_alias(item);
            }
            let span = KindSpan::new(
                alias_start,
                self.storage.types.storage.len() - alias_start.index(),
            );
            self.storage.kind_arg_spans.insert(row, span);
            return span;
        }
    }

    pub fn intern_life_span(&mut self, items: impl IntoIterator<Item = LifeId>) -> LifeSpan {
        let mut items = items.into_iter();
        let Some(start) = items.next() else {
            return LifeSpan::new(LifeId::new(0), 0);
        };

        let mut len = 1;
        let mut cur = start;
        loop {
            let Some(next) = items.next() else {
                return LifeSpan::new(start, len);
            };

            if next == cur.plus(1) {
                cur = next;
                len += 1;
                continue;
            }

            let (remaining, _) = items.size_hint();
            let mut row = Vec::with_capacity(len + 1 + remaining);
            row.extend((0..len).map(|offset| start.plus(offset)));
            row.push(next);
            row.extend(items);

            if let Some(span) = self.storage.life_arg_spans.get(&row).copied() {
                return span;
            }

            let alias_start = self.storage.life.next_index();
            for item in row.iter().copied() {
                self.add_life_alias(item);
            }
            let span = LifeSpan::new(alias_start, self.storage.life.len() - alias_start.index());
            self.storage.life_arg_spans.insert(row, span);
            return span;
        }
    }

    pub fn get(&mut self, id: KindId) -> Option<&TypeKind> {
        self.storage.get(&mut self.look, id)
    }

    pub fn unify(&mut self, found: KindId, wanted: KindId) -> Result<KindId, TypeClash> {
        let found_root = self.look.kinds.find_root(found);
        let wanted_root = self.look.kinds.find_root(wanted);

        if found_root == wanted_root {
            return Ok(wanted_root);
        }

        if self.__try_absorb_kind(wanted_root, found_root)? {
            let wanted_parent = self.look.kinds.find_root(wanted_root);
            if wanted_parent != found_root {
                self.look.kinds[found_root] = wanted_parent;
            }
            return Ok(wanted_parent);
        }

        if self
            .__try_absorb_kind(found_root, wanted_root)
            .map_err(TypeClash::swap)?
        {
            let found_parent = self.look.kinds.find_root(found_root);
            if found_parent != wanted_root {
                self.look.kinds[wanted_root] = found_parent;
            }
            return Ok(found_parent);
        }

        Err(self.clash(found, wanted))
    }

    /// args `dst` and `src` are required to be their own roots.
    fn __try_absorb_kind(&mut self, dst: KindId, src: KindId) -> Result<bool, TypeClash> {
        let dst_kind = self.storage.types[dst];
        let src_kind = self.storage.types[src];

        match (dst_kind, src_kind) {
            (_, None) => Ok(true),
            (None, Some(src_kind)) => {
                self.storage.types[dst] = Some(src_kind);
                Ok(true)
            }
            (Some(dst_kind), Some(src_kind)) => {
                self.unify_concrete_kinds(dst, dst_kind, src, src_kind)
            }
        }
    }

    fn unify_concrete_kinds(
        &mut self,
        dst: KindId,
        dst_kind: TypeKind,
        src: KindId,
        src_kind: TypeKind,
    ) -> Result<bool, TypeClash> {
        use TypeKind::*;

        match (dst_kind, src_kind) {
            (Builtin(dst_builtin), Builtin(src_builtin)) => {
                let Some(merged) = merge_builtin_kind(dst_builtin, src_builtin) else {
                    return Err(self.clash(src, dst));
                };
                self.refine_kind(dst, Builtin(merged));
                Ok(true)
            }
            (Generic(dst_gen), Generic(src_gen)) if dst_gen == src_gen => Ok(true),
            (Tuple(dst_items), Tuple(src_items)) => {
                if dst_items.len() != src_items.len() {
                    return Err(self.clash(src, dst));
                }
                let mut failed = false;
                for idx in 0..dst_items.len() {
                    let dst_item = self.kind_span_item(dst_items, idx);
                    let src_item = self.kind_span_item(src_items, idx);
                    if self.unify(src_item, dst_item).is_err() {
                        failed = true;
                    }
                }
                if failed {
                    return Err(self.clash(src, dst));
                }
                Ok(true)
            }
            (
                Struct {
                    id: dst_id,
                    gens: dst_gens,
                    lifes: dst_lifes,
                },
                Struct {
                    id: src_id,
                    gens: src_gens,
                    lifes: src_lifes,
                },
            ) => {
                if dst_id != src_id
                    || dst_gens.len() != src_gens.len()
                    || dst_lifes.len() != src_lifes.len()
                {
                    return Err(self.clash(src, dst));
                }

                let mut failed = false;
                for idx in 0..dst_gens.len() {
                    let dst_gen = self.kind_span_item(dst_gens, idx);
                    let src_gen = self.kind_span_item(src_gens, idx);
                    if self.unify(src_gen, dst_gen).is_err() {
                        failed = true;
                    }
                }

                for idx in 0..dst_lifes.len() {
                    let dst_life = self.life_span_item(dst_lifes, idx);
                    let src_life = self.life_span_item(src_lifes, idx);
                    if self.merge_life(dst_life, src_life).is_none() {
                        failed = true;
                    }
                }

                if failed {
                    return Err(self.clash(src, dst));
                }
                Ok(true)
            }
            (
                Func {
                    params: dst_params,
                    ret: dst_ret,
                },
                Func {
                    params: src_params,
                    ret: src_ret,
                },
            ) => {
                if dst_params.len() != src_params.len() {
                    return Err(self.clash(src, dst));
                }
                let mut failed = false;
                for idx in 0..dst_params.len() {
                    let dst_param = self.kind_span_item(dst_params, idx);
                    let src_param = self.kind_span_item(src_params, idx);
                    if self.unify(src_param, dst_param).is_err() {
                        failed = true;
                    }
                }
                if self.unify(src_ret, dst_ret).is_err() {
                    failed = true;
                }
                if failed {
                    return Err(self.clash(src, dst));
                }
                Ok(true)
            }
            (
                Array {
                    inner: dst_inner,
                    size: dst_size,
                },
                Array {
                    inner: src_inner,
                    size: src_size,
                },
            ) => {
                let size = merge_array_size(dst_size, src_size);
                let mut failed = size.is_none();
                if self.unify(src_inner, dst_inner).is_err() {
                    failed = true;
                }
                if let Some(size) = size {
                    self.refine_kind(
                        dst,
                        Array {
                            inner: dst_inner,
                            size,
                        },
                    );
                }
                if failed {
                    return Err(self.clash(src, dst));
                }
                Ok(true)
            }
            (
                Ptr {
                    tgt: dst_tgt,
                    style: dst_style,
                    mutable: dst_mut,
                },
                Ptr {
                    tgt: src_tgt,
                    style: src_style,
                    mutable: src_mut,
                },
            ) => {
                let style = self.merge_ptr_style(dst_style, src_style);
                let mut failed = style.is_none();
                failed |= self.look.mutable.try_unify(dst_mut, src_mut).is_some();
                if self.unify(src_tgt, dst_tgt).is_err() {
                    failed = true;
                }
                if let Some(style) = style {
                    self.storage.types[dst] = Some(Ptr {
                        tgt: dst_tgt,
                        style,
                        mutable: dst_mut,
                    });
                }
                if failed {
                    return Err(self.clash(src, dst));
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn merge_ptr_style(&mut self, dst: PtrId, src: PtrId) -> Option<PtrId> {
        let dst = self.look.ptr.find_root(dst);
        let src = self.look.ptr.find_root(src);

        if dst == src {
            return Some(dst);
        }

        let dst_style = self.storage.ptr[dst];
        let src_style = self.storage.ptr[src];

        let merged = match (dst_style, src_style) {
            (_, None) => dst_style,
            (None, Some(style)) => Some(style),
            (Some(PointerStyle::Raw(dst_null)), Some(PointerStyle::Raw(src_null))) => {
                Some(PointerStyle::Raw(merge_optional_equal(dst_null, src_null)?))
            }
            (Some(PointerStyle::Ref(dst_life)), Some(PointerStyle::Ref(src_life))) => {
                Some(PointerStyle::Ref(self.merge_life(dst_life, src_life)?))
            }
            (Some(dst_style), Some(src_style)) if dst_style == src_style => Some(dst_style),
            _ => return None,
        };

        self.storage.ptr[dst] = merged;
        self.storage.ptr[src] = merged;
        self.look.ptr[src] = dst;
        Some(dst)
    }

    fn refine_kind(&mut self, dst: KindId, kind: TypeKind) -> KindId {
        // Refinement links `dst` to an interned, more precise representative
        // instead of rewriting the hash-consed `TypeKind` key in place.
        let refined = self.intern(kind);
        if refined != dst {
            self.look.kinds[dst] = refined;
        }
        refined
    }

    fn merge_life(&mut self, dst: LifeId, src: LifeId) -> Option<LifeId> {

        if dst == src {
            return Some(dst);
        }

        //record the actual path so later finding things isnt confusing
        self.look.life_order.unify(dst, src,OutliveReason);
        Some(dst)
    }

    fn kind_span_item(&self, span: KindSpan, index: usize) -> KindId {
        span.at(index)
    }

    fn life_span_item(&self, span: LifeSpan, index: usize) -> LifeId {
        span.at(index)
    }

    fn clash(&mut self, found: KindId, wanted: KindId) -> TypeClash {
        TypeClash {
            found: Some(self.kind_to_clash_string(found)),
            wanted: Some(self.kind_to_clash_string(wanted)),
        }
    }

    fn kind_to_clash_string(&mut self, id: KindId) -> String {
        let mut out = String::new();
        self.write_kind_for_clash(id, 0, &mut out);
        out
    }

    fn write_kind_for_clash(&mut self, id: KindId, depth: usize, out: &mut String) {
        if depth > 32 {
            out.push_str("...");
            return;
        }

        match self.get(id).copied() {
            None => out.push('_'),
            Some(TypeKind::Builtin(builtin)) => out.push_str(builtin.name()),
            Some(TypeKind::Generic(gen_id)) => out.push_str(&format!("T{}", gen_id.0)),
            Some(TypeKind::Tuple(items)) => {
                out.push('(');
                for idx in 0..items.len() {
                    if idx > 0 {
                        out.push_str(", ");
                    }
                    self.write_kind_for_clash(self.kind_span_item(items, idx), depth + 1, out);
                }
                out.push(')');
            }
            Some(TypeKind::Struct { id, gens, lifes }) => {
                out.push_str(&format!("Struct{}", id.0));
                let has_lifes = !lifes.is_empty();
                let has_gens = !gens.is_empty();
                if has_lifes || has_gens {
                    out.push('[');
                    let mut needs_sep = false;
                    // // for now we dont want to show life for clashes
                    // for idx in 0..lifes.len() {
                    //     if needs_sep {
                    //         out.push_str(", ");
                    //     }
                    //     self.write_life_for_clash(self.life_span_item(lifes, idx), out);
                    //     needs_sep = true;
                    // }
                    for idx in 0..gens.len() {
                        if needs_sep {
                            out.push_str(", ");
                        }
                        self.write_kind_for_clash(self.kind_span_item(gens, idx), depth + 1, out);
                        needs_sep = true;
                    }
                    out.push(']');
                }
            }
            Some(TypeKind::Func { params, ret }) => {
                out.push_str("fn(");
                for idx in 0..params.len() {
                    if idx > 0 {
                        out.push_str(", ");
                    }
                    self.write_kind_for_clash(self.kind_span_item(params, idx), depth + 1, out);
                }
                out.push_str(") -> ");
                self.write_kind_for_clash(ret, depth + 1, out);
            }
            Some(TypeKind::Array { inner, size }) => {
                out.push('[');
                self.write_kind_for_clash(inner, depth + 1, out);
                match size {
                    Some(ArraySize::Sized(size)) => out.push_str(&format!("; {size}")),
                    Some(ArraySize::Unsized) => out.push_str("; _"),
                    None => {}
                }
                out.push(']');
            }
            Some(TypeKind::Ptr {
                tgt,
                style,
                mutable,
            }) => {
                let style = self.look.ptr.find_root(style);
                let style = self.storage.ptr.get(style).copied().flatten();
                let mutable = self.look.mutable.find_repr(mutable).try_answer();
                match style {
                    Some(PointerStyle::Raw(Some(Nullable::No))) => {
                        out.push_str("&'raw ");
                        if mutable == Some(false) {
                            out.push_str("const ");
                        } else if mutable.is_none() {
                            out.push_str("?mut ");
                        }
                    }
                    Some(PointerStyle::Ref(_life)) => {
                        out.push('&');
                        //we dont show lifetimes for type clashes.
                        if mutable == Some(true) {
                            out.push_str("mut ");
                        } else if mutable.is_none() {
                            out.push_str("?mut ");
                        }
                    }
                    Some(PointerStyle::Raw(Some(Nullable::Yes)) | PointerStyle::Raw(None))
                    | None => {
                        out.push('*');
                        if mutable == Some(false) {
                            out.push_str("const ");
                        } else if mutable.is_none() {
                            out.push_str("?mut ");
                        }
                    }
                }
                self.write_kind_for_clash(tgt, depth + 1, out);
            }
        }
    }

    pub fn kind_to_string(&mut self, program: &Program, id: KindId) -> String {
        self.storage.kind_to_string(&mut self.look, program, id)
    }

    pub fn kind_to_string_with_mut_guess(
        &mut self,
        program: &Program,
        id: KindId,
        mut_guess_mode: MutGuessMode,
    ) -> String {
        self.storage
            .kind_to_string_with_mut_guess(&mut self.look, program, id, mut_guess_mode)
    }

    pub fn deref_chain_to_string(
        &mut self,
        program: &Program,
        chain: &[(KindId, Projection)],
    ) -> String {
        self.storage
            .deref_chain_to_string(&mut self.look, program, chain)
    }
}

impl KindStorage {
    pub fn kind_to_string(&self, look: &mut KindLookUp, program: &Program, id: KindId) -> String {
        self.kind_to_string_with_mut_guess(look, program, id, MutGuessMode::UnknownAsConst)
    }

    pub fn kind_to_string_with_mut_guess(
        &self,
        look: &mut KindLookUp,
        program: &Program,
        id: KindId,
        mut_guess_mode: MutGuessMode,
    ) -> String {
        let mut out = String::new();
        self.write_kind(look, program, id, mut_guess_mode, &mut out);
        out
    }

    pub fn deref_chain_to_string(
        &self,
        look: &mut KindLookUp,
        program: &Program,
        chain: &[(KindId, Projection)],
    ) -> String {
        chain
            .iter()
            .map(|(id, projection)| {
                format!(
                    "{} [{}]",
                    self.kind_to_string(look, program, *id),
                    projection_name(*projection)
                )
            })
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    fn write_kind(
        &self,
        look: &mut KindLookUp,
        program: &Program,
        id: KindId,
        mut_guess_mode: MutGuessMode,
        out: &mut String,
    ) {
        match self.get(look, id) {
            None => out.push('_'),
            Some(TypeKind::Builtin(builtin)) => out.push_str(builtin.name()),
            Some(TypeKind::Generic(gen_id)) => out.push_str(&format!("T{}", gen_id.0)),
            Some(TypeKind::Tuple(items)) => {
                out.push('(');
                for (idx, item) in items.ids().enumerate() {
                    if idx > 0 {
                        out.push_str(", ");
                    }
                    self.write_kind(look, program, item, mut_guess_mode, out);
                }
                out.push(')');
            }
            Some(TypeKind::Struct { id, gens, lifes }) => {
                match self.struct_info(*id).and_then(|info| info.name) {
                    Some(name) => out.push_str(program.name_string(name)),
                    None => out.push_str("UnnamedStruct"),
                }

                let has_lifes = !lifes.is_empty();
                let has_gens = !gens.is_empty();
                if has_lifes || has_gens {
                    out.push('[');
                    let mut needs_sep = false;
                    for life in lifes.ids() {
                        if needs_sep {
                            out.push_str(", ");
                        }
                        self.write_life(look, program, life, out);
                        needs_sep = true;
                    }
                    for item in gens.ids() {
                        if needs_sep {
                            out.push_str(", ");
                        }
                        self.write_kind(look, program, item, mut_guess_mode, out);
                        needs_sep = true;
                    }
                    out.push(']');
                }
            }
            Some(TypeKind::Func { params, ret }) => {
                out.push_str("fn(");
                for (idx, param) in params.ids().enumerate() {
                    if idx > 0 {
                        out.push_str(", ");
                    }
                    self.write_kind(look, program, param, mut_guess_mode, out);
                }
                out.push_str(") -> ");
                self.write_kind(look, program, *ret, mut_guess_mode, out);
            }
            Some(TypeKind::Array { inner, size }) => {
                out.push('[');
                self.write_kind(look, program, *inner, mut_guess_mode, out);
                match size {
                    Some(ArraySize::Sized(size)) => out.push_str(&format!("; {size}")),
                    Some(ArraySize::Unsized) => out.push_str("; _"),
                    None => {}
                }
                out.push(']');
            }
            Some(TypeKind::Ptr {
                tgt,
                style,
                mutable,
            }) => {
                let style = look.ptr.find_root(*style);
                let style = self.ptr.get(style).copied().flatten();
                let mutable = match look.mutable.find_repr(*mutable).try_answer() {
                    Some(false) => MutGuess::Const,
                    Some(true) => MutGuess::Mut,
                    None => match mut_guess_mode {
                        MutGuessMode::UnknownAsUnknown => MutGuess::Unknown,
                        MutGuessMode::UnknownAsConst => match style {
                            Some(PointerStyle::Ref(_)) => MutGuess::Const,
                            Some(PointerStyle::Raw(_)) | None => MutGuess::Mut,
                        },
                    },
                };
                let write_mutability = |out: &mut String| match mutable {
                    MutGuess::Const => out.push_str("const "),
                    MutGuess::Mut => {}
                    MutGuess::Unknown => out.push_str("?mut "),
                };

                match style {
                    Some(PointerStyle::Raw(Some(Nullable::No))) => {
                        out.push_str("&'raw ");
                        write_mutability(out);
                    }
                    Some(PointerStyle::Ref(life)) => {
                        out.push('&');
                        if false {
                            self.write_life(look, program, life, out);
                        }
                        out.push(' ');
                        match mutable {
                            MutGuess::Const => {}
                            MutGuess::Mut => out.push_str("mut "),
                            MutGuess::Unknown => out.push_str("?mut "),
                        }
                    }
                    Some(PointerStyle::Raw(Some(Nullable::Yes)) | PointerStyle::Raw(None))
                    | None => {
                        out.push('*');
                        write_mutability(out);
                    }
                }

                self.write_kind(look, program, *tgt, mut_guess_mode, out);
            }
        }
    }

    fn write_life(&self, look: &mut KindLookUp, _program: &Program, id: LifeId, out: &mut String) {
        //TODO use the solved lifetime SCC+SparseMatrix to get a good aproximation
        //of what exactly is the original lifetime
        //this should probably not be used most the time... but its a nice feature
        todo!()

        // let id = look.life_root(id);
        // match self.__life_root(id) {
        //     Some(LifeKind::Static) => out.push_str("'static"),
        //     Some(LifeKind::Univeral(Some(id))) => out.push_str(&format!("'a{id}")),
        //     Some(LifeKind::Univeral(None)) => out.push_str("'a"),
        //     Some(LifeKind::Local) => out.push_str("'local"),
        //     None => out.push_str("'_"),
        // }
    }

    // /// arg `id` is required to be its own root.
    // fn __life_root(&self, id: LifeId) -> Option<LifeKind> {
    //     self.life.get(id).copied().flatten()
    // }
}

impl Default for TypeUniverse {
    fn default() -> Self {
        Self::new()
    }
}

fn merge_builtin_kind(dst: BuiltinKind, src: BuiltinKind) -> Option<BuiltinKind> {
    match (dst, src) {
        (BuiltinKind::Int(dst), BuiltinKind::Int(src)) => {
            Some(BuiltinKind::Int(merge_int_kind(dst, src)?))
        }
        (BuiltinKind::Float(dst), BuiltinKind::Float(src)) => {
            Some(BuiltinKind::Float(merge_float_kind(dst, src)?))
        }
        (dst, src) if dst == src => Some(dst),
        _ => None,
    }
}

fn merge_int_kind(dst: IntKind, src: IntKind) -> Option<IntKind> {
    Some(IntKind {
        size: merge_optional_equal(dst.size, src.size)?,
        sign: merge_optional_equal(dst.sign, src.sign)?,
    })
}

fn merge_float_kind(dst: FloatKind, src: FloatKind) -> Option<FloatKind> {
    Some(FloatKind {
        size: merge_optional_equal(dst.size, src.size)?,
    })
}

fn merge_array_size(dst: Option<ArraySize>, src: Option<ArraySize>) -> Option<Option<ArraySize>> {
    merge_optional_equal(dst, src)
}

fn merge_optional_equal<T: Copy + Eq>(dst: Option<T>, src: Option<T>) -> Option<Option<T>> {
    match (dst, src) {
        (Some(dst), Some(src)) if dst != src => None,
        (Some(dst), _) => Some(Some(dst)),
        (_, Some(src)) => Some(Some(src)),
        (None, None) => Some(None),
    }
}

#[cfg(test)]
mod mutability_tests {
    use super::*;

    fn reason() -> MutReasonPath {
        MutReasonPath::direct(MutReason)
    }

    #[test]
    fn mut_conflict_reports_mut_and_const_reasons() {
        let mut muts = MutInfo::new();
        let src = muts.add_unknown();
        let dst = muts.add_unknown();
        muts.add_edge(src, dst);

        muts.set_false(dst, reason()).unwrap();
        let conflict = match muts.set_true(src, reason()) {
            MutSetRes::Conflict(conflict) => conflict,
            MutSetRes::Ok(_) => panic!("expected mutability conflict"),
        };

        assert_eq!(conflict.const_reason, reason());
        assert_eq!(
            conflict.mut_reason,
            MutReasonPath {
                reason: MutReason,
                parent: Some(src),
                depth: 1,
            }
        );
    }

    #[test]
    fn edge_from_known_mut_propagates_immediately() {
        let mut muts = MutInfo::new();
        let dst = muts.add_unknown();

        muts.set_false(dst, reason()).unwrap();
        let conflict = muts
            .add_edge(MutId::TRUE, dst)
            .expect("expected mutability conflict");

        assert_eq!(conflict.const_reason, reason());
        assert_eq!(
            conflict.mut_reason,
            MutReasonPath {
                reason: MutReason,
                parent: Some(MutId::TRUE),
                depth: 1,
            }
        );
    }

    #[test]
    fn kind_string_can_show_unknown_mutability_or_default_const() {
        let mut types = TypeUniverse::new();
        let ptr_style = types.add_ptr_style(Some(PointerStyle::Raw(None)));
        let mutable = types.look.mutable.add_unknown();
        let ptr_ty = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: ptr_style,
            mutable,
        });
        let program = Program::new();

        assert_eq!(types.kind_to_string(&program, ptr_ty), "*bool");
        assert_eq!(
            types.kind_to_string_with_mut_guess(&program, ptr_ty, MutGuessMode::UnknownAsUnknown),
            "*?mut bool"
        );
    }

    #[test]
    fn kind_string_formats_pointer_styles_with_default_mutability() {
        let mut types = TypeUniverse::new();
        let raw_nullable = types.add_ptr_style(Some(PointerStyle::Raw(Some(Nullable::Yes))));
        let raw_non_null = types.add_ptr_style(Some(PointerStyle::Raw(Some(Nullable::No))));
        let life = types.add_life(Some(LifeKind::Static));
        let safe_ref = types.add_ptr_style(Some(PointerStyle::Ref(life)));
        let program = Program::new();

        let mut_raw_ptr = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: raw_nullable,
            mutable: MutId::TRUE,
        });
        let const_raw_ptr = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: raw_nullable,
            mutable: MutId::FALSE,
        });
        let mut_raw_ref = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: raw_non_null,
            mutable: MutId::TRUE,
        });
        let const_raw_ref = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: raw_non_null,
            mutable: MutId::FALSE,
        });
        let const_safe_ref = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: safe_ref,
            mutable: MutId::FALSE,
        });
        let mut_safe_ref = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: safe_ref,
            mutable: MutId::TRUE,
        });

        assert_eq!(types.kind_to_string(&program, mut_raw_ptr), "*bool");
        assert_eq!(types.kind_to_string(&program, const_raw_ptr), "*const bool");
        assert_eq!(types.kind_to_string(&program, mut_raw_ref), "&'raw bool");
        assert_eq!(
            types.kind_to_string(&program, const_raw_ref),
            "&'raw const bool"
        );
        assert_eq!(
            types.kind_to_string(&program, const_safe_ref),
            "&'static bool"
        );
        assert_eq!(
            types.kind_to_string(&program, mut_safe_ref),
            "&'static mut bool"
        );
    }

    #[test]
    fn kind_string_formats_struct_names() {
        let mut program = Program::new();
        let widget_name = program.str_intern.intern("Widget");
        let widget_name = program.insert_value_in_global_scope(widget_name);

        let mut types = TypeUniverse::new();
        let widget = types.storage.new_struct(StructInfo {
            name: Some(widget_name),
        });
        let anon = types.storage.new_struct(StructInfo { name: None });

        let empty_gens = types.intern_kind_span([]);
        let empty_lifes = types.intern_life_span([]);
        let widget_ty = types.intern(TypeKind::Struct {
            id: widget,
            gens: empty_gens,
            lifes: empty_lifes,
        });
        let anon_ty = types.intern(TypeKind::Struct {
            id: anon,
            gens: empty_gens,
            lifes: empty_lifes,
        });

        assert_eq!(types.kind_to_string(&program, widget_ty), "Widget");
        assert_eq!(types.kind_to_string(&program, anon_ty), "UnnamedStruct");
    }

    #[test]
    fn kind_string_formats_struct_parameters_after_name() {
        let mut program = Program::new();
        let box_name = program.str_intern.intern("Box");
        let box_name = program.insert_value_in_global_scope(box_name);

        let mut types = TypeUniverse::new();
        let sid = types.storage.new_struct(StructInfo {
            name: Some(box_name),
        });
        let life = types.add_life(Some(LifeKind::Static));
        let gens = types.intern_kind_span([KindId::BOOL]);
        let lifes = types.intern_life_span([life]);
        let boxed_bool = types.intern(TypeKind::Struct {
            id: sid,
            gens,
            lifes,
        });

        assert_eq!(
            types.kind_to_string(&program, boxed_bool),
            "Box['static, bool]"
        );
    }

    #[test]
    fn hard_coded_builtin_kind_ids_match_storage() {
        let mut types = TypeUniverse::new();

        for (idx, builtin) in HARD_CODED_BUILTIN_KINDS.iter().copied().enumerate() {
            let id = KindId::new(idx);
            assert_eq!(types.get(id), Some(&TypeKind::Builtin(builtin)));
            assert_eq!(types.intern_builtin(builtin), id);
        }

        assert_eq!(
            types.get(KindId::BOOL),
            Some(&TypeKind::Builtin(BuiltinKind::Bool))
        );
        assert_eq!(
            types.get(KindId::STR),
            Some(&TypeKind::Builtin(BuiltinKind::Str))
        );
        assert_eq!(
            types.get(KindId::VOID),
            Some(&TypeKind::Builtin(BuiltinKind::Void))
        );
    }

    #[test]
    fn kind_span_reuses_natural_rows_or_allocates_alias_rows() {
        let mut types = TypeUniverse::new();

        let natural = types.intern_kind_span([KindId::BOOL, KindId::STR]);
        assert_eq!(natural, KindSpan::new(KindId::BOOL, 2));

        let alias_span = types.intern_kind_span([KindId::BOOL, KindId::VOID]);
        let alias_span_again = types.intern_kind_span([KindId::BOOL, KindId::VOID]);
        assert_ne!(alias_span.start(), KindId::BOOL);
        assert_eq!(alias_span_again, alias_span);
        assert_eq!(alias_span.len(), 2);
        assert_eq!(types.look.kinds.find_root(alias_span.at(0)), KindId::BOOL);
        assert_eq!(types.look.kinds.find_root(alias_span.at(1)), KindId::VOID);
    }

    #[test]
    fn life_span_reuses_natural_rows_or_allocates_alias_rows() {
        let mut types = TypeUniverse::new();
        let first = types.add_life(Some(LifeKind::Static));
        let middle = types.add_life(None);
        let last = types.add_life(Some(LifeKind::Local));

        let natural = types.intern_life_span([first, middle]);
        assert_eq!(natural, LifeSpan::new(first, 2));

        let alias_span = types.intern_life_span([first, last]);
        let alias_span_again = types.intern_life_span([first, last]);
        assert_ne!(alias_span.start(), first);
        assert_eq!(alias_span_again, alias_span);
        assert_eq!(alias_span.len(), 2);
        assert_eq!(types.storage.life[alias_span.at(0)], Some(LifeKind::Static));
        assert_eq!(types.storage.life[alias_span.at(1)], Some(LifeKind::Local));
    }

    #[test]
    fn unify_unknown_absorbs_known_shape() {
        let mut types = TypeUniverse::new();
        let unknown = types.add_empty();

        let root = types.unify(unknown, KindId::BOOL).unwrap();

        assert_eq!(root, KindId::BOOL);
        assert_eq!(types.get(root), Some(&TypeKind::Builtin(BuiltinKind::Bool)));
        let program = Program::new();
        assert_eq!(types.kind_to_string(&program, unknown), "bool");
    }

    #[test]
    fn kind_string_follows_alias_span_roots() {
        let mut types = TypeUniverse::new();
        let program = Program::new();
        let items = types.intern_kind_span([KindId::BOOL, KindId::VOID]);
        let tuple = types.intern(TypeKind::Tuple(items));

        assert_eq!(types.kind_to_string(&program, tuple), "(bool, void)");
    }

    #[test]
    fn life_string_follows_alias_roots_after_refinement() {
        let mut types = TypeUniverse::new();
        let program = Program::new();
        let unknown_life = types.add_life(None);
        let _middle = types.add_life(Some(LifeKind::Local));
        let static_life = types.add_life(Some(LifeKind::Static));
        let lifes = types.intern_life_span([unknown_life, static_life]);
        let sid = types.storage.new_struct(StructInfo { name: None });
        let no_gens = types.intern_kind_span([]);
        let struct_ty = types.intern(TypeKind::Struct {
            id: sid,
            gens: no_gens,
            lifes,
        });
        let unknown_ref = types.add_ptr_style(Some(PointerStyle::Ref(unknown_life)));
        let static_ref = types.add_ptr_style(Some(PointerStyle::Ref(static_life)));
        let left = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: unknown_ref,
            mutable: MutId::FALSE,
        });
        let right = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: static_ref,
            mutable: MutId::FALSE,
        });

        types.unify(left, right).unwrap();

        assert_eq!(
            types.kind_to_string(&program, struct_ty),
            "UnnamedStruct['static, 'static]"
        );
    }

    #[test]
    fn unify_recurses_through_function_shape() {
        let mut types = TypeUniverse::new();
        let left_param = types.add_empty();
        let right_ret = types.add_empty();

        let left_params = types.intern_kind_span([left_param]);
        let right_params = types.intern_kind_span([KindId::BOOL]);

        let left = types.intern(TypeKind::Func {
            params: left_params,
            ret: KindId::STR,
        });
        let right = types.intern(TypeKind::Func {
            params: right_params,
            ret: right_ret,
        });

        types.unify(left, right).unwrap();
        let left_param = types.look.kinds.find_root(left_param);
        let right_ret = types.look.kinds.find_root(right_ret);

        assert_eq!(
            types.get(left_param),
            Some(&TypeKind::Builtin(BuiltinKind::Bool))
        );
        assert_eq!(
            types.get(right_ret),
            Some(&TypeKind::Builtin(BuiltinKind::Str))
        );
    }

    #[test]
    fn unify_function_continues_after_child_failure() {
        let mut types = TypeUniverse::new();
        let left_first_param = types.add_empty();
        let left_ret = types.add_empty();
        let left_params = types.intern_kind_span([left_first_param, KindId::BOOL]);
        let right_params = types.intern_kind_span([KindId::STR, KindId::VOID]);
        let left = types.intern(TypeKind::Func {
            params: left_params,
            ret: left_ret,
        });
        let right = types.intern(TypeKind::Func {
            params: right_params,
            ret: KindId::BOOL,
        });

        types.unify(left, right).unwrap_err();

        assert_eq!(
            types.get(left_first_param),
            Some(&TypeKind::Builtin(BuiltinKind::Str))
        );
        assert_eq!(
            types.get(left_ret),
            Some(&TypeKind::Builtin(BuiltinKind::Bool))
        );
    }

    #[test]
    fn unify_refines_partial_builtin_and_array_size() {
        let mut types = TypeUniverse::new();
        let int_like = types.intern(TypeKind::Builtin(BuiltinKind::Int(IntKind {
            size: None,
            sign: None,
        })));
        let int_root = types.unify(int_like, KindId::I32).unwrap();
        let int_root_kind = types.get(int_root).copied();
        let i32_kind = types.get(KindId::I32).copied();
        assert_eq!(int_root_kind, i32_kind);

        let unknown_size = types.intern(TypeKind::Array {
            inner: KindId::BOOL,
            size: None,
        });
        let sized = types.intern(TypeKind::Array {
            inner: KindId::BOOL,
            size: Some(ArraySize::Sized(3)),
        });
        let root = types.unify(sized, unknown_size).unwrap();

        assert_eq!(
            types.get(root),
            Some(&TypeKind::Array {
                inner: KindId::BOOL,
                size: Some(ArraySize::Sized(3)),
            })
        );
    }

    #[test]
    fn unify_pointer_merges_style_lifetime_and_mutability() {
        let mut types = TypeUniverse::new();
        let unknown_life = types.add_life(None);
        let static_life = types.add_life(Some(LifeKind::Static));
        let unknown_style = types.add_ptr_style(Some(PointerStyle::Ref(unknown_life)));
        let static_style = types.add_ptr_style(Some(PointerStyle::Ref(static_life)));
        let unknown_mut = types.look.mutable.add_unknown();

        let left = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: unknown_style,
            mutable: unknown_mut,
        });
        let right = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: static_style,
            mutable: MutId::TRUE,
        });

        let root = types.unify(left, right).unwrap();
        let TypeKind::Ptr { style, mutable, .. } = *types.get(root).unwrap() else {
            panic!("expected pointer root");
        };

        assert_eq!(
            types.storage.ptr[style],
            Some(PointerStyle::Ref(static_life))
        );
        assert_eq!(types.storage.life[static_life], Some(LifeKind::Static));
        assert!(types.look.mutable.must_mut(mutable));
    }

    #[test]
    fn unify_pointer_refines_raw_nullability() {
        let mut types = TypeUniverse::new();
        let unknown_raw = types.add_ptr_style(Some(PointerStyle::Raw(None)));
        let nullable_raw = types.add_ptr_style(Some(PointerStyle::Raw(Some(Nullable::Yes))));
        let non_null_raw = types.add_ptr_style(Some(PointerStyle::Raw(Some(Nullable::No))));

        let unknown_ty = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: unknown_raw,
            mutable: MutId::TRUE,
        });
        let nullable_ty = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: nullable_raw,
            mutable: MutId::TRUE,
        });
        let non_null_ty = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: non_null_raw,
            mutable: MutId::TRUE,
        });

        types.unify(unknown_ty, nullable_ty).unwrap();
        assert_eq!(
            types.storage.ptr[unknown_raw],
            Some(PointerStyle::Raw(Some(Nullable::Yes)))
        );
        assert_eq!(
            types.storage.ptr[nullable_raw],
            Some(PointerStyle::Raw(Some(Nullable::Yes)))
        );

        let err = types.unify(nullable_ty, non_null_ty).unwrap_err();
        assert_eq!(err.found(), Some("*bool"));
        assert_eq!(err.wanted(), Some("&'raw bool"));
    }

    #[test]
    fn unify_pointer_updates_shared_style_and_lifetime_ids() {
        let mut types = TypeUniverse::new();
        let unknown_life = types.add_life(None);
        let static_life = types.add_life(Some(LifeKind::Static));
        let shared_style = types.add_ptr_style(Some(PointerStyle::Ref(unknown_life)));
        let static_style = types.add_ptr_style(Some(PointerStyle::Ref(static_life)));
        let program = Program::new();

        let first = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: shared_style,
            mutable: MutId::FALSE,
        });
        let second = types.intern(TypeKind::Ptr {
            tgt: KindId::STR,
            style: shared_style,
            mutable: MutId::FALSE,
        });
        let static_bool = types.intern(TypeKind::Ptr {
            tgt: KindId::BOOL,
            style: static_style,
            mutable: MutId::FALSE,
        });

        types.unify(first, static_bool).unwrap();

        assert_eq!(types.storage.life[unknown_life], Some(LifeKind::Static));
        assert_eq!(types.storage.life[static_life], Some(LifeKind::Static));
        assert_eq!(
            types.storage.ptr[shared_style],
            Some(PointerStyle::Ref(static_life))
        );
        assert_eq!(
            types.storage.ptr[static_style],
            Some(PointerStyle::Ref(static_life))
        );
        assert_eq!(types.kind_to_string(&program, second), "&'static str");
    }

    #[test]
    fn unify_reports_shape_conflicts_without_linking_roots() {
        let mut types = TypeUniverse::new();

        let err = types.unify(KindId::BOOL, KindId::STR).unwrap_err();

        assert_eq!(err.found(), Some("bool"));
        assert_eq!(err.wanted(), Some("str"));
        assert_eq!(types.look.kinds.find_root(KindId::BOOL), KindId::BOOL);
        assert_eq!(types.look.kinds.find_root(KindId::STR), KindId::STR);
    }
}

fn projection_name(projection: Projection) -> &'static str {
    match projection {
        Projection::SimpleDeref => "deref",
        Projection::FieldReref(_) => "field_reref",
        Projection::RawReref => "raw_reref",
        Projection::ForgetSafe => "forget_safe",
        Projection::SmartCall => "smart_call",
        Projection::Casted => "casted",
    }
}

#[derive(Debug)]
pub struct SolvedTypes {
    pub typedef_types: IdHashMap<TExpId, KindId>,
    pub function_values: IdHashMap<ValId, SolvedFunctionTypes>,
    pub function_types: IdHashMap<NameId, ValId>,
    pub member_function_types: HashMap<(NameId, StrId), ValId>,
}

impl SolvedTypes {
    pub fn new(program: &Program) -> Self {
        let mut typedef_types = IdHashMap::default();
        typedef_types.reserve(program.definitions.len());

        Self {
            typedef_types,
            function_values: IdHashMap::default(),
            function_types: IdHashMap::default(),
            member_function_types: HashMap::new(),
        }
    }

    pub fn type_of(&self, value: ValId) -> Option<KindId> {
        self.function_values.get(&value).map(|f| f.ty)
    }

    pub fn pat_type(&self, id: PatId) -> Option<KindId> {
        self.function_values.values().find_map(|f| {
            f.arguments
                .iter()
                .find_map(|(p, _, t)| (*p == id).then_some(*t))
        })
    }

    pub fn function_types_by_name(&self, id: NameId) -> Option<&SolvedFunctionTypes> {
        self.function_types
            .get(&id)
            .and_then(|site| self.function_values.get(site))
    }

    pub fn inner_types_of_function(&self, function: ValId) -> Option<&InnerFunctionTypes> {
        self.function_values
            .get(&function)
            .and_then(|f| f.inner.as_ref())
    }

    pub fn implicit_deref_chain(&self, id: ValId) -> Option<&[(KindId, Projection)]> {
        self.function_values.values().find_map(|f| {
            f.inner
                .as_ref()
                .and_then(|inner| inner.implicit_derefs.get(&id).map(Vec::as_slice))
        })
    }

    pub fn implicit_deref_chain_in_function(
        &self,
        function: ValId,
        id: ValId,
    ) -> Option<&[(KindId, Projection)]> {
        self.inner_types_of_function(function)
            .and_then(|inner| inner.implicit_derefs.get(&id).map(Vec::as_slice))
    }

    pub fn implicit_deref_count(&self, id: ValId) -> Option<usize> {
        self.implicit_deref_chain(id).map(|chain| chain.len())
    }

    pub fn member_access_implicit_deref_chain(&self, id: ValId) -> Option<&[(KindId, Projection)]> {
        self.implicit_deref_chain(id)
    }

    pub fn member_access_implicit_deref_count(&self, id: ValId) -> Option<usize> {
        self.implicit_deref_count(id)
    }

    pub fn member_access_implicit_deref_count_in_function(
        &self,
        function: ValId,
        id: ValId,
    ) -> Option<usize> {
        self.implicit_deref_chain_in_function(function, id)
            .map(|chain| chain.len())
    }

    pub fn index_implicit_deref_chain(&self, id: ValId) -> Option<&[(KindId, Projection)]> {
        self.implicit_deref_chain(id)
    }

    pub fn index_implicit_deref_count(&self, id: ValId) -> Option<usize> {
        self.implicit_deref_count(id)
    }
}

#[derive(Debug)]
pub struct SolvedFunctionTypes {
    pub ty: KindId,
    pub impl_site: Option<ValId>,
    pub declaration_sites: Vec<ValId>,
    pub arguments: Vec<(PatId, Option<NameId>, KindId)>,
    pub generic_parameters: Vec<(PatId, Option<NameId>)>,
    pub lifetime_parameters: Vec<(PatId, Option<LifeTimeId>)>,
    pub inner: Option<InnerFunctionTypes>,
}

#[derive(Debug, Default)]
pub struct InnerFunctionTypes {
    pub my_universe: TypeUniverse,
    pub val_types: IdHashMap<ValId, KindId>,
    pub pat_types: IdHashMap<PatId, KindId>,
    pub member_method_types: IdHashMap<ValId, SolvedMemberMethodAccessType>,
    pub implicit_derefs: IdHashMap<ValId, Vec<(KindId, Projection)>>,
    pub value_origins: IdHashMap<ValId, OriginId>,
    pub pattern_origins: IdHashMap<PatId, OriginId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolvedMemberMethodAccessType {
    pub member: StrId,
    pub full_type: KindId,
}
