use super::Projection;
use super::{
    ArraySize, BuiltinKind, KindId, LifeId, LifeKind, MutId, Nullable, PointerStyle, PtrId,
    TypeKind,
};
use crate::data_structures::graph::BasicOrder;
use crate::data_structures::identity_hasher::IdHashMap;
use crate::data_structures::index::{Idx, IndexVec, UnionFind};
use crate::data_structures::string_intern::StrId;
use crate::ir::{LifeTimeId, NameId, PatId, TExpId, ValId};
use crate::program::Program;
use std::collections::{BTreeSet, HashMap};
use std::ops::{Index, IndexMut};

pub struct TypeUniverse {
    pub look: KindLookUp,
    pub storage: KindStorage,
}

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

    pub fn intern(&mut self, ty: TypeKind) -> KindId {
        if let Some(id) = self.map.get(&ty).copied() {
            return id;
        }

        let id = self.storage.push(Some(ty.clone()));
        self.map.insert(ty, id);
        id
    }

    pub fn add_empty(&mut self) -> KindId {
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

pub struct KindLookUp {
    pub kinds: UnionFind<KindId>,
    pub ptr: UnionFind<PtrId>,
    pub mutable: MutInfo,
    pub life: BasicOrder<LifeId>,
}

impl KindLookUp {
    pub fn new() -> Self {
        Self {
            kinds: UnionFind::new(),
            ptr: UnionFind::new(),
            mutable: MutInfo::new(),
            life: BasicOrder::new(),
        }
    }
}

impl Default for KindLookUp {
    fn default() -> Self {
        Self::new()
    }
}

pub struct KindStorage {
    pub types: TypeIntern,
    pub ptr: IndexVec<PtrId, Option<PointerStyle>>,
    pub life: IndexVec<LifeId, Option<LifeKind>>,
}

impl KindStorage {
    pub fn new() -> Self {
        Self {
            types: TypeIntern::new(),
            ptr: IndexVec::new(),
            life: IndexVec::new(),
        }
    }
}

impl Default for KindStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeUniverse {
    pub fn new() -> Self {
        Self {
            look: KindLookUp::new(),
            storage: KindStorage::new(),
        }
    }

    pub fn intern(&mut self, ty: TypeKind) -> KindId {
        if let Some(id) = self.storage.types.map.get(&ty).copied() {
            return id;
        }

        let id = self.storage.types.intern(ty);
        let uf_id = self.look.kinds.push_singleton();
        debug_assert_eq!(id, uf_id);
        id
    }

    pub fn intern_builtin(&mut self, builtin: BuiltinKind) -> KindId {
        self.intern(TypeKind::Builtin(builtin))
    }

    pub fn add_empty(&mut self) -> KindId {
        let id = self.storage.types.add_empty();
        let uf_id = self.look.kinds.push_singleton();
        debug_assert_eq!(id, uf_id);
        id
    }

    pub fn get(&self, id: KindId) -> Option<&TypeKind> {
        self.storage.types[id].as_ref()
    }

    pub fn kind_to_string(&self, program: &Program, id: KindId) -> String {
        self.kind_to_string_with_mut_guess(program, id, MutGuessMode::UnknownAsConst)
    }

    pub fn kind_to_string_with_mut_guess(
        &self,
        program: &Program,
        id: KindId,
        mut_guess_mode: MutGuessMode,
    ) -> String {
        let mut out = String::new();
        self.write_kind(program, id, mut_guess_mode, &mut out);
        out
    }

    pub fn deref_chain_to_string(
        &self,
        program: &Program,
        chain: &[(KindId, Projection)],
    ) -> String {
        chain
            .iter()
            .map(|(id, projection)| {
                format!(
                    "{} [{}]",
                    self.kind_to_string(program, *id),
                    projection_name(*projection)
                )
            })
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    fn write_kind(
        &self,
        program: &Program,
        id: KindId,
        mut_guess_mode: MutGuessMode,
        out: &mut String,
    ) {
        match self.get(id) {
            None => out.push('_'),
            Some(TypeKind::Builtin(builtin)) => out.push_str(builtin.name()),
            Some(TypeKind::Generic(gen_id)) => out.push_str(&format!("T{}", gen_id.0)),
            Some(TypeKind::Tuple(items)) => {
                out.push('(');
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(", ");
                    }
                    self.write_kind(program, *item, mut_guess_mode, out);
                }
                out.push(')');
            }
            Some(TypeKind::Struct { id, gens, lifes }) => {
                out.push_str(&format!("struct#{}", id.0));
                if let Some(gens) = gens {
                    write_angle_list(out, gens.iter().copied(), |out, item| {
                        self.write_kind(program, item, mut_guess_mode, out);
                    });
                }
                if let Some(lifes) = lifes {
                    write_angle_list(out, lifes.iter().copied(), |out, life| {
                        self.write_life(program, life, out);
                    });
                }
            }
            Some(TypeKind::Func { params, ret }) => {
                out.push_str("fn(");
                for (idx, param) in params.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(", ");
                    }
                    self.write_kind(program, *param, mut_guess_mode, out);
                }
                out.push_str(") -> ");
                self.write_kind(program, *ret, mut_guess_mode, out);
            }
            Some(TypeKind::Array { inner, size }) => {
                out.push('[');
                self.write_kind(program, *inner, mut_guess_mode, out);
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
                match self.storage.ptr.get(*style).and_then(|style| *style) {
                    Some(PointerStyle::Ref(life)) => {
                        out.push('&');
                        self.write_life(program, life, out);
                        out.push(' ');
                    }
                    Some(PointerStyle::Raw(nullable)) => {
                        out.push('*');
                        if matches!(nullable, Some(Nullable::Yes)) {
                            out.push('?');
                        }
                    }
                    None => out.push_str("ptr "),
                }
                match self.look.mutable.guess(*mutable, mut_guess_mode) {
                    MutGuess::Const => {}
                    MutGuess::Mut => out.push_str("mut "),
                    MutGuess::Unknown => out.push_str("?mut "),
                }
                self.write_kind(program, *tgt, mut_guess_mode, out);
            }
        }
    }

    fn write_life(&self, _program: &Program, id: LifeId, out: &mut String) {
        match self.storage.life.get(id).copied().flatten() {
            Some(LifeKind::Static) => out.push_str("'static"),
            Some(LifeKind::Univeral(Some(id))) => out.push_str(&format!("'u{id}")),
            Some(LifeKind::Univeral(None)) => out.push_str("'u"),
            Some(LifeKind::Local) => out.push_str("'local"),
            None => out.push_str("'_"),
        }
    }
}

impl Default for TypeUniverse {
    fn default() -> Self {
        Self::new()
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
        let bool_ty = types.intern_builtin(BuiltinKind::Bool);
        let ptr_style = types.storage.ptr.push(Some(PointerStyle::Raw(None)));
        let mutable = types.look.mutable.add_unknown();
        let ptr_ty = types.intern(TypeKind::Ptr {
            tgt: bool_ty,
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

fn write_angle_list<T, F>(out: &mut String, items: impl Iterator<Item = T>, mut write: F)
where
    F: FnMut(&mut String, T),
{
    out.push('<');
    for (idx, item) in items.enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        write(out, item);
    }
    out.push('>');
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

#[derive(Debug, Clone)]
pub struct SolvedFunctionTypes {
    pub ty: KindId,
    pub impl_site: Option<ValId>,
    pub declaration_sites: Vec<ValId>,
    pub arguments: Vec<(PatId, Option<NameId>, KindId)>,
    pub generic_parameters: Vec<(PatId, Option<NameId>)>,
    pub lifetime_parameters: Vec<(PatId, Option<LifeTimeId>)>,
    pub inner: Option<InnerFunctionTypes>,
}

#[derive(Debug, Clone, Default)]
pub struct InnerFunctionTypes {
    pub val_types: IdHashMap<ValId, KindId>,
    pub pat_types: IdHashMap<PatId, KindId>,
    pub member_method_types: IdHashMap<ValId, SolvedMemberMethodAccessType>,
    pub implicit_derefs: IdHashMap<ValId, Vec<(KindId, Projection)>>,
    pub origins: OriginVec<OriginNode>,
    pub value_origins: IdHashMap<ValId, OriginId>,
    pub pattern_origins: IdHashMap<PatId, OriginId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolvedMemberMethodAccessType {
    pub member: StrId,
    pub full_type: KindId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OriginId(pub u32);

#[derive(Debug, Clone)]
pub struct OriginVec<T>(pub Vec<T>);

impl<T> Default for OriginVec<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> OriginVec<T> {
    pub fn get(&self, id: OriginId) -> Option<&T> {
        self.0.get(id.0 as usize)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OriginNode {
    pub effective_mutability: Option<bool>,
}
