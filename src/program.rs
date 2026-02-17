use crate::identity_hasher::IdHashMap;
use crate::ir::{
    LABEL_ALREADY_DEFINED_MSG, LifeTimeId, Literal, MEMBER_METHOD_COLLISION_MSG, NameId, PatId,
    Pattern, PatternSpan, ValId, Value, ValueSpan,
};
use crate::ir::{LabelId, VarKind};
use crate::ir::{TExpId, TypeExpr, TypeExprSpan};
use crate::macros::{Macro, expand_macros_recursive};
use crate::parsing::{Expr, LExpr, Loc, Located, Parser, Token};
use crate::string_intern::StrId;
use crate::string_intern::StringInterner;
use crate::string_intern::{RAW_STR, STATIC_STR};
use crate::type_inference::TypeValue;
use thiserror::Error;

pub type CResult<T> = Result<T, CompileError>;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum CompileError {
    #[error("{s}")]
    SimpleError { loc: Loc, s: &'static str },

    #[error("Unresolved name")]
    UnresolvedNames { name: String, locs: Vec<Loc> },

    #[error("label `{name}` was used but never defined")]
    UnresolvedLabel { name: String, locs: Vec<Loc> },

    #[error("{call_name} expected {expected} arguments, got {got}")]
    Arity {
        loc: Loc,
        call_name: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("{message}")]
    UnsupportedForm {
        loc: Loc,
        op_loc: Option<Loc>,
        op: Option<&'static str>,
        message: &'static str,
    },

    #[error("repeated global assignment to `{name}`")]
    RepeatedGlobalAssignment {
        name: String,
        existing: Option<Loc>,
        new: Loc,
    },

    #[error(transparent)]
    Parse(#[from] crate::parsing::ParseError),
}

#[derive(Debug)]
pub enum Defined {
    ToBeDefined,
    // Value(ValId),
    Func(ValId),
    Type(TExpId),
    BuildinType(TypeValue),
    Macro(Macro),
}

#[derive(Debug)]
pub(crate) struct PendingLabel {
    pub id: LabelId,
    pub defined_loc: Option<Loc>,
    pub pending_uses: Vec<Loc>,
    pub pending_gotos: Vec<ValId>,
}

pub const SPECIAL_LIFETIMES: &[StrId] = &[STATIC_STR, RAW_STR];
impl LifeTimeId {
    pub const STATIC: Self = Self(0);
    pub const RAW: Self = Self(1);
}

fn insert_builtin_lifetimes(p: &mut Program) {
    for s in SPECIAL_LIFETIMES {
        p.insert_new_lifetiime(*s);
    }
}

#[derive(Debug)]
pub struct Program {
    pub definitions: IdHashMap<NameId, Defined>,
    definition_locs: IdHashMap<NameId, Loc>,
    // pub current_inference: Vec<TypeInfo>,
    // pub type_store: TypeStore,
    pub values: Vec<Value>,
    pub patterns: Vec<Pattern>,
    pub type_exprs: Vec<TypeExpr>,
    value_locs: Vec<Loc>,
    pattern_locs: Vec<Loc>,
    type_expr_locs: Vec<Loc>,

    names_strs: Vec<StrId>,
    lifetime_strs: Vec<StrId>,
    pub str_intern: StringInterner,

    pub scopes: Vec<(IdHashMap<StrId, NameId>, IdHashMap<StrId, LifeTimeId>)>,
    pub pending_names: IdHashMap<NameId, Vec<Loc>>,
    pub member_methods: IdHashMap<NameId, IdHashMap<StrId, ValId>>,

    label_names: Vec<StrId>,
    pub(crate) function_labels: Vec<IdHashMap<StrId, PendingLabel>>,

    pub lowering_errors: Vec<CompileError>,
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Program {
    pub fn new() -> Self {
        let mut program = Self {
            definitions: IdHashMap::default(),
            definition_locs: IdHashMap::default(),
            // type_store: TypeStore::new(),
            values: Vec::new(),
            patterns: Vec::new(),
            type_exprs: Vec::new(),
            value_locs: Vec::new(),
            pattern_locs: Vec::new(),
            type_expr_locs: Vec::new(),

            names_strs: Vec::new(),
            lifetime_strs: Vec::new(),
            str_intern: StringInterner::new(),

            scopes: vec![(IdHashMap::default(), IdHashMap::default())],
            pending_names: IdHashMap::default(),
            member_methods: IdHashMap::default(),
            label_names: Vec::new(),
            function_labels: Vec::new(),
            lowering_errors: Vec::new(),
        };
        program.insert_builtin_types();
        insert_builtin_lifetimes(&mut program);
        program
    }

    pub(crate) fn placeholder_loc() -> Loc {
        Loc {
            range: 0..0,
            file: 0,
        }
    }

    pub fn push_lowering_error(&mut self, err: CompileError) {
        self.lowering_errors.push(err);
    }

    pub fn lower_all(&mut self, parser: &mut Parser<'_>) -> Result<(), Vec<CompileError>> {
        while !parser.is_empty() {
            match parser.parse_with_macros(self) {
                Ok(Some(expr)) => {
                    self.gather_definition(expr);
                }
                Ok(None) => break,
                Err(e) => {
                    self.push_lowering_error(e);
                }
            }
        }

        self.check_pending_names();

        let errors = std::mem::take(&mut self.lowering_errors);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub(crate) fn poison_value(&mut self, loc: Loc) -> ValId {
        self.id_value(loc, Value::Poison)
    }

    pub(crate) fn poison_pattern(&mut self, loc: Loc) -> PatId {
        self.id_pattern(loc, Pattern::Poison)
    }

    pub(crate) fn poison_type_expr(&mut self, loc: Loc) -> TExpId {
        self.id_type_expr(loc, TypeExpr::Poison)
    }

    pub fn id_value(&mut self, loc: Loc, value: Value) -> ValId {
        let id = ValId(self.values.len());
        self.values.push(value);
        self.value_locs.push(loc);
        id
    }

    pub fn id_pattern(&mut self, loc: Loc, pattern: Pattern) -> PatId {
        let id = PatId(self.patterns.len());
        self.patterns.push(pattern);
        self.pattern_locs.push(loc);
        id
    }

    pub fn reserve_value_span(&mut self, count: usize) -> ValueSpan {
        let start = ValId(self.values.len());
        for _ in 0..count {
            self.values.push(Value::Literal(Literal::Void));
            self.value_locs.push(Self::placeholder_loc());
        }
        ValueSpan::new(start, count)
    }

    pub fn reserve_pattern_span(&mut self, count: usize) -> PatternSpan {
        let start = PatId(self.patterns.len());
        for _ in 0..count {
            self.patterns.push(Pattern::Wildcard(VarKind::Const));
            self.pattern_locs.push(Self::placeholder_loc());
        }
        PatternSpan::new(start, count)
    }

    pub fn reserve_type_expr_span(&mut self, count: usize) -> TypeExprSpan {
        let start = TExpId(self.type_exprs.len());
        for _ in 0..count {
            self.type_exprs.push(TypeExpr::Wildcard);
            self.type_expr_locs.push(Self::placeholder_loc());
        }
        TypeExprSpan::new(start, count)
    }

    pub fn id_type_expr(&mut self, loc: Loc, exp: TypeExpr) -> TExpId {
        let id = TExpId(self.type_exprs.len());
        self.type_exprs.push(exp);
        self.type_expr_locs.push(loc);
        id
    }

    pub fn set_value(&mut self, id: ValId, loc: Loc, value: Value) {
        self.value_locs[id.0] = loc;
        self.values[id.0] = value;
    }

    pub fn set_pattern(&mut self, id: PatId, loc: Loc, pattern: Pattern) {
        self.pattern_locs[id.0] = loc;
        self.patterns[id.0] = pattern;
    }

    pub fn set_type_expr(&mut self, id: TExpId, loc: Loc, exp: TypeExpr) {
        self.type_expr_locs[id.0] = loc;
        self.type_exprs[id.0] = exp;
    }

    pub fn value(&self, id: ValId) -> Value {
        self.values[id.0]
    }

    pub fn pattern(&self, id: PatId) -> Pattern {
        self.patterns[id.0]
    }

    pub fn type_expr(&self, id: TExpId) -> TypeExpr {
        self.type_exprs[id.0]
    }

    pub fn value_loc(&self, v: ValId) -> Loc {
        self.value_locs[v.0].clone()
    }

    pub fn pattern_loc(&self, p: PatId) -> Loc {
        self.pattern_locs[p.0].clone()
    }

    pub fn type_expr_loc(&self, t: TExpId) -> Loc {
        self.type_expr_locs[t.0].clone()
    }

    /// Push a new variable scope onto the stack
    pub fn push_scope(&mut self) {
        self.scopes
            .push((IdHashMap::default(), IdHashMap::default()));
    }

    /// Pop the current variable scope
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn try_get_lifetime(&self, name: StrId) -> Option<LifeTimeId> {
        for s in self.scopes.iter().rev() {
            if let Some(ans) = s.1.get(&name) {
                return Some(*ans);
            }
        }

        None
    }

    pub fn insert_new_lifetiime(&mut self, name: StrId) -> LifeTimeId {
        let id = LifeTimeId(self.lifetime_strs.len());
        self.lifetime_strs.push(name);
        let scope = self
            .scopes
            .last_mut()
            .expect("no scope available when inserting binding");
        scope.1.insert(name, id);
        id
    }

    /// Insert a new binding into the current (innermost) scope, always creating a fresh ID.
    pub fn insert_value_in_current_scope(&mut self, name: StrId) -> NameId {
        let id = self.fresh_name_id(name);
        let value_scope = self
            .scopes
            .last_mut()
            .expect("no scope available when inserting binding");
        value_scope.0.insert(name, id);
        id
    }

    /// Insert a new binding into the global scope, always creating a fresh ID.
    pub fn insert_value_in_global_scope(&mut self, name: StrId) -> NameId {
        let id = self.fresh_name_id(name);
        if let Some(value_scope) = self.scopes.first_mut() {
            value_scope.0.insert(name, id);
        } else {
            debug_assert!(false, "no scope available when inserting binding");
        }
        id
    }

    pub fn name_string(&self, n: NameId) -> &str {
        self.str_intern.resolve(self.names_strs[n.0])
    }

    pub fn name_str_id(&self, n: NameId) -> StrId {
        self.names_strs[n.0]
    }

    /// Generate a fresh unique name ID
    fn fresh_name_id(&mut self, s: StrId) -> NameId {
        let id = NameId(self.names_strs.len());
        self.names_strs.push(s);
        id
    }

    fn definition_loc(&self, id: NameId) -> Option<Loc> {
        if let Some(loc) = self.definition_locs.get(&id) {
            return Some(loc.clone());
        }

        match self.definitions.get(&id) {
            Some(Defined::Func(val)) => Some(self.value_loc(*val)),
            // Some(Defined::Raw(expr)) => Some(expr.loc.clone()),
            Some(Defined::Type(expr)) => Some(self.type_expr_loc(*expr)),
            // Macro(Macro),
            Some(Defined::Macro(m)) => Some(m.loc.clone()),

            Some(Defined::BuildinType(..) | Defined::ToBeDefined) | None => None,
        }
    }

    pub(crate) fn set_definition_loc(&mut self, id: NameId, loc: Loc) {
        self.definition_locs.insert(id, loc);
    }

    pub fn get_macro(&mut self, name: &str) -> Option<&Macro> {
        //TODO think if we wana do scopes
        let name = self.str_intern.intern(name);
        let id = self.scopes[0].0.get(&name)?;
        if let Some(Defined::Macro(ans)) = self.definitions.get(id) {
            Some(ans)
        } else {
            None
        }
    }
}

impl<'a> Parser<'a> {
    pub fn parse_with_macros(&mut self, program: &mut Program) -> CResult<Option<LExpr>> {
        let Some(mut expr) = self.parse_stmt()? else {
            return Ok(None);
        };
        expand_macros_recursive(&mut expr, program)?;
        Ok(Some(expr))
    }
}

impl Program {
    //this takes a String because a name is mentioned once as String and gets resolved here
    //it actually has a much nicer cache behvior becaused the String usually gets freed and is giving something else room
    //instead of a &str which would bring cache line to some random parse data
    pub(crate) fn resolve_name(&mut self, loc: &Loc, name: &str) -> NameId {
        let name = self.str_intern.intern(name);
        for value_scope in self.scopes.iter().skip(1).rev() {
            if let Some(id) = value_scope.0.get(&name) {
                return *id;
            }
        }

        if let Some(id) = self.scopes[0].0.get(&name) {
            //might still be empty.
            if let Some(spot) = self.pending_names.get_mut(id) {
                spot.push(loc.clone())
            }
            return *id;
        }

        //errors get reported later it can be there is just a late mention so we dont know yet
        let id = self.insert_value_in_global_scope(name);
        self.definitions.insert(id, Defined::ToBeDefined);
        self.pending_names.entry(id).or_default().push(loc.clone());
        id
    }

    #[inline]
    pub fn with_scope<T>(&mut self, f: impl FnOnce(&mut Program) -> CResult<T>) -> CResult<T> {
        self.push_scope();
        let result = f(self);
        self.pop_scope();
        result
    }

    #[inline]
    pub fn with_scope_value<T>(&mut self, f: impl FnOnce(&mut Program) -> T) -> T {
        self.push_scope();
        let result = f(self);
        self.pop_scope();
        result
    }

    pub fn gather_definition(&mut self, expr: LExpr) {
        let Located { loc, value } = expr;
        match value {
            Expr::Postfix(op, mut items) if op.value == ";" => {
                self.gather_definition(items.pop().expect("bad structure"))
            }
            Expr::Prefix(open, items) if open.value == "{" => {
                for item in items {
                    self.gather_definition(item);
                }
            }
            Expr::Bin(eq, box_pair) if eq.value == "=" => {
                let (lhs, rhs) = *box_pair;
                self.handle_assignment(lhs, rhs);
            }

            Expr::Prefix(open, mut items) if open.value == "type" => {
                debug_assert!(items.len() == 2);
                let rhs = items.pop().unwrap();
                let lhs = items.pop().unwrap();

                if let Ok(name) = self.get_ident_for_global(lhs) {
                    let v = self.with_scope_value(|this| {
                        this.lower_type_expr(Located {
                            loc: rhs.loc,
                            value: rhs.value,
                        })
                    });
                    self.definitions.insert(name, Defined::Type(v));
                }
            }
            _ => {
                self.lower_value(Located { loc, value });
            }
        }
    }

    pub fn check_pending_names(&mut self) {
        if self.pending_names.is_empty() {
            return;
        }

        let mut errors_to_push = Vec::new();

        for (id, locs_ref) in self.pending_names.iter_mut() {
            match self.definitions.entry(*id) {
                std::collections::hash_map::Entry::Occupied(o) => {
                    if !matches!(o.get(), Defined::ToBeDefined) {
                        continue;
                    }
                }
                _ => continue,
            };
            let name = self.str_intern.resolve(self.names_strs[id.0]).to_string();

            let mut locs = Vec::new();
            std::mem::swap(locs_ref, &mut locs);

            errors_to_push.push(CompileError::UnresolvedNames { name, locs });
        }

        for err in errors_to_push {
            self.push_lowering_error(err);
        }
    }

    fn fresh_label_id(&mut self, name: StrId) -> LabelId {
        let id = LabelId(self.label_names.len());
        self.label_names.push(name);
        id
    }

    pub(crate) fn in_function_body(&self) -> bool {
        !self.function_labels.is_empty()
    }

    pub(crate) fn with_function_labels<T>(
        &mut self,
        f: impl FnOnce(&mut Program) -> CResult<T>,
    ) -> CResult<T> {
        self.function_labels.push(IdHashMap::default());
        let result = f(self);
        let labels = self
            .function_labels
            .pop()
            .expect("function label scope missing");

        let Ok(result) = result else {
            return result;
        };

        for (name, state) in labels {
            if state.defined_loc.is_some() {
                continue;
            }

            let mut locs = state.pending_uses;
            if locs.is_empty() {
                continue;
            }

            let name = self.str_intern.resolve(name).to_string();
            locs.shrink_to_fit();
            return Err(CompileError::UnresolvedLabel { name, locs });
        }

        Ok(result)
    }

    pub(crate) fn with_function_labels_value<T>(&mut self, f: impl FnOnce(&mut Program) -> T) -> T {
        self.function_labels.push(IdHashMap::default());
        let result = f(self);
        let labels = self
            .function_labels
            .pop()
            .expect("function label scope missing");

        for (name, state) in labels {
            if state.defined_loc.is_some() {
                continue;
            }

            let mut locs = state.pending_uses;
            if locs.is_empty() {
                continue;
            }

            let name = self.str_intern.resolve(name).to_string();
            locs.shrink_to_fit();
            self.push_lowering_error(CompileError::UnresolvedLabel { name, locs });
        }

        result
    }

    pub(crate) fn use_label_name_for_goto(
        &mut self,
        goto_id: ValId,
        loc: &Loc,
        name: &str,
    ) -> LabelId {
        let name = self.str_intern.intern(name);

        {
            let Some(labels) = self.function_labels.last_mut() else {
                self.push_lowering_error(CompileError::SimpleError {
                    loc: loc.clone(),
                    s: "goto statements must stay inside function bodies",
                });
                return LabelId::PENDING;
            };

            if let Some(state) = labels.get_mut(&name) {
                if state.defined_loc.is_none() {
                    state.pending_uses.push(loc.clone());
                    state.pending_gotos.push(goto_id);
                    return LabelId::PENDING;
                }
                return state.id;
            }
        }

        let id = self.fresh_label_id(name);
        let labels = self
            .function_labels
            .last_mut()
            .expect("label usage outside function scope should be checked first");
        labels.insert(
            name,
            PendingLabel {
                id,
                defined_loc: None,
                pending_uses: vec![loc.clone()],
                pending_gotos: vec![goto_id],
            },
        );
        LabelId::PENDING
    }

    pub(crate) fn define_label_name(&mut self, loc: &Loc, name: &str) -> LabelId {
        let name = self.str_intern.intern(name);

        let mut pending_gotos = Vec::new();
        let mut id = None;

        {
            let Some(labels) = self.function_labels.last_mut() else {
                self.push_lowering_error(CompileError::SimpleError {
                    loc: loc.clone(),
                    s: "labels must be declared inside function bodies",
                });
                return LabelId::PENDING;
            };

            if let Some(state) = labels.get_mut(&name) {
                if state.defined_loc.is_some() {
                    self.push_lowering_error(CompileError::SimpleError {
                        loc: loc.clone(),
                        s: LABEL_ALREADY_DEFINED_MSG,
                    });
                    return LabelId::PENDING;
                }
                state.defined_loc = Some(loc.clone());
                state.pending_uses.clear();
                std::mem::swap(&mut pending_gotos, &mut state.pending_gotos);
                id = Some(state.id);
            }
        }

        let id = if let Some(id) = id {
            id
        } else {
            let id = self.fresh_label_id(name);
            let labels = self
                .function_labels
                .last_mut()
                .expect("label definition outside function scope should be checked first");
            labels.insert(
                name,
                PendingLabel {
                    id,
                    defined_loc: Some(loc.clone()),
                    pending_uses: Vec::new(),
                    pending_gotos: Vec::new(),
                },
            );
            id
        };

        for goto_id in pending_gotos {
            let goto_loc = self.value_loc(goto_id);
            self.set_value(goto_id, goto_loc, Value::Goto(id));
        }

        id
    }

    fn get_ident_for_global(&mut self, lhs: LExpr) -> CResult<NameId> {
        let ans = match lhs.value {
            Expr::Atom(Token::Ident(name)) => {
                let name = self.str_intern.intern(&name);
                if let Some(id) = self
                    .scopes
                    .last()
                    .and_then(|scope| scope.0.get(&name))
                    .copied()
                {
                    if matches!(self.definitions.get(&id), Some(Defined::ToBeDefined)) {
                        id
                    } else {
                        let existing_loc = self.definition_loc(id);
                        let name_string = self.str_intern.resolve(name).to_string();
                        return Err(CompileError::RepeatedGlobalAssignment {
                            name: name_string,
                            existing: existing_loc,
                            new: lhs.loc,
                        });
                    }
                } else {
                    self.insert_value_in_current_scope(name)
                }
            }
            _ => {
                return Err(CompileError::SimpleError {
                    loc: lhs.loc,
                    s: "the left-hand side of a global assignment must be a bare identifier",
                });
            }
        };

        self.set_definition_loc(ans, lhs.loc);
        self.pending_names.remove(&ans);
        Ok(ans)
    }

    fn handle_assignment(&mut self, lhs: LExpr, rhs: LExpr) {
        let Located {
            loc: rhs_loc,
            value: rhs_value,
        } = rhs;

        if let Some((struct_name_id, method_name)) = self.try_member_method_lhs(&lhs) {
            let Expr::Prefix(fn_kw, _) = &rhs_value else {
                self.push_lowering_error(CompileError::SimpleError {
                    loc: rhs_loc,
                    s: "member methods must be defined with `fn` or `cfn` literals",
                });
                return;
            };
            if fn_kw.value != "fn" && fn_kw.value != "cfn" {
                self.push_lowering_error(CompileError::SimpleError {
                    loc: rhs_loc,
                    s: "member methods must be defined with `fn` or `cfn` literals",
                });
                return;
            }

            let def_value = self.with_scope_value(|this| {
                this.lower_value(Located {
                    loc: rhs_loc.clone(),
                    value: rhs_value,
                })
            });

            let methods = self.member_methods.entry(struct_name_id).or_default();
            match methods.entry(method_name) {
                std::collections::hash_map::Entry::Occupied(_) => {
                    self.push_lowering_error(CompileError::SimpleError {
                        loc: rhs_loc,
                        s: MEMBER_METHOD_COLLISION_MSG,
                    });
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(def_value);
                }
            }
            return;
        }

        let Ok(name_id) = self.get_ident_for_global(lhs) else {
            return;
        };

        let def: Defined = match rhs_value {
            Expr::Prefix(macro_kw, args) if macro_kw.value == "macro" => {
                match Macro::new(args, rhs_loc) {
                    Ok(macro_def) => Defined::Macro(macro_def),
                    Err(e) => {
                        self.push_lowering_error(e);
                        return;
                    }
                }
            }
            Expr::Prefix(ref fn_kw, _) if fn_kw.value == "fn" || fn_kw.value == "cfn" => {
                let v = self.with_scope_value(|this| {
                    this.lower_value(Located {
                        loc: rhs_loc,
                        value: rhs_value,
                    })
                });
                Defined::Func(v)
            }

            Expr::Prefix(ref kw, _)
                if kw.value == "struct"
                    || kw.value == "cstruct"
                    || kw.value == "enum"
                    || kw.value == "union" =>
            {
                let v = self.with_scope_value(|this| {
                    this.lower_type_expr(Located {
                        loc: rhs_loc,
                        value: rhs_value,
                    })
                });
                Defined::Type(v)
            }
            _ => {
                self.push_lowering_error(CompileError::SimpleError {
                    loc: rhs_loc,
                    s: "global definitions must assign a macro, function, or type literal",
                });
                return;
            }
        };

        self.definitions.insert(name_id, def);
    }

    fn try_member_method_lhs(&mut self, lhs: &LExpr) -> Option<(NameId, StrId)> {
        let Expr::Bin(op, pair) = &lhs.value else {
            return None;
        };
        if op.value != "." {
            return None;
        }

        let (base, method) = pair.as_ref();
        let (Expr::Atom(Token::Ident(struct_name)), Expr::Atom(Token::Ident(method_name))) =
            (&base.value, &method.value)
        else {
            return None;
        };

        let Some(struct_name_id) = self
            .scopes
            .first()
            .and_then(|scope| scope.0.get(&self.str_intern.intern(struct_name)))
            .copied()
        else {
            self.push_lowering_error(CompileError::SimpleError {
                loc: base.loc.clone(),
                s: "member methods must be attached to a struct type",
            });
            return None;
        };

        let texp = match self.definitions.get(&struct_name_id) {
            Some(Defined::Type(texp)) => *texp,
            _ => {
                self.push_lowering_error(CompileError::SimpleError {
                    loc: base.loc.clone(),
                    s: "member methods must be attached to a struct type",
                });
                return None;
            }
        };

        let TypeExpr::Struct(def) = self.type_expr(texp) else {
            self.push_lowering_error(CompileError::SimpleError {
                loc: base.loc.clone(),
                s: "member methods must be attached to a struct type",
            });
            return None;
        };

        let method_name = self.str_intern.intern(method_name);

        for field in def.fields.ids() {
            if let Some(field_name) = self.field_name(field)
                && field_name == method_name
            {
                self.push_lowering_error(CompileError::SimpleError {
                    loc: method.loc.clone(),
                    s: MEMBER_METHOD_COLLISION_MSG,
                });
                return None;
            }
        }

        Some((struct_name_id, method_name))
    }

    fn field_name(&self, pat: PatId) -> Option<StrId> {
        match self.pattern(pat) {
            Pattern::Bind(id, _) => Some(self.name_str_id(id)),
            Pattern::TypeAnnotation { pat, .. } => self.field_name(pat),
            _ => None,
        }
    }
}
