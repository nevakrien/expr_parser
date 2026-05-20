use crate::data_structures::string_intern::StrId;
use crate::ir::{BinOp, NameId, PatId, TExpId, UnOp, ValId};
use crate::parsing::Loc;

#[derive(Debug)]
pub enum TypeError {
    Simple {
        loc: Loc,
        message: &'static str,
    },
    SimpleRelated {
        loc: Loc,
        message: &'static str,
        related: Loc,
        related_message: &'static str,
    },
    LifetimeError {
        loc: Loc,
        message: String,
        label: String,
        related: Option<Loc>,
        related_label: Option<String>,
    },
    LifetimeOrderingConflict {
        loc: Loc,
        operation: &'static str,
        shorter: String,
        longer: String,
        related: Option<Loc>,
    },
    IllegalGlobalLifetimeOrdering {
        loc: Loc,
        operation: &'static str,
        shorter: String,
        longer: String,
        related: Option<Loc>,
    },
    UnknownBuiltinMemberMethod {
        site: ValId,
        method: StrId,
    },
    Unresolved {
        value: ValId,
        found: Option<String>,
    },
    UnresolvedPattern {
        pattern: PatId,
        found: Option<String>,
    },
    UnresolvedTypeExpr {
        expr: TExpId,
        found: Option<String>,
    },
    UnknownField {
        field: StrId,
        site: ValId,
    },
    DuplicateField {
        field: StrId,
        site: ValId,
    },
    FieldAlreadyPositional {
        field: StrId,
        site: ValId,
    },
    MissingField {
        field: NameId,
        site: ValId,
    },
    TooManyArguments {
        site: ValId,
        expected: usize,
        found: usize,
    },
    FieldTypeMismatch {
        field: StrId,
        value: ValId,
        clash: TypeClash,
    },
    IlegalMethod {
        member_name: StrId,
        access_site: ValId,
    },
    IlegalToImplMethod {
        method_name: StrId,
        method_site: ValId,
    },
    ConstructorBaseNotGlobal {
        site: ValId,
    },
    ConstructorBaseNotTypeName {
        site: ValId,
    },
    ConstructorBaseNotStruct {
        site: ValId,
        found: Option<String>,
    },
    TypeClashBeforeMentioned {
        name: NameId,
        expr: TExpId,
        clash: TypeClash,
    },
    ExpectedTypeExpr {
        type_expr: TExpId,
    },
    ValuesContradict {
        expectation_reason: &'static str,
        site: ValId,
        found: ValId,
        expected_place: ValId,
        clash: TypeClash,
    },
    BinOpOverloadNotFound {
        site: ValId,
        op: BinOp,
        lhs: ValId,
        rhs: ValId,
        lhs_type: Option<String>,
        rhs_type: Option<String>,
    },
    UnOpOverloadNotFound {
        site: ValId,
        op: UnOp,
        operand: ValId,
        operand_type: Option<String>,
    },
    CannotDeref {
        site: ValId,
        operand: ValId,
        operand_type: Option<String>,
    },
    AnnotationMismatch {
        annotation: ValId,
        constrained: ValId,
        clash: TypeClash,
    },
    FunctionOutputAnnotationMismatch {
        output_type: Option<TExpId>,
        constrained: ValId,
        clash: TypeClash,
    },
    PatternAnnotationMismatch {
        annotation: PatId,
        constrained: PatId,
        clash: TypeClash,
    },
    TypeDefPatternMismatch {
        pattern: PatId,
        clash: TypeClash,
    },
    DuplicateFunctionImplementation {
        first_implementation: ValId,
        duplicate_implementation: ValId,
    },
    UnusedFunctionGeneric {
        function: ValId,
        generic_index: usize,
    },
    UnusedFunctionLifetime {
        function: ValId,
        lifetime_index: usize,
    },
    UnusedStructGeneric {
        type_expr: TExpId,
        generic_index: usize,
    },
    UnusedStructLifetime {
        type_expr: TExpId,
        lifetime_index: usize,
    },
}

#[derive(Debug)]
pub struct TypeClash {
    pub found: Option<String>,
    pub wanted: Option<String>,
}

impl TypeClash {
    pub fn found(&self) -> Option<&str> {
        self.found.as_deref()
    }

    pub fn wanted(&self) -> Option<&str> {
        self.wanted.as_deref()
    }

    pub fn swap(self) -> Self {
        Self {
            found: self.wanted,
            wanted: self.found,
        }
    }
}
