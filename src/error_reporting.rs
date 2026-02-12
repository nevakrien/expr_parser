use crate::ir::{BinOp, PatId, UnOp, ValId};
use crate::parsing::{Loc, OTok, ParseError};
use crate::program::{CompileError, Program};
use crate::type_inference::{
    BadTypeId, SolvedTypes, TypeClash, TypeError, TypeStore, UNKNOWN_TYPE,
};
use ariadne::{Cache, Color, Label, Report, ReportKind, Source};
use std::collections::HashMap;
use std::io;

const MAX_UNRESOLVED_NAME_LABELS: usize = 5;

pub struct ErrorReporter {
    sources: HashMap<usize, Source<String>>,
}

impl Cache<usize> for &ErrorReporter {
    type Storage = String;

    fn fetch(&mut self, id: &usize) -> Result<&Source<String>, Box<dyn std::fmt::Debug + '_>> {
        self.sources
            .get(id)
            .ok_or_else(|| panic!("missing source for file {id}"))
    }

    fn display<'b>(&self, id: &'b usize) -> Option<Box<dyn std::fmt::Display + 'b>> {
        Some(Box::new(format!("file {id}")))
    }
}

impl ErrorReporter {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
        }
    }

    pub fn add_source(&mut self, file_id: usize, source: String) {
        self.sources.insert(file_id, Source::from(source));
    }

    fn print_report(
        &self,
        primary_file: usize,
        report: Report<(usize, std::ops::Range<usize>)>,
    ) -> io::Result<()> {
        if !self.sources.contains_key(&primary_file) {
            return Ok(());
        }

        report.print(self)
    }

    pub fn report_parse_error(&self, error: &ParseError) -> io::Result<()> {
        match error {
            ParseError::UnexpectedChar { ch, loc } => {
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(format!("unexpected character `{}`", ch))
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("this character is not valid here"),
                    );

                self.print_report(loc.file, report.finish())
            }

            ParseError::UnterminatedString { loc } => {
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("unterminated string literal".to_string())
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("string starts here"),
                    );

                self.print_report(loc.file, report.finish())
            }

            ParseError::ExpectedExpr { got } => self.report_expected("expected expression", got),

            ParseError::ExpectedToken { expected, got } => {
                self.report_expected(&format!("expected {}", expected), got)
            }

            ParseError::OpenDelimiter { open, close, got } => {
                let open_loc = &open.loc;
                let close_loc = &got.loc;

                let report = Report::build(ReportKind::Error, open_loc.file, open_loc.range.start)
                    .with_message(format!("unclosed `{}` delimiter", open.value))
                    .with_label(
                        Label::new((open_loc.file, open_loc.range.clone()))
                            .with_message(format!("`{}` opened here", open.value)),
                    )
                    .with_label(
                        Label::new((close_loc.file, close_loc.range.clone()))
                            .with_message(format!("expected `{}`", close)),
                    );

                self.print_report(open_loc.file, report.finish())
            }
        }
    }

    fn report_expected(&self, message: &str, got: &OTok) -> io::Result<()> {
        let loc = &got.loc;
        let label_msg = match &got.value {
            Some(tok) => format!("found `{}` here", tok),
            None => "unexpected end of input here".to_string(),
        };

        let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
            .with_message(message)
            .with_label(Label::new((loc.file, loc.range.clone())).with_message(label_msg));

        self.print_report(loc.file, report.finish())
    }

    pub fn report_compile_error(&self, error: &CompileError) -> io::Result<()> {
        match error {
            CompileError::UnresolvedNames { locs, name } => {
                let Some(primary) = locs.first() else {
                    return Ok(());
                };

                let mut report = Report::build(
                    ReportKind::Error,
                    primary.file,
                    primary.range.start,
                )
                .with_message(if locs.len() < MAX_UNRESOLVED_NAME_LABELS {
                    format!("Unresolved name '{name}'")
                } else {
                    format!(
                        "Unresolved name '{name}' (showing {MAX_UNRESOLVED_NAME_LABELS}/{})",
                        locs.len()
                    )
                });

                for loc in locs.iter().take(MAX_UNRESOLVED_NAME_LABELS) {
                    report = report.with_label(
                        Label::new((loc.file, loc.range.clone())).with_message("used here"),
                    );
                }

                self.print_report(primary.file, report.finish())
            }
            CompileError::UnresolvedLabel { locs, name } => {
                let Some(primary) = locs.first() else {
                    return Ok(());
                };

                let mut report =
                    Report::build(ReportKind::Error, primary.file, primary.range.start)
                        .with_message(format!("label `{name}` was used but never defined"));

                for loc in locs.iter().take(MAX_UNRESOLVED_NAME_LABELS) {
                    report = report.with_label(
                        Label::new((loc.file, loc.range.clone())).with_message("used here"),
                    );
                }

                self.print_report(primary.file, report.finish())
            }
            CompileError::SimpleError { loc, .. } | CompileError::Arity { loc, .. } => {
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(error.to_string())
                    .with_label(Label::new((loc.file, loc.range.clone())).with_message("here"));

                self.print_report(loc.file, report.finish())
            }
            CompileError::UnsupportedForm {
                loc,
                op_loc,
                op,
                message,
            } => {
                let mut report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(*message)
                    .with_label(
                        // PRIMARY: whole expression, red, with operator message
                        Label::new((loc.file, loc.range.clone()))
                            .with_color(Color::Red)
                            .with_message(
                                op.map(|op| format!("operator `{}`", op))
                                    .unwrap_or_default(),
                            ),
                    );

                if let Some(op_loc) = op_loc {
                    // SECONDARY: operator token itself, cyan, no message
                    report = report.with_label(
                        Label::new((op_loc.file, op_loc.range.clone())).with_color(Color::Magenta),
                    );
                }

                self.print_report(loc.file, report.finish())
            }
            CompileError::RepeatedGlobalAssignment {
                name,
                existing: Some(existing),
                new,
            } => {
                let report = Report::build(ReportKind::Error, new.file, new.range.start)
                    .with_message(format!("repeated global assignment to `{name}`"))
                    .with_label(
                        Label::new((new.file, new.range.clone()))
                            .with_message("reassigned here")
                            .with_color(Color::Red),
                    )
                    .with_label(
                        Label::new((existing.file, existing.range.clone()))
                            .with_message("previous assignment here")
                            .with_color(Color::Yellow),
                    );

                self.print_report(new.file, report.finish())
            }

            CompileError::RepeatedGlobalAssignment {
                name,
                existing: None,
                new,
            } => {
                let report = Report::build(ReportKind::Error, new.file, new.range.start)
                    .with_message(format!("attempted global assignment to buildin `{name}`"))
                    .with_label(
                        Label::new((new.file, new.range.clone()))
                            .with_message("reassigned here")
                            .with_color(Color::Red),
                    );

                self.print_report(new.file, report.finish())
            }

            CompileError::Parse(parse_error) => self.report_parse_error(parse_error),
        }
    }

    pub fn report_type_error(
        &self,
        program: &Program,
        store: &TypeStore,
        error: &TypeError,
    ) -> io::Result<()> {
        match error {
            TypeError::Simple { loc, message } => {
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(*message)
                    .with_label(Label::new((loc.file, loc.range.clone())).with_message("here"));

                self.print_report(loc.file, report.finish())
            }
            TypeError::UnknownBuiltinMemberMethod { site, method } => {
                let loc = program.value_loc(*site);
                let method_name = program.str_intern.resolve(*method);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(format!("unknown builtin member method `{}`", method_name))
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("builtin member methods starting with `__` are reserved")
                            .with_color(Color::Red),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::Unresolved { value } => {
                let loc = program.value_loc(*value);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("could not infer type")
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("type is needed here"),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::UnresolvedPattern { pattern } => {
                let loc = program.pattern_loc(*pattern);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("could not infer pattern type")
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("pattern type is needed here"),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::UnresolvedTypeExpr { expr } => {
                let loc = program.type_expr_loc(*expr);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("could not infer state type");

                self.print_report(loc.file, report.finish())
            }
            TypeError::UnknownField { field, site } => {
                let loc = program.value_loc(*site);
                let field_name = program.str_intern.resolve(*field);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(format!("unknown field `{}`", field_name))
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("field not in struct")
                            .with_color(Color::Red),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::DuplicateField { field, site } => {
                let loc = program.value_loc(*site);
                let field_name = program.str_intern.resolve(*field);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(format!("field `{}` specified more than once", field_name))
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("duplicate field")
                            .with_color(Color::Red),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::FieldAlreadyPositional { field, site } => {
                let loc = program.value_loc(*site);
                let field_name = program.str_intern.resolve(*field);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(format!(
                        "field `{}` was already provided positionally",
                        field_name
                    ))
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("field already set")
                            .with_color(Color::Red),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::MissingField { field, site } => {
                let loc = program.value_loc(*site);
                let field_name = program.name_string(*field);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(format!("missing field `{}`", field_name))
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("field required")
                            .with_color(Color::Red),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::TooManyArguments {
                site,
                expected,
                found,
            } => {
                let loc = program.value_loc(*site);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(format!(
                        "too many arguments (expected {expected}, found {found})"
                    ))
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("extra argument")
                            .with_color(Color::Red),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::FieldTypeMismatch {
                field,
                value,
                clash,
            } => {
                let loc = program.value_loc(*value);
                let field_name = program.str_intern.resolve(*field);
                let (found_msg, expected_msg) = clash_messages(program, store, *clash);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(format!("field `{}` type mismatch", field_name))
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message(found_msg)
                            .with_color(Color::Yellow),
                    )
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message(expected_msg)
                            .with_color(Color::Cyan),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::ConstructorBaseNotGlobal { site } => {
                let loc = program.value_loc(*site);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("constructor base must be a global type name")
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("not a global name")
                            .with_color(Color::Red),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::ConstructorBaseNotTypeName { site } => {
                let loc = program.value_loc(*site);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("constructor base must be a type name")
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("not a type")
                            .with_color(Color::Red),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::ConstructorBaseNotStruct { site, found } => {
                let loc = program.value_loc(*site);
                let found_msg = found
                    .map(|t| format!("found {}", store.get_bad_type_string(program, t)))
                    .unwrap_or_else(|| "found unknown".to_string());
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("constructor base must be a struct type")
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message(found_msg)
                            .with_color(Color::Red),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::ExpectedTypeExpr { type_expr } => {
                let loc = program.type_expr_loc(*type_expr);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("unknown type expression")
                    .with_label(
                        Label::new((loc.file, loc.range.clone())).with_message("is this a type?"),
                    );

                self.print_report(loc.file, report.finish())
            }
            TypeError::ValuesContradict {
                expectation_reason,
                site,
                found,
                expected_place,
                clash,
            } => {
                let site_loc = program.value_loc(*site);
                let found_loc = program.value_loc(*found);
                let expected_loc = program.value_loc(*expected_place);
                let (found_msg, expected_msg) = clash_messages(program, store, *clash);

                let mut report =
                    Report::build(ReportKind::Error, site_loc.file, site_loc.range.start)
                        .with_message(format!("type mismatch: {expectation_reason}"))
                        .with_label(
                            Label::new((site_loc.file, site_loc.range.clone()))
                                .with_message("type mismatch here")
                                .with_color(Color::Red),
                        );

                report = report.with_label(
                    Label::new((found_loc.file, found_loc.range.clone()))
                        .with_message(found_msg)
                        .with_color(Color::Yellow),
                );

                report = report.with_label(
                    Label::new((expected_loc.file, expected_loc.range.clone()))
                        .with_message(expected_msg)
                        .with_color(Color::Cyan),
                );

                self.print_report(site_loc.file, report.finish())
            }
            TypeError::BinOpOverloadNotFound {
                site,
                op,
                lhs,
                rhs,
                lhs_type,
                rhs_type,
            } => {
                let site_loc = program.value_loc(*site);
                let lhs_loc = program.value_loc(*lhs);
                let rhs_loc = program.value_loc(*rhs);
                let lhs_msg = operand_type_message(program, store, "left operand", *lhs_type);
                let rhs_msg = operand_type_message(program, store, "right operand", *rhs_type);

                let mut report =
                    Report::build(ReportKind::Error, site_loc.file, site_loc.range.start)
                        .with_message(format!("no overload for operator `{}`", bin_op_symbol(*op)))
                        .with_label(
                            Label::new((site_loc.file, site_loc.range.clone()))
                                .with_message("operator used here")
                                .with_color(Color::Red),
                        );

                report = report.with_label(
                    Label::new((lhs_loc.file, lhs_loc.range.clone()))
                        .with_message(lhs_msg)
                        .with_color(Color::Yellow),
                );

                report = report.with_label(
                    Label::new((rhs_loc.file, rhs_loc.range.clone()))
                        .with_message(rhs_msg)
                        .with_color(Color::Cyan),
                );

                self.print_report(site_loc.file, report.finish())
            }
            TypeError::UnOpOverloadNotFound {
                site,
                op,
                operand,
                operand_type,
            } => {
                let site_loc = program.value_loc(*site);
                let operand_loc = program.value_loc(*operand);
                let operand_msg = operand_type_message(program, store, "operand", *operand_type);

                let mut report =
                    Report::build(ReportKind::Error, site_loc.file, site_loc.range.start)
                        .with_message(format!("no overload for operator `{}`", un_op_symbol(*op)))
                        .with_label(
                            Label::new((site_loc.file, site_loc.range.clone()))
                                .with_message("operator used here")
                                .with_color(Color::Red),
                        );

                report = report.with_label(
                    Label::new((operand_loc.file, operand_loc.range.clone()))
                        .with_message(operand_msg)
                        .with_color(Color::Yellow),
                );

                self.print_report(site_loc.file, report.finish())
            }
            TypeError::CannotDeref {
                site,
                operand,
                operand_type,
            } => {
                let site_loc = program.value_loc(*site);
                let operand_loc = program.value_loc(*operand);
                let operand_msg = operand_type_message(program, store, "operand", *operand_type);

                let report = Report::build(ReportKind::Error, site_loc.file, site_loc.range.start)
                    .with_message("cannot dereference this value")
                    .with_label(
                        Label::new((site_loc.file, site_loc.range.clone()))
                            .with_message("deref used here")
                            .with_color(Color::Red),
                    )
                    .with_label(
                        Label::new((operand_loc.file, operand_loc.range.clone()))
                            .with_message(operand_msg)
                            .with_color(Color::Yellow),
                    );

                self.print_report(site_loc.file, report.finish())
            }
            TypeError::AnnotationMismatch {
                annotation,
                constrained,
                clash,
            } => {
                let ann_loc = program.value_loc(*annotation);
                let constrained_loc = program.value_loc(*constrained);
                let (found_msg, expected_msg) = clash_messages(program, store, *clash);

                let report = Report::build(ReportKind::Error, ann_loc.file, ann_loc.range.start)
                    .with_message("type annotation mismatch")
                    .with_label(
                        Label::new((ann_loc.file, ann_loc.range.clone()))
                            .with_message(expected_msg)
                            .with_color(Color::Cyan),
                    )
                    .with_label(
                        Label::new((constrained_loc.file, constrained_loc.range.clone()))
                            .with_message(found_msg)
                            .with_color(Color::Yellow),
                    );

                self.print_report(ann_loc.file, report.finish())
            }
            TypeError::PatternAnnotationMismatch {
                annotation,
                constrained,
                clash,
            } => {
                let ann_loc = program.pattern_loc(*annotation);
                let constrained_loc = program.pattern_loc(*constrained);
                let (found_msg, expected_msg) = clash_messages(program, store, *clash);

                let report = Report::build(ReportKind::Error, ann_loc.file, ann_loc.range.start)
                    .with_message("pattern annotation mismatch")
                    .with_label(
                        Label::new((ann_loc.file, ann_loc.range.clone()))
                            .with_message(expected_msg)
                            .with_color(Color::Cyan),
                    )
                    .with_label(
                        Label::new((constrained_loc.file, constrained_loc.range.clone()))
                            .with_message(found_msg)
                            .with_color(Color::Yellow),
                    );

                self.print_report(ann_loc.file, report.finish())
            }

            TypeError::TypeDefPatternMismatch { pattern, clash } => {
                let loc = program.pattern_loc(*pattern);
                let (found_msg, expected_msg) = clash_messages(program, store, *clash);

                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("type definition name must be a type")
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message(found_msg)
                            .with_color(Color::Yellow),
                    )
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message(expected_msg)
                            .with_color(Color::Cyan),
                    );

                self.print_report(loc.file, report.finish())
            }

            TypeError::TypeClashBeforeMentioned { name, expr, clash } => {
                let loc = program.type_expr_loc(*expr);
                let (found_msg, expected_msg) = clash_messages(program, store, *clash);

                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(format!(
                        "could not infer type of `{}`",
                        program.name_string(*name)
                    ))
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message(format!("{} (this type was infered as)", found_msg))
                            .with_color(Color::Red),
                    )
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message(format!("{} but was defined as", expected_msg))
                            .with_color(Color::Cyan),
                    );

                self.print_report(loc.file, report.finish())
            }
        }
    }

    pub fn report_type_dump(
        &self,
        program: &Program,
        store: &TypeStore,
        solved: &SolvedTypes,
    ) -> io::Result<()> {
        self.report_type_dump_in_region(program, store, solved, None)
    }

    pub fn report_type_dump_in_region(
        &self,
        program: &Program,
        store: &TypeStore,
        solved: &SolvedTypes,
        region: Option<&Loc>,
    ) -> io::Result<()> {
        let mut labels_by_file: HashMap<usize, Vec<(std::ops::Range<usize>, String, Color)>> =
            HashMap::new();

        for index in 0..solved.val_types.len() {
            let Some(t) = solved.type_of(ValId(index)) else {
                continue;
            };
            let loc = program.value_loc(ValId(index));
            if !loc_in_region(&loc, region) {
                continue;
            }
            labels_by_file.entry(loc.file).or_default().push((
                loc.range.clone(),
                format!("value: {}", store.get_type_string(program, t)),
                Color::Yellow,
            ));
        }

        for index in 0..solved.pat_types.len() {
            let Some(t) = solved.pat_type(PatId(index)) else {
                continue;
            };
            let loc = program.pattern_loc(PatId(index));
            if !loc_in_region(&loc, region) {
                continue;
            }
            labels_by_file.entry(loc.file).or_default().push((
                loc.range.clone(),
                format!("pattern: {}", store.get_type_string(program, t)),
                Color::Cyan,
            ));
        }

        for (texp, t) in solved.typedef_types.iter() {
            if *t == UNKNOWN_TYPE {
                continue;
            }

            let loc = program.type_expr_loc(*texp);
            if !loc_in_region(&loc, region) {
                continue;
            }
            labels_by_file.entry(loc.file).or_default().push((
                loc.range.clone(),
                format!("type expr: {}", store.get_type_string(program, *t)),
                Color::Green,
            ));
        }

        for (site, member) in solved.member_method_types.iter() {
            let loc = program.value_loc(*site);
            if !loc_in_region(&loc, region) {
                continue;
            }
            let member_name = program.str_intern.resolve(member.member);
            labels_by_file.entry(loc.file).or_default().push((
                loc.range.clone(),
                format!(
                    "member method `{}`: {}",
                    member_name,
                    store.get_type_string(program, member.full_type)
                ),
                Color::Magenta,
            ));
        }

        for (site, chain) in solved.member_access_implicit_derefs.iter() {
            let loc = program.value_loc(*site);
            if !loc_in_region(&loc, region) {
                continue;
            }
            let chain_types = chain
                .iter()
                .map(|t| store.get_type_string(program, *t))
                .collect::<Vec<_>>()
                .join(" -> ");
            labels_by_file.entry(loc.file).or_default().push((
                loc.range.clone(),
                format!("member access implicit deref chain: {chain_types}"),
                Color::Blue,
            ));
        }

        if labels_by_file.is_empty() {
            return Ok(());
        }

        for (file, mut labels) in labels_by_file {
            labels.sort_by_key(|(range, _, _)| (range.start, range.end));

            let start = labels.first().map(|(r, _, _)| r.start).unwrap_or(0);
            let mut report =
                Report::build(ReportKind::Custom("Type Dump", Color::Yellow), file, start);

            for (range, message, color) in labels {
                report = report.with_label(
                    Label::new((file, range))
                        .with_message(message)
                        .with_color(color),
                );
            }

            self.print_report(file, report.finish())?;
        }

        Ok(())
    }
}

fn loc_in_region(loc: &Loc, region: Option<&Loc>) -> bool {
    let Some(region) = region else {
        return true;
    };

    loc.file == region.file
        && loc.range.start >= region.range.start
        && loc.range.end <= region.range.end
}

fn clash_messages(program: &Program, store: &TypeStore, clash: TypeClash) -> (String, String) {
    let found = type_string(program, store, clash.found).unwrap_or_else(|| "unknown".to_string());
    let wanted = type_string(program, store, clash.wanted).unwrap_or_else(|| "unknown".to_string());
    (format!("found {found}"), format!("expected {wanted}"))
}

fn operand_type_message(
    program: &Program,
    store: &TypeStore,
    label: &str,
    ty: Option<BadTypeId>,
) -> String {
    match type_string(program, store, ty) {
        Some(ty) => format!("{label} has type {ty}"),
        None => format!("{label} type is unknown"),
    }
}

fn bin_op_symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
    }
}

fn un_op_symbol(op: UnOp) -> &'static str {
    match op {
        UnOp::Neg => "-",
        UnOp::Not => "!",
        UnOp::BitNot => "~",
        // UnOp::Deref => "*",
        // UnOp::AddrOf(None) => "&",
        // UnOp::AddrOf(Some(VarKind::Mut)) => "&mut",
        // UnOp::AddrOf(Some(VarKind::Const)) => "&const",
    }
}

fn type_string(program: &Program, store: &TypeStore, ty: Option<BadTypeId>) -> Option<String> {
    ty.map(|t| store.get_bad_type_string(program, t))
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new()
    }
}
