use crate::ir::{BinOp, UnOp};
use crate::parsing::{Loc, OTok, ParseError};
use crate::program::{CompileError, Program};
use crate::type_inference::{SolvedTypes, TypeClash, TypeError, TypeStore, UNKNOWN_TYPE};
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
        if let Some(ans) = self.sources.get(id) {
            Ok(ans)
        } else {
            Err(Box::new(format!("missing source for file {id}")))
        }
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

    fn print_report(&self, report: Report<(usize, std::ops::Range<usize>)>) -> io::Result<()> {
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

                self.print_report(report.finish())
            }

            ParseError::UnterminatedString { loc } => {
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("unterminated string literal".to_string())
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("string starts here"),
                    );

                self.print_report(report.finish())
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

                self.print_report(report.finish())
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

        self.print_report(report.finish())
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

                self.print_report(report.finish())
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

                self.print_report(report.finish())
            }
            CompileError::SimpleError { loc, .. } | CompileError::Arity { loc, .. } => {
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(error.to_string())
                    .with_label(Label::new((loc.file, loc.range.clone())).with_message("here"));

                self.print_report(report.finish())
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

                self.print_report(report.finish())
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

                self.print_report(report.finish())
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

                self.print_report(report.finish())
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

                self.print_report(report.finish())
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

                self.print_report(report.finish())
            }
            TypeError::IlegalMethod {
                member_name,
                access_site,
            } => {
                let loc = program.value_loc(*access_site);
                let method_name = program.str_intern.resolve(*member_name);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(format!(
                        "{} is not allowed to be called in user code",
                        method_name
                    ))
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("bad call here")
                            .with_color(Color::Red),
                    );

                self.print_report(report.finish())
            }
            TypeError::IlegalToImplMethod {
                method_name,
                method_site,
            } => {
                let loc = program.value_loc(*method_site);
                let method_name = program.str_intern.resolve(*method_name);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(format!(
                        "{} is buildin and not allowed to be overwriten",
                        method_name
                    ))
                    .with_label(Label::new((loc.file, loc.range.clone())).with_color(Color::Red));

                self.print_report(report.finish())
            }
            TypeError::Unresolved { value, found } => {
                let loc = program.value_loc(*value);
                let mut report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("could not infer type")
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("type is needed here"),
                    );

                if let Some(found) = found {
                    report = report.with_note(format!("best known unresolved shape: {found}"));
                }

                self.print_report(report.finish())
            }
            TypeError::UnresolvedPattern { pattern, found } => {
                let loc = program.pattern_loc(*pattern);
                let mut report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("could not infer pattern type")
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("pattern type is needed here"),
                    );

                if let Some(found) = found {
                    report = report.with_note(format!("best known unresolved shape: {found}"));
                }

                self.print_report(report.finish())
            }
            TypeError::UnresolvedTypeExpr { expr, found } => {
                let loc = program.type_expr_loc(*expr);
                let mut report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("could not infer type")
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("this type is probably recursive in some way"),
                    );

                if let Some(found) = found {
                    report = report.with_note(format!("best known unresolved shape: {found}"));
                }

                self.print_report(report.finish())
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

                self.print_report(report.finish())
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

                self.print_report(report.finish())
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

                self.print_report(report.finish())
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

                self.print_report(report.finish())
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

                self.print_report(report.finish())
            }
            TypeError::FieldTypeMismatch {
                field,
                value,
                clash,
            } => {
                let loc = program.value_loc(*value);
                let field_name = program.str_intern.resolve(*field);
                let (found_msg, expected_msg) = clash_messages(program, store, clash);
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

                self.print_report(report.finish())
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

                self.print_report(report.finish())
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

                self.print_report(report.finish())
            }
            TypeError::ConstructorBaseNotStruct { site, found } => {
                let loc = program.value_loc(*site);
                let found_msg = found
                    .as_ref()
                    .map(|t| format!("found {t}"))
                    .unwrap_or_else(|| "found unknown".to_string());
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("constructor base must be a struct type")
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message(found_msg)
                            .with_color(Color::Red),
                    );

                self.print_report(report.finish())
            }
            TypeError::ExpectedTypeExpr { type_expr } => {
                let loc = program.type_expr_loc(*type_expr);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("unknown type expression")
                    .with_label(
                        Label::new((loc.file, loc.range.clone())).with_message("is this a type?"),
                    );

                self.print_report(report.finish())
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
                let (found_msg, expected_msg) = clash_messages(program, store, clash);

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

                self.print_report(report.finish())
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
                let lhs_msg =
                    operand_type_message(program, store, "left operand", lhs_type.as_deref());
                let rhs_msg =
                    operand_type_message(program, store, "right operand", rhs_type.as_deref());

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

                self.print_report(report.finish())
            }
            TypeError::UnOpOverloadNotFound {
                site,
                op,
                operand,
                operand_type,
            } => {
                let site_loc = program.value_loc(*site);
                let operand_loc = program.value_loc(*operand);
                let operand_msg =
                    operand_type_message(program, store, "operand", operand_type.as_deref());

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

                self.print_report(report.finish())
            }
            TypeError::CannotDeref {
                site,
                operand,
                operand_type,
            } => {
                let site_loc = program.value_loc(*site);
                let operand_loc = program.value_loc(*operand);
                let operand_msg =
                    operand_type_message(program, store, "operand", operand_type.as_deref());

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

                self.print_report(report.finish())
            }
            TypeError::AnnotationMismatch {
                annotation,
                constrained,
                clash,
            } => {
                let ann_loc = program.value_loc(*annotation);
                let constrained_loc = program.value_loc(*constrained);
                let (found_msg, expected_msg) = clash_messages(program, store, clash);

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

                self.print_report(report.finish())
            }
            TypeError::FunctionOutputAnnotationMismatch {
                output_type: Some(annotation),
                constrained,
                clash,
            } => {
                let ann_loc = program.type_expr_loc(*annotation);
                let constrained_loc = program.value_loc(*constrained);
                let (found_msg, expected_msg) = clash_messages(program, store, clash);

                let report = Report::build(ReportKind::Error, ann_loc.file, ann_loc.range.start)
                    .with_message("function output type annotation mismatch")
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

                self.print_report(report.finish())
            }
            TypeError::FunctionOutputAnnotationMismatch {
                output_type: None,
                constrained,
                clash,
            } => {
                let constrained_loc = program.value_loc(*constrained);
                let (found_msg, expected_msg) = clash_messages(program, store, clash);

                let report = Report::build(
                    ReportKind::Error,
                    constrained_loc.file,
                    constrained_loc.range.start,
                )
                .with_message("expected a void output for this function")
                .with_label(
                    Label::new((constrained_loc.file, constrained_loc.range.clone()))
                        .with_message(expected_msg)
                        .with_color(Color::Cyan),
                )
                .with_label(
                    Label::new((constrained_loc.file, constrained_loc.range.clone()))
                        .with_message(found_msg)
                        .with_color(Color::Yellow),
                );

                self.print_report(report.finish())
            }
            TypeError::PatternAnnotationMismatch {
                annotation,
                constrained,
                clash,
            } => {
                let ann_loc = program.pattern_loc(*annotation);
                let constrained_loc = program.pattern_loc(*constrained);
                let (found_msg, expected_msg) = clash_messages(program, store, clash);

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

                self.print_report(report.finish())
            }

            TypeError::TypeDefPatternMismatch { pattern, clash } => {
                let loc = program.pattern_loc(*pattern);
                let (found_msg, expected_msg) = clash_messages(program, store, clash);

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

                self.print_report(report.finish())
            }

            TypeError::DuplicateFunctionImplementation {
                first_implementation,
                duplicate_implementation,
            } => {
                let duplicate_loc = program.value_loc(*duplicate_implementation);
                let first_loc = program.value_loc(*first_implementation);
                let report = Report::build(
                    ReportKind::Error,
                    duplicate_loc.file,
                    duplicate_loc.range.start,
                )
                .with_message("multiple function implementations are not allowed")
                .with_label(
                    Label::new((duplicate_loc.file, duplicate_loc.range.clone()))
                        .with_message("duplicate implementation")
                        .with_color(Color::Red),
                )
                .with_label(
                    Label::new((first_loc.file, first_loc.range.clone()))
                        .with_message("first implementation")
                        .with_color(Color::Cyan),
                );

                self.print_report(report.finish())
            }

            TypeError::UnusedFunctionGeneric {
                function,
                generic_index,
            } => {
                let function_loc = program.value_loc(*function);
                let generic_loc = match program.value(*function) {
                    crate::ir::Value::Func { generics, .. } => generics
                        .generics()
                        .ids()
                        .nth(*generic_index)
                        .map(|pat| program.pattern_loc(pat))
                        .unwrap_or(function_loc.clone()),
                    _ => function_loc.clone(),
                };

                let report =
                    Report::build(ReportKind::Error, generic_loc.file, generic_loc.range.start)
                        .with_message(format!(
                            "unused generic parameter T{} in function signature",
                            generic_index
                        ))
                        .with_label(
                            Label::new((generic_loc.file, generic_loc.range.clone()))
                                .with_message("generic parameter declared here")
                                .with_color(Color::Red),
                        )
                        .with_label(
                            Label::new((function_loc.file, function_loc.range.clone()))
                                .with_message("function signature does not use this generic")
                                .with_color(Color::Cyan),
                        );

                self.print_report(report.finish())
            }

            TypeError::UnusedFunctionLifetime {
                function,
                lifetime_index,
            } => {
                let function_loc = program.value_loc(*function);
                let lifetime_loc = match program.value(*function) {
                    crate::ir::Value::Func { generics, .. } => generics
                        .lifetimes()
                        .ids()
                        .nth(*lifetime_index)
                        .map(|pat| program.pattern_loc(pat))
                        .unwrap_or(function_loc.clone()),
                    _ => function_loc.clone(),
                };

                let report = Report::build(
                    ReportKind::Error,
                    lifetime_loc.file,
                    lifetime_loc.range.start,
                )
                .with_message(format!(
                    "unused lifetime parameter 'a{} in function signature",
                    lifetime_index
                ))
                .with_label(
                    Label::new((lifetime_loc.file, lifetime_loc.range.clone()))
                        .with_message("lifetime parameter declared here")
                        .with_color(Color::Red),
                )
                .with_label(
                    Label::new((function_loc.file, function_loc.range.clone()))
                        .with_message("function signature does not use this lifetime")
                        .with_color(Color::Cyan),
                );

                self.print_report(report.finish())
            }

            TypeError::UnusedStructGeneric {
                type_expr,
                generic_index,
            } => {
                let type_loc = program.type_expr_loc(*type_expr);
                let generic_loc = match program.type_expr(*type_expr) {
                    crate::ir::TypeExpr::Struct(def) => def
                        .generics
                        .generics()
                        .ids()
                        .nth(*generic_index)
                        .map(|pat| program.pattern_loc(pat))
                        .unwrap_or(type_loc.clone()),
                    _ => type_loc.clone(),
                };

                let report =
                    Report::build(ReportKind::Error, generic_loc.file, generic_loc.range.start)
                        .with_message(format!(
                            "unused generic parameter T{} in struct signature",
                            generic_index
                        ))
                        .with_label(
                            Label::new((generic_loc.file, generic_loc.range.clone()))
                                .with_message("generic parameter declared here")
                                .with_color(Color::Red),
                        )
                        .with_label(
                            Label::new((type_loc.file, type_loc.range.clone()))
                                .with_message("struct signature does not use this generic")
                                .with_color(Color::Cyan),
                        );

                self.print_report(report.finish())
            }

            TypeError::UnusedStructLifetime {
                type_expr,
                lifetime_index,
            } => {
                let type_loc = program.type_expr_loc(*type_expr);
                let lifetime_loc = match program.type_expr(*type_expr) {
                    crate::ir::TypeExpr::Struct(def) => def
                        .generics
                        .lifetimes()
                        .ids()
                        .nth(*lifetime_index)
                        .map(|pat| program.pattern_loc(pat))
                        .unwrap_or(type_loc.clone()),
                    _ => type_loc.clone(),
                };

                let report = Report::build(
                    ReportKind::Error,
                    lifetime_loc.file,
                    lifetime_loc.range.start,
                )
                .with_message(format!(
                    "unused lifetime parameter 'a{} in struct signature",
                    lifetime_index
                ))
                .with_label(
                    Label::new((lifetime_loc.file, lifetime_loc.range.clone()))
                        .with_message("lifetime parameter declared here")
                        .with_color(Color::Red),
                )
                .with_label(
                    Label::new((type_loc.file, type_loc.range.clone()))
                        .with_message("struct signature does not use this lifetime")
                        .with_color(Color::Cyan),
                );

                self.print_report(report.finish())
            }

            TypeError::TypeClashBeforeMentioned { name, expr, clash } => {
                let loc = program.type_expr_loc(*expr);
                let (found_msg, expected_msg) = clash_messages(program, store, clash);

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

                self.print_report(report.finish())
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

        let mut typedef_entries = solved
            .typedef_types
            .iter()
            .map(|(texp, t)| (*texp, *t))
            .collect::<Vec<_>>();
        typedef_entries.sort_unstable_by_key(|(texp, _)| texp.0);

        for (texp, t) in typedef_entries {
            if t == UNKNOWN_TYPE {
                continue;
            }
            let loc = program.type_expr_loc(texp);
            if !loc_in_region(&loc, region) {
                continue;
            }
            labels_by_file.entry(loc.file).or_default().push((
                loc.range.clone(),
                format!("type expr: {}", store.get_type_string(program, t)),
                Color::Green,
            ));
        }

        let mut pattern_entries = solved
            .function_values
            .values()
            .flat_map(|f| {
                f.arguments
                    .iter()
                    .map(|(pat, _name, ty)| (*pat, *ty))
                    .chain(
                        f.inner
                            .as_ref()
                            .into_iter()
                            .flat_map(|inner| inner.pat_types.iter().map(|(pat, ty)| (*pat, *ty))),
                    )
            })
            .collect::<Vec<_>>();
        pattern_entries.sort_unstable_by_key(|(pat, _)| pat.0);
        pattern_entries.dedup_by_key(|(pat, _)| *pat);

        for (pat, t) in pattern_entries {
            let loc = program.pattern_loc(pat);
            if !loc_in_region(&loc, region) {
                continue;
            }
            labels_by_file.entry(loc.file).or_default().push((
                loc.range.clone(),
                format!("pattern: {}", store.get_type_string(program, t)),
                Color::Cyan,
            ));
        }

        let mut value_entries = solved
            .function_values
            .iter()
            .flat_map(|(function, f)| {
                std::iter::once((*function, f.ty)).chain(
                    f.inner
                        .as_ref()
                        .into_iter()
                        .flat_map(|inner| inner.val_types.iter().map(|(val, ty)| (*val, *ty))),
                )
            })
            .collect::<Vec<_>>();
        value_entries.sort_unstable_by_key(|(site, _)| site.0);
        value_entries.dedup_by_key(|(site, _)| *site);

        let mut member_entries = solved
            .function_values
            .values()
            .flat_map(|f| {
                f.inner.as_ref().into_iter().flat_map(|inner| {
                    inner
                        .member_method_types
                        .iter()
                        .map(|(site, member)| (*site, *member))
                })
            })
            .collect::<Vec<_>>();
        member_entries.sort_unstable_by_key(|(site, _)| site.0);

        let mut deref_entries = solved
            .function_values
            .values()
            .flat_map(|f| {
                f.inner.as_ref().into_iter().flat_map(|inner| {
                    inner
                        .implicit_derefs
                        .iter()
                        .map(|(site, chain)| (*site, chain.as_slice()))
                })
            })
            .collect::<Vec<_>>();
        deref_entries.sort_unstable_by_key(|(site, _)| site.0);

        for (site, t) in value_entries {
            let loc = program.value_loc(site);
            if !loc_in_region(&loc, region) {
                continue;
            }
            labels_by_file.entry(loc.file).or_default().push((
                loc.range.clone(),
                format!("value: {}", store.get_type_string(program, t)),
                Color::Yellow,
            ));
        }

        for (site, member) in member_entries {
            let loc = program.value_loc(site);
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

        for (site, chain) in deref_entries {
            let loc = program.value_loc(site);
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
                format!("implicit deref chain: {chain_types}"),
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

            self.print_report(report.finish())?;
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

fn clash_messages(_program: &Program, _store: &TypeStore, clash: &TypeClash) -> (String, String) {
    let found = clash.found().unwrap_or("unknown");
    let wanted = clash.wanted().unwrap_or("unknown");
    (format!("found {found}"), format!("expected {wanted}"))
}

fn operand_type_message(
    _program: &Program,
    _store: &TypeStore,
    label: &str,
    ty: Option<&str>,
) -> String {
    match ty {
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

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new()
    }
}
