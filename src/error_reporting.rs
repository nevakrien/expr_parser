use crate::parsing::{OTok, ParseError};
use crate::program::{CompileError, Program};
use crate::type_inference::{BadTypeId, TypeClash, TypeError, TypeStore};
use ariadne::{Cache, Color, Label, Report, ReportKind, Source};
use std::collections::HashMap;
use std::io;

const MAX_UNRESOLVED_NAME_LABELS: usize = 5;

pub struct ErrorReporter {
    sources: HashMap<usize, Source<String>>,
}

impl<'a> Cache<usize> for &'a ErrorReporter {
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

                let mut report =
                    Report::build(ReportKind::Error, primary.file, primary.range.start)
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
                existing,
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
            TypeError::ExpectedTypeExpr { type_expr } => {
                let loc = program.value_loc(*type_expr);
                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("expected a type expression")
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("this is not a type"),
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
                let (found_msg, expected_msg) = clash_messages(store, *clash);

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
            TypeError::AnnotationMismatch {
                annotation,
                constrained,
                clash,
            } => {
                let ann_loc = program.value_loc(*annotation);
                let constrained_loc = program.value_loc(*constrained);
                let (found_msg, expected_msg) = clash_messages(store, *clash);

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
                let (found_msg, expected_msg) = clash_messages(store, *clash);

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
        }
    }
}

fn clash_messages(store: &TypeStore, clash: TypeClash) -> (String, String) {
    let found = type_string(store, clash.found).unwrap_or_else(|| "unknown".to_string());
    let wanted = type_string(store, clash.wanted).unwrap_or_else(|| "unknown".to_string());
    (format!("found {found}"), format!("expected {wanted}"))
}

fn type_string(store: &TypeStore, ty: Option<BadTypeId>) -> Option<String> {
    ty.map(|t| store.get_bad_type_string(t))
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new()
    }
}
