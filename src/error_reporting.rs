use crate::parsing::{LexError, ParseError};
use crate::parsing::OTok;
use ariadne::{Label, Report, ReportKind, Source};
use std::collections::HashMap;
use std::io;

pub struct ErrorReporter {
    sources: HashMap<usize, String>,
}

impl ErrorReporter {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
        }
    }

    pub fn add_source(&mut self, file_id: usize, source: String) {
        self.sources.insert(file_id, source);
    }

    fn source(&self, file: usize) -> Option<Source<&str>> {
        self.sources
            .get(&file)
            .map(|s| Source::from(s.as_str()))
    }

    pub fn report_lex_error(&self, error: &LexError) -> io::Result<()> {
        let (loc, message, label) = match error {
            LexError::UnexpectedChar { ch, loc } => (
                loc,
                format!("unexpected character `{}`", ch),
                "this character is not valid here",
            ),
            LexError::UnterminatedString { loc } => (
                loc,
                "unterminated string literal".to_string(),
                "string starts here",
            ),
        };

        let Some(source) = self.source(loc.file) else {
            return Ok(());
        };

        let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
            .with_message(message)
            .with_label(
                Label::new((loc.file, loc.range.clone()))
                    .with_message(label),
            );

        report.finish().print((loc.file, source))
    }

    pub fn report_parse_error(&self, error: &ParseError) -> io::Result<()> {
        match error {
            ParseError::Lex(err) => {
                return self.report_lex_error(err);
            }

            ParseError::ExpectedExpr { got } => {
                self.report_expected(
                    "expected expression",
                    got,
                )
            }

            ParseError::ExpectedToken { expected, got } => {
                self.report_expected(
                    &format!("expected {}", expected),
                    got,
                )
            }

            ParseError::UnexpectedToken { got } => {
                let loc = &got.loc;
                let Some(source) = self.source(loc.file) else {
                    return Ok(());
                };

                let report = Report::build(
                    ReportKind::Error,
                    loc.file,
                    loc.range.start,
                )
                .with_message("unexpected token")
                .with_label(
                    Label::new((loc.file, loc.range.clone()))
                        .with_message(format!(
                            "`{}` is not valid here",
                            got.value
                        )),
                );

                report.finish().print((loc.file, source))
            }

            ParseError::OpenDelimiter { open, close, got } => {
                let open_loc = &open.loc;
                let Some(source) = self.source(open_loc.file) else {
                    return Ok(());
                };

                let mut report = Report::build(
                    ReportKind::Error,
                    open_loc.file,
                    open_loc.range.start,
                )
                .with_message(format!(
                    "unclosed `{}` delimiter",
                    open.value
                ))
                .with_label(
                    Label::new((open_loc.file, open_loc.range.clone()))
                        .with_message(format!(
                            "`{}` opened here",
                            open.value
                        )),
                );

                if let Some(tok) = got {
                    let loc = &tok.loc;
                    report = report.with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message(format!(
                                "expected `{}` before this",
                                close
                            )),
                    );
                } else {
                    // EOF: underline end-of-input span
                    let eof = open_loc.range.end;
                    report = report.with_label(
                        Label::new((open_loc.file, eof..eof))
                            .with_message(format!(
                                "expected `{}` before end of input",
                                close
                            )),
                    );
                }

                report.finish().print((open_loc.file, source))
            }
        }
    }

    fn report_expected(
        &self,
        message: &str,
        got: &OTok,
    ) -> io::Result<()> {
        let loc = &got.loc;
        let Some(source) = self.source(loc.file) else {
            return Ok(());
        };

        let label_msg = match &got.value {
            Some(tok) => format!("found `{}` here", tok),
            None => "unexpected end of input here".to_string(),
        };

        let report = Report::build(
            ReportKind::Error,
            loc.file,
            loc.range.start,
        )
        .with_message(message)
        .with_label(
            Label::new((loc.file, loc.range.clone()))
                .with_message(label_msg),
        );

        report.finish().print((loc.file, source))
    }
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new()
    }
}
