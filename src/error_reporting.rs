use crate::parsing::{OTok, ParseError};
use crate::program::CompileError;
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
        self.sources.get(&file).map(|s| Source::from(s.as_str()))
    }

    pub fn report_parse_error(&self, error: &ParseError) -> io::Result<()> {
        match error {
            ParseError::UnexpectedChar { ch, loc } => {
                let Some(source) = self.source(loc.file) else {
                    return Ok(());
                };

                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(format!("unexpected character `{}`", ch))
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("this character is not valid here"),
                    );

                report.finish().print((loc.file, source))
            }

            ParseError::UnterminatedString { loc } => {
                let Some(source) = self.source(loc.file) else {
                    return Ok(());
                };

                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message("unterminated string literal".to_string())
                    .with_label(
                        Label::new((loc.file, loc.range.clone()))
                            .with_message("string starts here"),
                    );

                report.finish().print((loc.file, source))
            }

            ParseError::ExpectedExpr { got } => self.report_expected("expected expression", got),

            ParseError::ExpectedToken { expected, got } => {
                self.report_expected(&format!("expected {}", expected), got)
            }

            ParseError::OpenDelimiter { open, close, got } => {
                let open_loc = &open.loc;
                let close_loc = &got.loc;

                let Some(source) = self.source(open_loc.file) else {
                    return Ok(());
                };

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

                report.finish().print((open_loc.file, source))
            }
        }
    }

    fn report_expected(&self, message: &str, got: &OTok) -> io::Result<()> {
        let loc = &got.loc;
        let Some(source) = self.source(loc.file) else {
            return Ok(());
        };

        let label_msg = match &got.value {
            Some(tok) => format!("found `{}` here", tok),
            None => "unexpected end of input here".to_string(),
        };

        let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
            .with_message(message)
            .with_label(Label::new((loc.file, loc.range.clone())).with_message(label_msg));

        report.finish().print((loc.file, source))
    }

    pub fn report_compile_error(&self, error: &CompileError) -> io::Result<()> {
        match error {
            CompileError::SimpleError { loc, .. } | CompileError::Arity { loc, .. } => {
                let Some(source) = self.source(loc.file) else {
                    return Ok(());
                };

                let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
                    .with_message(error.to_string())
                    .with_label(Label::new((loc.file, loc.range.clone())).with_message("here"));

                report.finish().print((loc.file, source))
            }
            CompileError::Parse(parse_error) => self.report_parse_error(parse_error),
        }
    }
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new()
    }
}
