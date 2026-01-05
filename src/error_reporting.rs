use crate::parsing::{LexError, Loc, ParseError};
use ariadne::{Label, Report, ReportKind, Source};
use std::collections::HashMap;

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

    pub fn report_lex_error(&self, error: &LexError) {
        let (loc, message) = match error {
            LexError::UnexpectedChar { ch, loc } => (loc, format!("unexpected character `{}`", ch)),
            LexError::UnterminatedString { loc } => {
                (loc, "unterminated string literal".to_string())
            }
        };

        let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
            .with_message(message)
            .with_label(
                Label::new((loc.file, loc.range.clone())).with_message("error occurred here"),
            );

        if let Some(source) = self.sources.get(&loc.file) {
            report
                .finish()
                .print((loc.file, Source::from(source.as_str())))
                .unwrap();
        }
    }

    pub fn report_parse_error(&self, error: &ParseError) {
        match error {
            ParseError::Lex(lex_err) => {
                self.report_lex_error(lex_err);
                return;
            }
            ParseError::Eof => {
                eprintln!("unexpected end of input");
                return;
            }
            _ => {}
        }

        let (loc, message) = match error {
            ParseError::ExpectedExpr { got } => {
                let loc = got.as_ref().map(|t| &t.loc).unwrap_or(&Loc {
                    range: 0..0,
                    file: 0,
                });
                (loc, "expected expression".to_string())
            }
            ParseError::ExpectedToken { expected, got } => {
                let loc = got.as_ref().map(|t| &t.loc).unwrap_or(&Loc {
                    range: 0..0,
                    file: 0,
                });
                (loc, format!("expected {}", expected))
            }
            ParseError::UnexpectedToken { got } => {
                (&got.loc, format!("unexpected token {:?}", got.value))
            }
            ParseError::OpenDelimiter { open, close, .. } => (
                &open.loc,
                format!("unclosed {}, missing {}", open.value, close),
            ),
            _ => unreachable!(),
        };

        let report = Report::build(ReportKind::Error, loc.file, loc.range.start)
            .with_message(&message)
            .with_label(
                Label::new((loc.file, loc.range.clone())).with_message("error occurred here"),
            );

        if let Some(source) = self.sources.get(&loc.file) {
            report
                .finish()
                .print((loc.file, Source::from(source.as_str())))
                .unwrap();
        }
    }
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new()
    }
}
