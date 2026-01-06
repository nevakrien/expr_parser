use std::fmt;
use std::ops::{Deref, Range};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Loc {
    pub range: Range<usize>,
    pub file: usize,
    // pub macro_site:Option<Box<MacroCtx>>
}

// #[derive(Debug,Clone,PartialEq,Eq,Hash)]
// pub struct MacroCtx {
//     pub loc: Loc,
//     ///macros produce [e0 e1 e2 e3 ...]
//     ///this fields indicates which id we are
//     pub expr_num:usize,
// }

#[derive(Debug, Clone, PartialEq)]
pub struct Located<T> {
    pub loc: Loc,
    pub value: T,
}

impl<T> Located<T> {
    pub fn with<U>(&self, value: U) -> Located<U> {
        Located {
            loc: self.loc.clone(),
            value,
        }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Located<U> {
        Located {
            loc: self.loc,
            value: f(self.value),
        }
    }
}

impl<T> Deref for Located<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}

pub type LStr<'a> = Located<&'a str>;
pub type LString = Located<String>;

pub type LExpr = Located<Expr>;
pub type LTok = Located<Token>;

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Atom(Token),
    Combo(LStr<'static>, Vec<LExpr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    NumLit(u64),
    FloatLit(f64),
    StrLit(String),
    Ident(String),
    Operator(&'static str),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::NumLit(n) => write!(f, "{}", n),
            Token::FloatLit(x) => write!(f, "{}", x),
            Token::StrLit(s) => write!(f, "{:?}", s), // quoted + escaped
            Token::Ident(name) => write!(f, "{}", name),
            Token::Operator(op) => write!(f, "{}", op),
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum LexError {
    #[error("unexpected character `{ch}`")]
    UnexpectedChar { ch: char, loc: Loc },

    #[error("unterminated string literal")]
    UnterminatedString { loc: Loc },
}

pub const KEYWORDS: &[&str] = &[
    "let",
    "const",
    "if",
    "else",
    "while",
    "for",
    "return",
    "break",
    "continue",
    "type",
    "as",
    "fn",
    "cfn",
    "struct",  
    "union",  
    "enum",  
];

pub const OPERATORS: &[&str] = &[
    // --- assignment (longest first) ---
    "<<=", ">>=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", // --- comparisons ---
    "==", "!=", "<=", ">=", // --- shifts ---
    "<<", ">>", // --- logical ---
    "&&", "||", // -- increments --
    "++", "--", // --- bitwise ---
    "&", "|", "^", "~", // --- arrows / paths ---
    "->", "::", ".", // --- arithmetic ---
    "+", "-", "*", "/", "%", // --- comparison / unary ---
    "=", "<", ">", "!", // --- delimiters ---
    "(", ")", "{", "}", "[", "]", ",", ";", ":",
];

pub struct Lexer<'a> {
    src: &'a str,
    file: usize,
    pos: usize,
    peeked: Option<LTok>,
}
impl<'a> Lexer<'a> {
    pub fn new(src: &'a str, file: usize) -> Self {
        Self {
            src,
            file,
            pos: 0,
            peeked: None,
        }
    }

    fn empty_loc(&self) -> Loc {
        Loc {
            range: self.pos..self.pos,
            file: self.file,
        }
    }

    /* =============================
     * Low-level character handling
     * ============================= */

    fn peek_char(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn remaining(&self) -> &'a str {
        &self.src[self.pos..]
    }

    /* =============================
     * Public helpers
     * ============================= */

    pub fn mark(&self) -> usize {
        self.pos
    }

    pub fn produce_loc(&self, start: usize) -> Loc {
        Loc {
            range: start..self.pos,
            file: self.file,
        }
    }

    pub fn skip_whitespace(&mut self) {
        while matches!(self.peek_char(), Some(c) if c.is_whitespace()) {
            self.bump();
        }
    }

    /* =============================
     * Tokenization
     * ============================= */

    fn lex_token(&mut self) -> Result<Option<LTok>, LexError> {
        let start = self.pos;
        let ch = match self.bump() {
            Some(c) => c,
            None => return Ok(None),
        };

        let value = match ch {
            // -------- numbers --------
            '0'..='9' => {
                while matches!(self.peek_char(), Some(c) if c.is_ascii_digit()) {
                    self.bump();
                }

                if self.peek_char() == Some('.') {
                    self.bump();
                    while matches!(self.peek_char(), Some(c) if c.is_ascii_digit()) {
                        self.bump();
                    }
                    let text = &self.src[start..self.pos];
                    Token::FloatLit(text.parse().unwrap())
                } else {
                    let text = &self.src[start..self.pos];
                    Token::NumLit(text.parse().unwrap())
                }
            }

            // -------- strings --------
            '"' => {
                while let Some(c) = self.bump() {
                    if c == '"' {
                        let text = &self.src[start + 1..self.pos - 1];
                        return Ok(Some(Located {
                            loc: self.produce_loc(start),
                            value: Token::StrLit(text.to_string()),
                        }));
                    }
                }

                // EOF without closing quote
                return Err(LexError::UnterminatedString {
                    loc: self.produce_loc(start),
                });
            }

            // -------- identifiers / keywords --------
            c if c.is_ascii_alphabetic() || c == '_' => {
                while matches!(
                    self.peek_char(),
                    Some(c) if c.is_ascii_alphanumeric() || c == '_'
                ) {
                    self.bump();
                }

                let text = &self.src[start..self.pos];
                if let Some(op) = KEYWORDS.iter().copied().find(|x| *x == text) {
                    Token::Operator(op)
                } else {
                    Token::Ident(text.to_string())
                }
            }

            // -------- operators --------
            _ => {
                self.pos = start; // rewind and try operator matching
                if let Some(op) = Self::match_operator(self.remaining()) {
                    self.pos += op.len();
                    Token::Operator(op)
                } else {
                    let bad = self.bump().unwrap();
                    return Err(LexError::UnexpectedChar {
                        ch: bad,
                        loc: self.produce_loc(start),
                    });
                }
            }
        };

        Ok(Some(Located {
            loc: self.produce_loc(start),
            value,
        }))
    }

    fn match_operator(input: &str) -> Option<&'static str> {
        OPERATORS.iter().copied().find(|op| input.starts_with(op))
    }

    /* =============================
     * Peek / consume
     * ============================= */

    pub fn peek(&mut self) -> Result<Option<&LTok>, LexError> {
        if self.peeked.is_none() {
            let saved = self.pos;
            self.skip_whitespace();
            let tok = self.lex_token()?;
            self.pos = saved;
            self.peeked = tok;
        }
        Ok(self.peeked.as_ref())
    }

    pub fn next(&mut self) -> Result<Option<LTok>, LexError> {
        if let Some(tok) = self.peeked.take() {
            self.pos = tok.loc.range.end;
            return Ok(Some(tok));
        }

        self.skip_whitespace();
        self.lex_token()
    }

    /* =============================
     * Convenience helpers
     * ============================= */

    pub fn try_ident(&mut self) -> Result<Option<Located<String>>, LexError> {
        match self.peek()? {
            Some(tok) => match tok.value.clone() {
                Token::Ident(name) => {
                    let tok = self.next()?.unwrap();
                    Ok(Some(Located {
                        loc: tok.loc,
                        value: name,
                    }))
                }
                _ => Ok(None),
            },
            None => Ok(None),
        }
    }

    pub fn try_op(&mut self) -> Result<Option<LStr<'static>>, LexError> {
        let Some(tok) = self.peek()? else {
            return Ok(None);
        };

        let Token::Operator(s) = tok.value else {
            return Ok(None);
        };

        let ans = tok.with(s);
        self.next()?;
        Ok(Some(ans))
    }

    pub fn try_operator(&mut self, op: &str) -> Result<Option<LStr<'static>>, LexError> {
        let Some(tok) = self.peek()? else {
            return Ok(None);
        };

        let Token::Operator(s) = tok.value else {
            return Ok(None);
        };

        if op != s {
            return Ok(None);
        }

        let ans = tok.with(s);
        self.next()?;
        Ok(Some(ans))
    }

    // pub fn try_token(
    //     &mut self,
    //     pred: impl FnOnce(&Token) -> bool,
    // ) -> Result<Option<LTok>, LexError> {
    //     match self.peek()? {
    //         Some(tok) if pred(&tok.value) => Ok(self.next()?),
    //         _ => Ok(None),
    //     }
    // }
}

#[cfg(test)]
mod lex_tests {
    use super::*;

    /* =========================================
     * 1) peek() must not affect mark()
     * ========================================= */
    #[test]
    fn peek_does_not_advance_mark() {
        let mut lex = Lexer::new("foo bar", 0);

        let m0 = lex.mark();

        // Peek should not move position
        let p = lex.peek().unwrap().unwrap();
        assert!(matches!(p.value, Token::Ident(ref s) if s == "foo"));
        assert_eq!(p.loc.range, 0..3);

        let m1 = lex.mark();
        assert_eq!(m0, m1, "mark changed after peek");

        // Now consume and ensure span is correct
        let tok = lex.next().unwrap().unwrap();
        assert_eq!(tok.loc.range, 0..3);

        let loc = lex.produce_loc(m1);
        assert_eq!(loc.range, 0..3);

        // Still sane afterwards
        let bar = lex.next().unwrap().unwrap();
        assert!(matches!(bar.value, Token::Ident(ref s) if s == "bar"));
    }

    /* =========================================
     * 2) keywords are operators, operators are greedy
     * ========================================= */
    #[test]
    fn keywords_and_multi_char_operators() {
        let src = "let x == y && return";
        let mut lex = Lexer::new(src, 0);

        let kinds: Vec<String> = std::iter::from_fn(|| lex.next().unwrap())
            .map(|tok| match tok.value {
                Token::Operator(s) => format!("op({})", s),
                Token::Ident(s) => format!("id({})", s),
                _ => "other".to_string(),
            })
            .collect();

        assert_eq!(
            kinds,
            vec![
                "op(let)",
                "id(x)",
                "op(==)",
                "id(y)",
                "op(&&)",
                "op(return)",
            ]
        );
    }

    /* =========================================
     * 3) hard lexer errors
     * ========================================= */
    #[test]
    fn lexer_error_cases() {
        // Unterminated string
        let mut lex = Lexer::new("\"hello", 0);
        let err = lex.next().unwrap_err();
        assert!(matches!(err, LexError::UnterminatedString { .. }));

        // Invalid operator / unexpected char
        let mut lex = Lexer::new("@", 0);
        let err = lex.next().unwrap_err();
        assert!(matches!(err, LexError::UnexpectedChar { ch: '@', .. }));
    }
}

pub type PResult<T> = Result<T, ParseError>;
pub type OTok = Located<Option<Token>>;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ParseError {
    #[error(transparent)]
    Lex(#[from] LexError),

    // #[error("unexpected end of input")]
    // Eof,
    #[error("expected expression, got {got:?}")]
    ExpectedExpr { got: OTok },

    #[error("expected {expected}, got {got:?}")]
    ExpectedToken { expected: &'static str, got: OTok },

    // #[error("unexpected token {got:?}")]
    // UnexpectedToken { got: LTok },

    #[error("opened {open} without closing with {close} but got {got:?}")]
    OpenDelimiter {
        open: LStr<'static>,
        close: &'static str,
        got: OTok,
    },
}

const BP_ASSIGN: u32 = 100;
const BP_PATTERN: u32 = 110;
const BP_PATH: u32 = 850; // ., ->, ::
const BP_CALL: u32 = 800; // (), []
const BP_POSTFIX_INC: u32 = 875;
const BP_PREFIX: u32 = 900;

fn prefix_bp(op: &str) -> Option<u32> {
    Some(match op {
        "!" | "-" | "*" | "&" | "~" | "++" | "--" | "const" => BP_PREFIX,
        _ => return None,
    })
}

fn infix_bp(op: &str) -> Option<(u32, u32)> {
    Some(match op {
        // assignment (right-assoc)
        "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>=" => {
            (BP_ASSIGN + 1, BP_ASSIGN)
        }

        // cast / annotation-ish
        "as" => (200, 201),
        ":" => (210, 211),

        // logical
        "||" => (300, 301),
        "&&" => (310, 311),

        // bitwise
        "|" => (400, 401),
        "^" => (410, 411),
        "&" => (420, 421),

        // comparisons
        "==" | "!=" | "<" | ">" | "<=" | ">=" => (500, 501),

        // shifts
        "<<" | ">>" => (600, 601),

        // arithmetic
        "+" | "-" => (700, 701),
        "*" | "/" | "%" => (800, 801),

        "." | "::" | "->" => (BP_PATH, BP_PATH + 1),

        _ => return None,
    })
}

fn postfix_bp(op: &str) -> Option<u32> {
    Some(match op {
        "++" | "--" => BP_POSTFIX_INC,
        "(" | "[" => BP_CALL,
        _ => return None,
    })
}

// fn token_starts_expr(tok: &Token) -> bool {
//     match tok {
//         Token::NumLit(_) | Token::FloatLit(_) | Token::StrLit(_) | Token::Ident(_) => true,
//         Token::Operator(op) => {
//             prefix_bp(op).is_some()
//                 || matches!(
//                     *op,
//                     "(" | "{"
//                         | "if"
//                         | "while"
//                         | "let"
//                         | "type"
//                         | "return"
//                         | "break"
//                         | "continue"
//                         | "fn"
//                         | "struct"
//                         | "union"
//                         | "enum"
//                         | "cfn"
//                         | "const"
//                 )
//         }
//     }
// }

pub struct Parser<'a> {
    lex: Lexer<'a>,
}

impl<'a> Parser<'a> {
    pub fn is_empty(&mut self) -> bool {
        matches!(self.peek(), Ok(None))
    }
    /// Try parse an expression.
    /// Ok(None) => no expression starts here
    /// Err(_)   => expression started but failed
    pub fn try_expr(&mut self) -> PResult<Option<LExpr>> {
        self.try_expr_bp(0)
    }

    /// Must parse an expression or error.
    pub fn consume_expr(&mut self) -> PResult<LExpr> {
        match self.try_expr()? {
            Some(e) => Ok(e),
            None => Err(ParseError::ExpectedExpr {
                got: self.peek_op()?,
            }),
        }
    }

    /// Parse a statement: expr [';']
    pub fn parse_stmt(&mut self) -> PResult<Option<LExpr>> {
        let start = self.expr_start();

        let Some(expr) = self.try_expr()? else {
            return Ok(None);
        };

        if let Some(semi) = self.try_operator(";")? {
            let loc = self.produce_loc(start);
            return Ok(Some(Located {
                loc,
                value: Expr::Combo(semi, vec![expr]),
            }));
        }

        Ok(Some(expr))
    }

    /// Must parse an expression or error.
    pub fn consume_stmt(&mut self) -> PResult<LExpr> {
        match self.parse_stmt()? {
            Some(e) => Ok(e),
            None => Err(ParseError::ExpectedExpr {
                got: self.peek_op()?,
            }),
        }
    }
}
impl<'a> Parser<'a> {
    pub fn new(src: &'a str, file: usize) -> Self {
        Self {
            lex: Lexer::new(src, file),
        }
    }

    /* =============================
     * Forwarding + invariants
     * ============================= */

    fn expr_start(&mut self) -> usize {
        self.lex.skip_whitespace();
        self.lex.mark()
    }
    fn produce_loc(&self, start: usize) -> Loc {
        self.lex.produce_loc(start)
    }

    fn peek(&mut self) -> PResult<Option<&LTok>> {
        Ok(self.lex.peek()?)
    }

    fn peek_op(&mut self) -> PResult<OTok> {
        Ok(match self.peek()? {
            Some(t) => t.clone().map(Some),
            None => Located {
                loc: self.lex.empty_loc(),
                value: None,
            },
        })
    }

    fn next(&mut self) -> PResult<Option<LTok>> {
        let t = self.lex.next()?;
        Ok(t)
    }

    fn try_ident(&mut self) -> PResult<Option<LString>> {
        Ok(self.lex.try_ident()?)
    }
    fn try_op(&mut self) -> Result<Option<LStr<'static>>, LexError> {
        self.lex.try_op()
    }
    fn try_operator(&mut self, op: &str) -> PResult<Option<LStr<'static>>> {
        Ok(self.lex.try_operator(op)?)
    }

    fn expect_operator(&mut self, op: &'static str) -> PResult<LStr<'static>> {
        match self.try_operator(op)? {
            Some(t) => Ok(t),
            None => Err(ParseError::ExpectedToken {
                expected: op,
                got: self.peek_op()?,
            }),
        }
    }

    fn err_open_delim(&mut self, open: LStr<'static>, close: &'static str) -> ParseError {
        let got = match self.peek_op() {
            Ok(x) => x,
            Err(_) => Located {
                loc: self.lex.empty_loc(),
                value: None,
            },
        };
        ParseError::OpenDelimiter { open, close, got }
    }
}

impl<'a> Parser<'a> {
    fn try_expr_bp(&mut self, min_bp: u32) -> PResult<Option<LExpr>> {
        let Some(peek) = self.peek()? else {
            return Ok(None);
        };
        // if !token_starts_expr(&peek.value) {
        //     return Ok(None);
        // }

        let start = self.expr_start();
        let Some(mut lhs) = self.parse_prefix(start)? else {
            return Ok(None);
        };

        loop {
            // postfix first
            if self.try_parse_postfix(start, &mut lhs, min_bp)? {
                continue;
            }

            // then infix
            if self.try_parse_infix(start, &mut lhs, min_bp)? {
                continue;
            }

            break;
        }

        Ok(Some(lhs))
    }

    fn consume_expr_bp(&mut self, min_bp: u32) -> PResult<LExpr> {
        match self.try_expr_bp(min_bp)? {
            Some(e) => Ok(e),
            None => Err(ParseError::ExpectedExpr {
                got: self.peek_op()?,
            }),
        }
    }
}

impl<'a> Parser<'a> {
    fn try_parse_infix(&mut self, start: usize, lhs: &mut LExpr, min_bp: u32) -> PResult<bool> {
        let Some(peek) = self.peek()? else {
            return Ok(false);
        };
        let Token::Operator(op) = &peek.value else {
            return Ok(false);
        };

        let Some((l_bp, r_bp)) = infix_bp(op) else {
            return Ok(false);
        };

        if l_bp < min_bp {
            return Ok(false);
        }

        let op_tok = self.try_op()?.unwrap();
        let rhs = self.consume_expr_bp(r_bp)?;

        let loc = self.produce_loc(start);
        let mut temp = Located {
            loc,
            value: Expr::Combo(op_tok, Vec::new()),
        };
        std::mem::swap(lhs, &mut temp);

        if let Expr::Combo(_, ref mut v) = lhs.value {
            v.push(temp); // old lhs
            v.push(rhs);
        }

        Ok(true)
    }

    fn try_parse_postfix(&mut self, start: usize, lhs: &mut LExpr, min_bp: u32) -> PResult<bool> {
        let Some(peek) = self.peek()? else {
            return Ok(false);
        };
        let Token::Operator(op) = &peek.value else {
            return Ok(false);
        };
        let Some(bp) = postfix_bp(op) else {
            return Ok(false);
        };

        //check if we need to special case later
        let end_op = match *op {
            "(" => ")",
            "[" => "]",
            _ => "",
        };

        if bp < min_bp {
            return Ok(false);
        }

        let open = self.try_op()?.unwrap();

        //swap the new lhs into place
        let loc = self.produce_loc(start);
        let mut temp = Located {
            loc,
            value: Expr::Combo(open.clone(), Vec::new()),
        };
        std::mem::swap(lhs, &mut temp);

        //put the old lhs on in the new
        let Expr::Combo(_, ref mut v) = lhs.value else {
            unreachable!()
        };
        v.push(temp);

        //handle common case
        if end_op.is_empty() {
            lhs.loc = self.produce_loc(start);
            return Ok(true);
        }

        //handle arg lists
        if self.try_operator(end_op)?.is_none() {
            loop {
                let Some(exp) = self.try_expr()? else {
                    return Err(self.err_open_delim(open, end_op));
                };
                v.push(exp);

                if self.try_operator(",")?.is_some() {
                    continue;
                }

                if self.try_operator(end_op)?.is_some() {
                    break;
                }

                return Err(self.err_open_delim(open, end_op));
            }
        }

        lhs.loc = self.produce_loc(start);
        Ok(true)
    }
}


impl<'a> Parser<'a> {
    fn parse_prefix(&mut self, start: usize) -> PResult<Option<LExpr>> {
        let Some(tok) = self.peek()? else {
            return Ok(None);
        };

        match tok.value {
            Token::NumLit(_) | Token::FloatLit(_) | Token::StrLit(_) | Token::Ident(_) => {
                let tok = self.next()?.unwrap();
                Ok(Some(Located {
                    loc: self.produce_loc(start),
                    value: Expr::Atom(tok.value),
                }))
            }

            Token::Operator(op) => {
                let op_s = tok.with(op);

                // grouping / blocks
                if op == "(" {
                    self.next()?.unwrap();
                    return self.parse_after_lparen(start, op_s).map(Some);
                }
                if op == "{" {
                    self.next()?.unwrap();
                    return self.parse_after_lbrace(start, op_s).map(Some);
                }

                // control keywords
                if op == "if" {
                    self.next()?.unwrap();
                    return self.parse_after_if(start, op_s).map(Some);
                }
                if op == "while" {
                    self.next()?.unwrap();
                    return self.parse_after_while(start, op_s).map(Some);
                }

                if op == "fn" || op == "cfn" {
                    self.next()?.unwrap();
                    return self.parse_after_fn(start, op_s).map(Some);
                }

                if op == "let" {
                    self.next()?.unwrap();
                    return self.parse_after_let(start, op_s).map(Some);
                }

                // generic prefix operator via BP
                if let Some(bp) = prefix_bp(op) {
                    self.next()?.unwrap();
                    let rhs = self.consume_expr_bp(bp)?;
                    let loc = self.produce_loc(start);
                    return Ok(Some(Located {
                        loc,
                        value: Expr::Combo(op_s, vec![rhs]),
                    }));
                }
                Ok(None)
                // Err(ParseError::ExpectedToken { got: tok.map(Some),expected:"value or prefix operator" })
            }
        }
    }

    fn parse_after_lparen(&mut self, start: usize, open: LStr<'static>) -> PResult<LExpr> {
        let mut parts = Vec::new();
        loop{
            let Some(exp) = self.try_expr()? else {
                break;
            };
            parts.push(exp);
            if self.try_operator(",")?.is_none(){
                break;
            }
        }

        if self.try_operator(")")?.is_none() {
            return Err(self.err_open_delim(open, ")"));
        }

        let loc = self.produce_loc(start);
        Ok(Located {
            loc,
            value: Expr::Combo(open, parts),
        })
    }

    fn parse_after_lbrace(&mut self, start: usize, open: LStr<'static>) -> PResult<LExpr> {
        let mut items = Vec::new();

        loop {
            if self.try_operator("}")?.is_some() {
                break;
            }

            match self.parse_stmt()? {
                Some(s) => items.push(s),
                None => return Err(self.err_open_delim(open, "}")),
            }
        }

        let loc = self.produce_loc(start);
        Ok(Located {
            loc,
            value: Expr::Combo(open, items),
        })
    }

    fn parse_after_if(&mut self, start: usize, if_tok: LStr<'static>) -> PResult<LExpr> {
        let cond = self.consume_expr()?;
        let then_expr = self.consume_stmt()?;

        let mut args = vec![cond, then_expr];

        if self.try_operator("else")?.is_some() {
            args.push(self.consume_stmt()?);
        }

        let loc = self.produce_loc(start);
        Ok(Located {
            loc,
            value: Expr::Combo(if_tok, args),
        })
    }

    fn parse_after_while(&mut self, start: usize, w: LStr<'static>) -> PResult<LExpr> {
        let cond = self.consume_expr()?;
        let body = self.consume_stmt()?;

        let loc = self.produce_loc(start);
        Ok(Located {
            loc,
            value: Expr::Combo(w, vec![cond, body]),
        })
    }

    fn parse_after_fn(&mut self, start: usize, fn_tok: LStr<'static>) -> PResult<LExpr> {
        let paren_start = self.expr_start();
        let open = self.expect_operator("(")?;
        let mut params: Vec<LExpr> = Vec::new();

        if self.try_operator(")")?.is_none() {
            loop {
                let vd = self.consume_expr()?;
                params.push(vd);

                if self.try_operator(",")?.is_some() {
                    continue;
                }
                if self.try_operator(")")?.is_some() {
                    break;
                }

                return Err(self.err_open_delim(open.clone(), ")"));
            }
        }

        let mut sig = Located {
            loc: self.produce_loc(paren_start),
            value: Expr::Combo(open, params),
        };

        if let Some(arrow) = self.try_operator("->")? {
            let output = self.consume_expr()?;
            sig = Located {
                loc: self.produce_loc(paren_start),
                value: Expr::Combo(arrow, vec![sig, output]),
            }
        }

        let mut v = vec![sig];
        if let Some(body) = self.try_expr()? {
            v.push(body)
        }
        Ok(Located {
            loc: self.produce_loc(start),
            value: Expr::Combo(fn_tok, v),
        })
    }

    fn parse_after_let(&mut self, start: usize, let_tok: LStr<'static>) -> PResult<LExpr> {
        let dec = self.consume_expr_bp(BP_PATTERN)?;
        self.expect_operator("=")?;
        let val = self.consume_expr()?;

        Ok(Located {
            loc: self.produce_loc(start),
            value: Expr::Combo(let_tok, vec![dec, val]),
        })
    }
}
#[cfg(test)]
mod parse_tests {
    use super::*;

    /* =============================
     * Helper for span assertions
     * ============================= */

    fn assert_loc(loc: &Loc, start: usize, end: usize) {
        assert_eq!(loc.file, 0);
        assert_eq!(loc.range.start, start);
        assert_eq!(loc.range.end, end);
    }

    /* =============================
     * Call / delimiter errors
     * ============================= */

    #[test]
    fn missing_close_paren_reports_correct_span() {
        let src = "f(a, b";
        let mut p = Parser::new(src, 0);
        let err = p.consume_expr().unwrap_err();

        match err {
            ParseError::OpenDelimiter { open, close, got } => {
                assert_eq!(open.value, "(");
                assert_eq!(close, ")");
                assert!(got.is_none());

                // f ( a ,   b
                // 0 1 2 3 4 5 6
                assert_loc(&open.loc, 1, 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn garbage_in_arg_list_reports_open_paren_span() {
        let src = "f(a b)";
        let mut p = Parser::new(src, 0);
        let err = p.consume_expr().unwrap_err();

        match err {
            ParseError::OpenDelimiter { open, close, got } => {
                assert_eq!(open.value, "(");
                assert_eq!(close, ")");

                let got = got.as_ref().expect("should have got token");
                assert_eq!(*got, Token::Ident("b".into()));

                // span of '('
                assert_loc(&open.loc, 1, 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /* =============================
     * If / else semantics
     * ============================= */

    #[test]
    fn if_without_else_consumes_exact_body_span() {
        let src = "if x y z";
        let mut p = Parser::new(src, 0);

        let first = p.consume_stmt().expect("if stmt");
        let second = p.consume_stmt().expect("z stmt");

        match first.value {
            Expr::Combo(tok, _) => {
                assert_eq!(tok.value, "if");
                assert_loc(&first.loc, 0, 6); // "if x y"
            }
            _ => panic!("expected if"),
        }

        match second.value {
            Expr::Atom(Token::Ident(ref s)) if s == "z" => {
                assert_loc(&second.loc, 7, 8);
            }
            _ => panic!("expected z"),
        }
    }

    #[test]
    fn if_else_span_covers_entire_expression() {
        let src = "if x y else z";
        let mut p = Parser::new(src, 0);
        let expr = p.consume_expr().expect("if-else");

        match expr.value {
            Expr::Combo(tok, _) => {
                assert_eq!(tok.value, "if");
                assert_loc(&expr.loc, 0, src.len());
            }
            _ => panic!("expected if-else"),
        }
    }

    /* =============================
     * Semicolon semantics
     * ============================= */

    #[test]
    fn semicolon_is_postfix_on_statement() {
        let src = "x; y";
        let mut p = Parser::new(src, 0);

        let first = p.consume_stmt().expect("x; stmt");
        let second = p.consume_stmt().expect("y stmt");

        match first.value {
            Expr::Combo(tok, args) => {
                assert_eq!(tok.value, ";");
                assert_eq!(args.len(), 1);

                // span includes the semicolon
                assert_eq!(first.loc.range.start, 0);
                assert_eq!(first.loc.range.end, 2);
            }
            _ => panic!("expected semicolon postfix expression"),
        }

        match second.value {
            Expr::Atom(Token::Ident(ref s)) if s == "y" => {
                assert_eq!(second.loc.range.start, 3);
                assert_eq!(second.loc.range.end, 4);
            }
            _ => panic!("expected y"),
        }
    }

    #[test]
    fn double_semicolon_reports_correct_error_span() {
        let src = "x;;y";
        let mut p = Parser::new(src, 0);

        let _x = p.consume_stmt().unwrap();
        assert!(p.parse_stmt().unwrap().is_none())
    }

    /* =============================
     * Block delimiter errors
     * ============================= */

    #[test]
    fn missing_close_brace_reports_open_brace_span() {
        let src = "{ a b";
        let mut p = Parser::new(src, 0);
        let err = p.consume_expr().unwrap_err();

        match err {
            ParseError::OpenDelimiter { open, close, got } => {
                assert_eq!(open.value, "{");
                assert_eq!(close, "}");
                assert!(got.is_none());

                assert_loc(&open.loc, 0, 1);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn fn_multi_param_mixed_types() {
        let src = "fn(x: T, y) y";
        let mut p = Parser::new(src, 0);
        let expr = p.consume_expr().unwrap();

        match expr.value {
            Expr::Combo(_, args) => {
                let sig = &args[0];
                match &sig.value {
                    Expr::Combo(_, params) => {
                        assert_eq!(params.len(), 2);

                        // x : T
                        assert_loc(&params[0].loc, 3, 7);

                        // y
                        match &params[1].value {
                            Expr::Atom(Token::Ident(name)) => {
                                assert_eq!(name, "y");
                                assert_loc(&params[1].loc, 9, 10);
                            }
                            _ => panic!("expected y"),
                        }
                    }
                    _ => panic!("expected param list"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn fn_arrow_does_not_consume_body() {
        let src = "fn(x) -> T x + 1";
        let mut p = Parser::new(src, 0);
        let expr = p.consume_expr().unwrap();

        match expr.value {
            Expr::Combo(_, args) => {
                assert_eq!(args.len(), 2);

                // signature is arrow
                match &args[0].value {
                    Expr::Combo(op, _) => assert_eq!(op.value, "->"),
                    _ => panic!("expected arrow sig"),
                }

                // body is x + 1
                match &args[1].value {
                    Expr::Combo(op, _) => assert_eq!(op.value, "+"),
                    _ => panic!("expected body expression"),
                }
            }
            _ => panic!("expected fn"),
        }
    }

    #[test]
    fn let_with_pointer_type() {
        let src = "let x: *char = c";
        let mut p = Parser::new(src, 0);

        let expr = p.consume_expr().expect("let expr");

        match expr.value {
            Expr::Combo(let_tok, args) => {
                assert_eq!(let_tok.value, "let");
                assert_eq!(args.len(), 2);

                // ---- declaration ----
                match &args[0].value {
                    Expr::Combo(colon, parts) => {
                        assert_eq!(colon.value, ":");
                        assert_eq!(parts.len(), 2);

                        // x
                        match &parts[0].value {
                            Expr::Atom(Token::Ident(name)) => assert_eq!(name, "x"),
                            _ => panic!("expected variable name"),
                        }

                        // *char
                        match &parts[1].value {
                            Expr::Combo(star, inner) => {
                                assert_eq!(star.value, "*");
                                assert_eq!(inner.len(), 1);
                                match &inner[0].value {
                                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "char"),
                                    _ => panic!("expected char"),
                                }
                            }
                            _ => panic!("expected pointer type"),
                        }
                    }
                    _ => panic!("expected typed declaration"),
                }

                // ---- value ----
                match &args[1].value {
                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "c"),
                    _ => panic!("expected initializer"),
                }

                assert_eq!(expr.loc.range, 0..src.len());
            }
            _ => panic!("expected let expression"),
        }
    }

    #[test]
    fn index_with_2_ranges() {
        let src = "x[0:2,1:2]";
        let mut p = Parser::new(src, 0);

        let expr = p.consume_expr().expect("index expr");

        match expr.value {
            Expr::Combo(open, args) => {
                assert_eq!(open.value, "[");
                assert_eq!(args.len(), 3);

                // ---- base expression ----
                match &args[0].value {
                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "x"),
                    _ => panic!("expected base identifier"),
                }

                // ---- first slice: 0:2 ----
                match &args[1].value {
                    Expr::Combo(colon, parts) => {
                        assert_eq!(colon.value, ":");
                        assert_eq!(parts.len(), 2);

                        match &parts[0].value {
                            Expr::Atom(Token::NumLit(0)) => {}
                            _ => panic!("expected 0"),
                        }
                        match &parts[1].value {
                            Expr::Atom(Token::NumLit(2)) => {}
                            _ => panic!("expected 2"),
                        }
                    }
                    _ => panic!("expected colon expression"),
                }

                // ---- second slice: 1:2 ----
                match &args[2].value {
                    Expr::Combo(colon, parts) => {
                        assert_eq!(colon.value, ":");
                        assert_eq!(parts.len(), 2);

                        match &parts[0].value {
                            Expr::Atom(Token::NumLit(1)) => {}
                            _ => panic!("expected 1"),
                        }
                        match &parts[1].value {
                            Expr::Atom(Token::NumLit(2)) => {}
                            _ => panic!("expected 2"),
                        }
                    }
                    _ => panic!("expected colon expression"),
                }

                // full span
                assert_eq!(expr.loc.range, 0..src.len());
            }
            _ => panic!("expected indexing expression"),
        }
    }

    #[test]
    fn unclosed_paren_at_eof_reports_open_delimiter() {
        let src = "( a";
        let mut p = Parser::new(src, 0);
        let err = p.consume_expr().unwrap_err();

        match err {
            ParseError::OpenDelimiter { open, close, got } => {
                assert_eq!(open.value, "(");
                assert_eq!(close, ")");

                // EOF is represented as Located { value: None }
                assert!(got.value.is_none());

                // ( a
                // 0 1 2
                assert_loc(&open.loc, 0, 1);

                // EOF location is well-formed
                assert_eq!(got.loc.file, 0);
                assert_eq!(got.loc.range.start, src.len());
                assert_eq!(got.loc.range.end, src.len());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn block_parse_right() {
        let src = "f((x)) { a; b } g";
        let mut p = Parser::new(src, 0);

        // ---- first expression: f((x)) ----
        let first = p.consume_stmt().unwrap();

        match first.value {
            Expr::Combo(open, args) => {
                assert_eq!(open.value, "(");
                assert_eq!(args.len(), 2);

                // f
                match &args[0].value {
                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "f"),
                    _ => panic!("expected identifier f"),
                }

                // (x)
                match &args[1].value {
                    Expr::Combo(inner_open, inner_args) => {
                        assert_eq!(inner_open.value, "(");
                        assert_eq!(inner_args.len(), 1);

                        match &inner_args[0].value {
                            Expr::Atom(Token::Ident(name)) => assert_eq!(name, "x"),
                            _ => panic!("expected identifier x"),
                        }
                    }
                    _ => panic!("expected parenthesized expression"),
                }
            }
            _ => panic!("expected call expression"),
        }

        // ---- second expression: { a; b } ----
        let block = p.consume_stmt().unwrap();

        match block.value {
            Expr::Combo(open, items) => {
                assert_eq!(open.value, "{");
                assert_eq!(items.len(), 2);

                // a;
                match &items[0].value {
                    Expr::Combo(tok, args) => {
                        assert_eq!(tok.value, ";");
                        assert_eq!(args.len(), 1);
                    }
                    _ => panic!("expected semicolon expression"),
                }

                // b
                match &items[1].value {
                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "b"),
                    _ => panic!("expected identifier b"),
                }
            }
            _ => panic!("expected block expression"),
        }

        // ---- third expression: g ----
        let last = p.consume_stmt().unwrap();

        match last.value {
            Expr::Atom(Token::Ident(name)) => assert_eq!(name, "g"),
            _ => panic!("expected identifier g"),
        }

        // ---- no trailing input ----
        assert!(p.is_empty());
    }

}
