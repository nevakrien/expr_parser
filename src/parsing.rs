use std::fmt;
use std::ops::{Deref, Range};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Loc {
    pub range: Range<usize>,
    pub file: usize,
}

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

pub type LExpr = Located<Expr>;
pub type LTok = Located<Token>;

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Atom(Token),
    Bin(LStr<'static>, Box<(LExpr, LExpr)>),
    Prefix(LStr<'static>, Vec<LExpr>),
    Postfix(LStr<'static>, Vec<LExpr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    NumLit(u64),
    FloatLit(f64),
    StrLit(String),
    Ident(String),
    Operator(FixedToken),
}

impl Token {
    pub const fn new_operator(s: &str) -> Self {
        Token::Operator(FixedToken::new(s))
    }
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

// Keywords are treated like operators during lexing.
pub const KEYWORDS: &[&str] = &[
    "let", "const", "type", "struct", "union", "enum", "fn", "cfn", "if", "else", "while", "for",
    "match", "return", "break", "continue", "as",
];

///greedy match
pub const OPERATORS: &[&str] = &[
    // --- 3-char operators ---
    "<<=", ">>=", // --- 2-char operators ---
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "|>", "==", "!=", "<=", ">=", "~=", "=>", "<<",
    ">>", "&&", "||", "++", "--", "->", "::", // --- 1-char operators ---
    "&", "|", "^", "~", "+", "-", "*", "/", "%", "=", "<", ">", "!", ".",
    // --- delimiters ---
    "(", ")", "{", "}", "[", "]", ",", ";", ":",
];

#[repr(C, align(8))]
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Pack8(pub [u8; 8]);


#[inline(always)]
pub const fn pack8(s: &str) -> Pack8 {
    let b = s.as_bytes();

    // Too long => sentinel that can't match any ASCII keyword
    if b.len() > 8 {
        return Pack8([0; 8]);
    }

    let mut out = [0u8; 8];
    let mut i = 0usize;
    while i < b.len() {
        out[i] = b[i];
        i += 1;
    }

    Pack8(out)
}


// ==================== Fixed tokens ====================

#[repr(u64)]
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum FixedToken {
    // ====================
    // Keywords
    // ====================
    Let      = u64::from_le_bytes(pack8("let").0),
    Const    = u64::from_le_bytes(pack8("const").0),
    Type     = u64::from_le_bytes(pack8("type").0),
    Struct   = u64::from_le_bytes(pack8("struct").0),
    Union    = u64::from_le_bytes(pack8("union").0),
    Enum     = u64::from_le_bytes(pack8("enum").0),
    Fn       = u64::from_le_bytes(pack8("fn").0),
    Cfn      = u64::from_le_bytes(pack8("cfn").0),
    If       = u64::from_le_bytes(pack8("if").0),
    Else     = u64::from_le_bytes(pack8("else").0),
    While    = u64::from_le_bytes(pack8("while").0),
    For      = u64::from_le_bytes(pack8("for").0),
    Match    = u64::from_le_bytes(pack8("match").0),
    Return   = u64::from_le_bytes(pack8("return").0),
    Break    = u64::from_le_bytes(pack8("break").0),
    Continue = u64::from_le_bytes(pack8("continue").0),
    As       = u64::from_le_bytes(pack8("as").0),

    // ====================
    // 3-char operators
    // ====================
    ShlEq = u64::from_le_bytes(pack8("<<=").0),
    ShrEq = u64::from_le_bytes(pack8(">>=").0),

    // ====================
    // 2-char operators
    // ====================
    AddEq   = u64::from_le_bytes(pack8("+=").0),
    SubEq   = u64::from_le_bytes(pack8("-=").0),
    MulEq   = u64::from_le_bytes(pack8("*=").0),
    DivEq   = u64::from_le_bytes(pack8("/=").0),
    ModEq   = u64::from_le_bytes(pack8("%=").0),
    AndEq   = u64::from_le_bytes(pack8("&=").0),
    OrEq    = u64::from_le_bytes(pack8("|=").0),
    XorEq   = u64::from_le_bytes(pack8("^=").0),
    PipeFwd = u64::from_le_bytes(pack8("|>").0),
    EqEq    = u64::from_le_bytes(pack8("==").0),
    Ne      = u64::from_le_bytes(pack8("!=").0),
    Le      = u64::from_le_bytes(pack8("<=").0),
    Ge      = u64::from_le_bytes(pack8(">=").0),
    TildeEq = u64::from_le_bytes(pack8("~=").0),
    FatArrow= u64::from_le_bytes(pack8("=>").0),
    Shl     = u64::from_le_bytes(pack8("<<").0),
    Shr     = u64::from_le_bytes(pack8(">>").0),
    AndAnd  = u64::from_le_bytes(pack8("&&").0),
    OrOr    = u64::from_le_bytes(pack8("||").0),
    Inc     = u64::from_le_bytes(pack8("++").0),
    Dec     = u64::from_le_bytes(pack8("--").0),
    Arrow   = u64::from_le_bytes(pack8("->").0),
    Path    = u64::from_le_bytes(pack8("::").0),

    // ====================
    // 1-char operators
    // ====================
    And     = u64::from_le_bytes(pack8("&").0),
    Or      = u64::from_le_bytes(pack8("|").0),
    Xor     = u64::from_le_bytes(pack8("^").0),
    Tilde   = u64::from_le_bytes(pack8("~").0),
    Add     = u64::from_le_bytes(pack8("+").0),
    Sub     = u64::from_le_bytes(pack8("-").0),
    Mul     = u64::from_le_bytes(pack8("*").0),
    Div     = u64::from_le_bytes(pack8("/").0),
    Mod     = u64::from_le_bytes(pack8("%").0),
    Assign = u64::from_le_bytes(pack8("=").0),
    Lt      = u64::from_le_bytes(pack8("<").0),
    Gt      = u64::from_le_bytes(pack8(">").0),
    Not     = u64::from_le_bytes(pack8("!").0),
    Dot     = u64::from_le_bytes(pack8(".").0),

    // ====================
    // Delimiters
    // ====================
    LParen   = u64::from_le_bytes(pack8("(").0),
    RParen   = u64::from_le_bytes(pack8(")").0),
    LBrace   = u64::from_le_bytes(pack8("{").0),
    RBrace   = u64::from_le_bytes(pack8("}").0),
    LBracket = u64::from_le_bytes(pack8("[").0),
    RBracket = u64::from_le_bytes(pack8("]").0),
    Comma    = u64::from_le_bytes(pack8(",").0),
    Semi     = u64::from_le_bytes(pack8(";").0),
    Colon    = u64::from_le_bytes(pack8(":").0),
}

impl FixedToken {
    pub const fn try_new(s: &str) -> Option<Self> {
        if let Some(w) = match_keyword(s) {
            return Some(w);
        }

        match match_operator(s) {
            Some(w) => {
                if w.as_str().len() == s.len() {
                    Some(w)
                } else {
                    None
                }
            }
            None => None,
        }
    }
    pub const fn new(s: &str) -> Self {
        match Self::try_new(s) {
            Some(x) => x,
            None => panic!("bad operator"),
        }
    }
    #[inline(always)]
    pub const fn as_str(self) -> &'static str {
        match self {
            FixedToken::Let => "let",
            FixedToken::Const => "const",
            FixedToken::Type => "type",
            FixedToken::Struct => "struct",
            FixedToken::Union => "union",
            FixedToken::Enum => "enum",
            FixedToken::Fn => "fn",
            FixedToken::Cfn => "cfn",
            FixedToken::If => "if",
            FixedToken::Else => "else",
            FixedToken::While => "while",
            FixedToken::For => "for",
            FixedToken::Match => "match",
            FixedToken::Return => "return",
            FixedToken::Break => "break",
            FixedToken::Continue => "continue",
            FixedToken::As => "as",

            FixedToken::ShlEq => "<<=",
            FixedToken::ShrEq => ">>=",
            FixedToken::AddEq => "+=",
            FixedToken::SubEq => "-=",
            FixedToken::MulEq => "*=",
            FixedToken::DivEq => "/=",
            FixedToken::ModEq => "%=",
            FixedToken::AndEq => "&=",
            FixedToken::OrEq => "|=",
            FixedToken::XorEq => "^=",
            FixedToken::PipeFwd => "|>",
            FixedToken::EqEq => "==",
            FixedToken::Ne => "!=",
            FixedToken::Le => "<=",
            FixedToken::Ge => ">=",
            FixedToken::TildeEq => "~=",
            FixedToken::FatArrow => "=>",
            FixedToken::Shl => "<<",
            FixedToken::Shr => ">>",
            FixedToken::AndAnd => "&&",
            FixedToken::OrOr => "||",
            FixedToken::Inc => "++",
            FixedToken::Dec => "--",
            FixedToken::Arrow => "->",
            FixedToken::Path => "::",

            FixedToken::And => "&",
            FixedToken::Or => "|",
            FixedToken::Xor => "^",
            FixedToken::Tilde => "~",
            FixedToken::Add => "+",
            FixedToken::Sub => "-",
            FixedToken::Mul => "*",
            FixedToken::Div => "/",
            FixedToken::Mod => "%",
            FixedToken::Assign => "=",
            FixedToken::Lt => "<",
            FixedToken::Gt => ">",
            FixedToken::Not => "!",
            FixedToken::Dot => ".",

            FixedToken::LParen => "(",
            FixedToken::RParen => ")",
            FixedToken::LBrace => "{",
            FixedToken::RBrace => "}",
            FixedToken::LBracket => "[",
            FixedToken::RBracket => "]",
            FixedToken::Comma => ",",
            FixedToken::Semi => ";",
            FixedToken::Colon => ":",
        }
    }
}

impl TryFrom<&str> for FixedToken{

type Error = ();
fn try_from(s: &str) -> Result<Self, ()> {
    if let Some(w) = match_keyword(s){
        return Ok(w);
    }

    let Some(w) = match_operator(s) else {
        return Err(())
    };

    if w.as_str()==s{
        Ok(w)
    }else{
        Err(())
    }
}
}

impl core::fmt::Display for FixedToken {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str((*self).as_str())
    }
}
impl core::fmt::Debug for FixedToken {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str((*self).as_str())
    }
}

#[inline(always)]
const fn match_operator(input: &str) -> Option<FixedToken> {
    let b = input.as_bytes();
    let len = b.len();
    if len == 0 {
        return None;
    }
    let b0 = b[0];
    let b1 = if len > 1 { b[1] } else { 0 };
    let b2 = if len > 2 { b[2] } else { 0 };

    match (b0, b1, b2) {
        // common arithmetic first
        (b'+', b'=', _) => Some(FixedToken::AddEq),
        (b'+', b'+', _) => Some(FixedToken::Inc),
        (b'+', _, _) => Some(FixedToken::Add),

        (b'-', b'=', _) => Some(FixedToken::SubEq),
        (b'-', b'-', _) => Some(FixedToken::Dec),
        (b'-', b'>', _) => Some(FixedToken::Arrow),
        (b'-', _, _) => Some(FixedToken::Sub),

        (b'*', b'=', _) => Some(FixedToken::MulEq),
        (b'*', _, _) => Some(FixedToken::Mul),

        (b'/', b'=', _) => Some(FixedToken::DivEq),
        (b'/', _, _) => Some(FixedToken::Div),

        (b'&', b'=', _) => Some(FixedToken::AndEq),
        (b'&', b'&', _) => Some(FixedToken::AndAnd),
        (b'&', _, _) => Some(FixedToken::And),

        (b'|', b'=', _) => Some(FixedToken::OrEq),
        (b'|', b'>', _) => Some(FixedToken::PipeFwd),
        (b'|', b'|', _) => Some(FixedToken::OrOr),
        (b'|', _, _) => Some(FixedToken::Or),

        // comparisons / assignment
        (b'=', b'=', _) => Some(FixedToken::EqEq),
        (b'=', b'>', _) => Some(FixedToken::FatArrow),
        (b'=', _, _) => Some(FixedToken::Assign),

        (b'!', b'=', _) => Some(FixedToken::Ne),
        (b'!', _, _) => Some(FixedToken::Not),

        (b'<', b'=', _) => Some(FixedToken::Le),
        (b'>', b'=', _) => Some(FixedToken::Ge),

        // shifts (greedy via 3-byte checks)
        (b'<', b'<', b'=') => Some(FixedToken::ShlEq),
        (b'>', b'>', b'=') => Some(FixedToken::ShrEq),
        (b'<', b'<', _) => Some(FixedToken::Shl),
        (b'>', b'>', _) => Some(FixedToken::Shr),
        (b'<', _, _) => Some(FixedToken::Lt),
        (b'>', _, _) => Some(FixedToken::Gt),

        // rest + delimiters
        (b'^', b'=', _) => Some(FixedToken::XorEq),
        (b'^', _, _) => Some(FixedToken::Xor),
        (b'%', b'=', _) => Some(FixedToken::ModEq),
        (b'%', _, _) => Some(FixedToken::Mod),
        (b'~', b'=', _) => Some(FixedToken::TildeEq),
        (b'~', _, _) => Some(FixedToken::Tilde),
        (b':', b':', _) => Some(FixedToken::Path),
        (b':', _, _) => Some(FixedToken::Colon),
        (b'.', _, _) => Some(FixedToken::Dot),

        (b'(', _, _) => Some(FixedToken::LParen),
        (b')', _, _) => Some(FixedToken::RParen),
        (b'{', _, _) => Some(FixedToken::LBrace),
        (b'}', _, _) => Some(FixedToken::RBrace),
        (b'[', _, _) => Some(FixedToken::LBracket),
        (b']', _, _) => Some(FixedToken::RBracket),
        (b',', _, _) => Some(FixedToken::Comma),
        (b';', _, _) => Some(FixedToken::Semi),

        _ => None,
    }
}

#[inline(always)]
const fn match_keyword(input: &str) -> Option<FixedToken> {
    const K_LET: Pack8      = pack8("let");
    const K_CONST: Pack8    = pack8("const");
    const K_TYPE: Pack8     = pack8("type");
    const K_STRUCT: Pack8   = pack8("struct");
    const K_UNION: Pack8    = pack8("union");
    const K_ENUM: Pack8     = pack8("enum");
    const K_FN: Pack8       = pack8("fn");
    const K_CFN: Pack8      = pack8("cfn");
    const K_IF: Pack8       = pack8("if");
    const K_ELSE: Pack8     = pack8("else");
    const K_WHILE: Pack8    = pack8("while");
    const K_FOR: Pack8      = pack8("for");
    const K_MATCH: Pack8    = pack8("match");
    const K_RETURN: Pack8   = pack8("return");
    const K_BREAK: Pack8    = pack8("break");
    const K_CONTINUE: Pack8 = pack8("continue");
    const K_AS: Pack8       = pack8("as");

    let k = pack8(input);

    match k {
        K_LET      => Some(FixedToken::Let),
        K_IF       => Some(FixedToken::If),
        K_ELSE     => Some(FixedToken::Else),
        K_WHILE    => Some(FixedToken::While),
        K_FOR      => Some(FixedToken::For),
        K_FN       => Some(FixedToken::Fn),
        K_RETURN   => Some(FixedToken::Return),
        K_MATCH    => Some(FixedToken::Match),
        K_BREAK    => Some(FixedToken::Break),
        K_CONTINUE => Some(FixedToken::Continue),
        K_CONST    => Some(FixedToken::Const),
        K_TYPE     => Some(FixedToken::Type),
        K_STRUCT   => Some(FixedToken::Struct),
        K_UNION    => Some(FixedToken::Union),
        K_ENUM     => Some(FixedToken::Enum),
        K_CFN      => Some(FixedToken::Cfn),
        K_AS       => Some(FixedToken::As),
        _ => None,
    }
}



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
     * Low-level byte handling
     * ============================= */

    #[inline]
    fn bytes(&self) -> &'a [u8] {
        self.src.as_bytes()
    }

    #[inline]
    fn peek_byte(&self) -> Option<u8> {
        self.bytes().get(self.pos).copied()
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

    #[cold]
    fn skip_unicode_whitespace_slow(&mut self) {
        let bytes = self.src.as_bytes();

        loop {
            let rest = &self.src[self.pos..];
            let Some(ch) = rest.chars().next() else { return; };

            if !ch.is_whitespace() {
                return;
            }

            self.pos += ch.len_utf8();

            // After consuming Unicode whitespace, gobble ASCII whitespace cheaply.
            while let Some(&b) = bytes.get(self.pos) {
                if b.is_ascii_whitespace() {
                    self.pos += 1;
                } else {
                    break;
                }
            }

            // If next byte is ASCII, it's definitely not Unicode whitespace
            // (and also not ASCII whitespace because we just consumed it).
            let Some(&b0) = bytes.get(self.pos) else { return; };
            if b0.is_ascii() {
                return;
            }
        }
    }

    /// Fast path: ASCII whitespace. Slow path: Unicode whitespace (`char::is_whitespace()`).
    #[inline]
    pub fn skip_whitespace(&mut self) {
        let bytes = self.src.as_bytes();

        // ---- fast ASCII whitespace loop ----
        while let Some(&b) = bytes.get(self.pos) {
            if b.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }

        // If EOF or next byte is ASCII, we're done.
        let Some(&b0) = bytes.get(self.pos) else { return; };
        if b0.is_ascii() {
            return;
        }

        // Otherwise, only now pay for UTF-8 decoding.
        self.skip_unicode_whitespace_slow();
    }

    /* =============================
     * Tokenization helpers
     * ============================= */
    #[inline(always)]
    fn lex_number(&mut self, start: usize) -> Token {
        // first digit already consumed by caller
        while matches!(self.peek_byte(), Some(b) if b.is_ascii_digit()) {
            self.pos += 1;
        }

        if self.peek_byte() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek_byte(), Some(b) if b.is_ascii_digit()) {
                self.pos += 1;
            }
            let text = &self.src[start..self.pos];
            Token::FloatLit(text.parse().unwrap())
        } else {
            let text = &self.src[start..self.pos];
            Token::NumLit(text.parse().unwrap())
        }
    }

    #[inline]
    fn is_ident_start(c: char) -> bool {
        c == '_' || c.is_alphabetic()
    }

    #[inline]
    fn is_ident_continue(c: char) -> bool {
        c == '_' || c.is_alphanumeric()
    }

    #[inline(always)]
    fn lex_string(&mut self, start: usize) -> Result<Token, LexError> {
        // opening quote already consumed by caller
        let bytes = self.src.as_bytes();
        let mut chunk_start = self.pos;
        let mut out: Option<String> = None;

        loop {
            let Some(&b) = bytes.get(self.pos) else {
                return Err(LexError::UnterminatedString {
                    loc: self.produce_loc(start),
                });
            };

            match b {
                b'"' => {
                    // end of string
                    let end = self.pos;
                    self.pos += 1; // consume closing quote

                    if let Some(mut s) = out {
                        s.push_str(&self.src[chunk_start..end]);
                        return Ok(Token::StrLit(s));
                    } else {
                        return Ok(Token::StrLit(self.src[chunk_start..end].to_string()));
                    }
                }

                b'\\' => {
                    if out.is_none() {
                        out = Some(String::new());
                    }
                    let s = out.as_mut().unwrap();

                    // push pending chunk before the backslash
                    s.push_str(&self.src[chunk_start..self.pos]);

                    self.pos += 1; // consume '\\'
                    let Some(&esc) = bytes.get(self.pos) else {
                        return Err(LexError::UnterminatedString {
                            loc: self.produce_loc(start),
                        });
                    };

                    match esc {
                        b'"' => { s.push('"'); self.pos += 1; }
                        b'\\' => { s.push('\\'); self.pos += 1; }
                        b'n' => { s.push('\n'); self.pos += 1; }
                        b'r' => { s.push('\r'); self.pos += 1; }
                        b't' => { s.push('\t'); self.pos += 1; }
                        b'0' => { s.push('\0'); self.pos += 1; }

                        // Unicode escape: \u{HEX...}
                        b'u' => {
                            self.pos += 1; // consume 'u'
                            if bytes.get(self.pos) != Some(&b'{') {
                                let bad = self.src[self.pos..].chars().next().unwrap();
                                self.pos += bad.len_utf8();
                                return Err(LexError::UnexpectedChar {
                                    ch: bad,
                                    loc: self.produce_loc(start),
                                });
                            }
                            self.pos += 1; // consume '{'

                            let hex_start = self.pos;
                            while let Some(&hb) = bytes.get(self.pos) {
                                if hb == b'}' {
                                    break;
                                }
                                if !(hb as char).is_ascii_hexdigit() {
                                    let bad = self.src[self.pos..].chars().next().unwrap();
                                    self.pos += bad.len_utf8();
                                    return Err(LexError::UnexpectedChar {
                                        ch: bad,
                                        loc: self.produce_loc(start),
                                    });
                                }
                                self.pos += 1;
                            }

                            if bytes.get(self.pos) != Some(&b'}') {
                                return Err(LexError::UnterminatedString {
                                    loc: self.produce_loc(start),
                                });
                            }

                            let hex = &self.src[hex_start..self.pos];
                            self.pos += 1; // consume '}'

                            let code = u32::from_str_radix(hex, 16).unwrap();
                            let ch = char::from_u32(code).ok_or_else(|| LexError::UnexpectedChar {
                                ch: '\u{FFFD}',
                                loc: self.produce_loc(start),
                            })?;
                            s.push(ch);
                        }

                        _ => {
                            let bad = self.src[self.pos..].chars().next().unwrap();
                            self.pos += bad.len_utf8();
                            return Err(LexError::UnexpectedChar {
                                ch: bad,
                                loc: self.produce_loc(start),
                            });
                        }
                    }

                    chunk_start = self.pos;
                }

                _ => {
                    // Advance by one byte. Safe: bytes for '"' (0x22) and '\\' (0x5C)
                    // never appear inside UTF-8 continuation bytes (0x80..=0xBF).
                    self.pos += 1;
                }
            }
        }
    }

    #[inline(always)]
    fn lex_ident_or_keyword(&mut self, start: usize, saw_non_ascii: bool) -> Token {
        let bytes = self.src.as_bytes();

        // ASCII fast path continues as long as we stay ASCII.
        while let Some(&b) = bytes.get(self.pos) {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.pos += 1;
                continue;
            }
            break;
        }

        // If the identifier began with non-ASCII, or if the next byte is non-ASCII,
        // we must switch to Unicode scanning.
        if saw_non_ascii || matches!(bytes.get(self.pos), Some(b) if !b.is_ascii()) {
            loop {
                let rest = &self.src[self.pos..];
                let Some(ch) = rest.chars().next() else { break; };
                if Self::is_ident_continue(ch) {
                    self.pos += ch.len_utf8();
                } else {
                    break;
                }
            }

            let text = &self.src[start..self.pos];
            return Token::Ident(text.to_string()); // not a keyword
        }

        // ASCII-only ident: keyword check is valid
        let text = &self.src[start..self.pos];
        if let Some(kw) = match_keyword(text) {
            Token::Operator(kw) // preserving your existing model
        } else {
            Token::Ident(text.to_string())
        }
    }

    #[inline(always)]
    fn lex_operator(&mut self, start: usize) -> Result<Token, LexError> {
        if let Some(op) = match_operator(&self.src[self.pos..]) {
            self.pos += op.as_str().len();
            Ok(Token::Operator(op))
        } else {
            let bad = self.src[self.pos..].chars().next().unwrap();
            self.pos += bad.len_utf8();
            Err(LexError::UnexpectedChar {
                ch: bad,
                loc: self.produce_loc(start),
            })
        }
    }

    // #[unsafe(no_mangle)]
    fn lex_token(&mut self) -> Result<Option<LTok>, LexError> {
        let start = self.pos;

        let Some(b0) = self.src.as_bytes().get(self.pos).copied() else {
            return Ok(None);
        };

        let value = match b0 {
            b'0'..=b'9' => {
                self.pos += 1;
                self.lex_number(start)
            }

            b'"' => {
                self.pos += 1;
                self.lex_string(start)?
            }

            // ASCII start of ident
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                self.pos += 1;
                self.lex_ident_or_keyword(start, false)
            }

            _ if !b0.is_ascii() => {
                // Non-ASCII: decode one char and decide if it's an identifier start.
                let rest = &self.src[self.pos..];
                let ch = rest.chars().next().unwrap();

                if Self::is_ident_start(ch) {
                    self.pos += ch.len_utf8();
                    self.lex_ident_or_keyword(start, true)
                } else {
                    // Not an identifier start; try operator (will likely error)
                    self.lex_operator(start)?
                }
            }

            _ => {
                // ASCII non-ident start: operator/delimiter/unknown
                self.lex_operator(start)?
            }
        };

        Ok(Some(Located {
            loc: self.produce_loc(start),
            value,
        }))
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

    pub fn try_op(&mut self) -> Result<Option<LStr<'static>>, LexError> {
        let Some(tok) = self.peek()? else {
            return Ok(None);
        };

        let Token::Operator(s) = tok.value else {
            return Ok(None);
        };

        let ans = tok.with(s.as_str());
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

        if op != s.as_str() {
            return Ok(None);
        }

        let ans = tok.with(s.as_str());
        self.next()?;
        Ok(Some(ans))
    }
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
        let src = "let x ~= y => z && match";
        let mut lex = Lexer::new(src, 0);

        let kinds: Vec<String> = std::iter::from_fn(|| lex.next().unwrap())
            .map(|tok| match tok.value {
                Token::Operator(s) => format!("op({})", s.as_str()),
                Token::Ident(s) => format!("id({})", s),
                _ => "other".to_string(),
            })
            .collect();

        assert_eq!(
            kinds,
            vec![
                "op(let)",
                "id(x)",
                "op(~=)",
                "id(y)",
                "op(=>)",
                "id(z)",
                "op(&&)",
                "op(match)",
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

    #[test]
    fn lexer_gets_all_operators(){
        for word in KEYWORDS.iter().chain(OPERATORS.iter()) {
            let mut lex = Lexer::new(word,0);
            let t = lex.next().unwrap().unwrap();
            assert_eq!(t.value, Token::new_operator(word));
            assert_eq!(lex.next().unwrap(),None);
        }
    }

     /* =========================================
     * 4) Unicode whitespace is skipped (not just ASCII)
     * ========================================= */
    #[test]
    fn unicode_whitespace_is_skipped() {
        // NBSP + EM SPACE around tokens
        let src = "  let x = 1;";
        //          ^ ^   ^ ^  ^ ^
        //        NBSP EM  NBSP  EM NBSP
        let mut lex = Lexer::new(src, 0);

        let t0 = lex.next().unwrap().unwrap();
        assert_eq!(t0.value, Token::new_operator("let"));

        let t1 = lex.next().unwrap().unwrap();
        assert!(matches!(t1.value, Token::Ident(ref s) if s == "x"));

        let t2 = lex.next().unwrap().unwrap();
        assert_eq!(t2.value, Token::new_operator("="));

        let t3 = lex.next().unwrap().unwrap();
        assert!(matches!(t3.value, Token::NumLit(1)));

        let t4 = lex.next().unwrap().unwrap();
        assert_eq!(t4.value, Token::new_operator(";"));

        assert_eq!(lex.next().unwrap(), None);
    }

    /* =========================================
     * 5) Unicode identifiers are allowed and are never keywords
     * ========================================= */
    #[test]
    fn unicode_ident_is_not_keyword() {
        let src = "let שלום = 3; match";
        let mut lex = Lexer::new(src, 0);

        let t0 = lex.next().unwrap().unwrap();
        assert_eq!(t0.value, Token::new_operator("let"));

        let t1 = lex.next().unwrap().unwrap();
        assert!(matches!(t1.value, Token::Ident(ref s) if s == "שלום"));

        let t2 = lex.next().unwrap().unwrap();
        assert_eq!(t2.value, Token::new_operator("="));

        let t3 = lex.next().unwrap().unwrap();
        assert!(matches!(t3.value, Token::NumLit(3)));

        let t4 = lex.next().unwrap().unwrap();
        assert_eq!(t4.value, Token::new_operator(";"));

        // "match" here should still be recognized as a keyword/operator
        let t5 = lex.next().unwrap().unwrap();
        assert_eq!(t5.value, Token::new_operator("match"));

        assert_eq!(lex.next().unwrap(), None);
    }

}

pub type PResult<T> = Result<T, ParseError>;
pub type OTok = Located<Option<Token>>;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ParseError {
    #[error(transparent)]
    Lex(#[from] LexError),

    #[error("expected expression, got {got:?}")]
    ExpectedExpr { got: OTok },

    #[error("expected {expected}, got {got:?}")]
    ExpectedToken { expected: &'static str, got: OTok },

    #[error("opened {open} without closing with {close} but got {got:?}")]
    OpenDelimiter {
        open: LStr<'static>,
        close: &'static str,
        got: OTok,
    },
}

const BP_ASSIGN: u32 = 100;
const BP_MATCH_ARM: u32 = 90;
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
        // match arm (right-assoc)
        "=>" => (BP_MATCH_ARM + 1, BP_MATCH_ARM),

        // assignment (right-assoc)
        "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>=" => {
            (BP_ASSIGN + 1, BP_ASSIGN)
        }
        "|>" => (120, 121),

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

pub struct Parser<'a> {
    lex: Lexer<'a>,
}

impl<'a> Parser<'a> {
    pub fn new(src: &'a str, file: usize) -> Self {
        Self {
            lex: Lexer::new(src, file),
        }
    }

    pub fn is_empty(&mut self) -> bool {
        matches!(self.peek(), Ok(None))
    }
    /// Try parse an expression.
    /// Ok(None) => no expression starts here
    /// Err(_)   => expression started but failed
    pub fn try_expr(&mut self) -> PResult<Option<LExpr>> {
        let mut out = self.dummy_expr();
        match self.try_expr_into(&mut out)? {
            Some(()) => Ok(Some(out)),
            None => Ok(None),
        }
    }

    /// Must parse an expression or error.
    pub fn consume_expr(&mut self) -> PResult<LExpr> {
        let mut out = self.dummy_expr();
        self.consume_expr_into(&mut out)?;
        Ok(out)
    }

    /// Parse a statement: expr [';']
    pub fn parse_stmt(&mut self) -> PResult<Option<LExpr>> {
        let mut out = self.dummy_expr();
        match self.try_stmt(&mut out)? {
            Some(()) => Ok(Some(out)),
            None => Ok(None),
        }
    }

    /// Must parse an expression or error.
    pub fn consume_stmt(&mut self) -> PResult<LExpr> {
        let mut out = self.dummy_expr();
        self.consume_stmt_into(&mut out)?;
        Ok(out)
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
    fn dummy_expr(&self) -> LExpr {
        Located {
            loc: self.lex.empty_loc(),
            value: Expr::Atom(Token::NumLit(0)),
        }
    }
    fn dummy_expr_with_loc(&self, loc: Loc) -> LExpr {
        Located {
            loc,
            value: Expr::Atom(Token::NumLit(0)),
        }
    }
    fn try_expr_into(&mut self, out: &mut LExpr) -> PResult<Option<()>> {
        self.try_expr_bp(0, out)
    }
    fn consume_expr_into(&mut self, out: &mut LExpr) -> PResult<()> {
        self.consume_expr_bp(0, out)
    }
    fn try_stmt(&mut self, out: &mut LExpr) -> PResult<Option<()>> {
        let start = self.expr_start();
        if self.try_expr_into(out)?.is_none() {
            return Ok(None);
        }

        if let Some(semi) = self.try_operator(";")? {
            let loc = self.produce_loc(start);
            let out_loc = out.loc.clone();
            let expr = std::mem::replace(out, self.dummy_expr_with_loc(out_loc));
            *out = Located {
                loc,
                value: Expr::Postfix(semi, vec![expr]),
            };
        }

        Ok(Some(()))
    }
    fn consume_stmt_into(&mut self, out: &mut LExpr) -> PResult<()> {
        match self.try_stmt(out)? {
            Some(()) => Ok(()),
            None => Err(ParseError::ExpectedExpr {
                got: self.peek_op()?,
            }),
        }
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
    fn try_expr_bp(&mut self, min_bp: u32, out: &mut LExpr) -> PResult<Option<()>> {
        let start = self.expr_start();
        if self.parse_prefix(start, out)?.is_none() {
            return Ok(None);
        }

        loop {
            // postfix first
            if self.try_parse_postfix(start, out, min_bp)? {
                continue;
            }

            // then infix
            if self.try_parse_infix(start, out, min_bp)? {
                continue;
            }

            break;
        }

        Ok(Some(()))
    }

    fn consume_expr_bp(&mut self, min_bp: u32, out: &mut LExpr) -> PResult<()> {
        match self.try_expr_bp(min_bp, out)? {
            Some(()) => Ok(()),
            None => Err(ParseError::ExpectedExpr {
                got: self.peek_op()?,
            }),
        }
    }
    fn try_parse_infix(&mut self, start: usize, lhs: &mut LExpr, min_bp: u32) -> PResult<bool> {
        let Some(peek) = self.peek()? else {
            return Ok(false);
        };
        let Token::Operator(op) = &peek.value else {
            return Ok(false);
        };

        let Some((l_bp, r_bp)) = infix_bp(op.as_str()) else {
            return Ok(false);
        };

        if l_bp < min_bp {
            return Ok(false);
        }

        let op_tok = self.try_op()?.unwrap();
        let mut rhs = self.dummy_expr();
        self.consume_expr_bp(r_bp, &mut rhs)?;

        let loc = self.produce_loc(start);
        let lhs_loc = lhs.loc.clone();
        let temp = std::mem::replace(lhs, self.dummy_expr_with_loc(lhs_loc));
        *lhs = Located {
            loc,
            value: Expr::Bin(op_tok, Box::new((temp, rhs))),
        };

        Ok(true)
    }

    fn try_parse_postfix(&mut self, start: usize, lhs: &mut LExpr, min_bp: u32) -> PResult<bool> {
        let Some(peek) = self.peek()? else {
            return Ok(false);
        };
        let Token::Operator(op) = &peek.value else {
            return Ok(false);
        };
        let Some(bp) = postfix_bp(op.as_str()) else {
            return Ok(false);
        };

        //check if we need to special case later
        let end_op = match op.as_str() {
            "(" => ")",
            "[" => "]",
            _ => "",
        };

        if bp < min_bp {
            return Ok(false);
        }

        let open = self.try_op()?.unwrap();

        let lhs_loc = lhs.loc.clone();
        let mut temp = std::mem::replace(lhs, self.dummy_expr_with_loc(lhs_loc));
        let mut args = Vec::with_capacity(1);
        args.push(self.dummy_expr());
        std::mem::swap(args.last_mut().unwrap(), &mut temp);

        //handle common case
        if end_op.is_empty() {
            *lhs = Located {
                loc: self.produce_loc(start),
                value: Expr::Postfix(open, args),
            };
            return Ok(true);
        }

        //handle arg lists
        while self.try_operator(end_op)?.is_none() {
            args.push(self.dummy_expr());
            let last = args.last_mut().unwrap();
            if self.try_expr_into(last)?.is_none() {
                return Err(self.err_open_delim(open, end_op));
            }
            self.try_operator(",")?;
        }

        *lhs = Located {
            loc: self.produce_loc(start),
            value: Expr::Postfix(open, args),
        };
        Ok(true)
    }
    fn parse_prefix(&mut self, start: usize, out: &mut LExpr) -> PResult<Option<()>> {
        let Some(tok) = self.peek()? else {
            return Ok(None);
        };

        match tok.value {
            Token::NumLit(_) | Token::FloatLit(_) | Token::StrLit(_) | Token::Ident(_) => {
                let tok = self.next()?.unwrap();
                *out = Located {
                    loc: self.produce_loc(start),
                    value: Expr::Atom(tok.value),
                };
                Ok(Some(()))
            }

            Token::Operator(op) => {
                let op_str = op.as_str();
                let op_s = tok.with(op_str);

                // grouping / blocks
                if op_str == "(" {
                    self.next()?.unwrap();
                    self.parse_after_lparen(start, op_s, out)?;
                    return Ok(Some(()));
                }
                if op_str == "{" {
                    self.next()?.unwrap();
                    self.parse_after_lbrace(start, op_s, out)?;
                    return Ok(Some(()));
                }

                // control keywords
                if op_str == "if" {
                    self.next()?.unwrap();
                    self.parse_after_if(start, op_s, out)?;
                    return Ok(Some(()));
                }
                if op_str == "while" {
                    self.next()?.unwrap();
                    self.parse_after_while(start, op_s, out)?;
                    return Ok(Some(()));
                }
                if op_str == "match" {
                    self.next()?.unwrap();
                    self.parse_after_match(start, op_s, out)?;
                    return Ok(Some(()));
                }

                if op_str == "fn" || op_str == "cfn" {
                    self.next()?.unwrap();
                    self.parse_after_fn(start, op_s, out)?;
                    return Ok(Some(()));
                }

                if op_str == "struct" || op_str == "enum" || op_str == "union" {
                    self.next()?.unwrap();
                    self.parse_after_struct(start, op_s, out)?;
                    return Ok(Some(()));
                }

                if op_str == "let" {
                    self.next()?.unwrap();
                    self.parse_after_let(start, op_s, out)?;
                    return Ok(Some(()));
                }

                // generic prefix operator via BP
                if let Some(bp) = prefix_bp(op_str) {
                    self.next()?.unwrap();
                    let mut args = Vec::with_capacity(1);
                    args.push(self.dummy_expr());
                    self.consume_expr_bp(bp, args.last_mut().unwrap())?;
                    let loc = self.produce_loc(start);
                    *out = Located {
                        loc,
                        value: Expr::Prefix(op_s, args),
                    };
                    return Ok(Some(()));
                }
                Ok(None)
            }
        }
    }

    fn parse_after_lparen(
        &mut self,
        start: usize,
        open: LStr<'static>,
        out: &mut LExpr,
    ) -> PResult<()> {
        let mut parts = Vec::new();
        let mut saw_comma = false;

        while self.try_operator(")")?.is_none() {
            parts.push(self.dummy_expr());
            let last = parts.last_mut().unwrap();
            if self.try_expr_into(last)?.is_none() {
                return Err(self.err_open_delim(open, ")"));
            }

            if self.try_operator(",")?.is_some() {
                saw_comma = true;
            }
        }

        let loc = self.produce_loc(start);
        if !saw_comma && parts.len() == 1 {
            let part = parts.pop().unwrap();
            *out = Located {
                loc,
                value: part.value,
            };
            return Ok(());
        }
        *out = Located {
            loc,
            value: Expr::Prefix(open, parts),
        };
        Ok(())
    }

    fn parse_after_lbrace(
        &mut self,
        start: usize,
        open: LStr<'static>,
        out: &mut LExpr,
    ) -> PResult<()> {
        let mut items = Vec::new();

        while self.try_operator("}")?.is_none() {
            items.push(self.dummy_expr());
            let last = items.last_mut().unwrap();
            if self.try_stmt(last)?.is_none() {
                return Err(self.err_open_delim(open, "}"));
            }
        }

        let loc = self.produce_loc(start);
        *out = Located {
            loc,
            value: Expr::Prefix(open, items),
        };
        Ok(())
    }

    fn parse_after_if(
        &mut self,
        start: usize,
        if_tok: LStr<'static>,
        out: &mut LExpr,
    ) -> PResult<()> {
        let mut args = Vec::with_capacity(3);
        args.push(self.dummy_expr());
        self.consume_expr_into(args.last_mut().unwrap())?;
        args.push(self.dummy_expr());
        self.consume_stmt_into(args.last_mut().unwrap())?;

        if self.try_operator("else")?.is_some() {
            args.push(self.dummy_expr());
            self.consume_stmt_into(args.last_mut().unwrap())?;
        }

        let loc = self.produce_loc(start);
        *out = Located {
            loc,
            value: Expr::Prefix(if_tok, args),
        };
        Ok(())
    }

    fn parse_after_while(
        &mut self,
        start: usize,
        w: LStr<'static>,
        out: &mut LExpr,
    ) -> PResult<()> {
        let mut args = Vec::with_capacity(2);
        args.push(self.dummy_expr());
        self.consume_expr_into(args.last_mut().unwrap())?;
        args.push(self.dummy_expr());
        self.consume_stmt_into(args.last_mut().unwrap())?;

        let loc = self.produce_loc(start);
        *out = Located {
            loc,
            value: Expr::Prefix(w, args),
        };
        Ok(())
    }

    fn parse_after_match(
        &mut self,
        start: usize,
        m: LStr<'static>,
        out: &mut LExpr,
    ) -> PResult<()> {
        let mut args = Vec::new();
        args.push(self.dummy_expr());
        self.consume_expr_into(args.last_mut().unwrap())?;
        let open = self.expect_operator("{")?;

        while self.try_operator("}")?.is_none() {
            let arm_start = self.expr_start();
            let mut pat = self.dummy_expr();
            if self.try_expr_bp(BP_PATTERN, &mut pat)?.is_none() {
                return Err(self.err_open_delim(open.clone(), "}"));
            }
            let arrow = self.expect_operator("=>")?;
            let mut body = self.dummy_expr();
            self.consume_expr_into(&mut body)?;

            args.push(self.dummy_expr());
            *args.last_mut().unwrap() = Located {
                loc: self.produce_loc(arm_start),
                value: Expr::Bin(arrow, Box::new((pat, body))),
            };

            if let Some(Token::Operator(op)) = self.peek()?.map(|l| &l.value) {
                if matches!(op.as_str(), "," | ";") {
                    self.next()?;
                }
            }
        }

        *out = Located {
            loc: self.produce_loc(start),
            value: Expr::Prefix(m, args),
        };
        Ok(())
    }

    fn parse_after_fn(
        &mut self,
        start: usize,
        fn_tok: LStr<'static>,
        out: &mut LExpr,
    ) -> PResult<()> {
        let paren_start = self.expr_start();
        let open = self.expect_operator("(")?;
        let mut params: Vec<LExpr> = Vec::new();

        while self.try_operator(")")?.is_none() {
            params.push(self.dummy_expr());
            let last = params.last_mut().unwrap();
            if self.try_expr_into(last)?.is_none() {
                return Err(self.err_open_delim(open.clone(), ")"));
            }
            self.try_operator(",")?;
        }
        let mut sig = Located {
            loc: self.produce_loc(paren_start),
            value: Expr::Prefix(open, params),
        };

        if let Some(arrow) = self.try_operator("->")? {
            let mut output = self.dummy_expr();
            self.consume_expr_into(&mut output)?;
            sig = Located {
                loc: self.produce_loc(paren_start),
                value: Expr::Bin(arrow, Box::new((sig, output))),
            }
        }

        let mut v = Vec::with_capacity(2);
        v.push(self.dummy_expr());
        std::mem::swap(v.last_mut().unwrap(), &mut sig);
        v.push(self.dummy_expr());
        if self.try_expr_into(v.last_mut().unwrap())?.is_none() {
            v.pop();
        }
        *out = Located {
            loc: self.produce_loc(start),
            value: Expr::Prefix(fn_tok, v),
        };
        Ok(())
    }

    fn parse_after_let(
        &mut self,
        start: usize,
        let_tok: LStr<'static>,
        out: &mut LExpr,
    ) -> PResult<()> {
        let mut args = Vec::with_capacity(2);
        args.push(self.dummy_expr());
        self.consume_expr_bp(BP_PATTERN, args.last_mut().unwrap())?;
        self.expect_operator("=")?;
        args.push(self.dummy_expr());
        self.consume_expr_into(args.last_mut().unwrap())?;

        *out = Located {
            loc: self.produce_loc(start),
            value: Expr::Prefix(let_tok, args),
        };
        Ok(())
    }

    fn parse_after_struct(
        &mut self,
        start: usize,
        def_tok: LStr<'static>,
        out: &mut LExpr,
    ) -> PResult<()> {
        let mut fields = Vec::new();

        let open = self.expect_operator("{")?;
        while self.try_operator("}")?.is_none() {
            fields.push(self.dummy_expr());
            let last = fields.last_mut().unwrap();
            if self.try_expr_into(last)?.is_none() {
                return Err(self.err_open_delim(open.clone(), ")"));
            }

            if let Some(Token::Operator(op)) = self.peek()?.map(|l| &l.value) {
                if matches!(op.as_str(), "," | ";") {
                    self.next()?;
                }
            };
        }
        *out = Located {
            loc: self.produce_loc(start),
            value: Expr::Prefix(def_tok, fields),
        };
        Ok(())
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
        let src = "f(a } b)";
        let mut p = Parser::new(src, 0);
        let err = p.consume_expr().unwrap_err();

        match err {
            ParseError::OpenDelimiter { open, close, got } => {
                assert_eq!(open.value, "(");
                assert_eq!(close, ")");

                let got = got.as_ref().unwrap();
                assert_eq!(*got, Token::new_operator("}"));

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

        let first = p.consume_stmt().unwrap();
        let second = p.consume_stmt().unwrap();

        match first.value {
            Expr::Prefix(tok, _) => {
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
        let expr = p.consume_expr().unwrap();

        match expr.value {
            Expr::Prefix(tok, _) => {
                assert_eq!(tok.value, "if");
                assert_loc(&expr.loc, 0, src.len());
            }
            _ => panic!("expected if-else"),
        }
    }

    #[test]
    fn match_parses_arms_and_allows_separators() {
        let src = "match x { 0 => y, Some(\"hi\"|\"by\") => w }";
        let mut p = Parser::new(src, 0);
        let expr = p.consume_expr().unwrap();

        match expr.value {
            Expr::Prefix(match_kw, args) => {
                assert_eq!(match_kw.value, "match");
                assert_eq!(args.len(), 3);

                match &args[0].value {
                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "x"),
                    _ => panic!("expected match scrutinee"),
                }

                match &args[1].value {
                    Expr::Bin(arrow, parts) => {
                        assert_eq!(arrow.value, "=>");
                        let (pat, body) = &**parts;
                        match &pat.value {
                            Expr::Atom(Token::NumLit(0)) => {}
                            _ => panic!("expected 0 pattern"),
                        }
                        match &body.value {
                            Expr::Atom(Token::Ident(name)) => assert_eq!(name, "y"),
                            _ => panic!("expected y body"),
                        }
                    }
                    _ => panic!("expected match arm"),
                }

                match &args[2].value {
                    Expr::Bin(arrow, parts) => {
                        assert_eq!(arrow.value, "=>");
                        let (pat, body) = &**parts;
                        match &pat.value {
                            Expr::Postfix(open, args) => {
                                assert_eq!(open.value, "(");
                                assert_eq!(args.len(), 2);
                                match &args[0].value {
                                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "Some"),
                                    _ => panic!("expected Some"),
                                }
                                match &args[1].value {
                                    Expr::Bin(pipe, parts) => {
                                        assert_eq!(pipe.value, "|");
                                        let (lhs, rhs) = &**parts;
                                        match &lhs.value {
                                            Expr::Atom(Token::StrLit(name)) => {
                                                assert_eq!(name, "hi")
                                            }
                                            _ => panic!("expected hi string"),
                                        }
                                        match &rhs.value {
                                            Expr::Atom(Token::StrLit(name)) => {
                                                assert_eq!(name, "by")
                                            }
                                            _ => panic!("expected by string"),
                                        }
                                    }
                                    _ => panic!("expected \"hi\"|\"by\""),
                                }
                            }
                            _ => panic!("expected Some(\"hi\"|\"by\") pattern"),
                        }
                        match &body.value {
                            Expr::Atom(Token::Ident(name)) => assert_eq!(name, "w"),
                            _ => panic!("expected w body"),
                        }
                    }
                    _ => panic!("expected match arm"),
                }

                assert_eq!(expr.loc.range, 0..src.len());
            }
            _ => panic!("expected match"),
        }

        let src = "match x { 0 => y; z => w }";
        let mut p = Parser::new(src, 0);
        let expr = p.consume_expr().unwrap();

        match expr.value {
            Expr::Prefix(match_kw, args) => {
                assert_eq!(match_kw.value, "match");
                assert_eq!(args.len(), 3);
            }
            _ => panic!("expected match"),
        }
    }

    /* =============================
     * Semicolon semantics
     * ============================= */

    #[test]
    fn semicolon_is_postfix_on_statement() {
        let src = "x; y";
        let mut p = Parser::new(src, 0);

        let first = p.consume_stmt().unwrap();
        let second = p.consume_stmt().unwrap();

        match first.value {
            Expr::Postfix(tok, args) => {
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
            Expr::Prefix(_, args) => {
                let sig = &args[0];
                match &sig.value {
                    Expr::Prefix(_, params) => {
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
            Expr::Prefix(_, args) => {
                assert_eq!(args.len(), 2);

                // signature is arrow
                match &args[0].value {
                    Expr::Bin(op, _) => assert_eq!(op.value, "->"),
                    _ => panic!("expected arrow sig"),
                }

                // body is x + 1
                match &args[1].value {
                    Expr::Bin(op, _) => assert_eq!(op.value, "+"),
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

        let expr = p.consume_expr().unwrap();

        match expr.value {
            Expr::Prefix(let_tok, args) => {
                assert_eq!(let_tok.value, "let");
                assert_eq!(args.len(), 2);

                // ---- declaration ----
                match &args[0].value {
                    Expr::Bin(colon, parts) => {
                        assert_eq!(colon.value, ":");
                        let (name, ty) = &**parts;

                        // x
                        match &name.value {
                            Expr::Atom(Token::Ident(name)) => assert_eq!(name, "x"),
                            _ => panic!("expected variable name"),
                        }

                        // *char
                        match &ty.value {
                            Expr::Prefix(star, inner) => {
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

        let expr = p.consume_expr().unwrap();

        match expr.value {
            Expr::Postfix(open, args) => {
                assert_eq!(open.value, "[");
                assert_eq!(args.len(), 3);

                // ---- base expression ----
                match &args[0].value {
                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "x"),
                    _ => panic!("expected base identifier"),
                }

                // ---- first slice: 0:2 ----
                match &args[1].value {
                    Expr::Bin(colon, parts) => {
                        assert_eq!(colon.value, ":");
                        let (lhs, rhs) = &**parts;

                        match &lhs.value {
                            Expr::Atom(Token::NumLit(0)) => {}
                            _ => panic!("expected 0"),
                        }
                        match &rhs.value {
                            Expr::Atom(Token::NumLit(2)) => {}
                            _ => panic!("expected 2"),
                        }
                    }
                    _ => panic!("expected colon expression"),
                }

                // ---- second slice: 1:2 ----
                match &args[2].value {
                    Expr::Bin(colon, parts) => {
                        assert_eq!(colon.value, ":");
                        let (lhs, rhs) = &**parts;

                        match &lhs.value {
                            Expr::Atom(Token::NumLit(1)) => {}
                            _ => panic!("expected 1"),
                        }
                        match &rhs.value {
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
            Expr::Postfix(open, args) => {
                assert_eq!(open.value, "(");
                assert_eq!(args.len(), 2);

                // f
                match &args[0].value {
                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "f"),
                    _ => panic!("expected identifier f"),
                }

                // (x)
                match &args[1].value {
                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "x"),
                    _ => panic!("expected identifier x"),
                }
            }
            _ => panic!("expected call expression"),
        }

        // ---- second expression: { a; b } ----
        let block = p.consume_stmt().unwrap();

        match block.value {
            Expr::Prefix(open, items) => {
                assert_eq!(open.value, "{");
                assert_eq!(items.len(), 2);

                // a;
                match &items[0].value {
                    Expr::Postfix(tok, args) => {
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

    #[test]
    fn struct_type_definition() {
        let src = "Point[f] = struct { x:f y }";
        let mut p = Parser::new(src, 0);

        let expr = p.consume_stmt().unwrap();

        match expr.value {
            Expr::Bin(eq, args) => {
                assert_eq!(eq.value, "=");
                let (lhs, rhs) = &*args;

                // ---- LHS: Point[f] ----
                match &lhs.value {
                    Expr::Postfix(bracket, gargs) => {
                        assert_eq!(bracket.value, "[");
                        assert_eq!(gargs.len(), 2);

                        match &gargs[0].value {
                            Expr::Atom(Token::Ident(name)) => assert_eq!(name, "Point"),
                            _ => panic!("expected identifier Point"),
                        }

                        match &gargs[1].value {
                            Expr::Atom(Token::Ident(name)) => assert_eq!(name, "f"),
                            _ => panic!("expected generic parameter f"),
                        }
                    }
                    _ => panic!("expected generic application Point[f]"),
                }

                // ---- RHS: struct { x:f y } ----
                match &rhs.value {
                    Expr::Prefix(struct_kw, fields) => {
                        assert_eq!(struct_kw.value, "struct");
                        assert_eq!(fields.len(), 2);

                        // x:f
                        match &fields[0].value {
                            Expr::Bin(colon, _parts) => {
                                assert_eq!(colon.value, ":");
                            }
                            _ => panic!("expected field definition x:f"),
                        }

                        // y
                        match &fields[1].value {
                            Expr::Atom(Token::Ident(name)) => assert_eq!(name, "y"),
                            _ => panic!("expected identifier y"),
                        }
                    }
                    _ => panic!("expected struct definition"),
                }
            }
            _ => panic!("expected assignment expression"),
        }

        assert!(p.is_empty());
    }

    #[test]
    fn struct_construction() {
        let src = "Point(4, y=2)";
        let mut p = Parser::new(src, 0);

        let expr = p.consume_stmt().unwrap();

        match expr.value {
            Expr::Postfix(open, args) => {
                assert_eq!(open.value, "(");
                assert_eq!(args.len(), 3);

                // Point
                match &args[0].value {
                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "Point"),
                    _ => panic!("expected identifier Point"),
                }

                // 4
                match &args[1].value {
                    Expr::Atom(Token::NumLit(n)) => assert_eq!(*n, 4),
                    _ => panic!("expected integer literal"),
                }

                // y=2
                match &args[2].value {
                    Expr::Bin(eq, parts) => {
                        assert_eq!(eq.value, "=");
                        let (lhs, _) = &**parts;

                        match &lhs.value {
                            Expr::Atom(Token::Ident(name)) => assert_eq!(name, "y"),
                            _ => panic!("expected identifier y"),
                        }
                    }
                    _ => panic!("expected named argument y=2"),
                }
            }
            _ => panic!("expected application expression"),
        }

        assert!(p.is_empty());
    }
}
