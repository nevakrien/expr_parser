pub mod error_reporting;
pub mod ir;
pub mod parsing;

pub use error_reporting::ErrorReporter;
pub use parsing::{Expr, LExpr, Parser, Token};


#[cfg(test)]
mod lex_tests {
    use crate::parsing::*;

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

#[cfg(test)]
mod parse_tests {
    use crate::parsing::*;

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
                assert_eq!(open.as_str(), "(");
                assert_eq!(close.as_str(), ")");
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
                assert_eq!(open.as_str(), "(");
                assert_eq!(close.as_str(), ")");

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
                assert_eq!(tok.as_str(), "if");
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
                assert_eq!(tok.as_str(), "if");
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
                assert_eq!(match_kw.as_str(), "match");
                assert_eq!(args.len(), 3);

                match &args[0].value {
                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "x"),
                    _ => panic!("expected match scrutinee"),
                }

                match &args[1].value {
                    Expr::Bin(arrow, parts) => {
                        assert_eq!(arrow.as_str(), "=>");
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
                        assert_eq!(arrow.as_str(), "=>");
                        let (pat, body) = &**parts;
                        match &pat.value {
                            Expr::Postfix(open, args) => {
                                assert_eq!(open.as_str(), "(");
                                assert_eq!(args.len(), 2);
                                match &args[0].value {
                                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "Some"),
                                    _ => panic!("expected Some"),
                                }
                                match &args[1].value {
                                    Expr::Bin(pipe, parts) => {
                                        assert_eq!(pipe.as_str(), "|");
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
                assert_eq!(match_kw.as_str(), "match");
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
                assert_eq!(tok.as_str(), ";");
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
                assert_eq!(open.as_str(), "{");
                assert_eq!(close.as_str(), "}");
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
                    Expr::Bin(op, _) => assert_eq!(op.as_str(), "->"),
                    _ => panic!("expected arrow sig"),
                }

                // body is x + 1
                match &args[1].value {
                    Expr::Bin(op, _) => assert_eq!(op.as_str(), "+"),
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
                assert_eq!(let_tok.as_str(), "let");
                assert_eq!(args.len(), 2);

                // ---- declaration ----
                match &args[0].value {
                    Expr::Bin(colon, parts) => {
                        assert_eq!(colon.as_str(), ":");
                        let (name, ty) = &**parts;

                        // x
                        match &name.value {
                            Expr::Atom(Token::Ident(name)) => assert_eq!(name, "x"),
                            _ => panic!("expected variable name"),
                        }

                        // *char
                        match &ty.value {
                            Expr::Prefix(star, inner) => {
                                assert_eq!(star.as_str(), "*");
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
                assert_eq!(open.as_str(), "[");
                assert_eq!(args.len(), 3);

                // ---- base expression ----
                match &args[0].value {
                    Expr::Atom(Token::Ident(name)) => assert_eq!(name, "x"),
                    _ => panic!("expected base identifier"),
                }

                // ---- first slice: 0:2 ----
                match &args[1].value {
                    Expr::Bin(colon, parts) => {
                        assert_eq!(colon.as_str(), ":");
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
                        assert_eq!(colon.as_str(), ":");
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
                assert_eq!(open.as_str(), "(");
                assert_eq!(close.as_str(), ")");

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
                assert_eq!(open.as_str(), "(");
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
                assert_eq!(open.as_str(), "{");
                assert_eq!(items.len(), 2);

                // a;
                match &items[0].value {
                    Expr::Postfix(tok, args) => {
                        assert_eq!(tok.as_str(), ";");
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
                assert_eq!(eq.as_str(), "=");
                let (lhs, rhs) = &*args;

                // ---- LHS: Point[f] ----
                match &lhs.value {
                    Expr::Postfix(bracket, gargs) => {
                        assert_eq!(bracket.as_str(), "[");
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
                        assert_eq!(struct_kw.as_str(), "struct");
                        assert_eq!(fields.len(), 2);

                        // x:f
                        match &fields[0].value {
                            Expr::Bin(colon, _parts) => {
                                assert_eq!(colon.as_str(), ":");
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
                assert_eq!(open.as_str(), "(");
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
                        assert_eq!(eq.as_str(), "=");
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
