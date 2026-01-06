use crate::parsing::Expr;
use crate::parsing::LExpr;
use crate::parsing::Loc;
use crate::parsing::Parser;
use crate::parsing::Token;
use std::io::{self, Write};
mod error_reporting;
mod parsing;
pub use error_reporting::ErrorReporter;

fn format_span(loc: &Loc) -> String {
    format!("{}..{}", loc.range.start, loc.range.end)
}

fn pretty_print_token(token: &Token) -> String {
    match token {
        Token::NumLit(n) => n.to_string(),
        Token::FloatLit(n) => n.to_string(),
        Token::StrLit(s) => format!("\"{}\"", s),
        Token::Ident(s) => s.clone(),
        Token::Operator(s) => s.to_string(),
    }
}

fn pretty_print_expr(expr: &LExpr, indent: usize) -> String {
    match &expr.value {
        Expr::Atom(token) => pretty_print_token(token),
        Expr::Combo(op, args) => {
            let mut result = String::new();
            let indent_str = "  ".repeat(indent);

            result.push('"');
            result.push_str(op.value);
            result.push('"');

            if args.is_empty() {
                result.push_str("()");
            } else if args.len() == 1 {
                result.push(' ');
                result.push_str(&pretty_print_expr(&args[0], 0));
            } else {
                result.push_str("(\n");
                for arg in args {
                    result.push_str(&indent_str);
                    result.push_str("  ");
                    result.push_str(&pretty_print_expr(arg, indent + 1));
                    result.push('\n');
                }
                result.push_str(&indent_str);
                result.push(')');
            }

            result
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reporter = ErrorReporter::new();
    let mut input = String::new();

    println!("Expression Parser REPL");
    println!("Type expressions to parse, or 'quit' to exit");

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        input.clear();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let input = input.trim();
                if input.is_empty() {
                    continue;
                }
                if input == "quit" || input == "exit" {
                    break;
                }

                reporter.add_source(0, input.to_string());
                let mut parser = Parser::new(input, 0);

                // Parse as many expressions as possible until we get None
                let mut expr_count = 0;
                while !parser.is_empty() {
                    match parser.consume_stmt() {
                        Ok(expr) => {
                            println!("Expr {}: [{}]", expr_count + 1, format_span(&expr.loc));
                            println!("{}", pretty_print_expr(&expr, 0));
                            expr_count += 1;
                        }

                        Err(err) => {
                            reporter.report_parse_error(&err)?;
                            break;
                        }
                    }
                }
            }
            Err(err) => {
                eprintln!("Error reading input: {}", err);
                break;
            }
        }
    }

    println!("Goodbye!");
    Ok(())
}
