use expr_parser::error_reporting::ErrorReporter;
use expr_parser::parsing::{Expr, LExpr, Parser, Token};
use expr_parser::program::Program;
use std::io::{self, Write};

fn pretty_print_token(token: &Token) -> String {
    match token {
        Token::NumLit(n) => n.to_string(),
        Token::FloatLit(n) => n.to_string(),
        Token::StrLit(s) => format!("\"{}\"", s),
        Token::Ident(s) => s.clone(),
        Token::Operator(s) => format!("\"{}\"", s),
    }
}

fn pretty_print_expr(expr: &LExpr, indent: usize) -> String {
    match &expr.value {
        Expr::Atom(token) => pretty_print_token(token),
        Expr::Bin(op, pair) => {
            let (lhs, rhs) = &**pair;
            let label = format!("_ \"{}\" _  ", op.value);
            pretty_print_node(&label, [lhs, rhs], indent)
        }
        Expr::Prefix(op, args) => {
            let label = format!("\"{}\" _  ", op.value);
            pretty_print_node(&label, args.iter(), indent)
        }
        Expr::Postfix(op, args) => {
            let label = format!("_ \"{}\"  ", op.value);
            pretty_print_node(&label, args.iter(), indent)
        }
    }
}

fn pretty_print_node<'a, I>(label: &str, args: I, indent: usize) -> String
where
    I: IntoIterator<Item = &'a LExpr>,
{
    let mut result = String::new();
    let indent_str = "  ".repeat(indent);

    result.push_str(label);

    result.push_str("(\n");
    for arg in args.into_iter() {
        result.push_str(&indent_str);
        result.push_str("  ");
        result.push_str(&pretty_print_expr(arg, indent + 1));
        result.push('\n');
    }
    result.push_str(&indent_str);
    result.push(')');

    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reporter = ErrorReporter::new();
    let mut input = String::new();
    let mut program = Program::new();

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
                let mut expr_count = 0;
                let mut compile_error = None;
                while !parser.is_empty() {
                    match parser.parse_with_macros(&mut program) {
                        Ok(None) => break,
                        Ok(Some(expr)) => {
                            println!(
                                "Expr {}: [{}..{}]",
                                expr_count + 1,
                                expr.loc.range.start,
                                expr.loc.range.end
                            );
                            println!("{}", pretty_print_expr(&expr, 0));
                            expr_count += 1;

                            if let Err(err) = program.gather_definition(expr) {
                                compile_error = Some(err);
                                break;
                            }
                        }
                        Err(err) => {
                            compile_error = Some(err);
                            break;
                        }
                    }
                }

                if compile_error.is_none()
                    && let Err(err) = program.check_pending_names()
                {
                    compile_error = Some(err);
                }

                if let Some(err) = compile_error {
                    reporter.report_compile_error(&err)?;
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
