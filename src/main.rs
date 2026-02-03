use expr_parser::error_reporting::ErrorReporter;
use expr_parser::parsing::{Expr, LExpr, ParseError, Parser, Token};
use expr_parser::program::CompileError;
use expr_parser::program::Defined;
use expr_parser::program::Program;
use expr_parser::type_inference::infer_value_internals;
use expr_parser::type_inference::TypeStore;
use std::fs;
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

enum ReplInput {
    Quit,
    Reset,
    Load(Vec<String>),
    Code(String),
}

fn is_incomplete_error(error: &ParseError) -> bool {
    match error {
        ParseError::UnterminatedString { .. } => true,
        ParseError::ExpectedExpr { got } | ParseError::ExpectedToken { got, .. } => {
            got.value.is_none()
        }
        ParseError::OpenDelimiter { got, .. } => got.value.is_none(),
        ParseError::UnexpectedChar { .. } => false,
    }
}

fn needs_more_input(input: &str) -> bool {
    let mut parser = Parser::new(input, 0);

    while !parser.is_empty() {
        match parser.parse_stmt() {
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(err) => return is_incomplete_error(&err),
        }
    }

    false
}

fn read_repl_input() -> io::Result<ReplInput> {
    let mut input = String::new();
    let mut line = String::new();
    let mut first = true;

    loop {
        if first {
            print!("> ");
        } else {
            print!("... ");
        }
        io::stdout().flush().unwrap();

        line.clear();
        let bytes = io::stdin().read_line(&mut line)?;
        if bytes == 0 {
            return Ok(ReplInput::Quit);
        }

        let trimmed = line.trim();
        if first {
            if trimmed.is_empty() {
                continue;
            }
            if trimmed == "quit" || trimmed == "exit" {
                return Ok(ReplInput::Quit);
            }
            if trimmed == ":reset" {
                return Ok(ReplInput::Reset);
            }
            if trimmed.starts_with(":load") {
                let args = trimmed.split_whitespace().skip(1);
                return Ok(ReplInput::Load(args.map(String::from).collect()));
            }
        }

        input.push_str(&line);
        if !needs_more_input(&input) {
            return Ok(ReplInput::Code(input));
        }

        first = false;
    }
}

fn parse_source(program: &mut Program, input: &str, file_id: usize) -> Result<usize, CompileError> {
    let mut parser = Parser::new(input, file_id);
    let mut expr_count = 0;

    while !parser.is_empty() {
        match parser.parse_with_macros(program)? {
            None => break,
            Some(expr) => {
                println!(
                    "Expr {}: [{}..{}]",
                    expr_count + 1,
                    expr.loc.range.start,
                    expr.loc.range.end
                );
                println!("{}", pretty_print_expr(&expr, 0));
                expr_count += 1;
                program.gather_definition(expr)?;
            }
        }
    }

    Ok(expr_count)
}

fn finalize_program(
    reporter: &mut ErrorReporter,
    program: &mut Program,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Err(err) = program.check_pending_names() {
        reporter.report_compile_error(&err)?;
        return Ok(());
    }

    run_typechecker(program, reporter)?;

    Ok(())
}

fn run_typechecker(
    program: &Program,
    reporter: &mut ErrorReporter,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut types = TypeStore::new();
    for (_, def) in program.definitions.iter() {
        let Defined::Value(v) = def else {
            continue;
        };
        let Err(errs) = infer_value_internals(program, &mut types, *v) else {
            continue;
        };

        for e in errs {
            reporter.report_type_error(program, &types, &e)?;
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reporter = ErrorReporter::new();
    let mut program = Program::new();
    let mut next_file_id = 0usize;

    println!("Expression Parser REPL");
    println!("Enter expressions; REPL waits for complete input.");
    println!("Commands: :load <path...>, :reset, quit, exit");

    loop {
        match read_repl_input() {
            Ok(ReplInput::Quit) => break,
            Ok(ReplInput::Reset) => {
                program = Program::new();
                println!("REPL state cleared.");
            }
            Ok(ReplInput::Load(paths)) => {
                if paths.is_empty() {
                    eprintln!("Usage: :load <path...>");
                    continue;
                }

                let mut had_error = false;
                for path in paths {
                    match fs::read_to_string(&path) {
                        Ok(contents) => {
                            let file_id = next_file_id;
                            next_file_id += 1;
                            reporter.add_source(file_id, contents.clone());

                            if let Err(err) = parse_source(&mut program, &contents, file_id) {
                                reporter.report_compile_error(&err)?;
                                had_error = true;
                                break;
                            }
                        }
                        Err(err) => {
                            eprintln!("Error reading {path}: {err}");
                            had_error = true;
                            break;
                        }
                    }
                }

                if !had_error {
                    finalize_program(&mut reporter, &mut program)?;
                }
            }
            Ok(ReplInput::Code(input)) => {
                let file_id = next_file_id;
                next_file_id += 1;
                reporter.add_source(file_id, input.clone());
                if let Err(err) = parse_source(&mut program, &input, file_id) {
                    reporter.report_compile_error(&err)?;
                    continue;
                }
                finalize_program(&mut reporter, &mut program)?;
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
