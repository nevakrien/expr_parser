use expr_parser::error_reporting::ErrorReporter;
use expr_parser::ir::NameId;
use expr_parser::parsing::{Expr, LExpr, ParseError, Parser, Token};
use expr_parser::program::Defined;
use expr_parser::program::Program;
use expr_parser::type_kinds::{SolvedTypes, TypeUniverse, run_typechecker};
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::ops::Range;

#[derive(Clone, Copy, Debug)]
struct CliOptions {
    stdin_batch: bool,
    show_ast: bool,
    type_dump: bool,
    origin_dump: bool,
}

fn cli_usage() -> &'static str {
    "Usage: expr_parser [--stdin-batch] [--show-ast] [--type-dump] [--origin-dump]\n\
     \n\
     Flags:\n\
       --stdin-batch  Read all stdin until EOF and compile once\n\
       --show-ast     Print parsed AST nodes while parsing\n\
       --type-dump    Print full type dump after successful typecheck\n\
       --origin-dump  Print full origin dump after successful typecheck\n\
       -h, --help     Show this help text"
}

fn parse_cli_options() -> Result<Option<CliOptions>, String> {
    let mut options = CliOptions {
        stdin_batch: false,
        show_ast: false,
        type_dump: false,
        origin_dump: false,
    };

    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--stdin-batch" => options.stdin_batch = true,
            "--show-ast" => options.show_ast = true,
            "--type-dump" => options.type_dump = true,
            "--origin-dump" => options.origin_dump = true,
            "-h" | "--help" => return Ok(None),
            _ => {
                return Err(format!("Unknown argument: {arg}\n\n{}", cli_usage()));
            }
        }
    }

    Ok(Some(options))
}

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
    SetShowAst(bool),
    SetTypeDump(bool),
    SetOriginDump(bool),
    ShowModes,
    ShowType(Vec<String>),
    DumpTypes,
    DumpTypesOf(String),
    DumpOrigins,
    DumpOriginsOf(String),
    Code(String),
}

fn parse_on_off(rest: &str) -> Option<bool> {
    match rest.trim() {
        "on" => Some(true),
        "off" => Some(false),
        _ => None,
    }
}

struct ParsedExprInfo {
    index: usize,
    loc: expr_parser::parsing::Loc,
    values_added: Range<usize>,
    defined_names: Vec<String>,
}

struct ParseBatch {
    expr_count: usize,
    infos: Vec<ParsedExprInfo>,
}

fn collect_defined_names(expr: &LExpr, out: &mut Vec<String>) {
    match &expr.value {
        Expr::Postfix(op, items) if op.value == ";" => {
            if let Some(last) = items.last() {
                collect_defined_names(last, out);
            }
        }
        Expr::Prefix(open, items) if open.value == "{" => {
            for item in items {
                collect_defined_names(item, out);
            }
        }
        Expr::Bin(eq, pair) if eq.value == "=" => {
            if let Expr::Atom(Token::Ident(name)) = &pair.0.value {
                out.push(name.clone());
            }
        }
        Expr::Prefix(open, items) if open.value == "type" && items.len() == 2 => {
            if let Expr::Atom(Token::Ident(name)) = &items[0].value {
                out.push(name.clone());
            }
        }
        _ => {}
    }
}

fn lookup_global_name_id(program: &Program, name: &str) -> Option<NameId> {
    let scope = program.scopes.first()?;
    scope
        .0
        .iter()
        .find_map(|(sid, id)| (program.str_intern.resolve(*sid) == name).then_some(*id))
}

fn def_type_string(
    program: &Program,
    types: &TypeUniverse,
    solved: &SolvedTypes,
    id: NameId,
) -> Option<String> {
    let def = program.definitions.get(&id)?;
    match def {
        Defined::Func(_funcs) => solved
            .function_types_by_name(id)
            .map(|f| f.ty)
            .map(|ty| types.kind_to_string(program, ty)),
        Defined::Type(texp) => solved
            .typedef_types
            .get(texp)
            .copied()
            .map(|ty| types.kind_to_string(program, ty)),
        Defined::BuildinType(_) => Some("builtin type".to_string()),
        Defined::BuildinInterface(_) => Some("builtin interface".to_string()),
        Defined::Macro(_) => Some("macro".to_string()),
        Defined::ToBeDefined => None,
    }
}

fn print_expr_types(
    program: &Program,
    types: &TypeUniverse,
    solved: &SolvedTypes,
    batch: &ParseBatch,
) {
    if batch.expr_count == 0 {
        return;
    }

    println!("Type info:");
    for info in &batch.infos {
        let expr_type = info
            .values_added
            .clone()
            .next_back()
            .and_then(|idx| solved.type_of(expr_parser::ir::ValId(idx)))
            .map(|ty| types.kind_to_string(program, ty))
            .unwrap_or_else(|| "<unknown>".to_string());

        println!(
            "  Expr {} [{}..{}]: {}",
            info.index, info.loc.range.start, info.loc.range.end, expr_type
        );

        for name in &info.defined_names {
            let t = lookup_global_name_id(program, name)
                .and_then(|id| def_type_string(program, types, solved, id))
                .unwrap_or_else(|| "<unknown>".to_string());
            println!("    {}: {}", name, t);
        }
    }
}

fn print_named_types(
    program: &Program,
    types: &TypeUniverse,
    solved: &SolvedTypes,
    names: &[String],
) {
    for name in names {
        let Some(id) = lookup_global_name_id(program, name) else {
            println!("{}: <not found>", name);
            continue;
        };
        let t =
            def_type_string(program, types, solved, id).unwrap_or_else(|| "<unknown>".to_string());
        println!("{}: {}", name, t);
    }
}

fn definition_loc_for_type_dump(
    program: &Program,
    id: NameId,
) -> Option<expr_parser::parsing::Loc> {
    match program.definitions.get(&id)? {
        Defined::Func(funcs) => funcs
            .declarations
            .first()
            .copied()
            .or_else(|| funcs.implementations.first().copied())
            .map(|v| program.value_loc(v)),
        Defined::Type(t) => Some(program.type_expr_loc(*t)),
        _ => None,
    }
}

fn member_method_loc_for_type_dump(
    program: &Program,
    query: &str,
) -> Option<expr_parser::parsing::Loc> {
    let (struct_name, method_name) = query.split_once('.')?;
    if struct_name.is_empty() || method_name.is_empty() || method_name.contains('.') {
        return None;
    }

    let struct_id = lookup_global_name_id(program, struct_name)?;
    let methods = program.member_methods.get(&struct_id)?;
    let funcs = methods.iter().find_map(|(sid, funcs)| {
        (program.str_intern.resolve(*sid) == method_name).then_some(funcs)
    })?;

    funcs
        .declarations
        .first()
        .copied()
        .or_else(|| funcs.implementations.first().copied())
        .map(|v| program.value_loc(v))
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
            if trimmed == ":quit" || trimmed == ":exit" {
                return Ok(ReplInput::Quit);
            }
            if trimmed == ":reset" || trimmed == ":clear" || trimmed == ":c" || trimmed == ":r" {
                return Ok(ReplInput::Reset);
            }
            if trimmed.starts_with(":load") {
                let args = trimmed.split_whitespace().skip(1);
                return Ok(ReplInput::Load(args.map(String::from).collect()));
            }
            if let Some(rest) = trimmed.strip_prefix(":show-ast") {
                let Some(value) = parse_on_off(rest) else {
                    eprintln!("Usage: :show-ast <on|off>");
                    continue;
                };
                return Ok(ReplInput::SetShowAst(value));
            }
            if let Some(rest) = trimmed.strip_prefix(":type-dump") {
                let Some(value) = parse_on_off(rest) else {
                    eprintln!("Usage: :type-dump <on|off>");
                    continue;
                };
                return Ok(ReplInput::SetTypeDump(value));
            }
            if let Some(rest) = trimmed.strip_prefix(":origin-dump") {
                let Some(value) = parse_on_off(rest) else {
                    eprintln!("Usage: :origin-dump <on|off>");
                    continue;
                };
                return Ok(ReplInput::SetOriginDump(value));
            }
            if trimmed == ":modes" {
                return Ok(ReplInput::ShowModes);
            }
            if let Some(rest) = trimmed.strip_prefix(":types-of") {
                let names: Vec<String> = rest.split_whitespace().map(String::from).collect();
                match names.as_slice() {
                    [name] => return Ok(ReplInput::DumpTypesOf(name.clone())),
                    _ => {
                        eprintln!("Usage: :types-of <name>");
                        continue;
                    }
                }
            }
            if trimmed == ":types" {
                return Ok(ReplInput::DumpTypes);
            }
            if let Some(rest) = trimmed.strip_prefix(":types-of") {
                let names: Vec<String> = rest.split_whitespace().map(String::from).collect();
                match names.as_slice() {
                    [name] => return Ok(ReplInput::DumpTypesOf(name.clone())),
                    _ => {
                        eprintln!("Usage: :types-of <name>");
                        continue;
                    }
                }
            }
            if trimmed == ":origins" {
                return Ok(ReplInput::DumpOrigins);
            }
            if let Some(rest) = trimmed.strip_prefix(":origins-of") {
                let names: Vec<String> = rest.split_whitespace().map(String::from).collect();
                match names.as_slice() {
                    [name] => return Ok(ReplInput::DumpOriginsOf(name.clone())),
                    _ => {
                        eprintln!("Usage: :origins-of <name>");
                        continue;
                    }
                }
            }
            if let Some(rest) = trimmed.strip_prefix(":type") {
                let names: Vec<String> = rest.split_whitespace().map(String::from).collect();
                if names.is_empty() {
                    eprintln!("Usage: :type <name...>");
                    continue;
                }
                return Ok(ReplInput::ShowType(names));
            }
        }

        input.push_str(&line);
        if !needs_more_input(&input) {
            return Ok(ReplInput::Code(input));
        }

        first = false;
    }
}

fn parse_source(program: &mut Program, input: &str, file_id: usize, show_ast: bool) -> ParseBatch {
    let mut parser = Parser::new(input, file_id);
    let mut expr_count = 0;
    let mut infos = Vec::new();

    while !parser.is_empty() {
        match parser.parse_with_macros(program) {
            Ok(Some(expr)) => {
                let mut defined_names = Vec::new();
                collect_defined_names(&expr, &mut defined_names);
                let expr_loc = expr.loc.clone();
                let value_start = program.values.len();

                if show_ast {
                    println!(
                        "Expr {}: [{}..{}]",
                        expr_count + 1,
                        expr.loc.range.start,
                        expr.loc.range.end
                    );
                    println!("{}", pretty_print_expr(&expr, 0));
                }
                expr_count += 1;
                program.gather_definition(expr);

                let value_end = program.values.len();
                infos.push(ParsedExprInfo {
                    index: expr_count,
                    loc: expr_loc,
                    values_added: value_start..value_end,
                    defined_names,
                });
            }
            Ok(None) => break,
            Err(e) => {
                program.push_lowering_error(e);
            }
        }
    }

    ParseBatch { expr_count, infos }
}

fn report_all_errors(reporter: &mut ErrorReporter, program: &mut Program) -> bool {
    let errors = std::mem::take(&mut program.lowering_errors);
    let mut has_errors = false;
    for err in errors {
        let _ = reporter.report_compile_error(&err);
        has_errors = true;
    }
    has_errors
}

fn finalize_program(
    reporter: &mut ErrorReporter,
    program: &mut Program,
) -> Result<Option<(TypeUniverse, SolvedTypes)>, Box<dyn std::error::Error>> {
    program.check_pending_names();
    if report_all_errors(reporter, program) {
        return Ok(None);
    }

    let (result, _checked) = run_typechecker(program, reporter)?;
    let Ok(ans) = result else {
        return Ok(None);
    };

    Ok(Some(ans))
}

fn run_stdin_batch(options: CliOptions) -> Result<(), Box<dyn std::error::Error>> {
    let mut reporter = ErrorReporter::new();
    let mut program = Program::new();
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    if input.trim().is_empty() {
        return Ok(());
    }

    reporter.add_source(0, input.clone());
    let _batch = parse_source(&mut program, &input, 0, options.show_ast);

    if let Some((types, solved)) = finalize_program(&mut reporter, &mut program)? {
        if options.type_dump {
            reporter.report_type_dump(&program, &types, &solved)?;
        }
        if options.origin_dump {
            reporter.report_origin_dump(&program, &types, &solved)?;
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(options) = parse_cli_options().map_err(io::Error::other)? else {
        println!("{}", cli_usage());
        return Ok(());
    };

    if options.stdin_batch {
        return run_stdin_batch(options);
    }

    let mut reporter = ErrorReporter::new();
    let mut program = Program::new();
    let mut next_file_id = 0usize;
    let mut last_typecheck: Option<(TypeUniverse, SolvedTypes)> = None;
    let mut show_ast = options.show_ast;
    let mut type_dump = options.type_dump;
    let mut origin_dump = options.origin_dump;

    println!("Expression Parser REPL");
    println!("Enter expressions; REPL waits for complete input.");
    println!("AST printing is off by default. Pass --show-ast or use :show-ast on.");
    println!(
        "Commands: :load <path...>, :reset, :show-ast <on|off>, :type-dump <on|off>, :origin-dump <on|off>, :modes, :types, :types-of <name>, :type <name...>, :origins, :origins-of <name>, :quit, :exit"
    );

    loop {
        match read_repl_input() {
            Ok(ReplInput::Quit) => break,
            Ok(ReplInput::Reset) => {
                program = Program::new();
                last_typecheck = None;
                println!("REPL state cleared.");
            }
            Ok(ReplInput::SetShowAst(value)) => {
                show_ast = value;
                println!("show-ast: {}", if show_ast { "on" } else { "off" });
            }
            Ok(ReplInput::SetTypeDump(value)) => {
                type_dump = value;
                println!("type-dump: {}", if type_dump { "on" } else { "off" });
            }
            Ok(ReplInput::SetOriginDump(value)) => {
                origin_dump = value;
                println!("origin-dump: {}", if origin_dump { "on" } else { "off" });
            }
            Ok(ReplInput::ShowModes) => {
                println!(
                    "modes: show-ast={}, type-dump={}, origin-dump={}",
                    if show_ast { "on" } else { "off" },
                    if type_dump { "on" } else { "off" },
                    if origin_dump { "on" } else { "off" }
                );
            }
            Ok(ReplInput::ShowType(names)) => {
                let Some((types, solved)) = last_typecheck.as_ref() else {
                    println!("No successful typecheck yet. Enter code first.");
                    continue;
                };
                print_named_types(&program, types, solved, &names);
            }
            Ok(ReplInput::DumpTypes) => {
                let Some((types, solved)) = last_typecheck.as_ref() else {
                    println!("No successful typecheck yet. Enter code first.");
                    continue;
                };
                reporter.report_type_dump(&program, types, solved)?;
            }
            Ok(ReplInput::DumpTypesOf(name)) => {
                let Some((types, solved)) = last_typecheck.as_ref() else {
                    println!("No successful typecheck yet. Enter code first.");
                    continue;
                };

                let loc = if let Some(id) = lookup_global_name_id(&program, &name) {
                    let Some(loc) = definition_loc_for_type_dump(&program, id) else {
                        println!("{}: <no value/type definition span>", name);
                        continue;
                    };
                    loc
                } else {
                    let Some(loc) = member_method_loc_for_type_dump(&program, &name) else {
                        println!("{}: <not found>", name);
                        continue;
                    };
                    loc
                };

                reporter.report_type_dump_in_region(&program, types, solved, Some(&loc))?;
            }
            Ok(ReplInput::DumpOrigins) => {
                let Some((types, solved)) = last_typecheck.as_ref() else {
                    println!("No successful typecheck yet. Enter code first.");
                    continue;
                };
                reporter.report_origin_dump(&program, &types, &solved)?;
            }
            Ok(ReplInput::DumpOriginsOf(name)) => {
                let Some((_types, solved)) = last_typecheck.as_ref() else {
                    println!("No successful typecheck yet. Enter code first.");
                    continue;
                };

                let loc = if let Some(id) = lookup_global_name_id(&program, &name) {
                    let Some(loc) = definition_loc_for_type_dump(&program, id) else {
                        println!("{}: <no value/type definition span>", name);
                        continue;
                    };
                    loc
                } else {
                    let Some(loc) = member_method_loc_for_type_dump(&program, &name) else {
                        println!("{}: <not found>", name);
                        continue;
                    };
                    loc
                };

                reporter.report_origin_dump_in_region(&program, &solved, Some(&loc))?;
            }
            Ok(ReplInput::Load(paths)) => {
                if paths.is_empty() {
                    eprintln!("Usage: :load <path...>");
                    continue;
                }

                let mut batches = Vec::new();
                let mut parse_failed = false;
                for path in paths {
                    match fs::read_to_string(&path) {
                        Ok(contents) => {
                            let file_id = next_file_id;
                            next_file_id += 1;
                            reporter.add_source(file_id, contents.clone());

                            let batch = parse_source(&mut program, &contents, file_id, show_ast);
                            batches.push(batch);
                        }
                        Err(err) => {
                            eprintln!("Error reading {path}: {err}");
                            last_typecheck = None;
                            parse_failed = true;
                            break;
                        }
                    }
                }

                if parse_failed {
                    continue;
                }

                if let Some((types, solved)) = finalize_program(&mut reporter, &mut program)? {
                    for batch in &batches {
                        print_expr_types(&program, &types, &solved, batch);
                    }
                    if type_dump {
                        reporter.report_type_dump(&program, &types, &solved)?;
                    }
                    if origin_dump {
                        reporter.report_origin_dump(&program, &types, &solved)?;
                    }
                    last_typecheck = Some((types, solved));
                } else {
                    last_typecheck = None;
                }
            }
            Ok(ReplInput::Code(input)) => {
                let file_id = next_file_id;
                next_file_id += 1;
                reporter.add_source(file_id, input.clone());
                let batch = parse_source(&mut program, &input, file_id, show_ast);
                if let Some((types, solved)) = finalize_program(&mut reporter, &mut program)? {
                    print_expr_types(&program, &types, &solved, &batch);
                    if type_dump {
                        reporter.report_type_dump(&program, &types, &solved)?;
                    }
                    if origin_dump {
                        reporter.report_origin_dump(&program, &types, &solved)?;
                    }
                    last_typecheck = Some((types, solved));
                } else {
                    last_typecheck = None;
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
