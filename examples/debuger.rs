use expr_parser::error_reporting::ErrorReporter;
use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use expr_parser::type_inference::run_typechecker;

const SOURCE: &str = r#"
S=struct{x:int}
f=fn(){
    let s=S{2};
    let i = 1+s.x;
}
"#;

fn main() {
    println!("Running debug pipeline");

    let mut program = Program::new();
    let mut parser = Parser::new(SOURCE, 0);
    let mut reporter = ErrorReporter::new();
    reporter.add_source(0, SOURCE.to_string());

    if let Err(err) = program.lower_all(&mut parser) {
        let _ = reporter.report_compile_error(&err);
        return;
    }

    if let Ok((result, _)) = run_typechecker(&program, &mut reporter) {
        if let Err(type_error_count) = result {
            println!("Type errors: {type_error_count}");
        }
    }
}
