use expr_parser::error_reporting::ErrorReporter;
use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use expr_parser::type_inference::run_typechecker;

const SOURCE: &str = r#"
free = cfn(p:*void);
Box = struct[T]{ptr:*T};
Box.__free = fn[T](b:&mut Box[T]){free(b->ptr as *void)}
Box.__deref = fn[T](b:&const Box[T])->&T{&*b.ptr}
Box.__deref_mut = fn[T](b:&mut Box[T])->&mut T{&*b.ptr}
Box.get = fn[T](b:Box[T])->T {*b}


f=fn(b:Box[Box[Box[int]]])->int {*b}

"#;

fn main() {
    println!("Running debug pipeline");

    let mut program = Program::new();
    let mut parser = Parser::new(SOURCE, 0);
    let mut reporter = ErrorReporter::new();
    reporter.add_source(0, SOURCE.to_string());

    if let Err(errs) = program.lower_all(&mut parser) {
        for err in errs {
            let _ = reporter.report_compile_error(&err);
        }
        return;
    }

    // unsafe{asm!("int3");}
    if let Ok((result, _)) = run_typechecker(&program, &mut reporter) {
        if let Err(type_error_count) = result {
            println!("Type errors: {type_error_count}");
        }
    }
}
