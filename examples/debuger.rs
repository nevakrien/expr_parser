use expr_parser::error_reporting::ErrorReporter;
use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use expr_parser::type_inference::run_typechecker;

// const SOURCE: &str = "Box=struct{inner:&[int;2]}; f=fn(){};";
// const SOURCE: &str = r#"
// Box=struct['a]{inner:&'a [int;2]};
// Box.__deref_mut =
//   fn['a](self:&mut Box['a])->&mut &'a [int;2]
//     { &mut self.inner };
// f = fn['rand,'a](b:Box['a],random:&'rand int)->int { let y:int = b[1:usize]; y };
// "#;
const SOURCE: &str = r#"
type Pair = struct['a, 'b, T] {
    left: &'a T,
    right: &'b T,
}

f = fn['x](x:&'x int, y:&int)->Pair['x, '_, int] {
    Pair{ left = x, right = y }
}
"#;

// const SOURCE: &str = r#"
// // type Pair = struct['a, 'b, T] {
// //user code
//             Box = struct[T]{ptr:*T};

//             free = cfn(p:*void);
//             Box.__free = fn[T](b:&mut Box[T]){
//             (&*b.ptr).__free()
//             free(b->ptr as *void)
//             };

//             Box.__deref = fn[T](b:&const Box[T])->&T{&*b.ptr};
//             Box.__deref_mut = fn[T](b:&mut Box[T])->&mut T{&*b.ptr};

//             f=fn(b:Box[[int]])->int { let y:int = b[0]; y };
// "#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running debug pipeline");

    let mut program = Program::new();
    let mut parser = Parser::new(SOURCE, 0);
    let mut reporter = ErrorReporter::new();
    reporter.add_source(0, SOURCE.to_string());

    if let Err(errs) = program.lower_all(&mut parser) {
        for err in errs {
            let _ = reporter.report_compile_error(&err);
        }
        return Ok(());
    }

    // unsafe{asm!("int3");}
    if let Ok((result, _)) = run_typechecker(&program, &mut reporter) {
        if let Err(type_error_count) = result {
            println!("Type errors: {type_error_count}");
        }
        if let Ok((solved, store)) = result {
            reporter.report_type_dump(&program, &solved, &store)?;
        }
    }
    Ok(())
}
