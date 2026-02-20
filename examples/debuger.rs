use expr_parser::error_reporting::ErrorReporter;
use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use expr_parser::type_inference::run_typechecker;

const SOURCE: &str = r#"
// type Pair = struct['a, 'b, T] {
//                 left: &'a T,
//                 right: &'b T,
//             }

//             f = fn['x](x:&'x int, y:&int)->Pair['x, '_, int] {
//                 Pair{ left = x, right = y }
//             }
// Wrapper = struct {inner:int};
//             Wrapper.get = fn(self:&mut Wrapper)->&mut int {&mut self.inner}

//             Unsafe = struct { inner: &'raw Wrapper };
//             Unsafe.__deref_mut = fn['a](self: &'raw mut Unsafe) -> &'a mut Wrapper  { &*self.inner };

//             RawCalc = struct { inner: &'raw Unsafe };
//             RawCalc.__deref_mut = fn(self: &'raw mut RawCalc) -> &'raw Unsafe { self.inner };

//             Raw = struct { inner: &'raw RawCalc };
//             Raw.__deref_mut = fn(self: &mut Raw) -> &'raw RawCalc { self.inner };

//             Safe = struct { inner: &'raw Raw };
//             Safe.__deref_mut = fn(self: &mut Safe) -> &mut Raw { &*self.inner };

//             f = fn(s: &mut Safe) {
//                 let out : &mut int = s->get();
//             };

Box=struct['a]{inner:&'a [int;2]}; Box.__deref_mut = fn['a](self:&mut Box['a])->&mut &'a [int;2] { &mut self.inner }; f = fn['a](b:Box['a])->int { let y:int = b[1:usize]; y };
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
