use expr_parser::error_reporting::ErrorReporter;
use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use expr_parser::type_kinds::run_typechecker;

// const SOURCE: &str = r#"
// get_static = fn()->&'static &'static void;
// weird_func = fn['a,'b,T](r:&'a &'b void,y:&'a T)->&'a T{
//     y
// }

// cheat = fn[T](x:&T)->&T {
//     weird_func(&*get_static(),x)
// }
// "#;
// const SOURCE: &str = r#"
// get_static = fn()->&'static &'static void;
// weird_func = fn['a,'b,T](r:&'a &'b void,y:&'b T)->&'a T{
//     y
// }

// cheat = fn['a,'b,T](x:&'a T)->&'b T {
//     weird_func(get_static(),x)
// }
// "#;
const SOURCE: &str = r#"
    f=fn['a,'b](r1:&'a &'a int,r2:&'a &'b int)->&'a &'a int {
        & & * * r2
    }
"#;

// const SOURCE: &str = r#"
//     f=fn['a,'b](r1:&'a &'a int)->&'b &'a int {
//         & & * * r1
//     }
// "#;
// const SOURCE: &str = r#"
// sqrt = fn(f:float)->float;
// Point = struct{x:float,y:float}
// Point.dist = fn(self:&Point,other:Point)->float {
//     let dx = self.x-other.x
//     let dy = self.y-other.y
//     sqrt(dx*dx+dy*dy)
// }
// __user_free = fn(self:&mut Point) {}
// Point.new = fn()->Point {Point{0.0,0.0}}
// Point.__add = fn(p1:&Point,p2:Point)->Point {Point{p1.x+p2.x,p1.y+p2.y}}
// f=fn(p1:Point,p2:Point)->Point {p1+p2}
// "#;

// const SOURCE: &str = r#"
// Box = struct[T]{ptr:&'raw T};

// free = cfn(p:*void);
// no_fail_alloc = cfn(s:usize)->*void;
// Box.new = fn[T](x:T)->Box[T] {
//   let p=no_fail_alloc(x.__size_of());
//   Box{p as &'raw _}
// }
// Box.__free = fn[T](b:&mut Box[T]){
// (*b.ptr).__free()
// free(b->ptr as *void)
// }

// Box.__deref = fn[T](b:&const Box[T])->&T{&*b.ptr}
// Box.__deref_mut = fn[T](b:&mut Box[T])->&mut T{&*b.ptr}

// S=struct{x:bool};

// f=fn(b:&mut Box[Box[S]])->&mut bool {
//     // b->x=false;
//     let ans = &mut b->x;
//     *ans=true
//     ans
// };
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
