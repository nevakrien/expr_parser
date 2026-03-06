use expr_parser::error_reporting::ErrorReporter;
use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use expr_parser::type_inference::run_typechecker;

const SOURCE: &str = r#"
// f=fn(){
//     var x = 2;
//     var p = &x;
//     *p=3;
//     p: &const int;
// }

S=struct{x:int,y:bool};
N=struct{n:int,s:S};

// const_struct_borrow = fn(self:&const S)->&mut int { &mut self.x };
const_two_members = fn(self:&const S){ self.x = 1; self.y = true; };
"#;
// const SOURCE: &str = r#"
// Wrapper = struct {baba:[int;1]};
// Wrapper.get = fn(self:&mut Wrapper)->&mut int {&mut self.baba[0]}

// Unsafe = struct { inner: &'raw Wrapper };
// Unsafe.__deref_mut = fn['a](self: &'raw mut Unsafe) -> &'a mut Wrapper  { &*self.inner };

// RawCalc = struct { inner: &'raw Unsafe };
// RawCalc.__deref_mut = fn(self: &'raw mut RawCalc) -> &'raw Unsafe { self.inner };

// Raw = struct { inner: &'static mut RawCalc };
// Raw.__deref_mut = fn(self: &mut Raw) -> &'static mut RawCalc { self.inner };

// Safe = struct { inner: &'raw Raw };
// Safe.__deref_mut = fn(self: &mut Safe) -> &mut Raw { &*self.inner };

// f = fn(s: &mut Safe) {
//     let out : &mut int = s->get();
//     let arr : [int;1] = s->baba;
// };
// // "#;
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
