use expr_parser::parsing::Parser;
use expr_parser::program::{Defined, Program};
use expr_parser::type_inference::{
    infer_global_types, infer_value_internals, InferState, TypeStore,
};
use std::time::Instant;

const ITERATIONS: usize = 40000;

const SOURCE: &str = r#"
f = fn() -> i32 {
    let a: i32 = 1 + 2;
    let b = a + 3;
    let c = b + 4;
    c
}
"#;

fn main() {
    println!("Running typechecker benchmark (single simple function)");

    let start = Instant::now();
    let mut ok_count = 0usize;
    let mut error_count = 0usize;

    for _ in 0..ITERATIONS {
        let mut program = Program::new();
        let mut parser = Parser::new(SOURCE, 0);
        if program.lower_all(&mut parser).is_err() {
            error_count += 1;
            continue;
        }

        let mut types = TypeStore::new();
        let mut ctx = InferState::new(&mut types, &program, ());
        let Ok(globals) = infer_global_types(&mut ctx) else {
            error_count += 1;
            continue;
        };
        let mut ctx = ctx.map_global(&globals);
        for (_, def) in program.definitions.iter() {
            let Defined::Func(v) = def else {
                continue;
            };
            match infer_value_internals(&mut ctx, *v) {
                Ok(_) => ok_count += 1,
                Err(_) => error_count += 1,
            }
        }
    }

    let duration = start.elapsed();
    let total_iterations = ok_count + error_count;
    let exprs_per_second = total_iterations as f64 / duration.as_secs_f64();

    println!("Results:");
    println!("  Ok: {}", ok_count);
    println!("  Errors: {}", error_count);
    println!("  Total iterations: {}", total_iterations);
    println!("  Time: {:?}", duration);
    println!("  Iterations per second: {:.2}", exprs_per_second);
    println!(
        "  Microseconds per iteration: {:.2}",
        duration.as_micros() as f64 / total_iterations as f64
    );
}
