use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use std::time::Instant;

const ITERATIONS: usize = 60000;

const SOURCE: &str = r#"
mk = macro(name, body) { name = macro() { body } }
wrap = macro(name, body) { name = macro(x) { body(x) } }

mk(m0, 0)
mk(m1, m0())
mk(m2, m1())
mk(m3, m2())
mk(m4, m3())
mk(m5, m4())
mk(m6, m5())
mk(m7, m6())
mk(m8, m7())
mk(m9, m8())
mk(m10, m9())
mk(m11, m10())
mk(m12, m11())
mk(m13, m12())
mk(m14, m13())
mk(m15, m14())
mk(m16, m15())
mk(m17, m16())
mk(m18, m17())
mk(m19, m18())

wrap(w0, m19)
wrap(w1, w0)
wrap(w2, w1)
wrap(w3, w2)
wrap(w4, w3)
wrap(w5, w4)
wrap(w6, w5)
wrap(w7, w6)
wrap(w8, w7)
wrap(w9, w8)
wrap(w10, w9)
wrap(w11, w10)
wrap(w12, w11)
wrap(w13, w12)
wrap(w14, w13)
wrap(w15, w14)
wrap(w16, w15)
wrap(w17, w16)
wrap(w18, w17)
wrap(w19, w18)

w19(1)
"#;

fn main() {
    println!("Running macro expansion benchmark (nested macro definitions)");

    let start = Instant::now();
    let mut expanded_count = 0usize;
    let mut error_count = 0usize;

    for _ in 0..ITERATIONS {
        let mut program = Program::new();
        let mut parser = Parser::new(SOURCE, 0);

        match program.lower_all(&mut parser) {
            Ok(()) => expanded_count += 1,
            Err(_) => error_count += 1,
        }
    }

    let duration = start.elapsed();
    let total_iterations = expanded_count + error_count;
    let exprs_per_second = total_iterations as f64 / duration.as_secs_f64();

    println!("Results:");
    println!("  Expanded expressions: {}", expanded_count);
    println!("  Errors: {}", error_count);
    println!("  Total iterations: {}", total_iterations);
    println!("  Time: {:?}", duration);
    println!("  Expressions per second: {:.2}", exprs_per_second);
    println!(
        "  Microseconds per expression: {:.2}",
        duration.as_micros() as f64 / total_iterations as f64
    );
}
