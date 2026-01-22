use expr_parser::parsing::Parser;
use std::fs::File;
use std::io::{BufWriter, Write};

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn pick<'a>(&mut self, items: &'a [&'a str]) -> &'a str {
        let idx = (self.next_u32() as usize) % items.len();
        items[idx]
    }

    fn range(&mut self, max: u32) -> u32 {
        if max == 0 { 0 } else { self.next_u32() % max }
    }
}

fn ident(rng: &mut Rng) -> String {
    let bases = [
        "foo", "bar", "baz", "alpha", "beta", "gamma", "delta", "theta",
    ];
    let suffix = rng.range(10); //most code isnt that unique
    format!("{}_{}", rng.pick(&bases), suffix)
}

fn number(rng: &mut Rng) -> String {
    if rng.range(4) == 0 {
        format!("{}.{:03}", rng.range(1000), rng.range(1000))
    } else {
        format!("{}", rng.range(1_000_000))
    }
}

fn string_lit(rng: &mut Rng) -> String {
    let parts = ["hello", "world", "value", "expr", "parser", "bench"];
    format!("\"{}_{}\"", rng.pick(&parts), rng.range(1000))
}

fn bin_op(rng: &mut Rng) -> &'static str {
    let ops = [
        "+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "|", "^", "&",
    ];
    ops[(rng.next_u32() as usize) % ops.len()]
}

fn gen_expr(rng: &mut Rng, depth: u32) -> String {
    if depth == 0 {
        return match rng.range(4) {
            0 => ident(rng),
            1 => number(rng),
            2 => string_lit(rng),
            _ => format!("({})", ident(rng)),
        };
    }

    match rng.range(10) {
        0 => format!("({})", gen_expr(rng, depth - 1)),
        1 => format!(
            "{} {} {}",
            gen_expr(rng, depth - 1),
            bin_op(rng),
            gen_expr(rng, depth - 1)
        ),
        2 => format!(
            "{}({}, {})",
            ident(rng),
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        3 => format!("{}[{}:{}]", ident(rng), rng.range(10), rng.range(10) + 1),
        4 => format!(
            "if {} {} else {}",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        5 => format!(
            "match {} {{ 0 => {}, 1 => {}, _ => {} }}",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        6 => format!("let {} = {}", ident(rng), gen_expr(rng, depth - 1)),
        7 => format!(
            "fn({}: {}, {}) {}",
            ident(rng),
            ident(rng),
            ident(rng),
            gen_expr(rng, depth - 1)
        ),
        8 => format!(
            "{{ {} {} }}",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        _ => format!("{} = {}", ident(rng), gen_expr(rng, depth - 1)),
    }
}

fn is_valid_line(line: &str) -> bool {
    let mut parser = Parser::new(line, 0);
    let Ok(expr) = parser.consume_stmt() else {
        return false;
    };
    expr.loc.range.end == line.len() && parser.is_empty()
}

fn main() {
    let output_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmark_data.txt".to_string());
    let expr_count = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(300_000);
    let max_lines = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);

    let file = File::create(&output_path).expect("failed to create output file");
    let mut writer = BufWriter::new(file);
    let mut rng = Rng::new(0xC0FFEE);
    let mut written = 0usize;
    let mut rejected = 0usize;

    while written < expr_count {
        let depth = 2 + rng.range(3);
        let candidate = format!("{};", gen_expr(&mut rng, depth));
        if !is_valid_line(&candidate) {
            rejected += 1;
            continue;
        }

        let line_count = candidate.lines().count();
        if line_count > max_lines {
            rejected += 1;
            continue;
        }

        writeln!(writer, "{candidate}").expect("failed to write line");
        written += 1;
    }

    println!(
        "Wrote {} expressions to {} (rejected {}, max lines {})",
        written, output_path, rejected, max_lines
    );
}
