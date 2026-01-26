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

    fn range(&mut self, max: u32) -> u32 {
        if max == 0 { 0 } else { self.next_u32() % max }
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        if values.len() <= 1 {
            return;
        }
        for idx in (1..values.len()).rev() {
            let swap_idx = (self.next_u32() as usize) % (idx + 1);
            values.swap(idx, swap_idx);
        }
    }
}

struct FunctionDef {
    name: String,
    arity: usize,
}

struct MacroDef {
    name: String,
    params: Vec<String>,
    body: String,
}

fn number(rng: &mut Rng) -> String {
    if rng.range(5) == 0 {
        format!("{}.{:03}", rng.range(1000), rng.range(1000))
    } else {
        format!("{}", rng.range(1_000_000))
    }
}

fn bin_op(rng: &mut Rng) -> &'static str {
    let _ = rng;
    "+"
}

fn gen_params(arity: usize) -> Vec<String> {
    (0..arity).map(|idx| format!("p{}", idx)).collect()
}

fn gen_call(
    rng: &mut Rng,
    macros: &[MacroDef],
    funcs: &[FunctionDef],
    params: &[String],
    locals: &[String],
    depth: u32,
) -> String {
    let idx = rng.range(funcs.len() as u32) as usize;
    let func = &funcs[idx];
    let mut args = Vec::with_capacity(func.arity);
    for _ in 0..func.arity {
        args.push(gen_value_expr(rng, macros, funcs, params, locals, depth));
    }
    format!("{}({})", func.name, args.join(", "))
}

fn gen_macro_call(
    rng: &mut Rng,
    macros: &[MacroDef],
    funcs: &[FunctionDef],
    params: &[String],
    locals: &[String],
    depth: u32,
) -> String {
    let idx = rng.range(macros.len() as u32) as usize;
    let mac = &macros[idx];
    let mut args = Vec::with_capacity(mac.params.len());
    for _ in 0..mac.params.len() {
        args.push(gen_value_expr(rng, macros, funcs, params, locals, depth));
    }
    format!("{}({})", mac.name, args.join(", "))
}

fn pick_atom(
    rng: &mut Rng,
    macros: &[MacroDef],
    funcs: &[FunctionDef],
    params: &[String],
    locals: &[String],
    depth: u32,
) -> String {
    let mut choices = 2;
    if !params.is_empty() {
        choices += 1;
    }
    if !locals.is_empty() {
        choices += 1;
    }
    if !macros.is_empty() {
        choices += 1;
    }
    if !funcs.is_empty() {
        choices += 1;
    }

    match rng.range(choices) {
        0 => number(rng),
        1 => number(rng),
        2 if !params.is_empty() => params[rng.range(params.len() as u32) as usize].clone(),
        3 if !locals.is_empty() => locals[rng.range(locals.len() as u32) as usize].clone(),
        4 if !macros.is_empty() => gen_macro_call(rng, macros, funcs, params, locals, depth),
        _ if !funcs.is_empty() => gen_call(rng, macros, funcs, params, locals, depth),
        _ => number(rng),
    }
}

fn gen_value_expr(
    rng: &mut Rng,
    macros: &[MacroDef],
    funcs: &[FunctionDef],
    params: &[String],
    locals: &[String],
    depth: u32,
) -> String {
    if depth == 0 {
        return pick_atom(rng, macros, funcs, params, locals, depth);
    }

    format!(
        "{} {} {}",
        gen_value_expr(rng, macros, funcs, params, locals, depth - 1),
        bin_op(rng),
        gen_value_expr(rng, macros, funcs, params, locals, depth - 1)
    )
}

fn gen_local_name(rng: &mut Rng, used: &[String]) -> String {
    loop {
        let candidate = format!("v{}_{}", rng.range(1000), rng.range(1000));
        if !used.iter().any(|name| name == &candidate) {
            return candidate;
        }
    }
}

fn gen_function_body(
    rng: &mut Rng,
    macros: &[MacroDef],
    funcs: &[FunctionDef],
    params: &[String],
    max_statements: u32,
    max_depth: u32,
) -> String {
    let max_locals = max_statements.saturating_sub(1).min(3) as u32;
    let local_count = rng.range(max_locals + 1) as usize;
    let mut locals: Vec<String> = Vec::new();
    let mut statements: Vec<String> = Vec::new();

    for _ in 0..local_count {
        let mut used = Vec::new();
        used.extend_from_slice(params);
        used.extend_from_slice(&locals);
        let local_name = gen_local_name(rng, &used);
        let value = gen_value_expr(rng, macros, funcs, params, &locals, max_depth);
        statements.push(format!("let {local_name} = {value}"));
        locals.push(local_name);
    }

    let base_count = local_count as u32 + 1;
    let extra_allowed = max_statements.saturating_sub(base_count);
    let extra_count = rng.range(extra_allowed + 1) as usize;
    for _ in 0..extra_count {
        let expr = gen_value_expr(rng, macros, funcs, params, &locals, max_depth);
        statements.push(expr);
    }

    let tail_expr = gen_value_expr(rng, macros, funcs, params, &locals, max_depth);
    if statements.is_empty() {
        return format!("{{ {tail_expr} }}");
    }
    format!("{{ {}; {} }}", statements.join("; "), tail_expr)
}

fn is_valid_line(line: &str) -> bool {
    let mut parser = Parser::new(line, 0);
    let Ok(expr) = parser.consume_stmt() else {
        return false;
    };
    let _ = expr;
    parser.is_empty()
}

// Default number of examples to generate
const DEFAULT_FUNCTION_COUNT: usize = 100_000;

fn main() {
    let output_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "compile_benchmark_data.txt".to_string());
    let function_count = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_FUNCTION_COUNT);
    let max_statements = std::env::args()
        .nth(3)
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(4);
    let max_depth = std::env::args()
        .nth(4)
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(2);

    let file = File::create(&output_path).expect("failed to create output file");
    let mut writer = BufWriter::new(file);
    let mut rng = Rng::new(0xDEC0_DEAD);
    let mut functions = Vec::with_capacity(function_count + 2);
    let macros = vec![
        MacroDef {
            name: "inc".to_string(),
            params: vec!["x".to_string()],
            body: "x + 1".to_string(),
        },
        MacroDef {
            name: "wrap".to_string(),
            params: vec!["x".to_string()],
            body: "{ let _ = x; x }".to_string(),
        },
    ];

    for mac in &macros {
        let line = format!(
            "{} = macro({}) {{ {} }};",
            mac.name,
            mac.params.join(", "),
            mac.body
        );
        if !is_valid_line(&line) {
            panic!("Generated invalid macro: {line}");
        }
        writeln!(writer, "{line}").expect("failed to write line");
    }

    let forward_a = "mut_a".to_string();
    let forward_b = "mut_b".to_string();
    let mut def_lines = Vec::with_capacity(function_count + 2);
    let forward_lines = [
        format!("{forward_a} = fn(x) {{ let v = inc(x); {forward_b}(v) + v }};"),
        format!("{forward_b} = fn(x) {{ let v = wrap(x); {forward_a}(v) + v }};"),
    ];
    for line in forward_lines {
        if !is_valid_line(&line) {
            panic!("Generated invalid forward definition: {line}");
        }
        def_lines.push(line);
    }
    functions.push(FunctionDef {
        name: forward_a.clone(),
        arity: 1,
    });
    functions.push(FunctionDef {
        name: forward_b.clone(),
        arity: 1,
    });

    for idx in 0..function_count {
        let arity = 1 + rng.range(3) as usize;
        let name = format!("f{}", idx);
        functions.push(FunctionDef { name, arity });
    }

    for func in &functions {
        if func.name == forward_a || func.name == forward_b {
            continue;
        }
        let params = gen_params(func.arity);
        let body = gen_function_body(
            &mut rng,
            &macros,
            &functions,
            &params,
            max_statements,
            max_depth,
        );
        let line = format!("{} = fn({}) {};", func.name, params.join(", "), body);
        if !is_valid_line(&line) {
            panic!("Generated invalid definition: {line}");
        }
        def_lines.push(line);
    }

    rng.shuffle(&mut def_lines);
    for line in def_lines {
        writeln!(writer, "{line}").expect("failed to write line");
    }

    println!(
        "Wrote {} functions to {} (max statements {}, max depth {})",
        function_count + 2,
        output_path,
        max_statements,
        max_depth
    );
}
