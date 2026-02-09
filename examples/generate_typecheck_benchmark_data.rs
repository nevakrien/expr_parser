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
        if max == 0 {
            0
        } else {
            self.next_u32() % max
        }
    }
}

const DEFAULT_LINE_TARGET: usize = 100_000;

fn write_line(writer: &mut BufWriter<File>, line_count: &mut usize, line: &str) {
    writeln!(writer, "{line}").expect("failed to write line");
    *line_count += 1;
}

fn write_header(writer: &mut BufWriter<File>, line_count: &mut usize) {
    let lines = [
        "inc = macro(x) { x + 1:int };",
        "dec = macro(x) { x - 1:int };",
        "add = macro(a, b) { a + b };",
        "mul = macro(a, b) { a * b };",
        "bitmix = macro(a, b) { (a & b) ^ (a | b) };",
        "to_int = macro(x) { x as int };",
        "to_float = macro(x) { x as float };",
        "type Pair = struct[T, U] { a:T, b:U };",
        "type Box = struct[T] { value:T };",
        "type Point = struct{ x:int, y:int };",
        "id = fn[T](x:T)->T { x };",
        "wrap = fn[T](x:T)->Box[T] { Box{ value = x } };",
        "make_pair = fn[A, B](a:A, b:B)->Pair[A, B] { Pair{ a = a, b = b } };",
        "make_point = fn(a:int, b:int)->Point { Point{ x = a, y = b } };",
    ];

    for line in lines {
        write_line(writer, line_count, line);
    }
}

fn write_int_math(writer: &mut BufWriter<File>, line_count: &mut usize, name: &str) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn(a:int, b:int)->int {{"),
    );
    write_line(writer, line_count, "    let x:int = inc(a);");
    write_line(writer, line_count, "    let y:int = bitmix(x, b);");
    write_line(writer, line_count, "    let z:int = add(y, 3:int);");
    write_line(writer, line_count, "    let w:bool = (z & 1:int) == 0:int;");
    write_line(writer, line_count, "    if w { z } else { dec(z) }");
    write_line(writer, line_count, "};");
}

fn write_float_math(writer: &mut BufWriter<File>, line_count: &mut usize, name: &str) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn(a:float, b:float)->float {{"),
    );
    write_line(writer, line_count, "    let x:float = add(a, b);");
    write_line(writer, line_count, "    let y:float = mul(x, 1.25:float);");
    write_line(writer, line_count, "    let z:int = to_int(y);");
    write_line(writer, line_count, "    let w:float = to_float(z);");
    write_line(writer, line_count, "    add(w, y)");
    write_line(writer, line_count, "};");
}

fn write_mixed_math(writer: &mut BufWriter<File>, line_count: &mut usize, name: &str) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn(a:int, b:float)->float {{"),
    );
    write_line(writer, line_count, "    let x:float = to_float(a);");
    write_line(writer, line_count, "    let y:float = add(x, b);");
    write_line(writer, line_count, "    let z:int = to_int(y);");
    write_line(writer, line_count, "    let q:int = id(z);");
    write_line(
        writer,
        line_count,
        "    let r:int = (fn(x:int)->int { x + 1:int })(q);",
    );
    write_line(writer, line_count, "    to_float(r)");
    write_line(writer, line_count, "};");
}

fn write_point_fn(writer: &mut BufWriter<File>, line_count: &mut usize, name: &str) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn(a:int, b:int)->Point {{"),
    );
    write_line(writer, line_count, "    let p = Point{ x = a, y = b };");
    write_line(writer, line_count, "    p");
    write_line(writer, line_count, "};");
}

fn write_pair_fn(writer: &mut BufWriter<File>, line_count: &mut usize, name: &str) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn(a:int, b:float) {{"),
    );
    write_line(writer, line_count, "    let p = Pair{ a = a, b = b };");
    write_line(writer, line_count, "    let _ = wrap(a);");
    write_line(writer, line_count, "    let _ = make_pair(a, b);");
    write_line(writer, line_count, "    p");
    write_line(writer, line_count, "};");
}

fn write_bool_fn(writer: &mut BufWriter<File>, line_count: &mut usize, name: &str) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn(a:int, b:int)->bool {{"),
    );
    write_line(writer, line_count, "    let x:int = add(a, b);");
    write_line(writer, line_count, "    (x & 1:int) == 0:int");
    write_line(writer, line_count, "};");
}

fn main() {
    let output_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "typecheck_benchmark_data.txt".to_string());
    let target_lines = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_LINE_TARGET);

    let file = File::create(&output_path).expect("failed to create output file");
    let mut writer = BufWriter::new(file);
    let mut line_count = 0usize;

    write_header(&mut writer, &mut line_count);

    let mut rng = Rng::new(0xBADA_5515);
    let mut function_count = 0usize;
    while line_count < target_lines {
        let name = format!("f_{function_count}");
        match rng.range(6) {
            0 => write_int_math(&mut writer, &mut line_count, &name),
            1 => write_float_math(&mut writer, &mut line_count, &name),
            2 => write_mixed_math(&mut writer, &mut line_count, &name),
            3 => write_point_fn(&mut writer, &mut line_count, &name),
            4 => write_pair_fn(&mut writer, &mut line_count, &name),
            _ => write_bool_fn(&mut writer, &mut line_count, &name),
        }
        function_count += 1;
    }

    println!(
        "Wrote {} functions to {} (total lines {}, target lines {})",
        function_count, output_path, line_count, target_lines
    );
}
