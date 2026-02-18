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

fn write_noise_block(writer: &mut BufWriter<File>, line_count: &mut usize, rng: &mut Rng) {
    match rng.range(8) {
        0 => {
            write_line(
                writer,
                line_count,
                "    { let cl = fn(x:int)->int { x + 1:int }; cl(3:int); id(seed); };",
            );
        }
        1 => {
            write_line(
                writer,
                line_count,
                "    { let pick = fn(a:int, b:int)->int { a + b }; pick(2:int, 5:int); id(extra); };",
            );
        }
        2 => {
            write_line(
                writer,
                line_count,
                "    { let local = fn(x:int)->int { x - 1:int }; local(9:int); let keep = pick_left(seed, seed); id(keep); };",
            );
        }
        3 => {
            write_line(
                writer,
                line_count,
                "    { let call = fn(x:int)->int { x }; call(0:int); let v = id(extra); id(v); };",
            );
        }
        4 => {
            write_line(
                writer,
                line_count,
                "    { let p = Pair{ left = 1:int, right = 2:int }; p.left + p.right; let keep = id(seed); id(keep); };",
            );
        }
        5 => {
            write_line(
                writer,
                line_count,
                "    { let v = Vec2{ x = 3:int, y = 4:int }; let w = Vec2{ x = 1:int, y = 2:int }; let z = v + w; z.norm1(); id(extra); };",
            );
        }
        6 => {
            write_line(
                writer,
                line_count,
                "    { let cond = (1:int + 2:int) == 3:int; if cond { id(seed); } else { id(seed); }; id(extra); };",
            );
        }
        _ => {
            write_line(
                writer,
                line_count,
                "    { let raw = opaque_alloc(8:usize); let p = raw as *Node; cast_void(p); let l = Late{ value = 2:int }; late_score(l); };",
            );
        }
    }
}

fn write_header(writer: &mut BufWriter<File>, line_count: &mut usize) {
    let lines = [
        "free = cfn(p:*void);",
        "opaque_alloc = cfn(n:usize)->*void;",
        "id = fn[T](x:T)->T { x };",
        "pick_left = fn[A, B](a:A, _b:B)->A { a };",
        "cast_void = fn[T](p:*T)->*void { p as *void };",
        "late_score = fn(l:Late)->int { l.value + 1:int };",
        "Pair = struct[A, B] { left:A, right:B };",
        "Vec2 = struct { x:int, y:int };",
        "Node = struct { value:int };",
        "Point = struct { x:int, y:int };",
        "Box = struct[T] { ptr:*T };",
        "Wrap = struct[T] { boxed:Box[T] };",
        "Deep = struct[T] { inner:Wrap[T] };",
        "ArrBox = struct { arr:[int;4] };",
        "Late = struct { value:int };",
        "Pair.swap = fn[A, B](p:Pair[A, B])->Pair[B, A] { Pair{ left = p.right, right = p.left } };",
        "Vec2.__add = fn(a:Vec2, b:Vec2)->Vec2 { Vec2{ x = a.x + b.x, y = a.y + b.y } };",
        "Vec2.__sub = fn(a:Vec2, b:Vec2)->Vec2 { Vec2{ x = a.x - b.x, y = a.y - b.y } };",
        "Vec2.norm1 = fn(v:Vec2)->int { v.x + v.y };",
        "Point.sum = fn(p:Point)->int { p.x + p.y };",
        "Point.shift = fn(p:Point, dx:int, dy:int)->Point { Point{ x = p.x + dx, y = p.y + dy } };",
        "Box.__free = fn[T](b:&mut Box[T]) { free(b->ptr as *void) };",
        "Box.__deref = fn[T](b:&const Box[T])->&T { &*b.ptr };",
        "Box.__deref_mut = fn[T](b:&mut Box[T])->&mut T { &*b.ptr };",
        "Box.get = fn[T](b:Box[T])->T { *b };",
        "Wrap.__deref = fn[T](w:&const Wrap[T])->&Box[T] { &w.boxed };",
        "Wrap.__deref_mut = fn[T](w:&mut Wrap[T])->&mut Box[T] { &mut w.boxed };",
        "Deep.__deref = fn[T](d:&const Deep[T])->&Wrap[T] { &d.inner };",
        "Deep.__deref_mut = fn[T](d:&mut Deep[T])->&mut Wrap[T] { &mut d.inner };",
        "ArrBox.__deref = fn(a:&const ArrBox)->&[int;4] { &a.arr };",
        "ArrBox.__deref_mut = fn(a:&mut ArrBox)->&mut [int;4] { &mut a.arr };",
    ];

    for line in lines {
        write_line(writer, line_count, line);
    }
}

fn write_dynamic_struct(writer: &mut BufWriter<File>, line_count: &mut usize, idx: usize) {
    write_line(
        writer,
        line_count,
        &format!("S{idx} = struct {{ value:int, delta:int }};"),
    );
    write_line(
        writer,
        line_count,
        &format!(
            "S{idx}.score = fn[T, U](s:S{idx}, seed:T, extra:U)->int {{ {{ let c = fn(x:int)->int {{ x + 1:int }}; c(s.value); id(seed); id(extra); }}; s.value + s.delta }};"
        ),
    );
}

fn write_box_chain_fn(
    writer: &mut BufWriter<File>,
    line_count: &mut usize,
    name: &str,
    rng: &mut Rng,
) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn[T, U](seed:T, extra:U, b:Box[Box[Node]])->int {{"),
    );
    write_line(writer, line_count, "    let out = b->value;");
    write_noise_block(writer, line_count, rng);
    write_line(writer, line_count, "    out");
    write_line(writer, line_count, "};");
}

fn write_deep_chain_fn(
    writer: &mut BufWriter<File>,
    line_count: &mut usize,
    name: &str,
    rng: &mut Rng,
) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn[T, U](seed:T, extra:U, d:Deep[Node])->int {{"),
    );
    write_line(writer, line_count, "    let out = d->value;");
    write_noise_block(writer, line_count, rng);
    write_line(writer, line_count, "    out");
    write_line(writer, line_count, "};");
}

fn write_vec_overload_fn(
    writer: &mut BufWriter<File>,
    line_count: &mut usize,
    name: &str,
    rng: &mut Rng,
) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn[T, U](seed:T, extra:U, a:Vec2, b:Vec2)->int {{"),
    );
    write_line(writer, line_count, "    let c = a + b;");
    write_noise_block(writer, line_count, rng);
    write_line(writer, line_count, "    let d = c - a;");
    write_line(writer, line_count, "    d.norm1()");
    write_line(writer, line_count, "};");
}

fn write_pair_swap_fn(
    writer: &mut BufWriter<File>,
    line_count: &mut usize,
    name: &str,
    rng: &mut Rng,
) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn[T, U](seed:T, extra:U, b:Box[Pair[int, bool]])->int {{"),
    );
    write_line(writer, line_count, "    let p = b->swap();");
    write_noise_block(writer, line_count, rng);
    write_line(
        writer,
        line_count,
        "    if p.left { p.right } else { 0:int }",
    );
    write_line(writer, line_count, "};");
}

fn write_array_index_fn(
    writer: &mut BufWriter<File>,
    line_count: &mut usize,
    name: &str,
    rng: &mut Rng,
) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn[T, U](seed:T, extra:U, a:ArrBox)->int {{"),
    );
    write_line(writer, line_count, "    let x = a[1:usize];");
    write_noise_block(writer, line_count, rng);
    write_line(writer, line_count, "    x + a[2:usize]");
    write_line(writer, line_count, "};");
}

fn write_point_shift_fn(
    writer: &mut BufWriter<File>,
    line_count: &mut usize,
    name: &str,
    rng: &mut Rng,
) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn[T, U](seed:T, extra:U, w:Wrap[Point])->int {{"),
    );
    write_line(writer, line_count, "    let q = w->shift(1:int, 2:int);");
    write_noise_block(writer, line_count, rng);
    write_line(writer, line_count, "    q.sum()");
    write_line(writer, line_count, "};");
}

fn write_cast_fn(writer: &mut BufWriter<File>, line_count: &mut usize, name: &str, rng: &mut Rng) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn[T, U](seed:T, extra:U, p:*Point)->*void {{"),
    );
    write_noise_block(writer, line_count, rng);
    write_line(writer, line_count, "    cast_void(p)");
    write_line(writer, line_count, "};");
}

fn write_generic_box_fn(
    writer: &mut BufWriter<File>,
    line_count: &mut usize,
    name: &str,
    rng: &mut Rng,
) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn[T, U](seed:T, extra:U, b:Box[T], fallback:T)->T {{"),
    );
    write_line(writer, line_count, "    let x = b.get();");
    write_noise_block(writer, line_count, rng);
    write_line(writer, line_count, "    let _ = id(extra);");
    write_line(writer, line_count, "    pick_left(x, fallback)");
    write_line(writer, line_count, "};");
}

fn write_late_type_fn(
    writer: &mut BufWriter<File>,
    line_count: &mut usize,
    name: &str,
    rng: &mut Rng,
) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn[T, U](seed:T, extra:U, v:int)->int {{"),
    );
    write_line(writer, line_count, "    let l = Late{ value = v };");
    write_noise_block(writer, line_count, rng);
    write_line(writer, line_count, "    late_score(l)");
    write_line(writer, line_count, "};");
}

fn write_alloc_cast_fn(
    writer: &mut BufWriter<File>,
    line_count: &mut usize,
    name: &str,
    rng: &mut Rng,
) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn[T, U](seed:T, extra:U)->*Node {{"),
    );
    write_noise_block(writer, line_count, rng);
    write_line(writer, line_count, "    opaque_alloc(8:usize) as *Node");
    write_line(writer, line_count, "};");
}

fn write_dynamic_struct_use_fn(
    writer: &mut BufWriter<File>,
    line_count: &mut usize,
    name: &str,
    sid: usize,
    rng: &mut Rng,
) {
    write_line(
        writer,
        line_count,
        &format!("{name} = fn[T, U](seed:T, extra:U, s:S{sid})->int {{"),
    );
    write_line(writer, line_count, "    let base = s.score(seed, extra);");
    write_noise_block(writer, line_count, rng);
    write_line(writer, line_count, "    base");
    write_line(writer, line_count, "};");
}

#[allow(unreachable_code)]
fn main() {
    panic!("this has closures we dont do closures");
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
    let mut struct_count = 0usize;
    let mut dynamic_structs: Vec<usize> = Vec::new();

    while line_count < target_lines {
        if dynamic_structs.is_empty() || rng.range(4) == 0 {
            write_dynamic_struct(&mut writer, &mut line_count, struct_count);
            dynamic_structs.push(struct_count);
            struct_count += 1;
            if line_count >= target_lines {
                break;
            }
        }

        let name = format!("f_{function_count}");
        let sid = dynamic_structs[rng.range(dynamic_structs.len() as u32) as usize];
        match rng.range(11) {
            0 => write_box_chain_fn(&mut writer, &mut line_count, &name, &mut rng),
            1 => write_deep_chain_fn(&mut writer, &mut line_count, &name, &mut rng),
            2 => write_vec_overload_fn(&mut writer, &mut line_count, &name, &mut rng),
            3 => write_pair_swap_fn(&mut writer, &mut line_count, &name, &mut rng),
            4 => write_array_index_fn(&mut writer, &mut line_count, &name, &mut rng),
            5 => write_point_shift_fn(&mut writer, &mut line_count, &name, &mut rng),
            6 => write_cast_fn(&mut writer, &mut line_count, &name, &mut rng),
            7 => write_generic_box_fn(&mut writer, &mut line_count, &name, &mut rng),
            8 => write_late_type_fn(&mut writer, &mut line_count, &name, &mut rng),
            9 => write_alloc_cast_fn(&mut writer, &mut line_count, &name, &mut rng),
            _ => write_dynamic_struct_use_fn(&mut writer, &mut line_count, &name, sid, &mut rng),
        }
        function_count += 1;
    }

    println!(
        "Wrote {} functions and {} dynamic structs to {} (total lines {}, target lines {})",
        function_count, struct_count, output_path, line_count, target_lines
    );
}
