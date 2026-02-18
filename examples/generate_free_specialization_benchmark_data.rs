use std::fs::File;
use std::io::{BufWriter, Write};

const DEFAULT_FAMILY_COUNT: usize = 2000;

fn write_line(writer: &mut BufWriter<File>, line_count: &mut usize, line: &str) {
    writeln!(writer, "{line}").expect("failed to write line");
    *line_count += 1;
}

fn write_prelude(writer: &mut BufWriter<File>, line_count: &mut usize) {
    let lines = [
        "__user_free=fn[T](p:&mut T);",
        "free = cfn(p:*void);",
        "opaque_alloc = cfn(n:usize)->*void;",
    ];

    for line in lines {
        write_line(writer, line_count, line);
    }
}

fn write_family(writer: &mut BufWriter<File>, line_count: &mut usize, i: usize) {
    write_line(
        writer,
        line_count,
        &format!("Base{i} = struct {{ raw:*void, payload:int }};"),
    );
    write_line(
        writer,
        line_count,
        &format!("Box{i} = struct[T] {{ inner:T }};"),
    );
    write_line(
        writer,
        line_count,
        &format!("Deep{i} = struct[T] {{ inner:Box{i}[Box{i}[T]] }};"),
    );
    write_line(
        writer,
        line_count,
        &format!("Shell{i} = struct[K, V] {{ key:K, inner:Deep{i}[V] }};"),
    );

    write_line(
        writer,
        line_count,
        &format!("free_base_impl_{i} = fn(x:&mut Base{i}) {{ free(x.raw) }};"),
    );
    write_line(
        writer,
        line_count,
        &format!("free_box_impl_{i} = fn[T](x:&mut Box{i}[T]) {{ __user_free(&mut x.inner) }};"),
    );
    write_line(
        writer,
        line_count,
        &format!(
            "free_deep_impl_{i} = fn[T](x:&mut Deep{i}[T]) {{ free_box_impl_{i}(&mut x.inner) }};"
        ),
    );
    write_line(
        writer,
        line_count,
        &format!(
            "free_shell_impl_{i} = fn[K, V](x:&mut Shell{i}[K, V]) {{ __user_free(&mut x.key); free_deep_impl_{i}(&mut x.inner) }};"
        ),
    );

    write_line(
        writer,
        line_count,
        &format!("free_base_{i} = fn(x:&mut Base{i}) {{ free_base_impl_{i}(x) }};"),
    );
    write_line(
        writer,
        line_count,
        &format!("free_box_{i} = fn[T](x:&mut Box{i}[T]) {{ free_box_impl_{i}(x) }};"),
    );
    write_line(
        writer,
        line_count,
        &format!("free_deep_{i} = fn[T](x:&mut Deep{i}[T]) {{ free_deep_impl_{i}(x) }};"),
    );
    write_line(
        writer,
        line_count,
        &format!("free_shell_{i} = fn[K, V](x:&mut Shell{i}[K, V]) {{ free_shell_impl_{i}(x) }};"),
    );

    write_line(
        writer,
        line_count,
        &format!(
            "build_and_free_{i} = fn()->int {{ let x = Base{i}{{ raw = opaque_alloc(8:usize), payload = {i}:int }}; free_base_impl_{i}(&mut x); x.payload }};"
        ),
    );
}

fn main() {
    let output_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "free_specialization_benchmark_data.txt".to_string());
    let family_count = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_FAMILY_COUNT);

    let file = File::create(&output_path).expect("failed to create output file");
    let mut writer = BufWriter::new(file);
    let mut line_count = 0usize;

    write_prelude(&mut writer, &mut line_count);

    for i in 0..family_count {
        write_family(&mut writer, &mut line_count, i);
    }

    println!(
        "Wrote {} specialization families to {} (total lines {})",
        family_count, output_path, line_count
    );
    println!(
        "Run benchmark with: cargo run --release --example file_typecheck_benchmark -- {}",
        output_path
    );
}
