use expr_parser::type_inference::run_typechecker;
use expr_parser::error_reporting::ErrorReporter;
use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use std::time::Instant;

mod mapped_file;
use mapped_file::MappedFile;

fn main() {
    println!("Running file-based typecheck benchmark (single large file)");

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "typecheck_benchmark_data.txt".to_string());

    let mapping = match MappedFile::map(&path, "typechecker_input") {
        Ok(mapping) => mapping,
        Err(message) => {
            eprintln!("{message}");
            return;
        }
    };

    let file_content = match mapping.as_str() {
        Ok(content) => content,
        Err(message) => {
            eprintln!("{message}");
            return;
        }
    };

    let line_count = file_content.lines().count();
    let start = Instant::now();

    let mut program = Program::new();
    let mut parser = Parser::new(file_content, 0);
    let mut reporter = ErrorReporter::new();
    reporter.add_source(0, file_content.to_string());

    let mut compile_error_count = 0usize;
    if let Err(err) = program.lower_all(&mut parser) {
        compile_error_count = 1;
        let _ = reporter.report_compile_error(&err);
    }

    let mut type_error_count = 0usize;
    let mut functions_checked = 0usize;

    if compile_error_count == 0 {
        let (r,checked) = run_typechecker(&program,&mut reporter).unwrap();
        functions_checked+=checked;
        if let Err(ec) = r {
            type_error_count+=ec;
        }

    }

    let duration = start.elapsed();
    let total_errors = compile_error_count + type_error_count;
    let funcs_per_second = if functions_checked == 0 {
        0.0
    } else {
        functions_checked as f64 / duration.as_secs_f64()
    };

    let lines_per_second = if line_count == 0 {
        0.0
    } else {
        line_count as f64 / duration.as_secs_f64()
    };

    println!("Results:");
    println!("  Lines in file: {}", line_count);
    println!("  Functions checked: {}", functions_checked);
    println!("  Compile errors: {}", compile_error_count);
    println!("  Type errors: {}", type_error_count);
    println!("  Total errors: {}", total_errors);
    println!("  Time: {:?}", duration);
    println!("  Lines per second: {:.2}", lines_per_second);
    println!("  Functions per second: {:.2}", funcs_per_second);
    if line_count > 0 {
        println!(
            "  Microseconds per line: {:.2}",
            duration.as_micros() as f64 / line_count as f64
        );
    }
}
