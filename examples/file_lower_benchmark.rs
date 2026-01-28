use expr_parser::error_reporting::ErrorReporter;
use expr_parser::parsing::Parser;
use expr_parser::program::Program;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::time::Instant;

struct MappedFile {
    ptr: *mut libc::c_void,
    len: usize,
}

impl MappedFile {
    fn map(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|err| format!("Error opening {path}: {err}"))?;
        let metadata = file
            .metadata()
            .map_err(|err| format!("Error reading metadata for {path}: {err}"))?;
        let len = metadata.len() as usize;
        if len == 0 {
            return Err(format!("File {path} is empty"));
        }

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(format!("mmap failed for {path}"));
        }

        let mapping = Self { ptr, len };
        mapping.name_mapping("compiler_input");
        Ok(mapping)
    }

    fn name_mapping(&self, name: &str) {
        #[cfg(target_os = "linux")]
        unsafe {
            let cstr = std::ffi::CString::new(name).ok();
            if let Some(cstr) = cstr {
                let _ = libc::prctl(
                    libc::PR_SET_VMA,
                    libc::PR_SET_VMA_ANON_NAME,
                    self.ptr,
                    self.len,
                    cstr.as_ptr(),
                );
            }
        }
    }

    fn as_str(&self) -> Result<&str, String> {
        let bytes = unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) };
        std::str::from_utf8(bytes).map_err(|err| format!("Invalid UTF-8: {err}"))
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}

fn main() {
    println!("Running file-based lower benchmark (single large file)");

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "lower_benchmark_data.txt".to_string());

    let mapping = match MappedFile::map(&path) {
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

    let statement_count = file_content.lines().count();
    let start = Instant::now();
    let mut program = Program::new();
    let mut parser = Parser::new(file_content, 0);
    let mut reporter = ErrorReporter::new();
    reporter.add_source(0, file_content.to_string());

    let mut compiled_count = 0;
    let mut error_count = 0;
    if let Err(err) = program.lower_all(&mut parser) {
        error_count = 1;
        let _ = reporter.report_compile_error(&err);
    } else {
        compiled_count = statement_count;
    }

    let duration = start.elapsed();
    let total_iterations = compiled_count + error_count;
    let exprs_per_second = total_iterations as f64 / duration.as_secs_f64();

    println!("Results:");
    println!("  Statements in file: {}", statement_count);
    println!("  Successfully compiled: {}", compiled_count);
    println!("  Errors: {}", error_count);
    println!("  Total iterations: {}", total_iterations);
    println!("  Time: {:?}", duration);
    println!("  Statements per second: {:.2}", exprs_per_second);
    println!(
        "  Microseconds per statement: {:.2}",
        duration.as_micros() as f64 / total_iterations as f64
    );
    println!(
        "  Milliseconds per statement: {:.4}",
        duration.as_secs_f64() * 1000.0 / total_iterations as f64
    );
}
