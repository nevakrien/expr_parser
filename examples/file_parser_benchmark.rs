use expr_parser::parsing::Parser;
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
        mapping.name_mapping("parser_input");
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
    println!("Running file-based benchmark (single large file)");

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmark_data.txt".to_string());

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

    let start = Instant::now();
    let mut parsed_count = 0;
    let mut error_count = 0;
    let mut expressions = Vec::new();

    let mut parser = Parser::new(file_content, 0);
    while !parser.is_empty() {
        match parser.consume_stmt() {
            Ok(expr) => {
                expressions.push(expr);
                parsed_count += 1;
            }
            Err(_) => {
                error_count += 1;
                break;
            }
        }
    }

    let duration = start.elapsed();
    let total_iterations = parsed_count + error_count;
    let exprs_per_second = total_iterations as f64 / duration.as_secs_f64();

    println!("Results:");
    println!("  Successfully parsed: {}", parsed_count);
    println!("  Errors: {}", error_count);
    println!("  Total iterations: {}", total_iterations);
    println!("  Time: {:?}", duration);
    println!("  Expressions per second: {:.2}", exprs_per_second);
    println!("  Stored expressions: {}", expressions.len());
    println!(
        "  Microseconds per expression: {:.2}",
        duration.as_micros() as f64 / total_iterations as f64
    );
}
