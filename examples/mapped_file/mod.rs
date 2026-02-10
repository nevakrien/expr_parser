#[cfg(target_os = "linux")]
mod platform {
    use simdutf8::basic::from_utf8;
    use std::fs::File;
    use std::os::fd::AsRawFd;

    pub struct MappedFile {
        ptr: *mut libc::c_void,
        len: usize,
    }

    impl MappedFile {
        pub fn map(path: &str, name: &str) -> Result<Self, String> {
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
            mapping.name_mapping(name);
            Ok(mapping)
        }

        pub fn as_str(&self) -> Result<&str, String> {
            let bytes = unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) };
            from_utf8(bytes).map_err(|err| format!("Invalid UTF-8: {err}"))
        }

        fn name_mapping(&self, name: &str) {
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
    }

    impl Drop for MappedFile {
        fn drop(&mut self) {
            unsafe {
                libc::munmap(self.ptr, self.len);
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    pub struct MappedFile {
        contents: Vec<u8>,
    }

    impl MappedFile {
        pub fn map(path: &str, _name: &str) -> Result<Self, String> {
            let contents =
                std::fs::read(path).map_err(|err| format!("Error opening {path}: {err}"))?;
            if contents.is_empty() {
                return Err(format!("File {path} is empty"));
            }

            Ok(Self { contents })
        }

        pub fn as_str(&self) -> Result<&str, String> {
            simdutf8::basic::from_utf8(&self.contents)
                .map_err(|err| format!("Invalid UTF-8: {err}"))
        }
    }
}

pub use platform::MappedFile;
