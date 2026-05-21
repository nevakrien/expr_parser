use std::time::SystemTime;

fn main() {
        // println!("cargo:rerun-if-changed=perf_counter.c");
        // println!("cargo:rerun-if-env-changed=FORCE_REBUILD");

        // let now = SystemTime::now();
        // println!("cargo:rustc-env=PERF_BUILD_TIME={:?}", now);

        // cc::Build::new()
        //     .file("perf_counter.c")
        //     .flag_if_supported("-Wno-unused-function")
        //     .flag_if_supported("-Wno-unused-variable")
        //     .flag_if_supported("-Wno-unused-parameter")
        //     .flag_if_supported("-Wno-comment")
        //     .compile("perf_counter");
}
// fn main() {}
