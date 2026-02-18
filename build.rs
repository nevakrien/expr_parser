use std::time::{SystemTime};

fn main() {
    // Force rebuild every time
    println!("cargo:rerun-if-changed=perf_counter.c");
    println!("cargo:rerun-if-env-changed=FORCE_REBUILD");

    // Also trick Cargo by emitting current timestamp
    let now = SystemTime::now();
    println!("cargo:rustc-env=PERF_BUILD_TIME={:?}", now);

    cc::Build::new()
        .file("perf_counter.c")
        .compile("perf_counter");
}
// fn main() {}
