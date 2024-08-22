extern crate bindgen;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    // Specify the header file you want to generate bindings for
    let header_file = "nccl/nccl_net_v8.h";

    // Tell cargo to invalidate the built crate whenever the header changes
    println!("cargo:rerun-if-changed={}", header_file);

    // Generate the bindings
    let bindings = bindgen::Builder::default()
        .header(header_file)
        .clang_arg("-I/usr/include") // Add any necessary include paths here
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    //let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    // out path is src/bindings/
    let out_path = PathBuf::from("src/bindings/");
    let bindings_path = out_path.join("bindings.rs");
    bindings
        .write_to_file(bindings_path.clone())
        .expect("Couldn't write bindings!");

    let mut file = OpenOptions::new()
    .append(true)
    .open(&bindings_path)
    .expect("Couldn't open bindings file for appending");

    writeln!(file, "unsafe impl Send for ncclNet_v8_t {{}}").unwrap();
    writeln!(file, "unsafe impl Sync for ncclNet_v8_t {{}}").unwrap();
}