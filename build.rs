extern crate bindgen;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;

fn main() -> io::Result<()>{
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
    let mut file = File::create(&bindings_path)?;
    writeln!(file, "#![allow(non_camel_case_types)]")?;
    writeln!(file, "#![allow(non_upper_case_globals)]")?;
    writeln!(file, "#![allow(non_snake_case)]")?;
    writeln!(file, "#![allow(dead_code)]")?;
    
    // Write the bindings to the file
    file.write_all(bindings.to_string().as_bytes())?;

    // Open the file for appending and add the unsafe impls
    let mut file = OpenOptions::new()
        .append(true)
        .open(&bindings_path)
        .expect("Couldn't open bindings file for appending");

    writeln!(file, "unsafe impl Send for ncclNet_v8_t {{}}")?;
    writeln!(file, "unsafe impl Sync for ncclNet_v8_t {{}}")?;
    Ok(())
}