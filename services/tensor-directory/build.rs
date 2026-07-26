fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/proto")
        .compile(
            &["../../protocols/memory.proto"],
            &["../../protocols"],
        )?;
    println!("cargo:rerun-if-changed=../../protocols/memory.proto");
    Ok(())
}
