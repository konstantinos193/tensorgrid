fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .out_dir("src/proto")
        .compile(
            &["../../protocols/control.proto"],
            &["../../protocols"],
        )?;
    println!("cargo:rerun-if-changed=../../protocols/control.proto");
    Ok(())
}
