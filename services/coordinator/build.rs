fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Skip proto compilation for now - coordinator will use HTTP-only API
    // This is a temporary workaround for development without protoc
    println!("cargo:warning=Skipping proto compilation - using HTTP-only coordinator");
    Ok(())
}
