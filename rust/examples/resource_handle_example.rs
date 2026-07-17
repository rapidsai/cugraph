mod utils;

#[cfg(feature = "runtime-link")]
fn main() {
    match utils::create_resource_handle() {
        Ok(handle) => {
            println!("resource handle created");
            utils::free_resource_handle(handle);
        }
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "runtime-link"))]
fn main() {
    println!("Enable feature runtime-link to run this example against libcugraph_c");
}
