use rocmforge::loader::ModelFile;

pub(crate) fn list_tensors(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = ModelFile::open(path)?;
    let mut names: Vec<String> = file.tensor_names().map(|s| s.to_string()).collect();
    names.sort_unstable();

    println!("{:<45} {:<20} SHAPE", "NAME", "TYPE");
    println!("{}", "-".repeat(80));
    for n in &names {
        if let Some(t) = file.tensor_desc(n) {
            println!("{:<45} {:<20} {:?}", n, t.type_name, t.dims);
        }
    }
    println!("\nTotal: {} tensors", names.len());
    Ok(())
}
