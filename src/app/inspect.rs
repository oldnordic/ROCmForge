use crate::GgufFile;

pub(crate) fn list_tensors(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if path.ends_with(".rfm") {
        use rocmforge::loader::RfmFile;
        let file = RfmFile::open(path)?;
        let mut names: Vec<&str> = file.tensor_names().collect();
        names.sort_unstable();

        println!("{:<45} {:<20} SHAPE", "NAME", "TYPE");
        println!("{}", "-".repeat(80));
        for n in &names {
            if let Ok(Some(t)) = file.tensor(n) {
                println!("{:<45} {:<20?} {:?}", n, t.wtype, t.dims);
            }
        }
        println!("\nTotal: {} tensors", names.len());
        Ok(())
    } else {
        let file = GgufFile::open(path)?;
        let mut names: Vec<&str> = file.tensor_names().collect();
        names.sort_unstable();

        println!("{:<45} {:<20} SHAPE", "NAME", "TYPE");
        println!("{}", "-".repeat(80));
        for n in &names {
            if let Ok(Some(t)) = file.tensor(n) {
                println!("{:<45} {:<20} {:?}", n, t.ggml_type, t.dims);
            }
        }
        println!("\nTotal: {} tensors", names.len());
        Ok(())
    }
}
