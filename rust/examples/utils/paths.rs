use std::path::PathBuf;

pub const DEFAULT_DATASET_DIR: &str = "../datasets";

pub fn resolve_dataset_path(file_name: &str, directory: Option<&str>) -> PathBuf {
    let base_dir = PathBuf::from(directory.unwrap_or(DEFAULT_DATASET_DIR));
    base_dir.join(file_name)
}
