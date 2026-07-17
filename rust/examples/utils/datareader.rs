use polars::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::paths::resolve_dataset_path;

fn infer_separator(path: &Path) -> PolarsResult<u8> {
    let file = File::open(path).map_err(|e| {
        PolarsError::ComputeError(
            format!("failed to open dataset file '{}': {e}", path.display()).into(),
        )
    })?;

    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line.map_err(|e| {
            PolarsError::ComputeError(
                format!("failed reading dataset file '{}': {e}", path.display()).into(),
            )
        })?;

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.contains(',') {
            return Ok(b',');
        }
        if trimmed.contains('\t') {
            return Ok(b'\t');
        }
        return Ok(b' ');
    }

    Err(PolarsError::ComputeError(
        format!("dataset file '{}' is empty", path.display()).into(),
    ))
}

/// Read an edge dataset and split it into src, dst, wt DataFrames.
///
/// The dataset is expected to contain at least three columns in this order:
/// source vertex, destination vertex, and edge weight.
///
/// `directory` defaults to ../datasets when `None`.
pub fn read_triplet_dataframes(
    file_name: &str,
    directory: Option<&str>,
) -> PolarsResult<(DataFrame, DataFrame, DataFrame)> {
    let dataset_path = resolve_dataset_path(file_name, directory);
    let separator = infer_separator(&dataset_path)?;

    let df = CsvReader::from_path(&dataset_path)?
        .has_header(false)
        .with_separator(separator)
        .finish()?;

    if df.width() < 3 {
        return Err(PolarsError::ComputeError(
            format!(
                "dataset '{}' must contain at least 3 columns (src, dst, wt), found {}",
                dataset_path.display(),
                df.width()
            )
            .into(),
        ));
    }

    let src_name = df.get_columns()[0].name().to_owned();
    let dst_name = df.get_columns()[1].name().to_owned();
    let wt_name = df.get_columns()[2].name().to_owned();

    let mut src_df = df.select([src_name.as_str()])?;
    src_df.set_column_names(&["src"])?;

    let mut dst_df = df.select([dst_name.as_str()])?;
    dst_df.set_column_names(&["dst"])?;

    let mut wt_df = df.select([wt_name.as_str()])?;
    wt_df.set_column_names(&["wt"])?;

    Ok((src_df, dst_df, wt_df))
}
