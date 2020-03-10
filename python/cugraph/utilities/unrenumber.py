import cudf


def unrenumber(renumber_map, df, col):
    if isinstance(renumber_map, cudf.DataFrame):
        unrenumered_df = df.merge(renumber_map, left_on=col,
                                  right_on='id',
                                  how='left').drop(['id', col])
        cols = unrenumered_df.columns
        df = unrenumered_df[[cols[1:], cols[0]]]
    else:
        df[col] = renumber_map[df[col]].reset_index(drop=True)
    return df
