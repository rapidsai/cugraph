import cudf


def unrenumber(renumber_map, df, col):
    if isinstance(renumber_map, cudf.DataFrame):
        unrenumbered_df = df.merge(renumber_map, left_on=col,
                                   right_on='id',
                                   how='left').drop(['id', col])
        cols = unrenumbered_df.columns.to_list()
        df = unrenumbered_df[cols[1:] + [cols[0]]]
    else:
        df[col] = renumber_map[df[col]]
    return df
