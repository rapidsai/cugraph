import cudf


def unrenumber(renumber_map, df, col):
    if isinstance(renumber_map, cudf.DataFrame):
        unrenumbered_df = df.merge(renumber_map, left_on=col,
                                   right_on='id',
                                   how='left').drop(['id', col])
        cols = unrenumbered_df.columns.to_list()
        df = unrenumbered_df[cols[1:] + [cols[0]]]
    else:
        tmp = cudf.DataFrame()
        tmp['OuTpUt'] = renumber_map
        tmp['id'] = tmp.index
        df['OrIgInAl'] = df.index
        unrenumbered_df = df.merge(
            tmp, left_on=col, right_on='id', how='left'
        ).sort_values('OrIgInAl').reset_index(drop=True)
        df[col] = unrenumbered_df['OuTpUt']
        df = df.drop('OrIgInAl')
    return df
