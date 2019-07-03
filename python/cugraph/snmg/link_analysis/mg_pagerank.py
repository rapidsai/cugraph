from cugraph.snmg.link_analysis import mg_pagerank_wrapper


def mg_pagerank(src_ptrs_info, dest_ptrs_info):
    df = mg_pagerank_wrapper.mg_pagerank(src_ptrs_info, dest_ptrs_info)

    return df
