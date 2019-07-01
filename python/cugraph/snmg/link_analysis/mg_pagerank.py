import cugraph.snmg.link_analysis.mg_pagerank as cpp_mg_pagerank


def mg_pagerank(src_ptrs_info, dest_ptrs_info):
    df = cpp_mg_pagerank.mg_pagerank(src_ptrs_info, dest_ptrs_info)

    return df
