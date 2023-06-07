from cugraph.experimental.gnn import BulkSampler

def bulk_sample(graph, batch_size, fanout, seeds_per_call, batches_per_partition, batch_df, samples_dir, seed):
    sampler = BulkSampler(
        batch_size=batch_size,
        output_path=samples_dir,
        graph=graph,
        fanout_vals=fanout,
        with_replacement=False,
        random_state=seed,
        seeds_per_call=seeds_per_call,
        batches_per_partition=batches_per_partition,
    )

    sampler.add_batches(batch_df, start_col_name='node', batch_col_name='batch')
    sampler.flush()