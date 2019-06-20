#pragma once

#include "HornetAlg.hpp"

namespace hornets_nest {

using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;

struct KCoreData {
    vid_t *src;
    vid_t *dst;
    int   *counter;
};

class KCore : public StaticAlgorithm<HornetGraph> {
public:
    KCore(HornetGraph &hornet);
    ~KCore();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }
    void set_hcopy(HornetGraph *h_copy);

private:
    HostDeviceVar<KCoreData> hd_data;

    long edge_vertex_count;

    load_balancing::VertexBased1 load_balancing;

    TwoLevelQueue<vid_t> vqueue;
    TwoLevelQueue<vid_t> peel_vqueue;
    TwoLevelQueue<vid_t> active_queue;
    TwoLevelQueue<vid_t> iter_queue;

    vid_t *vertex_pres { nullptr };
    vid_t *vertex_color { nullptr };
    vid_t *vertex_deg { nullptr };
};

}
