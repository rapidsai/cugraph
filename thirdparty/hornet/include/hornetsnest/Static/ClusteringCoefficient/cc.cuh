#pragma once

#include "HornetAlg.hpp"

#include "Static/TriangleCounting/triangle2.cuh"


namespace hornets_nest {


using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;
using clusterCoeff_t =  float;
//==============================================================================

class ClusteringCoefficient : public TriangleCounting2 {
public:
    ClusteringCoefficient(HornetGraph& hornet);
    ~ClusteringCoefficient();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    void init();

    /// Array needs to be pre-allocated by user
    void copyLocalClusCoeffToHost(clusterCoeff_t* h_tcs);


private:
    // One value per vertex
   clusterCoeff_t* d_ccLocal { nullptr };

    // One value per graph
   clusterCoeff_t* d_ccGlobal { nullptr };   
   clusterCoeff_t  h_ccGlobal;   
};

//==============================================================================

} // namespace hornets_nest
