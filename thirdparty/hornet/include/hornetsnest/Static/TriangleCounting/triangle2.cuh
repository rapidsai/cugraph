#pragma once

#include "HornetAlg.hpp"


namespace hornets_nest {

//using triangle_t = int;
using triangle_t = unsigned long long;
using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;


//==============================================================================

class TriangleCounting2 : public StaticAlgorithm<HornetGraph> {
public:
    TriangleCounting2(HornetGraph& hornet);
    ~TriangleCounting2();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    void run(const int WORK_FACTOR);
    void init();
    void copyTCToHost(triangle_t* h_tcs);

    triangle_t countTriangles();

protected:
   triangle_t* triPerVertex { nullptr };

};

//==============================================================================

} // namespace hornets_nest
