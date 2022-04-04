/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <utilities/base_fixture.hpp>
#include <utilities/high_res_timer.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <raft/cudart_utils.h>
#include <rmm/exec_policy.hpp>
#include <tuple>
#include <vector>

#include <cuda_profiler_api.h>

typedef struct InducedEgo_Usecase_t {
  std::string graph_file_full_path{};
  std::vector<int32_t> ego_sources{};
  int32_t radius;
  bool test_weighted{false};

  InducedEgo_Usecase_t(std::string const& graph_file_path,
                       std::vector<int32_t> const& ego_sources,
                       int32_t radius,
                       bool test_weighted)
    : ego_sources(ego_sources), radius(radius), test_weighted(test_weighted)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} InducedEgo_Usecase;

class Tests_InducedEgo : public ::testing::TestWithParam<InducedEgo_Usecase> {
 public:
  Tests_InducedEgo() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(InducedEgo_Usecase const& configuration)
  {
    int n_streams    = std::min(configuration.ego_sources.size(), static_cast<std::size_t>(128));
    auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(n_streams);
    raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

    cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, false> graph(handle);
    std::tie(graph, std::ignore) = cugraph::test::
      read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, configuration.graph_file_full_path, configuration.test_weighted, false);
    auto graph_view = graph.view();

    rmm::device_uvector<vertex_t> d_ego_sources(configuration.ego_sources.size(),
                                                handle.get_stream());

    raft::update_device(d_ego_sources.data(),
                        configuration.ego_sources.data(),
                        configuration.ego_sources.size(),
                        handle.get_stream());

    HighResTimer hr_timer;
    hr_timer.start("egonet");
    cudaProfilerStart();
    auto [d_ego_edgelist_src, d_ego_edgelist_dst, d_ego_edgelist_weights, d_ego_edge_offsets] =
      cugraph::extract_ego(handle,
                           graph_view,
                           d_ego_sources.data(),
                           static_cast<vertex_t>(configuration.ego_sources.size()),
                           configuration.radius);
    cudaProfilerStop();
    hr_timer.stop();
    hr_timer.display(std::cout);
    std::vector<size_t> h_cugraph_ego_edge_offsets(d_ego_edge_offsets.size());
    std::vector<vertex_t> h_cugraph_ego_edgelist_src(d_ego_edgelist_src.size());
    std::vector<vertex_t> h_cugraph_ego_edgelist_dst(d_ego_edgelist_dst.size());
    raft::update_host(h_cugraph_ego_edgelist_src.data(),
                      d_ego_edgelist_src.data(),
                      d_ego_edgelist_src.size(),
                      handle.get_stream());
    raft::update_host(h_cugraph_ego_edgelist_dst.data(),
                      d_ego_edgelist_dst.data(),
                      d_ego_edgelist_dst.size(),
                      handle.get_stream());
    raft::update_host(h_cugraph_ego_edge_offsets.data(),
                      d_ego_edge_offsets.data(),
                      d_ego_edge_offsets.size(),
                      handle.get_stream());
    ASSERT_TRUE(d_ego_edge_offsets.size() == (configuration.ego_sources.size() + 1));
    ASSERT_TRUE(d_ego_edgelist_src.size() == d_ego_edgelist_dst.size());
    if (configuration.test_weighted)
      ASSERT_TRUE(d_ego_edgelist_src.size() == (*d_ego_edgelist_weights).size());
    ASSERT_TRUE(h_cugraph_ego_edge_offsets[configuration.ego_sources.size()] ==
                d_ego_edgelist_src.size());
    for (size_t i = 0; i < configuration.ego_sources.size(); i++)
      ASSERT_TRUE(h_cugraph_ego_edge_offsets[i] <= h_cugraph_ego_edge_offsets[i + 1]);
    auto n_vertices = graph_view.number_of_vertices();
    for (size_t i = 0; i < d_ego_edgelist_src.size(); i++) {
      ASSERT_TRUE(cugraph::is_valid_vertex(n_vertices, h_cugraph_ego_edgelist_src[i]));
      ASSERT_TRUE(cugraph::is_valid_vertex(n_vertices, h_cugraph_ego_edgelist_dst[i]));
    }
  }
};

TEST_P(Tests_InducedEgo, CheckInt32Int32FloatTransposeFalse)
{
  run_current_test<int32_t, int32_t, float, false>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_InducedEgo,
  ::testing::Values(
    InducedEgo_Usecase("test/datasets/karate.mtx", std::vector<int32_t>{0}, 1, false),
    InducedEgo_Usecase("test/datasets/karate.mtx", std::vector<int32_t>{0}, 2, false),
    InducedEgo_Usecase("test/datasets/karate.mtx", std::vector<int32_t>{1}, 3, false),
    InducedEgo_Usecase("test/datasets/karate.mtx", std::vector<int32_t>{10, 0, 5}, 2, false),
    InducedEgo_Usecase("test/datasets/karate.mtx", std::vector<int32_t>{9, 3, 10}, 2, false),
    InducedEgo_Usecase(
      "test/datasets/karate.mtx", std::vector<int32_t>{5, 9, 3, 10, 12, 13}, 2, true)));

// For perf analysis
/*
INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_InducedEgo,
  ::testing::Values(
    InducedEgo_Usecase("test/datasets/soc-LiveJournal1.mtx", std::vector<int32_t>{0}, 1, false),
    InducedEgo_Usecase("test/datasets/soc-LiveJournal1.mtx", std::vector<int32_t>{0}, 2, false),
    InducedEgo_Usecase("test/datasets/soc-LiveJournal1.mtx", std::vector<int32_t>{0}, 3, false),
    InducedEgo_Usecase("test/datasets/soc-LiveJournal1.mtx", std::vector<int32_t>{0}, 4, false),
    InducedEgo_Usecase("test/datasets/soc-LiveJournal1.mtx", std::vector<int32_t>{0}, 5, false),
    InducedEgo_Usecase(
      "test/datasets/soc-LiveJournal1.mtx", std::vector<int32_t>{363617}, 2, false),
    InducedEgo_Usecase(
      "test/datasets/soc-LiveJournal1.mtx",
      std::vector<int32_t>{
        363617, 722214, 2337449, 2510183, 2513389, 225853, 2035807, 3836330, 1865496, 28755},
      2,
      false),
    InducedEgo_Usecase(
      "test/datasets/soc-LiveJournal1.mtx",
      std::vector<int32_t>{
        363617,  722214,  2337449, 2510183, 2513389, 225853,  2035807, 3836330, 1865496, 28755,
        2536834, 3070144, 3888415, 3131712, 2382526, 1040771, 2631543, 4607218, 4465829, 3341686,
        2772973, 2611175, 4526129, 2624421, 1220593, 2593137, 3270705, 1503899, 1213033, 4840102,
        4529036, 3421116, 4264831, 4089751, 4272322, 3486998, 2830318, 320953,  2388331, 520808,
        3023094, 1600294, 3631119, 1716614, 4829213, 1175844, 960680,  847662,  3277365, 3957318,
        3455123, 2454259, 670953,  4465677, 1027332, 2560721, 89061,   1163406, 3109528, 3221856,
        4714426, 2382774, 37828,   4433616, 3283229, 591911,  4200188, 442522,  872207,  2437601,
        741003,  266241,  914618,  3626195, 2021080, 4679624, 777476,  2527796, 1114017, 640142,
        49259,   4069879, 3869098, 1105040, 4707804, 3208582, 3325885, 1450601, 4072548, 2037062,
        2029646, 4575891, 1488598, 79105,   4827273, 3795434, 4647518, 4733397, 3980718, 1184627},
      2,
      false),
    InducedEgo_Usecase(
      "test/datasets/soc-LiveJournal1.mtx",
      std::vector<int32_t>{
        363617,  722214,  2337449, 2510183, 2513389, 225853,  2035807, 3836330, 1865496, 28755,
        2536834, 3070144, 3888415, 3131712, 2382526, 1040771, 2631543, 4607218, 4465829, 3341686,
        2772973, 2611175, 4526129, 2624421, 1220593, 2593137, 3270705, 1503899, 1213033, 4840102,
        4529036, 3421116, 4264831, 4089751, 4272322, 3486998, 2830318, 320953,  2388331, 520808,
        3023094, 1600294, 3631119, 1716614, 4829213, 1175844, 960680,  847662,  3277365, 3957318,
        3455123, 2454259, 670953,  4465677, 1027332, 2560721, 89061,   1163406, 3109528, 3221856,
        4714426, 2382774, 37828,   4433616, 3283229, 591911,  4200188, 442522,  872207,  2437601,
        741003,  266241,  914618,  3626195, 2021080, 4679624, 777476,  2527796, 1114017, 640142,
        49259,   4069879, 3869098, 1105040, 4707804, 3208582, 3325885, 1450601, 4072548, 2037062,
        2029646, 4575891, 1488598, 79105,   4827273, 3795434, 4647518, 4733397, 3980718, 1184627,
        984983,  3114832, 1967741, 1599818, 144593,  2698770, 2889449, 2495550, 1053813, 1193622,
        686026,  3989015, 2040719, 4693428, 3190376, 2926728, 3399030, 1664419, 662429,  4526841,
        2186957, 3752558, 2440046, 2930226, 3633006, 4058166, 3137060, 3499296, 2126343, 148971,
        2199672, 275811,  2813976, 2274536, 1189239, 1335942, 2465624, 2596042, 829684,  193400,
        2682845, 3691697, 4022437, 4051170, 4195175, 2876420, 3984220, 2174475, 326134,  2606530,
        2493046, 4706121, 1498980, 4576225, 1271339, 44832,   1875673, 4664940, 134931,  736397,
        4333554, 2751031, 2163610, 2879676, 3174153, 3317403, 2052464, 1881883, 4757859, 3596257,
        2358088, 2578758, 447504,  590720,  1717038, 1869795, 1133885, 3027521, 840312,  2818881,
        3654321, 2730947, 353585,  1134903, 2223378, 1508824, 3662521, 1363776, 2712071, 288441,
        1204581, 3502242, 4645567, 2767267, 1514366, 3956099, 1422145, 1216608, 2253360, 189132,
        4238225, 1345783, 451571,  1599442, 3237284, 4711405, 929446,  1857675, 150759,  1277633,
        761210,  138628,  1026833, 2599544, 2464737, 989203,  3399615, 2144292, 216142,  637312,
        2044964, 716256,  1660632, 1762919, 4784357, 2213415, 2764769, 291806,  609772,  3264819,
        1870953, 1516385, 235647,  1045474, 2664957, 819095,  1824119, 4045271, 4448109, 1676788,
        4285177, 1580502, 3546548, 2771971, 3927086, 1339779, 3156204, 1730998, 1172522, 2433024,
        4533449, 479930,  2010695, 672994,  3542039, 3176455, 26352,   2137735, 866910,  4410835,
        2623982, 3603159, 2555625, 2765653, 267865,  2015523, 1009052, 4713994, 1600667, 2176195,
        3179631, 4570390, 2018424, 3356384, 1784287, 894861,  3622099, 1647273, 3044136, 950354,
        1491760, 3416929, 3757300, 2244912, 4129215, 1600848, 3867343, 72329,   919189,  992521,
        3445975, 4712557, 4680974, 188419,  2612093, 1991268, 3566207, 2281468, 3859078, 2492806,
        3398628, 763441,  2679107, 2554420, 2130132, 4664374, 1182901, 3890770, 4714667, 4209303,
        4013060, 3617653, 2040022, 3296519, 4190671, 1693353, 2678411, 3788834, 2781815, 191965,
        1083926, 503974,  3529226, 1650522, 1900976, 542080,  3423929, 3418905, 878165,  4701703,
        3022790, 4316365, 76365,   4053672, 1358185, 3830478, 4445661, 3210024, 1895915, 4541133,
        2938808, 562788,  3920065, 1458776, 4052046, 2967475, 1092809, 3203538, 159626,  3399464,
        214467,  3343982, 1811854, 3189045, 4272117, 4701563, 424807,  4341116, 760545,  4674683,
        1538018, 386762,  194237,  2162719, 1694433, 943728,  2389036, 2196653, 3085571, 1513424,
        3689413, 3278747, 4197291, 3324063, 3651090, 1737936, 2768803, 2768889, 3108096, 4311775,
        3569480, 886705,  733256,  2477493, 1735412, 2960895, 1983781, 1861797, 3566460, 4537673,
        1164093, 3499764, 4553071, 3518985, 847658,  918948,  2922351, 1056144, 652895,  1013195,
        780505,  1702928, 3562838, 1432719, 2405207, 1054920, 641647,  2240939, 3617702, 383165,
        652641,  879593,  1810739, 2096385, 4497865, 4768530, 1743968, 3582014, 1025009, 3002122,
        2422190, 527647,  1251821, 2571153, 4095874, 3705333, 3637407, 1385567, 4043855, 4041930,
        2433139, 1710383, 1127734, 4362316, 711588,  817839,  3214775, 910077,  1313768, 2382229,
        16864,   2081770, 3095420, 3195272, 548711,  2259860, 1167323, 2435974, 425238,  2085179,
        2630042, 2632881, 2867923, 3703565, 1037695, 226617,  4379130, 1541468, 3581937, 605965,
        1137674, 4655221, 4769963, 1394370, 4425315, 2990132, 2364485, 1561137, 2713384, 481509,
        2900382, 934766,  2986774, 1767669, 298593,  2502539, 139296,  3794229, 4002180, 4718138,
        2909238, 423691,  3023810, 2784924, 2760160, 1971980, 316683,  3828090, 3253691, 4839313,
        1203624, 584938,  3901482, 1747543, 1572737, 3533226, 774708,  1691195, 1037110, 1557763,
        225120,  4424243, 3524086, 1717663, 4332507, 3513592, 4274932, 1232118, 873498,  1416042,
        2488925, 111391,  4704545, 4492545, 445317,  1584812, 2187737, 2471948, 3731678, 219255,
        2282627, 2589971, 2372185, 4609096, 3673961, 2524410, 12823,   2437155, 3015974, 4188352,
        3184084, 3690756, 1222341, 1278376, 3652030, 4162647, 326548,  3930062, 3926100, 1551222,
        2722165, 4526695, 3997534, 4815513, 3139056, 2547644, 3028915, 4149092, 3656554, 2691582,
        2676699, 1878842, 260174,  3129900, 4379993, 182347,  2189338, 3783616, 2616666, 2596952,
        243007,  4179282, 2730,    1939894, 2332032, 3335636, 182332,  3112260, 2174584, 587481,
        4527368, 3154106, 3403059, 673206,  2150292, 446521,  1600204, 4819428, 2591357, 48490,
        2917012, 2285923, 1072926, 2824281, 4364250, 956033,  311938,  37251,   3729300, 2726300,
        644966,  1623020, 1419070, 4646747, 2417222, 2680238, 2561083, 1793801, 2349366, 339747,
        611366,  4684147, 4356907, 1277161, 4510381, 3218352, 4161658, 3200733, 1172372, 3997786,
        3169266, 3353418, 2248955, 2875885, 2365369, 498208,  2968066, 2681505, 2059048, 2097106,
        3607540, 1121504, 2016789, 1762605, 3138431, 866081,  3705757, 3833066, 2599788, 760816,
        4046672, 1544367, 2983906, 4842911, 209599,  1250954, 3333704, 561212,  4674336, 2831841,
        3690724, 2929360, 4830834, 1177524, 2487687, 3525137, 875283,  651241,  2110742, 1296646,
        1543739, 4349417, 2384725, 1931751, 1519208, 1520034, 3385008, 3219962, 734912,  170230,
        1741419, 729913,  2860117, 2362381, 1199807, 2424230, 177824,  125948,  2722701, 4687548,
        1140771, 3232742, 4522020, 4376360, 1125603, 590312,  2481884, 138951,  4086775, 615155,
        3395781, 4587272, 283209,  568470,  4296185, 4344150, 2454321, 2672602, 838828,  4051647,
        1709120, 3074610, 693235,  4356087, 3018806, 239410,  2431497, 691186,  766276,  4462126,
        859155,  2370304, 1571808, 1938673, 1694955, 3871296, 4245059, 3987376, 301524,  2512461,
        3410437, 3300380, 684922,  4581995, 3599557, 683515,  1850634, 3704678, 1937490, 2035591,
        3718533, 2065879, 3160765, 1467884, 1912241, 2501509, 3668572, 3390469, 2501150, 612319,
        713633,  1976262, 135946,  3641535, 632083,  13414,   4217765, 4137712, 2550250, 3281035,
        4179598, 961045,  2020694, 4380006, 1345936, 289162,  1359035, 770872,  4509911, 3947317,
        4719693, 248568,  2625660, 1237232, 2153208, 4814282, 1259954, 3677369, 861222,  2883506,
        3339149, 3998335, 491017,  1609022, 2648112, 742132,  649609,  4206953, 3131106, 3504814,
        3344486, 611721,  3215620, 2856233, 4447505, 1949222, 1868345, 712710,  6966,    4730666,
        3181872, 2972889, 3038521, 3525444, 4385208, 1845613, 1124187, 2030476, 4468651, 2478792,
        3473580, 3783357, 1852991, 1648485, 871319,  1670723, 4458328, 3218600, 1811100, 3443356,
        2233873, 3035207, 2548692, 3337891, 3773674, 1552957, 4782811, 3144712, 3523466, 1491315,
        3955852, 1838410, 3164028, 1092543, 776459,  2959379, 2541744, 4064418, 3908320, 2854145,
        3960709, 1348188, 977678,  853619,  1304291, 2848702, 1657913, 1319826, 3322665, 788037,
        2913686, 4471279, 1766285, 348304,  56570,   1892118, 4017244, 401006,  3524539, 4310134,
        1624693, 4081113, 957511,  849400,  129975,  2616130, 378537,  1556787, 3916162, 1039980,
        4407778, 2027690, 4213675, 839863,  683134,  75805,   2493150, 4215796, 81587,   751845,
        1255588, 1947964, 1950470, 859401,  3077088, 3931110, 2316256, 1523761, 4527477, 4237511,
        1123513, 4209796, 3584772, 4250563, 2091754, 1618766, 2139944, 4525352, 382159,  2955887,
        41760,   2313998, 496912,  3791570, 3904792, 3613654, 873959,  127076,  2537797, 2458107,
        4543265, 3661909, 26828,   271816,  17854,   2461269, 1776042, 1573899, 3409957, 4335712,
        4534313, 3392751, 1230124, 2159031, 4444015, 3373087, 3848014, 2026600, 1382747, 3537242,
        4536743, 4714155, 3788371, 3570849, 173741,  211962,  4377778, 119369,  2856973, 2945854,
        1508054, 4503932, 3141566, 1842177, 3448683, 3384614, 2886508, 1573965, 990618,  3053734,
        2918742, 4508753, 1032149, 60943,   4291620, 722607,  2883224, 169359,  4356585, 3725543,
        3678729, 341673,  3592828, 4077251, 3382936, 3885685, 4630994, 1286698, 4449616, 1138430,
        3113385, 4660578, 2539973, 4562286, 4085089, 494737,  3967610, 2130702, 1823755, 1369324,
        3796951, 956299,  141730,  935144,  4381893, 4412545, 1382250, 3024476, 2364546, 3396164,
        3573511, 314081,  577688,  4154135, 1567018, 4047761, 2446220, 1148833, 4842497, 3967186,
        1175290, 3749667, 1209593, 3295627, 3169065, 2460328, 1838486, 1436923, 2843887, 3676426,
        2079145, 2975635, 535071,  4287509, 3281107, 39606,   3115500, 3204573, 722131,  3124073},
      2,
      false)));
*/
CUGRAPH_TEST_PROGRAM_MAIN()
