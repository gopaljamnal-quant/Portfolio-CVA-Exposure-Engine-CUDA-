// =============================================================================
// CVAEngine.cuh
//
// GPU exposure engine: for every simulated path and every time step it
//   1. prices the full portfolio (warp-coherent, tagged-dispatch pricing
//      of the flattened TradeSpec array -- see the design note in
//      Portfolio.hpp on why this avoids on-device virtual dispatch),
//   2. nets trade-level MtMs into a single counterparty exposure,
//   3. applies unilateral CSA collateral mechanics (Threshold + Minimum
//      Transfer Amount) to the netted exposure,
//   4. accumulates the collateralised exposure into the Expected Exposure
//      (EE) profile via atomics, and
//   5. retains the full path x time exposure grid so the 95th-percentile
//      Potential Future Exposure (PFE) profile can be extracted with a
//      per-time-step order statistic (via Thrust) once the kernel
//      completes.
//
// The CVA charge is then computed on the host from the reduced EE profile
// against a discrete hazard-rate curve:
//     CVA = (1-R) * sum_i EE(t_i) * D(0,t_i) * [PD(t_i) - PD(t_{i-1})]
// with D(0,t) the deterministic (t=0) Hull-White discount curve and PD(t)
// derived from a piecewise-constant hazard-rate term structure.
// =============================================================================
#pragma once

#include "MarketModels.cuh"
#include "Portfolio.hpp"

#include <vector>

namespace qlib {

// -----------------------------------------------------------------------
// CSAParams
//
// Unilateral Credit Support Annex: the counterparty posts collateral to
// us whenever our netted, uncollateralised exposure to them exceeds the
// threshold H, subject to a Minimum Transfer Amount (calls/returns below
// MTA are not exchanged).
// -----------------------------------------------------------------------
struct CSAParams {
    float threshold;   // H
    float mta;          // Minimum Transfer Amount
};

// -----------------------------------------------------------------------
// ExposureResult
// -----------------------------------------------------------------------
struct ExposureResult {
    std::vector<float> time_grid;          // size num_steps + 1, in years
    std::vector<float> ee_profile;         // Expected (collateralised) Exposure
    std::vector<float> pfe_profile;        // 95th percentile Potential Future Exposure
    float               cva = 0.0f;         // (1-R) * sum EE(t) D(t) dPD(t)
};

// -----------------------------------------------------------------------
// Device-callable trade pricer used by the exposure kernel. Mirrors the
// closed-form formulas implemented on the host in Portfolio.cpp (see
// InterestRateSwap::price / FXForward::price) exactly, but dispatches via
// a plain switch on TradeSpec::type instead of a vtable, which keeps
// execution warp-coherent (all lanes in a warp typically iterate the same
// portfolio, so the divergence cost of the switch itself is negligible
// and paid once per trade rather than once per virtual call).
// -----------------------------------------------------------------------
__device__ float price_trade_gpu(const TradeSpec& trade, float r, float S, float t);

// -----------------------------------------------------------------------
// compute_exposure_kernel
//
// Grid/block geometry: one thread per Monte Carlo path (same mapping used
// by simulate_paths_kernel), each thread sequentially walking all
// (num_steps + 1) time steps of its own path. This is embarrassingly
// parallel across paths; the only sequential dependency (the running CSA
// collateral balance) lives entirely within a single thread's registers,
// so there is no inter-thread synchronisation required inside the kernel.
//
// exposures is laid out time-major: exposures[step * num_paths + path].
// This layout makes each time step's cross-path distribution contiguous
// in memory, which is what the host-side PFE (order-statistic) reduction
// needs for an efficient per-column Thrust sort.
// -----------------------------------------------------------------------
__global__ void compute_exposure_kernel(const TradeSpec* trades, int num_trades,
                                         const float* r_paths, const float* s_paths,
                                         int num_paths, int num_steps, float dt,
                                         float csa_threshold, float csa_mta,
                                         float* exposures,
                                         float* ee_sum);

// -----------------------------------------------------------------------
// run_cva_engine
//
// End-to-end pipeline: simulate risk factors -> upload portfolio -> GPU
// exposure/CSA kernel -> EE/PFE reduction -> host-side CVA integration.
// hazard_curve must have size market_params.num_steps + 1 and gives the
// (annualised, piecewise-constant-per-interval) instantaneous hazard rate
// applicable up to and including each grid point.
// -----------------------------------------------------------------------
ExposureResult run_cva_engine(const MarketModelParams& market_params,
                               const Portfolio& portfolio,
                               const CSAParams& csa,
                               const std::vector<float>& hazard_curve,
                               float recovery_rate,
                               unsigned long long seed);

} // namespace qlib
