#include "CVAEngine.cuh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace qlib {

namespace {

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) +
                                  " at " + file + ":" + std::to_string(line));
    }
}

} // namespace

#define QLIB_CUDA_CHECK(call) qlib::cuda_check((call), __FILE__, __LINE__)

// ===========================================================================
// Device-side trade pricing (tagged dispatch, see header for rationale).
// Mirrors InterestRateSwap::price / FXForward::price in Portfolio.cpp.
// ===========================================================================
__device__ float price_trade_gpu(const TradeSpec& trade, float r, float S, float t) {
    if (t >= trade.maturity) {
        return 0.0f;
    }

    switch (trade.type) {
        case TradeType::InterestRateSwap: {
            const float t0_eff = fmaxf(trade.start_time, t);
            const float floating_leg = trade.notional * (
                hw_zero_bond_price(trade.hw_a, trade.hw_sigma, trade.hw_mean_level, r, t, t0_eff) -
                hw_zero_bond_price(trade.hw_a, trade.hw_sigma, trade.hw_mean_level, r, t, trade.maturity)
            );

            float fixed_leg_annuity = 0.0f;
            for (int i = 1; i <= trade.num_payments; ++i) {
                const float Ti = trade.start_time + static_cast<float>(i) * trade.payment_tau;
                if (Ti <= t) {
                    continue;
                }
                fixed_leg_annuity += trade.payment_tau *
                    hw_zero_bond_price(trade.hw_a, trade.hw_sigma, trade.hw_mean_level, r, t, Ti);
            }
            const float fixed_leg = trade.notional * trade.rate_or_strike * fixed_leg_annuity;

            return trade.pay_receive_sign * (fixed_leg - floating_leg);
        }
        case TradeType::FXForward: {
            const float tau = trade.maturity - t;
            const float p_dom = hw_zero_bond_price(trade.hw_a, trade.hw_sigma, trade.hw_mean_level, r, t, trade.maturity);
            const float value = trade.notional * (S * expf(-trade.foreign_rate * tau) - trade.rate_or_strike * p_dom);
            return trade.pay_receive_sign * value;
        }
    }
    return 0.0f; // unreachable for a well-formed TradeSpec
}

// ===========================================================================
// Exposure + CSA collateral kernel
// ===========================================================================
__global__ void compute_exposure_kernel(const TradeSpec* trades, int num_trades,
                                         const float* r_paths, const float* s_paths,
                                         int num_paths, int num_steps, float dt,
                                         float csa_threshold, float csa_mta,
                                         float* exposures,
                                         float* ee_sum) {
    const int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= num_paths) {
        return;
    }

    const int path_stride = num_steps + 1;
    float collateral_balance = 0.0f; // sequential per-path state, register-resident

    for (int step = 0; step <= num_steps; ++step) {
        const float r = r_paths[path * path_stride + step];
        const float s = s_paths[path * path_stride + step];
        const float t = static_cast<float>(step) * dt;

        float portfolio_mtm = 0.0f;
        for (int k = 0; k < num_trades; ++k) {
            portfolio_mtm += price_trade_gpu(trades[k], r, s, t);
        }

        // Net Positive Exposure: E_net = max(sum MtM_i, 0)
        const float net_exposure = fmaxf(portfolio_mtm, 0.0f);

        // CSA margining: bring the uncollateralised exposure down to the
        // threshold H whenever the required transfer exceeds the MTA.
        const float target_collateral = fmaxf(net_exposure - csa_threshold, 0.0f);
        const float call_amount = target_collateral - collateral_balance;
        if (fabsf(call_amount) >= csa_mta) {
            collateral_balance = target_collateral;
        }
        const float collateralised_exposure = fmaxf(net_exposure - collateral_balance, 0.0f);

        exposures[step * num_paths + path] = collateralised_exposure;
        atomicAdd(&ee_sum[step], collateralised_exposure);
    }
}

// ===========================================================================
// Host-side CVA integration
//
//   CVA = (1-R) * sum_i EE(t_i) * D(0,t_i) * [PD(t_i) - PD(t_{i-1})]
//
// with PD(t) = 1 - exp(-cumulative hazard up to t), hazard piecewise
// constant on each grid interval, and D(0,t) the deterministic (r = r0,
// t = 0) Hull-White discount curve.
// ===========================================================================
namespace {

float integrate_cva(const std::vector<float>& ee_profile,
                     const std::vector<float>& time_grid,
                     const std::vector<float>& hazard_curve,
                     float recovery_rate,
                     float hw_a, float hw_sigma, float hw_mean_level, float r0) {
    float cva = 0.0f;
    float cumulative_hazard = 0.0f;
    float pd_prev = 0.0f;

    for (size_t i = 1; i < ee_profile.size(); ++i) {
        const float t_prev = time_grid[i - 1];
        const float t_curr = time_grid[i];
        cumulative_hazard += hazard_curve[i] * (t_curr - t_prev);

        const float pd_curr = 1.0f - std::exp(-cumulative_hazard);
        const float delta_pd = pd_curr - pd_prev;
        pd_prev = pd_curr;

        const float discount = hw_zero_bond_price(hw_a, hw_sigma, hw_mean_level, r0, 0.0f, t_curr);

        cva += ee_profile[i] * discount * delta_pd;
    }

    return (1.0f - recovery_rate) * cva;
}

} // namespace

// ===========================================================================
// run_cva_engine: full pipeline orchestration
// ===========================================================================
ExposureResult run_cva_engine(const MarketModelParams& market_params,
                               const Portfolio& portfolio,
                               const CSAParams& csa,
                               const std::vector<float>& hazard_curve,
                               float recovery_rate,
                               unsigned long long seed) {
    if (hazard_curve.size() != static_cast<size_t>(market_params.num_steps + 1)) {
        throw std::invalid_argument("hazard_curve must have num_steps + 1 entries");
    }
    if (portfolio.size() == 0) {
        throw std::invalid_argument("portfolio must contain at least one trade");
    }

    // --- 1. Simulate joint risk factors ------------------------------------
    SimulatedPaths paths = simulate_market_paths(market_params, seed);

    // --- 2. Upload the flattened trade book ---------------------------------
    const std::vector<TradeSpec> specs = portfolio.to_specs();
    TradeSpec* d_trades = nullptr;
    QLIB_CUDA_CHECK(cudaMalloc(&d_trades, specs.size() * sizeof(TradeSpec)));
    QLIB_CUDA_CHECK(cudaMemcpy(d_trades, specs.data(), specs.size() * sizeof(TradeSpec),
                                cudaMemcpyHostToDevice));

    // --- 3. Allocate exposure grid + EE accumulator -------------------------
    const int num_steps = market_params.num_steps;
    const int num_paths = market_params.num_paths;
    const size_t grid_size = static_cast<size_t>(num_steps + 1) * static_cast<size_t>(num_paths);

    float* d_exposures = nullptr;
    float* d_ee_sum = nullptr;
    QLIB_CUDA_CHECK(cudaMalloc(&d_exposures, grid_size * sizeof(float)));
    QLIB_CUDA_CHECK(cudaMalloc(&d_ee_sum, static_cast<size_t>(num_steps + 1) * sizeof(float)));
    QLIB_CUDA_CHECK(cudaMemset(d_ee_sum, 0, static_cast<size_t>(num_steps + 1) * sizeof(float)));

    // --- 4. Launch the exposure/CSA kernel -----------------------------------
    constexpr int kThreadsPerBlock = 256;
    const int blocks = (num_paths + kThreadsPerBlock - 1) / kThreadsPerBlock;

    compute_exposure_kernel<<<blocks, kThreadsPerBlock>>>(
        d_trades, static_cast<int>(specs.size()),
        paths.r, paths.s,
        num_paths, num_steps, market_params.dt,
        csa.threshold, csa.mta,
        d_exposures, d_ee_sum);
    QLIB_CUDA_CHECK(cudaGetLastError());
    QLIB_CUDA_CHECK(cudaDeviceSynchronize());

    // --- 5. Reduce to Expected Exposure (mean) --------------------------------
    std::vector<float> ee_sum_host(num_steps + 1);
    QLIB_CUDA_CHECK(cudaMemcpy(ee_sum_host.data(), d_ee_sum,
                                static_cast<size_t>(num_steps + 1) * sizeof(float),
                                cudaMemcpyDeviceToHost));

    ExposureResult result;
    result.time_grid.resize(num_steps + 1);
    result.ee_profile.resize(num_steps + 1);
    result.pfe_profile.resize(num_steps + 1);

    for (int step = 0; step <= num_steps; ++step) {
        result.time_grid[step] = static_cast<float>(step) * market_params.dt;
        result.ee_profile[step] = ee_sum_host[step] / static_cast<float>(num_paths);
    }

    // --- 6. Potential Future Exposure (95th percentile) via per-step sort ----
    // The time-major layout of d_exposures makes each time step's cross-path
    // distribution a contiguous block, so a Thrust device-pointer sort in
    // place gives the order statistic directly with no extra copies.
    const int pfe_index = static_cast<int>(std::ceil(0.95 * (num_paths - 1)));
    for (int step = 0; step <= num_steps; ++step) {
        thrust::device_ptr<float> column(d_exposures + static_cast<size_t>(step) * num_paths);
        thrust::sort(thrust::device, column, column + num_paths);
        float pfe_value = 0.0f;
        QLIB_CUDA_CHECK(cudaMemcpy(&pfe_value, thrust::raw_pointer_cast(column) + pfe_index,
                                    sizeof(float), cudaMemcpyDeviceToHost));
        result.pfe_profile[step] = pfe_value;
    }

    // --- 7. CVA integration (host-side, off the reduced EE profile) ---------
    result.cva = integrate_cva(result.ee_profile, result.time_grid, hazard_curve,
                                recovery_rate,
                                market_params.hw_a, market_params.hw_sigma,
                                market_params.hw_mean_level, market_params.r0);

    // --- 8. Cleanup -----------------------------------------------------------
    free_simulated_paths(paths);
    cudaFree(d_trades);
    cudaFree(d_exposures);
    cudaFree(d_ee_sum);

    return result;
}

} // namespace qlib
