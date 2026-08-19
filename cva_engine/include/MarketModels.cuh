// =============================================================================
// MarketModels.cuh
//
// Joint risk-factor simulation for the CVA exposure engine:
//   - 1-Factor Hull-White (time-homogeneous / Vasicek-form) short rate model
//         dr_t = a * (b - r_t) dt + sigma_IR dW1_t
//     (equivalently dr_t = (theta - a r_t) dt + sigma_IR dW1_t with the
//      constant drift level theta = a * b, as permitted by the spec's
//      "constant analytical theta(t)" simplification).
//   - Geometric Brownian Motion FX model
//         dS_t = mu * S_t dt + sigma_FX * S_t dW2_t
//   - dW1_t and dW2_t are instantaneously correlated with coefficient rho.
//
// Path generation is fully GPU resident: one CUDA thread simulates one
// complete path (all 520 weekly steps over a 10Y horizon), using a
// per-thread cuRAND Philox state for correlated Gaussian draws. This keeps
// occupancy high (each thread does a modest, register-resident amount of
// sequential work) and avoids any host round-trips during simulation.
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

namespace qlib {

// -----------------------------------------------------------------------
// MarketModelParams
//
// All time quantities are annualised (ACT/365-style year fractions). The
// struct is a plain POD so it can be passed by value into a __global__
// kernel launch without any marshalling.
// -----------------------------------------------------------------------
struct MarketModelParams {
    // Hull-White (Vasicek-form) short-rate parameters
    float hw_a;           // mean reversion speed
    float hw_sigma;       // short-rate volatility
    float hw_mean_level;  // long-run mean level b, i.e. theta(t) = a * b
    float r0;              // initial short rate

    // FX GBM parameters (domestic risk-neutral measure)
    float fx_mu;            // drift of the FX spot
    float fx_sigma;         // FX volatility
    float s0;                // initial FX spot

    // Joint dynamics
    float rho;                // corr(dW1, dW2), must lie in [-1, 1]

    // Time grid
    int   num_paths;          // number of Monte Carlo paths
    int   num_steps;          // number of time steps (520 => 10Y weekly)
    float dt;                  // step size in years (horizon_years / num_steps)
};

// -----------------------------------------------------------------------
// SimulatedPaths
//
// Row-major storage: element (path, step) lives at index
// path * (num_steps + 1) + step. Step 0 holds the initial values r0 / s0.
// Memory is CUDA Unified Memory (cudaMallocManaged) so it is directly
// addressable from host code (e.g. for debugging/inspection) as well as
// from any downstream device kernel (the CVA exposure engine).
// -----------------------------------------------------------------------
struct SimulatedPaths {
    float* r = nullptr;   // short-rate paths,  size num_paths * (num_steps + 1)
    float* s = nullptr;   // FX spot paths,     size num_paths * (num_steps + 1)
};

// Allocates unified memory for the joint path set and launches the
// simulation kernel. Synchronises before returning, so the paths are
// immediately readable on return. Throws std::runtime_error on any CUDA
// failure.
SimulatedPaths simulate_market_paths(const MarketModelParams& params,
                                      unsigned long long seed);

// Releases device memory owned by a SimulatedPaths instance.
void free_simulated_paths(SimulatedPaths& paths);

// -----------------------------------------------------------------------
// Analytic Hull-White (Vasicek-form) zero-coupon bond price P(t,T), given
// the short rate r observed at time t:
//
//   B(t,T) = (1 - exp(-a*(T-t))) / a
//   A(t,T) = exp( (B(t,T)-(T-t)) * (a^2*b - sigma^2/2) / a^2
//                 - sigma^2 * B(t,T)^2 / (4a) )
//   P(t,T) = A(t,T) * exp(-B(t,T) * r)
//
// Marked __host__ __device__ so the exact same formula is used both for
// on-GPU trade pricing (CVAEngine.cu, called millions of times) and for
// any host-side diagnostics/discount-curve computations. Degenerates
// gracefully to P = 1 when T <= t.
// -----------------------------------------------------------------------
__device__ __host__ inline float hw_zero_bond_price(float a, float sigma, float b,
                                                      float r, float t, float T) {
    float tau = T - t;
    if (tau <= 0.0f) {
        return 1.0f;
    }
    float Bt = (1.0f - expf(-a * tau)) / a;
    float exponent = (Bt - tau) * (a * a * b - 0.5f * sigma * sigma) / (a * a)
                      - (sigma * sigma * Bt * Bt) / (4.0f * a)
                      - Bt * r;
    return expf(exponent);
}

// -----------------------------------------------------------------------
// simulate_paths_kernel
//
// One thread per Monte Carlo path. Each thread owns a private cuRAND
// Philox4_32_10 generator (fast, high-quality, good statistical
// properties for MC, and cheap to initialise per-thread) and walks the
// full 520-step timeline sequentially, writing both risk factors at
// every step. Correlation between the rate and FX Brownian increments is
// imposed via a Cholesky decomposition of the 2x2 correlation matrix:
//
//   Z1 = X1
//   Z2 = rho * X1 + sqrt(1 - rho^2) * X2 ,   X1, X2 ~ iid N(0,1)
//
// The FX leg uses the exact log-Euler (geometric) discretisation of GBM,
// which is unconditionally stable and keeps every simulated spot strictly
// positive; the rate leg uses a standard Euler-Maruyama step, which is
// exact in distribution for the Vasicek/Hull-White SDE's conditional mean
// and introduces only a (typically negligible at weekly steps) variance
// discretisation error.
// -----------------------------------------------------------------------
__global__ void simulate_paths_kernel(MarketModelParams params,
                                       float* r_paths,
                                       float* s_paths,
                                       unsigned long long seed);

} // namespace qlib
