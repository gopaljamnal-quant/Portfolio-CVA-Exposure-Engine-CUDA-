#include "MarketModels.cuh"

#include <stdexcept>
#include <string>

namespace qlib {

namespace {

// Uniform CUDA error-checking helper. Throws so failures surface cleanly
// through the pybind11 boundary as a Python exception rather than a silent
// wrong answer or a hard process abort.
inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) +
                                  " at " + file + ":" + std::to_string(line));
    }
}

} // namespace

#define QLIB_CUDA_CHECK(call) qlib::cuda_check((call), __FILE__, __LINE__)

__global__ void simulate_paths_kernel(MarketModelParams params,
                                       float* r_paths,
                                       float* s_paths,
                                       unsigned long long seed) {
    const int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= params.num_paths) {
        return;
    }

    // Independent, statistically decorrelated stream per path: distinct
    // subsequence per thread, offset 0.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(path), 0ULL, &state);

    const int stride = params.num_steps + 1;
    float r = params.r0;
    float s = params.s0;

    r_paths[path * stride + 0] = r;
    s_paths[path * stride + 0] = s;

    const float sqrt_dt = sqrtf(params.dt);
    const float rho = params.rho;
    const float chol_off_diag = sqrtf(fmaxf(0.0f, 1.0f - rho * rho));

    for (int step = 1; step <= params.num_steps; ++step) {
        const float x1 = curand_normal(&state);
        const float x2 = curand_normal(&state);

        // Cholesky mapping to correlated standard normals.
        const float z1 = x1;
        const float z2 = rho * x1 + chol_off_diag * x2;

        // Hull-White (Vasicek-form) Euler-Maruyama update.
        const float drift_r = params.hw_a * (params.hw_mean_level - r) * params.dt;
        r = r + drift_r + params.hw_sigma * sqrt_dt * z1;

        // GBM FX: exact log-Euler step (positivity preserving).
        const float drift_s = (params.fx_mu - 0.5f * params.fx_sigma * params.fx_sigma) * params.dt;
        s = s * expf(drift_s + params.fx_sigma * sqrt_dt * z2);

        r_paths[path * stride + step] = r;
        s_paths[path * stride + step] = s;
    }
}

SimulatedPaths simulate_market_paths(const MarketModelParams& params, unsigned long long seed) {
    if (params.num_paths <= 0 || params.num_steps <= 0) {
        throw std::invalid_argument("num_paths and num_steps must both be positive");
    }

    SimulatedPaths paths;
    const size_t n = static_cast<size_t>(params.num_paths) * static_cast<size_t>(params.num_steps + 1);

    QLIB_CUDA_CHECK(cudaMallocManaged(&paths.r, n * sizeof(float)));
    QLIB_CUDA_CHECK(cudaMallocManaged(&paths.s, n * sizeof(float)));

    constexpr int kThreadsPerBlock = 256;
    const int blocks = (params.num_paths + kThreadsPerBlock - 1) / kThreadsPerBlock;

    simulate_paths_kernel<<<blocks, kThreadsPerBlock>>>(params, paths.r, paths.s, seed);
    QLIB_CUDA_CHECK(cudaGetLastError());
    QLIB_CUDA_CHECK(cudaDeviceSynchronize());

    return paths;
}

void free_simulated_paths(SimulatedPaths& paths) {
    if (paths.r) {
        cudaFree(paths.r);
        paths.r = nullptr;
    }
    if (paths.s) {
        cudaFree(paths.s);
        paths.s = nullptr;
    }
}

} // namespace qlib
