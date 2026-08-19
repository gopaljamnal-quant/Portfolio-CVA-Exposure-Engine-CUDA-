// =============================================================================
// pybind_wrapper.cpp
//
// Exposes the CVA exposure engine to Python as the `cva_engine` module.
// Simulation parameters (paths, steps, volatilities, correlation, CSA
// terms, credit inputs) are passed as keyword arguments with sensible
// institutional defaults; the 520-element Expected Exposure and PFE
// profiles are returned as genuine NumPy arrays (via py::array_t), and
// the scalar CVA charge as a Python float.
// =============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "CVAEngine.cuh"
#include "MarketModels.cuh"
#include "Portfolio.hpp"

namespace py = pybind11;

namespace {

py::array_t<float> to_numpy(const std::vector<float>& v) {
    py::array_t<float> arr(static_cast<py::ssize_t>(v.size()));
    py::buffer_info buf = arr.request();
    std::memcpy(buf.ptr, v.data(), v.size() * sizeof(float));
    return arr;
}

py::dict run_cva_simulation(
    int num_paths,
    int num_steps,
    float horizon_years,
    float hw_a,
    float hw_sigma,
    float hw_mean_level,
    float r0,
    float fx_mu,
    float fx_sigma,
    float s0,
    float rho,
    float csa_threshold,
    float csa_mta,
    float recovery_rate,
    float flat_hazard_rate,
    unsigned long long seed) {

    if (num_paths <= 0 || num_steps <= 0) {
        throw std::invalid_argument("num_paths and num_steps must be positive");
    }
    if (rho < -1.0f || rho > 1.0f) {
        throw std::invalid_argument("rho must lie in [-1, 1]");
    }

    qlib::MarketModelParams market_params{};
    market_params.hw_a = hw_a;
    market_params.hw_sigma = hw_sigma;
    market_params.hw_mean_level = hw_mean_level;
    market_params.r0 = r0;
    market_params.fx_mu = fx_mu;
    market_params.fx_sigma = fx_sigma;
    market_params.s0 = s0;
    market_params.rho = rho;
    market_params.num_paths = num_paths;
    market_params.num_steps = num_steps;
    market_params.dt = horizon_years / static_cast<float>(num_steps);

    qlib::Portfolio portfolio = qlib::Portfolio::build_sample_book(hw_a, hw_sigma, hw_mean_level);

    qlib::CSAParams csa{};
    csa.threshold = csa_threshold;
    csa.mta = csa_mta;

    // Flat hazard-rate term structure by default; the engine itself
    // supports an arbitrary piecewise-constant curve (see CVAEngine.cuh).
    std::vector<float> hazard_curve(static_cast<size_t>(num_steps) + 1, flat_hazard_rate);

    qlib::ExposureResult result;
    {
        py::gil_scoped_release release; // GPU work does not need the GIL held
        result = qlib::run_cva_engine(market_params, portfolio, csa, hazard_curve,
                                       recovery_rate, seed);
    }

    py::dict out;
    out["time_grid"] = to_numpy(result.time_grid);
    out["expected_exposure"] = to_numpy(result.ee_profile);
    out["pfe_95"] = to_numpy(result.pfe_profile);
    out["cva"] = result.cva;
    out["num_trades"] = static_cast<int>(portfolio.size());
    return out;
}

} // namespace

PYBIND11_MODULE(cva_engine, m) {
    m.doc() = "GPU-accelerated Portfolio CVA Exposure Engine "
              "(1F Hull-White rates, GBM FX, CSA collateral, Monte Carlo on CUDA)";

    m.def("run_cva_simulation", &run_cva_simulation,
          py::arg("num_paths") = 10000,
          py::arg("num_steps") = 520,
          py::arg("horizon_years") = 10.0f,
          py::arg("hw_a") = 0.05f,
          py::arg("hw_sigma") = 0.01f,
          py::arg("hw_mean_level") = 0.03f,
          py::arg("r0") = 0.03f,
          py::arg("fx_mu") = 0.0f,
          py::arg("fx_sigma") = 0.12f,
          py::arg("s0") = 1.10f,
          py::arg("rho") = 0.20f,
          py::arg("csa_threshold") = 500000.0f,
          py::arg("csa_mta") = 50000.0f,
          py::arg("recovery_rate") = 0.40f,
          py::arg("flat_hazard_rate") = 0.02f,
          py::arg("seed") = 42ULL,
          "Run the full Monte Carlo CVA exposure simulation on the GPU.\n\n"
          "Returns a dict with:\n"
          "  time_grid          -- ndarray[float32], size num_steps+1, years\n"
          "  expected_exposure  -- ndarray[float32], collateralised EE profile\n"
          "  pfe_95             -- ndarray[float32], 95th percentile PFE profile\n"
          "  cva                -- float, the aggregate CVA charge\n"
          "  num_trades         -- int, size of the sample portfolio priced\n");
}
