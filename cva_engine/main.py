#!/usr/bin/env python3
"""
main.py
=======
Driver script for the GPU Portfolio CVA Exposure Engine.

Usage:
    python main.py

If the compiled `cva_engine` extension is not importable, build it first:

    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j
    # then either copy the resulting cva_engine*.so next to this script,
    # or run this file from within the build directory.

The script initialises the simulation parameters (Hull-White rate model,
GBM FX model, correlation, CSA terms, credit inputs), triggers the GPU
Monte Carlo exposure simulation, and plots the Expected Exposure (EE) and
95th-percentile Potential Future Exposure (PFE) profiles over the 10-year
horizon.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

try:
    import cva_engine
except ImportError as exc:
    sys.exit(
        "Could not import the compiled 'cva_engine' extension.\n"
        "Build it first:\n"
        "    mkdir build && cd build\n"
        "    cmake .. -DCMAKE_BUILD_TYPE=Release\n"
        "    cmake --build . -j\n"
        f"(original import error: {exc})"
    )


def run_and_plot() -> None:
    # ------------------------------------------------------------------
    # Simulation configuration
    # ------------------------------------------------------------------
    params = dict(
        num_paths=10_000,
        num_steps=520,        # 10Y horizon, weekly time steps
        horizon_years=10.0,
        # Hull-White 1F short rate (Vasicek form): dr = a(b-r)dt + sigma dW
        hw_a=0.05,
        hw_sigma=0.01,
        hw_mean_level=0.03,
        r0=0.03,
        # GBM FX: dS = mu*S dt + sigma_fx*S dW
        fx_mu=0.0,
        fx_sigma=0.12,
        s0=1.10,
        # Correlation between the rate and FX Brownian motions
        rho=0.20,
        # Unilateral CSA: counterparty posts collateral above this threshold,
        # subject to a minimum transfer amount
        csa_threshold=500_000.0,
        csa_mta=50_000.0,
        # Credit inputs
        recovery_rate=0.40,
        flat_hazard_rate=0.02,
        seed=42,
    )

    print("Running GPU Monte Carlo CVA exposure simulation...")
    print(f"  paths={params['num_paths']:,}  steps={params['num_steps']}  "
          f"horizon={params['horizon_years']}Y")

    result = cva_engine.run_cva_simulation(**params)

    time_grid = np.asarray(result["time_grid"])
    ee = np.asarray(result["expected_exposure"])
    pfe = np.asarray(result["pfe_95"])
    cva = float(result["cva"])
    num_trades = int(result["num_trades"])

    print(f"Portfolio: {num_trades} trades")
    print(f"Peak EE:  {ee.max():,.0f}  at t = {time_grid[np.argmax(ee)]:.2f}y")
    print(f"Peak PFE: {pfe.max():,.0f}  at t = {time_grid[np.argmax(pfe)]:.2f}y")
    print(f"CVA charge: {cva:,.2f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_grid, ee, label="Expected Exposure (EE)", linewidth=2, color="#1f77b4")
    ax.plot(time_grid, pfe, label="Potential Future Exposure (PFE, 95%)",
            linewidth=2, color="#d62728", linestyle="--")
    ax.fill_between(time_grid, ee, pfe, alpha=0.08, color="#d62728")

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Collateralised Exposure")
    ax.set_title(f"Counterparty Exposure Profile  |  CVA = {cva:,.0f}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = "exposure_profile.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

    plt.show()


if __name__ == "__main__":
    run_and_plot()
