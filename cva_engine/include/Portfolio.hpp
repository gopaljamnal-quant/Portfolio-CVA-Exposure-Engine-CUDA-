// =============================================================================
// Portfolio.hpp
//
// Host-side analytic trade library. Deliberately free of any CUDA
// dependency (no <cuda_runtime.h>, no __host__/__device__ qualifiers) so
// it can be built, unit-tested and linked with a plain C++17 compiler even
// on machines without the CUDA toolkit installed -- this is the "quant
// analytics" layer a desk would use for standalone pricing/greeks.
//
// The GPU exposure engine (CVAEngine.cuh/.cu) does NOT call these classes
// directly: on-device polymorphic (vtable) dispatch is a well known
// anti-pattern for CUDA kernels, because divergent virtual calls across a
// warp serialise execution. Instead each trade exposes a flat, POD
// TradeSpec via to_spec(); the exposure engine uploads an array of specs
// and prices them on-device with warp-coherent switch-based dispatch
// (see CVAEngine.cuh: price_trade_gpu). Both code paths implement the
// same closed-form Hull-White / covered-interest-parity valuation
// formulas, so host and device prices agree to floating point precision.
// =============================================================================
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace qlib {

// -----------------------------------------------------------------------
// TradeType / TradeSpec
//
// TradeSpec is the GPU-friendly, tagged-union-style flattening of any
// trade in the book. It is a trivially-copyable POD (only int/float/enum
// members) so an array of these can be uploaded to device memory with a
// single cudaMemcpy and indexed directly inside a __global__ kernel.
// -----------------------------------------------------------------------
enum class TradeType : int {
    InterestRateSwap = 0,
    FXForward = 1
};

struct TradeSpec {
    TradeType type;

    float notional;           // trade notional, domestic currency
    float rate_or_strike;     // fixed rate (IRS) or FX strike (FX Forward)
    float start_time;         // T0: first reset / trade effective time (IRS only)
    float maturity;           // final maturity Tn / T
    int   num_payments;       // number of fixed-leg payments (IRS only)
    float payment_tau;        // year fraction per payment period (IRS only)
    float foreign_rate;       // flat foreign discount rate r_f (FX Forward only)
    float pay_receive_sign;   // +1.0f or -1.0f, direction of the trade

    // Each trade carries its own copy of the discounting model
    // parameters. In this single-curve design they are simply the
    // Hull-White parameters used to build the simulation, but keeping
    // them on the trade itself allows, e.g., cross-currency books where
    // different trades discount off different curves without changing
    // the kernel's dispatch logic.
    float hw_a;
    float hw_sigma;
    float hw_mean_level;
};

// -----------------------------------------------------------------------
// BaseTrade
//
// Abstract analytic instrument. price() gives the time-t mark-to-market
// given a realisation of the short rate and FX spot; to_spec() bridges
// to the GPU exposure engine.
// -----------------------------------------------------------------------
class BaseTrade {
public:
    virtual ~BaseTrade() = default;

    // Analytical evaluation at future time t, given the simulated short
    // rate r and FX spot S at that time.
    virtual float price(float r, float S, float t) const = 0;

    // Flattened GPU representation of this trade.
    virtual TradeSpec to_spec() const = 0;

    virtual float maturity() const = 0;
    virtual const std::string& name() const = 0;
};

// -----------------------------------------------------------------------
// InterestRateSwap
//
// Single-curve fixed-vs-floating interest rate swap, priced off the
// Hull-White short rate via the analytic zero-coupon bond formula.
// Under the single-curve idealisation the floating leg telescopes to
//     floatingLeg(t) = notional * ( P(t, max(T0,t)) - P(t, Tn) )
// and the fixed leg is
//     fixedLeg(t) = notional * K * sum_i tau_i * P(t, T_i)   (T_i > t)
// The receiver-fixed value is fixedLeg - floatingLeg; a payer swap is the
// negative of that.
// -----------------------------------------------------------------------
class InterestRateSwap final : public BaseTrade {
public:
    InterestRateSwap(std::string name, float notional, float fixed_rate,
                      float start_time, float maturity, int num_payments,
                      bool pay_fixed, float hw_a, float hw_sigma, float hw_mean_level);

    float price(float r, float S, float t) const override;
    TradeSpec to_spec() const override;
    float maturity() const override { return maturity_; }
    const std::string& name() const override { return name_; }

private:
    std::string name_;
    float notional_;
    float fixed_rate_;
    float start_time_;
    float maturity_;
    int   num_payments_;
    float payment_tau_;
    float pay_receive_sign_;   // -1 if pay_fixed, +1 if receive_fixed
    float hw_a_, hw_sigma_, hw_mean_level_;
};

// -----------------------------------------------------------------------
// FXForward
//
// FX forward priced by covered interest parity, discounting domestically
// off the stochastic Hull-White short rate and off a flat foreign rate:
//     value(t) = sign * notional * ( S_t * exp(-r_f * (T-t)) - K * P_dom(t,T) )
// which is equivalent to notional * (F(t,T) - K) * P_dom(t,T) with
// F(t,T) = S_t * exp(-r_f*(T-t)) / P_dom(t,T).
// -----------------------------------------------------------------------
class FXForward final : public BaseTrade {
public:
    FXForward(std::string name, float notional, float strike, float maturity,
              float foreign_rate, bool is_long,
              float hw_a, float hw_sigma, float hw_mean_level);

    float price(float r, float S, float t) const override;
    TradeSpec to_spec() const override;
    float maturity() const override { return maturity_; }
    const std::string& name() const override { return name_; }

private:
    std::string name_;
    float notional_;
    float strike_;
    float maturity_;
    float foreign_rate_;
    float pay_receive_sign_;   // +1 if long, -1 if short
    float hw_a_, hw_sigma_, hw_mean_level_;
};

// -----------------------------------------------------------------------
// Portfolio
//
// Owns a heterogeneous book of trades. Provides both the host-side
// aggregate mark-to-market (for standalone analytics / sanity checks) and
// the flattened TradeSpec array consumed by the GPU exposure engine.
// -----------------------------------------------------------------------
class Portfolio {
public:
    void add_trade(std::unique_ptr<BaseTrade> trade);

    // Sum of all trade-level prices at (r, S, t). Host-side utility, not
    // used on the hot path of the exposure engine.
    float mark_to_market(float r, float S, float t) const;

    std::vector<TradeSpec> to_specs() const;

    size_t size() const { return trades_.size(); }
    const std::vector<std::unique_ptr<BaseTrade>>& trades() const { return trades_; }

    // Builds a representative 8-trade book (mixed IRS/FX Forward, mixed
    // payer/receiver and long/short directions, staggered maturities and
    // notionals) that produces a deliberately asymmetric net exposure
    // profile -- useful for exercising the netting and CSA collateral
    // logic in the exposure engine.
    static Portfolio build_sample_book(float hw_a, float hw_sigma, float hw_mean_level);

private:
    std::vector<std::unique_ptr<BaseTrade>> trades_;
};

} // namespace qlib
