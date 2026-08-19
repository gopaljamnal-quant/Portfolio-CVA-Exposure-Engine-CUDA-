#include "Portfolio.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qlib {

namespace {

// Plain-C++ re-implementation of the analytic Hull-White (Vasicek-form)
// zero-coupon bond price, independent of the CUDA-qualified version in
// MarketModels.cuh so that this translation unit has zero CUDA-toolkit
// dependency. See MarketModels.cuh for the closed-form derivation; the
// two implementations are mathematically identical:
//
//   B(t,T) = (1 - exp(-a*(T-t))) / a
//   P(t,T) = exp( (B(t,T)-(T-t))*(a^2 b - sigma^2/2)/a^2
//                 - sigma^2 B(t,T)^2/(4a) - B(t,T)*r )
float vasicek_zero_bond_price(float a, float sigma, float b, float r, float t, float T) {
    const float tau = T - t;
    if (tau <= 0.0f) {
        return 1.0f;
    }
    const float Bt = (1.0f - std::exp(-a * tau)) / a;
    const float exponent = (Bt - tau) * (a * a * b - 0.5f * sigma * sigma) / (a * a)
                            - (sigma * sigma * Bt * Bt) / (4.0f * a)
                            - Bt * r;
    return std::exp(exponent);
}

} // namespace

// ===========================================================================
// InterestRateSwap
// ===========================================================================
InterestRateSwap::InterestRateSwap(std::string name, float notional, float fixed_rate,
                                    float start_time, float maturity, int num_payments,
                                    bool pay_fixed, float hw_a, float hw_sigma, float hw_mean_level)
    : name_(std::move(name)),
      notional_(notional),
      fixed_rate_(fixed_rate),
      start_time_(start_time),
      maturity_(maturity),
      num_payments_(num_payments),
      payment_tau_((maturity - start_time) / static_cast<float>(std::max(1, num_payments))),
      pay_receive_sign_(pay_fixed ? -1.0f : 1.0f),
      hw_a_(hw_a), hw_sigma_(hw_sigma), hw_mean_level_(hw_mean_level) {
    if (num_payments <= 0) {
        throw std::invalid_argument("InterestRateSwap requires num_payments > 0");
    }
    if (maturity <= start_time) {
        throw std::invalid_argument("InterestRateSwap requires maturity > start_time");
    }
}

// Single-curve valuation:
//   floatingLeg(t) = notional * ( P(t, max(T0,t)) - P(t, Tn) )
//   fixedLeg(t)    = notional * K * sum_{T_i > t} tau_i * P(t, T_i)
//   value(t)       = sign * ( fixedLeg(t) - floatingLeg(t) )
// where sign = +1 for a receiver (receives fixed) and -1 for a payer.
// Past cashflows (T_i <= t) are correctly excluded as the trade seasons.
float InterestRateSwap::price(float r, float /*S*/, float t) const {
    if (t >= maturity_) {
        return 0.0f;
    }

    const float t0_eff = std::max(start_time_, t);
    const float floating_leg = notional_ * (
        vasicek_zero_bond_price(hw_a_, hw_sigma_, hw_mean_level_, r, t, t0_eff) -
        vasicek_zero_bond_price(hw_a_, hw_sigma_, hw_mean_level_, r, t, maturity_)
    );

    float fixed_leg_annuity = 0.0f;
    for (int i = 1; i <= num_payments_; ++i) {
        const float Ti = start_time_ + static_cast<float>(i) * payment_tau_;
        if (Ti <= t) {
            continue;
        }
        fixed_leg_annuity += payment_tau_ * vasicek_zero_bond_price(hw_a_, hw_sigma_, hw_mean_level_, r, t, Ti);
    }
    const float fixed_leg = notional_ * fixed_rate_ * fixed_leg_annuity;

    return pay_receive_sign_ * (fixed_leg - floating_leg);
}

TradeSpec InterestRateSwap::to_spec() const {
    TradeSpec spec{};
    spec.type = TradeType::InterestRateSwap;
    spec.notional = notional_;
    spec.rate_or_strike = fixed_rate_;
    spec.start_time = start_time_;
    spec.maturity = maturity_;
    spec.num_payments = num_payments_;
    spec.payment_tau = payment_tau_;
    spec.foreign_rate = 0.0f; // unused for IRS
    spec.pay_receive_sign = pay_receive_sign_;
    spec.hw_a = hw_a_;
    spec.hw_sigma = hw_sigma_;
    spec.hw_mean_level = hw_mean_level_;
    return spec;
}

// ===========================================================================
// FXForward
// ===========================================================================
FXForward::FXForward(std::string name, float notional, float strike, float maturity,
                      float foreign_rate, bool is_long,
                      float hw_a, float hw_sigma, float hw_mean_level)
    : name_(std::move(name)),
      notional_(notional),
      strike_(strike),
      maturity_(maturity),
      foreign_rate_(foreign_rate),
      pay_receive_sign_(is_long ? 1.0f : -1.0f),
      hw_a_(hw_a), hw_sigma_(hw_sigma), hw_mean_level_(hw_mean_level) {
    if (maturity <= 0.0f) {
        throw std::invalid_argument("FXForward requires maturity > 0");
    }
}

// Covered interest parity valuation, domestic leg discounted stochastically
// via the Hull-White short rate, foreign leg discounted off a flat foreign
// rate:
//   value(t) = sign * notional * ( S_t * exp(-r_f*(T-t)) - K * P_dom(t,T) )
float FXForward::price(float r, float S, float t) const {
    if (t >= maturity_) {
        return 0.0f;
    }
    const float tau = maturity_ - t;
    const float p_dom = vasicek_zero_bond_price(hw_a_, hw_sigma_, hw_mean_level_, r, t, maturity_);
    const float value = notional_ * (S * std::exp(-foreign_rate_ * tau) - strike_ * p_dom);
    return pay_receive_sign_ * value;
}

TradeSpec FXForward::to_spec() const {
    TradeSpec spec{};
    spec.type = TradeType::FXForward;
    spec.notional = notional_;
    spec.rate_or_strike = strike_;
    spec.start_time = 0.0f; // unused for FX Forward
    spec.maturity = maturity_;
    spec.num_payments = 0;   // unused for FX Forward
    spec.payment_tau = 0.0f; // unused for FX Forward
    spec.foreign_rate = foreign_rate_;
    spec.pay_receive_sign = pay_receive_sign_;
    spec.hw_a = hw_a_;
    spec.hw_sigma = hw_sigma_;
    spec.hw_mean_level = hw_mean_level_;
    return spec;
}

// ===========================================================================
// Portfolio
// ===========================================================================
void Portfolio::add_trade(std::unique_ptr<BaseTrade> trade) {
    trades_.push_back(std::move(trade));
}

float Portfolio::mark_to_market(float r, float S, float t) const {
    float total = 0.0f;
    for (const auto& trade : trades_) {
        total += trade->price(r, S, t);
    }
    return total;
}

std::vector<TradeSpec> Portfolio::to_specs() const {
    std::vector<TradeSpec> specs;
    specs.reserve(trades_.size());
    for (const auto& trade : trades_) {
        specs.push_back(trade->to_spec());
    }
    return specs;
}

Portfolio Portfolio::build_sample_book(float hw_a, float hw_sigma, float hw_mean_level) {
    Portfolio book;

    // -- Interest Rate Swaps --------------------------------------------
    book.add_trade(std::make_unique<InterestRateSwap>(
        "IRS_PAYER_5Y_50M", 50'000'000.0f, 0.032f, 0.0f, 5.0f, 10,
        /*pay_fixed=*/true, hw_a, hw_sigma, hw_mean_level));

    book.add_trade(std::make_unique<InterestRateSwap>(
        "IRS_RECEIVER_10Y_30M", 30'000'000.0f, 0.028f, 0.0f, 10.0f, 20,
        /*pay_fixed=*/false, hw_a, hw_sigma, hw_mean_level));

    book.add_trade(std::make_unique<InterestRateSwap>(
        "IRS_PAYER_2Y_20M", 20'000'000.0f, 0.035f, 0.0f, 2.0f, 4,
        /*pay_fixed=*/true, hw_a, hw_sigma, hw_mean_level));

    book.add_trade(std::make_unique<InterestRateSwap>(
        "IRS_RECEIVER_7Y_15M", 15'000'000.0f, 0.030f, 0.0f, 7.0f, 14,
        /*pay_fixed=*/false, hw_a, hw_sigma, hw_mean_level));

    // -- FX Forwards ------------------------------------------------------
    book.add_trade(std::make_unique<FXForward>(
        "FXFWD_LONG_3Y_10M", 10'000'000.0f, 1.15f, 3.0f, 0.015f,
        /*is_long=*/true, hw_a, hw_sigma, hw_mean_level));

    book.add_trade(std::make_unique<FXForward>(
        "FXFWD_SHORT_1Y_25M", 25'000'000.0f, 1.08f, 1.0f, 0.015f,
        /*is_long=*/false, hw_a, hw_sigma, hw_mean_level));

    book.add_trade(std::make_unique<FXForward>(
        "FXFWD_LONG_5Y_8M", 8'000'000.0f, 1.20f, 5.0f, 0.018f,
        /*is_long=*/true, hw_a, hw_sigma, hw_mean_level));

    book.add_trade(std::make_unique<FXForward>(
        "FXFWD_SHORT_2Y_12M", 12'000'000.0f, 1.10f, 2.0f, 0.012f,
        /*is_long=*/false, hw_a, hw_sigma, hw_mean_level));

    return book;
}

} // namespace qlib
