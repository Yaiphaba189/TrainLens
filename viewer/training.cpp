#include "training.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

const char* kValPrefixes[] = {"val_", "valid_", "validation_", "test_", "eval_"};

std::string fmt(const char* f, double v) {
    char b[48];
    std::snprintf(b, sizeof b, f, v);
    return b;
}

bool is_loss(const std::string& name) { return chart_key(name).find("loss") != std::string::npos; }

// Smallest finite value in [from, to).
double min_in(const Series& s, size_t from, size_t to) {
    double m = INFINITY;
    for (size_t i = from; i < to; ++i)
        if (std::isfinite(s.value[i])) m = std::min(m, s.value[i]);
    return m;
}

double mean_tail(const Series& s, size_t n) {
    double sum = 0;
    size_t c = 0;
    for (size_t i = s.value.size() - std::min(n, s.value.size()); i < s.value.size(); ++i)
        if (std::isfinite(s.value[i])) { sum += s.value[i]; ++c; }
    return c ? sum / (double)c : NAN;
}

}  // namespace

void Series::add(double s, double v) {
    if (value.size() >= kMaxPoints) {
        size_t k = 0;
        for (size_t i = 1; i < value.size(); i += 2, ++k) {
            step[k] = step[i];
            value[k] = value[i];
        }
        step.resize(k);
        value.resize(k);
    }
    step.push_back(s);
    value.push_back(v);
}

Series& Training::get(const std::string& name) {
    for (Series& s : series)
        if (s.name == name) return s;
    series.push_back({});
    series.back().name = name;
    return series.back();
}

const Series* Training::find(const std::string& name) const {
    for (const Series& s : series)
        if (s.name == name) return &s;
    return nullptr;
}

void Training::clear() {
    series.clear();
    params.clear();
}

bool is_validation(const std::string& name) {
    for (const char* p : kValPrefixes)
        if (starts_with(name, p)) return true;
    return false;
}

std::string chart_key(const std::string& name) {
    for (const char* p : kValPrefixes)
        if (starts_with(name, p)) return name.substr(std::string(p).size());
    if (starts_with(name, "train_")) return name.substr(6);
    return name;
}

std::vector<Alert> find_alerts(const Training& t) {
    std::vector<Alert> out;

    for (const Series& s : t.series) {
        if (s.value.empty() || !is_loss(s.name)) continue;
        const double last = s.value.back();
        const size_t n = s.value.size();
        if (!std::isfinite(last)) {
            out.push_back({2, s.name + " IS " + (std::isnan(last) ? "NAN" : "INF") + " AT STEP " + fmt("%.0f", s.step.back()) +
                                  ": LOWER THE LR OR CHECK THE DATA"});
            continue;
        }
        if (is_validation(s.name)) continue;
        const double best = min_in(s, 0, n);
        // Spike: some point far above where training started. Often the run
        // never recovers (dead ReLUs, zero gradients) even if the curve settles.
        size_t peak = 0;
        for (size_t i = 1; i < n; ++i)
            if (std::isfinite(s.value[i]) && s.value[i] > s.value[peak]) peak = i;
        if (s.value[0] > 0 && s.value[peak] > s.value[0] * 10.0)
            out.push_back({2, s.name + " SPIKED TO " + fmt("%.3g", s.value[peak]) + " AT STEP " + fmt("%.0f", s.step[peak]) +
                                  ": LR IS LIKELY TOO HIGH, OR ADD GRADIENT CLIPPING"});
        // Diverging: well above the best value seen, after a warm-up.
        if (n >= 20 && best > 0 && mean_tail(s, 5) > best * 3.0)
            out.push_back({2, s.name + " IS DIVERGING (" + fmt("%.3g", mean_tail(s, 5)) + " VS BEST " + fmt("%.3g", best) +
                                  "): LR MAY BE TOO HIGH"});
        // Plateau: the last quarter did not beat the best before it by 1%.
        else if (n >= 60) {
            const size_t cut = n - n / 4;
            const double before = min_in(s, 0, cut), recent = min_in(s, cut, n);
            if (std::isfinite(before) && recent > before - std::fabs(before) * 0.01)
                out.push_back({1, s.name + " HAS NOT IMPROVED SINCE STEP " + fmt("%.0f", s.step[cut]) +
                                      ": TRY A LOWER LR OR STOP EARLY"});
        }
    }

    // Overfitting: validation loss climbing from its best while training loss keeps falling.
    for (const Series& v : t.series) {
        if (!is_validation(v.name) || !is_loss(v.name) || v.value.size() < 6) continue;
        const std::string key = chart_key(v.name);
        const Series* tr = t.find(key);
        if (!tr) tr = t.find("train_" + key);
        if (!tr || tr->value.size() < 10) continue;
        const double vbest = min_in(v, 0, v.value.size()), vnow = mean_tail(v, 3);
        const double tr_old = mean_tail(*tr, tr->value.size() / 2), tr_now = mean_tail(*tr, 5);
        if (vbest > 0 && vnow > vbest * 1.05 && tr_now < tr_old)
            out.push_back({1, v.name + " IS RISING (" + fmt("%.3g", vnow) + " VS BEST " + fmt("%.3g", vbest) +
                                  ") WHILE TRAINING LOSS FALLS: OVERFITTING"});
    }

    // Per-layer gradient and update size. A healthy update is roughly 1e-3 of the weights per step.
    int exploding = 0, silent = 0, big = 0, tiny = 0;
    std::string ex_name, si_name, big_name, tiny_name;
    for (const ParamStat& p : t.params) {
        if (p.grad > 100.0) { if (!exploding++) ex_name = p.name; }
        else if (p.grad >= 0 && p.grad < 1e-7 && p.weight > 0) { if (!silent++) si_name = p.name; }
        if (p.update > 0.05) { if (!big++) big_name = p.name; }
        else if (p.update >= 0 && p.update < 1e-6 && p.grad > 0) { if (!tiny++) tiny_name = p.name; }
    }
    auto more = [](int n) { return n > 1 ? " AND " + std::to_string(n - 1) + " MORE" : std::string(); };
    if (exploding) out.push_back({2, "EXPLODING GRADIENT IN " + ex_name + more(exploding) + ": CLIP GRADIENTS OR LOWER THE LR"});
    if (silent) out.push_back({1, "NO GRADIENT REACHES " + si_name + more(silent) + ": VANISHING OR DISCONNECTED"});
    if (big) out.push_back({1, "UPDATES ARE OVER 5% OF THE WEIGHTS IN " + big_name + more(big) + ": LR MAY BE TOO HIGH"});
    if (tiny) out.push_back({1, "UPDATES BELOW 1E-6 OF THE WEIGHTS IN " + tiny_name + more(tiny) + ": LR MAY BE TOO LOW"});

    std::stable_sort(out.begin(), out.end(), [](const Alert& a, const Alert& b) { return a.level > b.level; });
    return out;
}
