#pragma once

#include <string>
#include <vector>

// A scalar curve sent with `metric` (loss, val_loss, acc, lr, grad_norm, ...).
struct Series {
    static const size_t kMaxPoints = 4000;
    std::string name;
    std::vector<double> step, value;

    // Appends a point. Past kMaxPoints the history is thinned to every other
    // point, so a long run keeps its whole shape at lower resolution.
    void add(double s, double v);
};

// Health of one parameter group (usually one layer), sent with `pstat`.
// Negative values mean unknown; grad < 0 means the group got no gradient.
struct ParamStat {
    std::string name;
    double grad = -1, weight = -1, update = -1;  // update = |dW| / |W| for one optimizer step
};

struct Alert {
    int level;  // 1 = warning, 2 = critical
    std::string text;
};

struct Training {
    std::vector<Series> series;
    std::vector<ParamStat> params;
    bool any = false;  // something training-related has arrived

    Series& get(const std::string& name);
    const Series* find(const std::string& name) const;
    void clear();
};

// The name a series is charted under: "val_loss" and "train_loss" both chart as "loss".
std::string chart_key(const std::string& name);
bool is_validation(const std::string& name);

// Rule-of-thumb checks on the curves and layer stats.
std::vector<Alert> find_alerts(const Training& t);
