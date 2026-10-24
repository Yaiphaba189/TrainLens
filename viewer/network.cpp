#include "network.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>

bool parse_act(const std::string& name, Act& out) {
    std::string t;
    for (char c : name) t += (char)std::tolower((unsigned char)c);
    if (t == "linear" || t == "none" || t == "identity") out = Act::Linear;
    else if (t == "relu") out = Act::ReLU;
    else if (t == "tanh") out = Act::Tanh;
    else if (t == "sigmoid") out = Act::Sigmoid;
    else if (t == "softmax") out = Act::Softmax;
    else return false;
    return true;
}

const char* act_name(Act a) {
    switch (a) {
        case Act::Linear: return "linear";
        case Act::ReLU: return "relu";
        case Act::Tanh: return "tanh";
        case Act::Sigmoid: return "sigmoid";
        case Act::Softmax: return "softmax";
    }
    return "?";
}

void Network::set_sizes(const std::vector<int>& s) {
    sizes = s;
    const int edges = std::max(0, (int)s.size() - 1);
    acts.assign(edges, Act::ReLU);
    if (edges > 0) acts.back() = Act::Sigmoid;
    W.assign(edges, {});
    B.assign(edges, {});
    has_w.assign(edges, false);
    for (int l = 0; l < edges; ++l) {
        W[l].assign((size_t)s[l + 1] * s[l], 0.f);
        B[l].assign(s[l + 1], 0.f);
    }
}

void Network::randomize(unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    for (size_t l = 0; l < W.size(); ++l) {
        const float sd = std::sqrt(2.f / sizes[l]);  // He init, suits ReLU
        for (float& w : W[l]) w = normal(rng) * sd;
        for (float& b : B[l]) b = normal(rng) * 0.15f;
        has_w[l] = true;
    }
}

void Network::forward(std::vector<std::vector<float>>& a, const std::vector<bool>& fixed,
                      const std::vector<std::vector<char>>& muted) const {
    auto mute = [&](int l) {
        if (l >= (int)muted.size()) return;
        for (size_t j = 0; j < a[l].size() && j < muted[l].size(); ++j)
            if (muted[l][j]) a[l][j] = 0.f;
    };
    mute(0);
    for (int l = 0; l + 1 < layers(); ++l) {
        std::vector<float>& out = a[l + 1];
        const bool is_fixed = l + 1 < (int)fixed.size() && fixed[l + 1];
        if (!is_fixed) {
            if (!has_w[l]) {
                std::fill(out.begin(), out.end(), 0.f);
            } else {
                const std::vector<float>& in = a[l];
                const int nin = sizes[l], nout = sizes[l + 1];
                for (int j = 0; j < nout; ++j) {
                    const float* w = &W[l][(size_t)j * nin];
                    float z = B[l][j];
                    for (int i = 0; i < nin; ++i) z += w[i] * in[i];
                    out[j] = z;
                }
                switch (acts[l]) {
                    case Act::Linear: break;
                    case Act::ReLU: for (float& v : out) v = std::max(0.f, v); break;
                    case Act::Tanh: for (float& v : out) v = std::tanh(v); break;
                    case Act::Sigmoid: for (float& v : out) v = 1.f / (1.f + std::exp(-v)); break;
                    case Act::Softmax: {
                        const float m = *std::max_element(out.begin(), out.end());
                        float sum = 0.f;
                        for (float& v : out) sum += (v = std::exp(v - m));
                        for (float& v : out) v /= sum;
                        break;
                    }
                }
            }
        }
        mute(l + 1);
    }
}
