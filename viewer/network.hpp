#pragma once

#include <string>
#include <vector>

enum class Act { Linear, ReLU, Tanh, Sigmoid, Softmax };

bool parse_act(const std::string& name, Act& out);
const char* act_name(Act a);

// Dense feed-forward network. W[l] maps layer l to layer l+1 and is stored
// row-major as sizes[l+1] x sizes[l], the same layout as torch.nn.Linear.weight.
struct Network {
    std::vector<int> sizes;
    std::vector<Act> acts;               // one per non-input layer
    std::vector<std::vector<float>> W, B;
    std::vector<bool> has_w;             // weights for this edge layer were supplied

    int layers() const { return (int)sizes.size(); }
    void set_sizes(const std::vector<int>& s);
    void randomize(unsigned seed);

    // a[0] holds the input. Layers with fixed[l] set keep their values (they
    // came from a client); the rest are computed. muted[l][j] forces a neuron
    // to zero, so you can see what the network does without it.
    void forward(std::vector<std::vector<float>>& a, const std::vector<bool>& fixed,
                 const std::vector<std::vector<char>>& muted) const;
};
