#pragma once

#include <string>
#include <vector>

#include "network.hpp"
#include "training.hpp"

// Everything the viewer knows about the network it is showing. It is built
// from text commands, which arrive from a .nlens file, a TCP client or stdin
// (see README.md for the command list).
struct State {
    Network net;
    std::vector<float> input;              // last `input` received
    bool input_set = false;
    std::vector<std::vector<float>> ext;   // activations sent with `acts`, per layer
    std::vector<double> ext_time;          // when each arrived, in seconds; < 0 = never
    std::vector<std::string> labels;       // output neuron names
    std::string status;                    // free text from the client (epoch, loss, ...)
    unsigned version = 0;                  // bumped whenever the layer sizes change
    Training train;                        // curves and layer stats, independent of the network
};

// Passed as `now` while loading a file, so activations stored in the file
// never go stale.
constexpr double kPinned = 1e300;

// How long activations sent with `acts` are shown before the viewer goes
// back to computing that layer itself.
constexpr double kFreshSeconds = 2.0;

void set_topology(State& st, const std::vector<int>& sizes);

// Applies one command line. On a malformed line, leaves the state unchanged,
// sets err and returns false.
bool apply_line(State& st, const std::string& line, double now, std::string& err);

bool load_file(State& st, const std::string& path, std::string& err);
