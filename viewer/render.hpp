#pragma once

#include <string>
#include <vector>

#include "network.hpp"
#include "training.hpp"

struct View {
    float yaw = 0.f, pitch = 0.f, dist = 3.1f;
    float mix = 0.f;      // 0 = flat columns, 1 = each layer as a 3D grid
    float gain = 1.f;     // brightness of connections
    int max_show = 64;    // larger layers are drawn as an evenly spaced sample
};

// Neuron under the mouse; index is the neuron's index in the network.
struct Hover {
    int layer = -1, index = -1;
};

struct Scene {
    const Network* net = nullptr;
    std::vector<const std::vector<float>*> acts;  // activations shown, per layer
    const std::vector<std::vector<char>>* muted = nullptr;
    const std::vector<std::string>* labels = nullptr;
    View view;
    int fb_w = 0, fb_h = 0;
    float scale = 1.f;                      // framebuffer pixels per window point
    float mouse_x = -1.f, mouse_y = -1.f;   // framebuffer pixels; negative = outside
    std::vector<std::string> hud;   // top-left lines; the first is drawn in gold
    int hud_highlight = -1;         // index of a HUD line drawn brighter
    std::string help;

    bool show_net = true;           // draw the network
    bool show_panel = false;        // draw the training panel (right side, or full window without a network)
    const Training* train = nullptr;
    std::vector<Alert> alerts;
};

// Draws one frame into the current OpenGL context.
Hover draw_scene(const Scene& s);
