// TrainLens: real-time neural network visualizer and training monitor (C++ / OpenGL / SDL2).
//
// Shows a dense feed-forward network as glowing neurons and connections, and
// training curves with layer health checks. Data comes from a .tlens file,
// from Python code over TCP (the train_lens package), or from stdin; without
// any, it runs a random demo network.

#define SDL_MAIN_HANDLED
#include <SDL.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "network.hpp"
#include "png.hpp"
#include "protocol.hpp"
#include "render.hpp"
#include "server.hpp"

#ifndef TRAIN_LENS_VERSION
#define TRAIN_LENS_VERSION "dev"
#endif

namespace {

const char* kUsage =
    "usage: train-lens [options] [model.tlens]\n"
    "\n"
    "With no model file, shows a random demo network and waits for a client.\n"
    "\n"
    "options:\n"
    "  --port N          listen on 127.0.0.1:N for clients (default 5757)\n"
    "  --no-server       do not listen for clients\n"
    "  --stdin           also read commands from stdin\n"
    "  --layers A,B,...  sizes of the demo network (default 16,26,26,38,24)\n"
    "  --seed N          random seed for the demo weights\n"
    "  --max-show N      draw at most N neurons per layer (default 64)\n"
    "  --3d              start in the 3D view (layers as grids)\n"
    "  --check FILE      validate a .tlens file, print a summary and exit\n"
    "  --selftest        run built-in checks and exit\n"
    "  --frames N        quit after N frames (for scripts)\n"
    "  --screenshot FILE save a PNG of the last frame (use with --frames)\n"
    "  -V, --version     print the version\n"
    "  -h, --help        show this help\n";

const char* kHelpLine =
    "DRAG ROTATE  WHEEL ZOOM  CLICK MUTE  V 2D/3D  T TRAINING PANEL  N NETWORK  SPACE PAUSE  A SPIN  D DELAY  R RANDOMIZE  C UNMUTE  [ ] GAIN  0 CAMERA  S SCREENSHOT  H HIDE";

struct Options {
    std::string file;
    std::string check_file;
    std::string screenshot;
    int port = 5757;
    bool server = true;
    bool use_stdin = false;
    std::vector<int> demo_layers{16, 26, 26, 38, 24};
    unsigned seed = 7;
    int max_show = 64;
    long frames = -1;
    bool selftest = false;
    bool three_d = false;
};

bool parse_layers(const char* s, std::vector<int>& out) {
    out.clear();
    while (*s) {
        char* end;
        long v = std::strtol(s, &end, 10);
        if (end == s || v < 1) return false;
        out.push_back((int)v);
        s = end;
        if (*s == ',') ++s;
        else if (*s) return false;
    }
    return out.size() >= 2;
}

// Returns -1 to keep running, or an exit code.
int parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "train-lens: %s needs %s\n", a.c_str(), what);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { std::fputs(kUsage, stdout); return 0; }
        else if (a == "-V" || a == "--version") { std::printf("train-lens viewer %s\n", TRAIN_LENS_VERSION); return 0; }
        else if (a == "--port") { const char* v = next("a number"); if (!v) return 2; o.port = std::atoi(v); }
        else if (a == "--no-server") o.server = false;
        else if (a == "--stdin") o.use_stdin = true;
        else if (a == "--layers") {
            const char* v = next("sizes like 16,26,24");
            if (!v || !parse_layers(v, o.demo_layers)) { std::fprintf(stderr, "train-lens: --layers wants two or more sizes, like 16,26,24\n"); return 2; }
        }
        else if (a == "--seed") { const char* v = next("a number"); if (!v) return 2; o.seed = (unsigned)std::strtoul(v, nullptr, 10); }
        else if (a == "--max-show") { const char* v = next("a number"); if (!v) return 2; o.max_show = std::max(1, std::atoi(v)); }
        else if (a == "--check") { const char* v = next("a file"); if (!v) return 2; o.check_file = v; }
        else if (a == "--frames") { const char* v = next("a number"); if (!v) return 2; o.frames = std::atol(v); }
        else if (a == "--screenshot") { const char* v = next("a file"); if (!v) return 2; o.screenshot = v; }
        else if (a == "--selftest") o.selftest = true;
        else if (a == "--3d") o.three_d = true;
        else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "train-lens: unknown option %s\n\n%s", a.c_str(), kUsage); return 2; }
        else if (o.file.empty()) o.file = a;
        else { std::fprintf(stderr, "train-lens: only one model file can be given\n"); return 2; }
    }
    if (o.port < 1 || o.port > 65535) { std::fprintf(stderr, "train-lens: port must be 1-65535\n"); return 2; }
    if (!o.screenshot.empty() && o.frames < 1) { std::fprintf(stderr, "train-lens: --screenshot needs --frames N\n"); return 2; }
    return -1;
}

double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

std::string join_sizes(const std::vector<int>& v, const char* sep) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? sep : "") + std::to_string(v[i]);
    return s;
}

std::string base_name(const std::string& path) {
    size_t p = path.find_last_of('/');
    return p == std::string::npos ? path : path.substr(p + 1);
}

int run_selftest() {
    int failures = 0;
    auto expect = [&](bool ok, const char* what) {
        std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++failures;
    };
    State st;
    std::string err;
    // Hand-set weights that solve XOR: h0 = relu(x+y), h1 = relu(x+y-1), out = sigmoid(10 h0 - 20 h1 - 5).
    const char* lines[] = {"layers 2 2 1", "act relu sigmoid", "w 0 1 1 1 1", "b 0 0 -1", "w 1 10 -20", "b 1 -5",
                           "labels xor"};
    bool parsed = true;
    for (const char* l : lines) parsed = apply_line(st, l, 0.0, err) && parsed;
    expect(parsed, "parse a small network");

    std::vector<std::vector<float>> a = {{0, 0}, {0, 0}, {0}};
    const std::vector<bool> fixed(3, false);
    const std::vector<std::vector<char>> none;
    const float cases[4][3] = {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 0}};
    bool xor_ok = true;
    for (const auto& c : cases) {
        a[0] = {c[0], c[1]};
        st.net.forward(a, fixed, none);
        xor_ok = xor_ok && ((a[2][0] > 0.5f) == (c[2] > 0.5f));
    }
    expect(xor_ok, "forward pass computes XOR");

    std::vector<std::vector<char>> muted = {{0, 0}, {1, 0}, {0}};
    a[0] = {1, 0};
    st.net.forward(a, fixed, muted);
    expect(a[1][0] == 0.f && a[2][0] < 0.5f, "muting a neuron changes the output");

    expect(!apply_line(st, "w 0 1 2", 0.0, err), "reject a weight line with the wrong count");
    expect(!apply_line(st, "act relu", 0.0, err), "reject an act line with the wrong count");
    expect(!apply_line(st, "w 5 1", 0.0, err), "reject an out-of-range layer");
    expect(!apply_line(st, "input 1 x", 0.0, err), "reject a non-number value");
    expect(!apply_line(st, "fly away", 0.0, err), "reject an unknown command");
    expect(apply_line(st, "acts 1 0.5 0.25", 3.0, err) && st.ext[1][1] == 0.25f && st.ext_time[1] == 3.0,
           "store activations sent by a client");
    expect(apply_line(st, "status epoch=2 loss=0.1", 0.0, err) && st.status == "epoch=2 loss=0.1", "store status text");
    const unsigned v = st.version;
    expect(apply_line(st, "layers 2 2 1", 0.0, err) && st.version == v && st.net.has_w[0],
           "same layer sizes keep the weights");
    expect(apply_line(st, "layers 3 4 2", 0.0, err) && st.version == v + 1 && !st.net.has_w[0],
           "new layer sizes reset the network");

    // training monitor
    auto has_alert = [](const std::vector<Alert>& v, const char* word) {
        for (const Alert& a : v)
            if (a.text.find(word) != std::string::npos) return true;
        return false;
    };
    State tr;
    bool ok = true;
    for (int i = 0; i < 40; ++i) {
        ok = apply_line(tr, "metric loss " + std::to_string(1.0 / (1 + i)) + " " + std::to_string(i), 0.0, err) && ok;
        if (i % 5 == 4) ok = apply_line(tr, "metric val_loss " + std::to_string(i < 20 ? 1.0 / (1 + i) : 0.05 + 0.01 * i) + " " + std::to_string(i), 0.0, err) && ok;
    }
    const Series* loss = tr.train.find("loss");
    expect(ok && loss && loss->value.size() == 40 && loss->step.back() == 39.0, "store metric curves");
    expect(chart_key("val_loss") == "loss" && chart_key("train_acc") == "acc" && is_validation("val_loss"),
           "chart train and validation curves together");
    expect(has_alert(find_alerts(tr.train), "OVERFITTING"), "detect validation loss rising while training loss falls");
    expect(apply_line(tr, "metric loss nan 40", 0.0, err) && has_alert(find_alerts(tr.train), "NAN"), "detect a NaN loss");
    expect(apply_line(tr, "pstats 3", 0.0, err) && apply_line(tr, "pstat 0 fc1 500 3 0.001", 0.0, err) &&
               apply_line(tr, "pstat 1 fc2 0 2 0", 0.0, err) && apply_line(tr, "pstat 2 head 0.1 1 0.2", 0.0, err),
           "store layer health");
    const std::vector<Alert> al = find_alerts(tr.train);
    expect(has_alert(al, "EXPLODING") && has_alert(al, "NO GRADIENT") && has_alert(al, "TOO HIGH"),
           "detect exploding, missing and oversized updates");
    expect(!apply_line(tr, "pstat 9 x 1 1 1", 0.0, err), "reject a layer stat outside the count");
    State sp;
    for (const char* l : {"metric loss 1.4 1", "metric loss 9000 2", "metric loss 1.5 3"}) apply_line(sp, l, 0.0, err);
    expect(has_alert(find_alerts(sp.train), "SPIKED"), "detect a loss spike even after it settles");
    expect(apply_line(tr, "clearmetrics", 0.0, err) && tr.train.series.empty(), "clear curves");
    Series big;
    for (int i = 0; i < 10000; ++i) big.add(i, i);
    expect(big.value.size() <= Series::kMaxPoints && big.step.back() == 9999.0, "thin long curves but keep the latest point");

    std::printf(failures ? "%d check(s) failed\n" : "all checks passed\n", failures);
    return failures ? 1 : 0;
}

int run_check(const std::string& path) {
    State st;
    std::string err;
    if (!load_file(st, path, err)) {
        std::fprintf(stderr, "train-lens: %s\n", err.c_str());
        return 1;
    }
    const Network& n = st.net;
    std::printf("%s: OK\n  layers:  %s\n  act:    ", path.c_str(), join_sizes(n.sizes, " -> ").c_str());
    for (Act a : n.acts) std::printf(" %s", act_name(a));
    std::printf("\n  weights:");
    for (int l = 0; l + 1 < n.layers(); ++l) std::printf(" %d:%s", l, n.has_w[l] ? "yes" : "no");
    std::printf("\n  labels:  %zu\n  input:   %s\n", st.labels.size(), st.input_set ? "yes" : "no (inputs will animate)");
    return 0;
}

// Runs the network at a fixed rate and keeps recent frames, so each layer can
// be shown a few frames behind the one before it. That delay is what makes
// the signal visibly travel left to right.
struct Sim {
    static const int kHist = 128;
    std::vector<std::vector<std::vector<float>>> ring;
    std::vector<std::vector<char>> muted;
    std::vector<std::vector<int>> zero_run;  // ticks each neuron has output exactly zero
    int head = 0;
    unsigned version = ~0u;
    static const int kDeadTicks = 180;       // 3 s at 60 Hz

    void reset(const Network& net, unsigned v) {
        ring.assign(kHist, {});
        for (auto& frame : ring) {
            frame.resize(net.sizes.size());
            for (size_t l = 0; l < net.sizes.size(); ++l) frame[l].assign(net.sizes[l], 0.f);
        }
        muted.assign(net.sizes.size(), {});
        zero_run.assign(net.sizes.size(), {});
        for (size_t l = 0; l < net.sizes.size(); ++l) {
            muted[l].assign(net.sizes[l], 0);
            zero_run[l].assign(net.sizes[l], 0);
        }
        head = 0;
        version = v;
    }

    void tick(const State& st, double t, double now) {
        if (version != st.version) reset(st.net, st.version);
        head = (head + 1) % kHist;
        std::vector<std::vector<float>>& a = ring[head];
        const int L = st.net.layers();
        auto fresh = [&](int l) { return st.ext_time[l] >= 0.0 && now - st.ext_time[l] < kFreshSeconds; };
        if (fresh(0)) {
            a[0] = st.ext[0];
        } else if (st.input_set) {
            a[0] = st.input;
        } else {
            // No input yet: feed each input a sine wave so something moves.
            for (size_t i = 0; i < a[0].size(); ++i) {
                const double f = 0.15 + 0.9 * std::fmod(std::fabs(std::sin(i * 12.9898) * 43758.5453), 1.0);
                a[0][i] = (float)std::sin(2.0 * M_PI * f * t + i * 2.3999);
            }
        }
        std::vector<bool> fixed(L, false);
        for (int l = 1; l < L; ++l)
            if (fresh(l)) {
                a[l] = st.ext[l];
                fixed[l] = true;
            }
        st.net.forward(a, fixed, muted);
        for (int l = 1; l < L; ++l)
            for (size_t j = 0; j < a[l].size(); ++j)
                zero_run[l][j] = (a[l][j] == 0.f && !muted[l][j]) ? zero_run[l][j] + 1 : 0;
    }

    // "H2: 60% OF RELU NEURONS ARE DEAD" for hidden ReLU layers where at
    // least half the neurons have output zero for every input lately.
    std::vector<Alert> dead_relu_alerts(const Network& net) const {
        std::vector<Alert> out;
        for (int l = 1; l + 1 < net.layers(); ++l) {
            if (net.acts[l - 1] != Act::ReLU || zero_run[l].empty()) continue;
            int dead = 0;
            for (int z : zero_run[l]) dead += z >= kDeadTicks;
            const int pct = (int)(100.0 * dead / (double)zero_run[l].size());
            if (pct >= 50)
                out.push_back({1, "H" + std::to_string(l) + ": " + std::to_string(pct) +
                                      "% OF RELU NEURONS ARE DEAD (ALWAYS 0): TRY A LOWER LR OR LEAKYRELU"});
        }
        return out;
    }

    const std::vector<float>* layer(int l, int delay) const {
        return &ring[((head - l * delay) % kHist + kHist) % kHist][l];
    }
};

struct Window {
    SDL_Window* win = nullptr;
    SDL_GLContext ctx = nullptr;

    bool open(bool multisample) {
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, multisample ? 1 : 0);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, multisample ? 4 : 0);
        win = SDL_CreateWindow("TrainLens", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1400, 860,
                               SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
        if (!win) return false;
        ctx = SDL_GL_CreateContext(win);
        if (!ctx) {
            SDL_DestroyWindow(win);
            win = nullptr;
            return false;
        }
        SDL_GL_SetSwapInterval(1);
        return true;
    }
    ~Window() {
        if (ctx) SDL_GL_DeleteContext(ctx);
        if (win) SDL_DestroyWindow(win);
    }
};

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    int rc = parse_args(argc, argv, opt);
    if (rc >= 0) return rc;
    if (opt.selftest) return run_selftest();
    if (!opt.check_file.empty()) return run_check(opt.check_file);

    State st;
    std::string err, source;
    if (!opt.file.empty()) {
        if (!load_file(st, opt.file, err)) {
            std::fprintf(stderr, "train-lens: %s\n", err.c_str());
            return 1;
        }
        source = "FILE " + base_name(opt.file);
    } else {
        set_topology(st, opt.demo_layers);
        st.net.randomize(opt.seed);
        source = "DEMO, RANDOM WEIGHTS";
    }
    const unsigned boot_version = st.version;

    LineServer srv;
    if (opt.server && !srv.listen(opt.port, err))
        std::fprintf(stderr, "train-lens: not listening for clients (%s)\n", err.c_str());
    if (opt.use_stdin) srv.enable_stdin();

    std::printf("TrainLens - real-time neural network visualizer\n");
    std::printf("  showing:   %s (%s)\n", join_sizes(st.net.sizes, "-").c_str(), source.c_str());
    if (srv.listening())
        std::printf("  listening: 127.0.0.1:%d  (Python: train_lens.monitor(model, optimizer) or train_lens.watch(model))\n", srv.port());
    std::printf("  keys:      %s, ESC quits\n", kHelpLine);
    std::fflush(stdout);

    SDL_SetMainReady();
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "train-lens: cannot start SDL: %s\n", SDL_GetError());
        return 1;
    }
    int exit_code = 0;
    {
        Window w;
        if (!w.open(true) && !w.open(false)) {
            std::fprintf(stderr, "train-lens: cannot open an OpenGL window: %s\n", SDL_GetError());
            SDL_Quit();
            return 1;
        }

        Sim sim;
        View view;
        view.max_show = opt.max_show;
        float mix_target = 0.f, cam_yaw = 0.f, cam_pitch = 0.f;
        bool cam_anim = false, auto_rotate = false, paused = false, delay_on = true, show_help = true;
        bool dragging = false, mouse_in = false, want_shot = false;
        int panel_choice = -1, net_choice = -1;  // -1 = automatic, else what the user picked with T / N
        bool show_panel_now = false, show_net_now = true;
        float drag_moved = 0.f, mouse_x = 0.f, mouse_y = 0.f;
        Hover hover;
        int bad_lines = 0;
        double t_sim = 0.0, acc = 0.0, last = now_seconds(), fps_t = last, fps = 0.0;
        long frame = 0, fps_frames = 0;
        const double kStep = 1.0 / 60.0;

        auto set_mode = [&](bool three_d) {
            mix_target = three_d ? 1.f : 0.f;
            cam_yaw = three_d ? -0.65f : 0.f;
            cam_pitch = three_d ? 0.35f : 0.f;
            cam_anim = true;
            auto_rotate = false;
        };
        if (opt.three_d) {
            set_mode(true);
            view.mix = mix_target;
            view.yaw = cam_yaw;
            view.pitch = cam_pitch;
        }

        for (bool running = true; running;) {
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                switch (e.type) {
                    case SDL_QUIT: running = false; break;
                    case SDL_KEYDOWN:
                        switch (e.key.keysym.sym) {
                            case SDLK_ESCAPE: case SDLK_q: running = false; break;
                            case SDLK_SPACE: paused = !paused; break;
                            case SDLK_v: set_mode(mix_target < 0.5f); break;
                            case SDLK_a: auto_rotate = !auto_rotate; cam_anim = false; break;
                            case SDLK_d: delay_on = !delay_on; break;
                            case SDLK_h: show_help = !show_help; break;
                            case SDLK_s: want_shot = true; break;
                            case SDLK_t: panel_choice = show_panel_now ? 0 : 1; break;
                            case SDLK_n: net_choice = show_net_now ? 0 : 1; break;
                            case SDLK_c: for (auto& m : sim.muted) std::fill(m.begin(), m.end(), 0); break;
                            case SDLK_r:
                                st.net.randomize((unsigned)SDL_GetTicks());
                                std::printf("train-lens: weights randomized\n");
                                break;
                            case SDLK_RIGHTBRACKET: case SDLK_EQUALS: view.gain = std::min(4.f, view.gain * 1.25f); break;
                            case SDLK_LEFTBRACKET: case SDLK_MINUS: view.gain = std::max(0.1f, view.gain / 1.25f); break;
                            case SDLK_0: view.dist = 3.1f; set_mode(mix_target > 0.5f); break;
                            default: break;
                        }
                        break;
                    case SDL_MOUSEBUTTONDOWN:
                        if (e.button.button == SDL_BUTTON_LEFT) { dragging = true; drag_moved = 0.f; }
                        break;
                    case SDL_MOUSEBUTTONUP:
                        if (e.button.button == SDL_BUTTON_LEFT) {
                            dragging = false;
                            if (drag_moved < 5.f && hover.layer >= 0 && hover.layer < (int)sim.muted.size())
                                sim.muted[hover.layer][hover.index] ^= 1;
                        }
                        break;
                    case SDL_MOUSEMOTION:
                        mouse_x = (float)e.motion.x;
                        mouse_y = (float)e.motion.y;
                        mouse_in = true;
                        if (dragging) {
                            view.yaw += (float)e.motion.xrel * 0.008f;
                            view.pitch = std::max(-1.5f, std::min(1.5f, view.pitch + (float)e.motion.yrel * 0.008f));
                            drag_moved += std::fabs((float)e.motion.xrel) + std::fabs((float)e.motion.yrel);
                            cam_anim = false;
                        }
                        break;
                    case SDL_MOUSEWHEEL:
                        view.dist = std::max(1.2f, std::min(15.f, view.dist * std::pow(0.9f, (float)e.wheel.y)));
                        break;
                    case SDL_WINDOWEVENT:
                        if (e.window.event == SDL_WINDOWEVENT_LEAVE) mouse_in = false;
                        break;
                    default: break;
                }
            }

            const double now = now_seconds();
            srv.poll([&](const std::string& line) {
                std::string e2;
                if (!apply_line(st, line, now, e2) && ++bad_lines <= 20)
                    std::fprintf(stderr, "train-lens: ignored a line: %s%s\n", e2.c_str(),
                                 bad_lines == 20 ? " (further errors hidden)" : "");
            });
            if (sim.version != st.version) {
                sim.reset(st.net, st.version);
                hover = {};
            }

            const double dt = std::min(0.1, now - last);
            last = now;
            if (!paused) {
                acc += dt;
                while (acc >= kStep) {
                    t_sim += kStep;
                    sim.tick(st, t_sim, now);
                    acc -= kStep;
                }
            }

            const float k = std::min(1.f, (float)dt * 6.f);
            view.mix += (mix_target - view.mix) * k;
            if (cam_anim) {
                view.yaw += (cam_yaw - view.yaw) * k;
                view.pitch += (cam_pitch - view.pitch) * k;
                if (std::fabs(cam_yaw - view.yaw) + std::fabs(cam_pitch - view.pitch) < 1e-3f) cam_anim = false;
            }
            if (auto_rotate) view.yaw += (float)dt * 0.3f;

            int win_w, win_h, fb_w, fb_h;
            SDL_GetWindowSize(w.win, &win_w, &win_h);
            SDL_GL_GetDrawableSize(w.win, &fb_w, &fb_h);
            const float scale = (float)fb_w / (float)std::max(1, win_w);

            const int L = st.net.layers();
            const int delay = delay_on ? std::min(6, (Sim::kHist - 1) / std::max(1, L - 1)) : 0;

            Scene scene;
            scene.net = &st.net;
            for (int l = 0; l < L; ++l) scene.acts.push_back(sim.layer(l, delay));
            scene.muted = &sim.muted;
            scene.labels = &st.labels;
            scene.view = view;
            scene.fb_w = fb_w;
            scene.fb_h = fb_h;
            scene.scale = scale;
            scene.mouse_x = mouse_in ? mouse_x * scale : -1.f;
            scene.mouse_y = mouse_in ? mouse_y * scale : -1.f;

            // A monitor-only client (any model, no network sent) replaces the demo network.
            const bool demo_net = opt.file.empty() && st.version == boot_version;
            show_net_now = net_choice >= 0 ? net_choice == 1 : !(demo_net && st.train.any);
            show_panel_now = panel_choice >= 0 ? panel_choice == 1 : st.train.any;
            if (!show_net_now && !show_panel_now) show_net_now = true;
            scene.show_net = show_net_now;
            scene.show_panel = show_panel_now;
            scene.train = &st.train;
            if (show_panel_now) {
                scene.alerts = find_alerts(st.train);
                if (show_net_now)
                    for (Alert& a : sim.dead_relu_alerts(st.net)) scene.alerts.push_back(a);
            }

            std::string src = demo_net && !st.train.any ? source : "LIVE DATA";
            if (srv.listening())
                src += "   TCP 127.0.0.1:" + std::to_string(srv.port()) + "  " + std::to_string(srv.clients()) +
                       (srv.clients() == 1 ? " CLIENT" : " CLIENTS");
            if (srv.reading_stdin()) src += "   STDIN";
            char line[160];
            if (show_net_now) {
                std::string act_line;
                for (Act a : st.net.acts) act_line += std::string(act_line.empty() ? "" : " ") + act_name(a);
                scene.hud = {"TRAINLENS  " + join_sizes(st.net.sizes, "-"), act_line, src};
            } else {
                scene.hud = {"TRAINLENS  TRAINING MONITOR", src};
            }
            if (!st.status.empty()) {
                scene.hud_highlight = (int)scene.hud.size();
                scene.hud.push_back(st.status);
            }
            if (show_net_now) {
                const std::vector<float>& out = *scene.acts.back();
                const int top = (int)(std::max_element(out.begin(), out.end()) - out.begin());
                std::snprintf(line, sizeof line, "TOP %s  %.3f",
                              top < (int)st.labels.size() ? st.labels[top].c_str() : ("OUT" + std::to_string(top)).c_str(),
                              out[top]);
                scene.hud.push_back(line);
            }
            std::snprintf(line, sizeof line, "%.0f FPS  GAIN %.2f%s%s", fps, view.gain, paused ? "  PAUSED" : "",
                          delay_on ? "" : "  NO DELAY");
            scene.hud.push_back(line);
            if (show_help) scene.help = kHelpLine;

            hover = draw_scene(scene);

            ++frame;
            const bool last_frame = opt.frames > 0 && frame >= opt.frames;
            if (want_shot || (last_frame && !opt.screenshot.empty())) {
                std::string path = opt.screenshot;
                if (want_shot) {
                    char name[64];
                    const std::time_t t = std::time(nullptr);
                    std::strftime(name, sizeof name, "train-lens-%Y%m%d-%H%M%S.png", std::localtime(&t));
                    path = name;
                }
                std::string e2;
                if (save_framebuffer_png(path, fb_w, fb_h, e2)) std::printf("train-lens: saved %s\n", path.c_str());
                else std::fprintf(stderr, "train-lens: screenshot failed: %s\n", e2.c_str());
                want_shot = false;
            }
            SDL_GL_SwapWindow(w.win);

            ++fps_frames;
            if (now - fps_t >= 0.5) {
                fps = fps_frames / (now - fps_t);
                fps_t = now;
                fps_frames = 0;
            }
            if (last_frame) running = false;
        }
        if (opt.frames > 0) std::printf("train-lens: drew %ld frames\n", frame);
    }
    SDL_Quit();
    return exit_code;
}
