#include "render.hpp"

#define GL_SILENCE_DEPRECATION
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "font5x7.hpp"

namespace {

const float kBg[3] = {0.027f, 0.031f, 0.051f};
const float kGold[3] = {1.00f, 0.69f, 0.20f};   // positive signal
const float kEmber[3] = {0.93f, 0.29f, 0.23f};  // negative signal
const float kIce[3] = {0.51f, 0.75f, 1.00f};    // input neurons
const float kDim[3] = {0.16f, 0.17f, 0.23f};    // a silent neuron
const float kMutedRGBA[4] = {0.30f, 0.32f, 0.40f, 0.75f};
const float kText[4] = {0.55f, 0.57f, 0.66f, 1.f};
const float kTextBright[4] = {0.92f, 0.93f, 0.97f, 1.f};
const float kTextGold[4] = {1.00f, 0.69f, 0.20f, 1.f};
const float kBarTrack[4] = {0.55f, 0.57f, 0.66f, 0.25f};
const float kTooltipBg[4] = {0.07f, 0.08f, 0.12f, 0.94f};

struct V3 {
    float x, y, z;
};

struct Mat4 {
    float m[16];  // column-major, as OpenGL expects
};

Mat4 identity() {
    Mat4 r{};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.f;
    return r;
}

Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0.f;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = s;
        }
    return r;
}

Mat4 perspective(float fovy, float aspect, float n, float f) {
    const float t = 1.f / std::tan(fovy / 2.f);
    Mat4 r{};
    r.m[0] = t / aspect;
    r.m[5] = t;
    r.m[10] = (f + n) / (n - f);
    r.m[11] = -1.f;
    r.m[14] = 2.f * f * n / (n - f);
    return r;
}

Mat4 translate(float x, float y, float z) {
    Mat4 r = identity();
    r.m[12] = x;
    r.m[13] = y;
    r.m[14] = z;
    return r;
}

Mat4 rot_x(float a) {
    Mat4 r = identity();
    const float c = std::cos(a), s = std::sin(a);
    r.m[5] = c; r.m[6] = s;
    r.m[9] = -s; r.m[10] = c;
    return r;
}

Mat4 rot_y(float a) {
    Mat4 r = identity();
    const float c = std::cos(a), s = std::sin(a);
    r.m[0] = c; r.m[2] = -s;
    r.m[8] = s; r.m[10] = c;
    return r;
}

// World point to framebuffer pixels (origin top-left). False if behind the camera.
bool project(const Mat4& mvp, const V3& p, int w, int h, float& sx, float& sy) {
    const float* m = mvp.m;
    const float cx = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
    const float cy = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
    const float cw = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
    if (cw <= 1e-4f) return false;
    sx = (cx / cw * 0.5f + 0.5f) * (float)w;
    sy = (0.5f - cy / cw * 0.5f) * (float)h;
    return true;
}

// Layers run left to right. mix blends a flat column into a square grid in
// the y-z plane, which is easier to read for wide layers once rotated.
V3 node_pos(int l, int k, int n, int L, float mix) {
    // The flat view sits left of centre to leave room for the output labels.
    const float t = L > 1 ? (float)l / (float)(L - 1) : 0.5f;
    const float x = (-1.65f + 2.95f * t) * (1.f - mix) + (-1.5f + 3.f * t) * mix;
    const float col_y = n > 1 ? 0.76f - 1.52f * (float)k / (float)(n - 1) : 0.f;
    const int cols = (int)std::ceil(std::sqrt((double)n));
    const int rows = (n + cols - 1) / cols;
    const float sp = std::min(0.22f, 1.5f / (float)std::max(cols, rows));
    const int i = k % cols, j = k / cols;
    const float gy = ((float)(rows - 1) * 0.5f - (float)j) * sp;
    const float gz = ((float)i - (float)(cols - 1) * 0.5f) * sp;
    return {x, col_y + (gy - col_y) * mix, gz * mix};
}

std::vector<int> shown_indices(int n, int max_show) {
    const int m = std::max(1, std::min(n, max_show));
    std::vector<int> idx(m);
    for (int k = 0; k < m; ++k)
        idx[k] = m == n ? k : (int)((long long)k * (n - 1) / std::max(1, m - 1));
    return idx;
}

std::string layer_name(int l, int L) {
    if (l == 0) return "INPUT";
    if (l == L - 1) return "OUTPUT";
    return "H" + std::to_string(l);
}

// ---- 2D overlay text (5x7 bitmap font drawn as quads) ----

void push_text(std::vector<float>& q, float x, float y, float px, const std::string& s) {
    for (char ch : s) {
        const unsigned char* g = glyph5x7(ch);
        for (int col = 0; col < 5; ++col)
            for (int row = 0; row < 7; ++row)
                if ((g[col] >> row) & 1) {
                    const float x0 = x + (float)col * px, y0 = y + (float)row * px;
                    const float quad[8] = {x0, y0, x0 + px, y0, x0 + px, y0 + px, x0, y0 + px};
                    q.insert(q.end(), quad, quad + 8);
                }
        x += 6.f * px;
    }
}

float text_width(const std::string& s, float px) {
    return s.empty() ? 0.f : (float)s.size() * 6.f * px - px;
}

void flush_quads(std::vector<float>& q, const float* rgba) {
    if (q.empty()) return;
    glColor4fv(rgba);
    glVertexPointer(2, GL_FLOAT, 0, q.data());
    glDrawArrays(GL_QUADS, 0, (GLsizei)(q.size() / 2));
    q.clear();
}

void fill_rect(float x, float y, float w, float h, const float* rgba) {
    std::vector<float> q = {x, y, x + w, y, x + w, y + h, x, y + h};
    flush_quads(q, rgba);
}

void draw_lines(const std::vector<float>& xy, GLenum mode, const float* rgba) {
    if (xy.size() < 4) return;
    glColor4fv(rgba);
    glVertexPointer(2, GL_FLOAT, 0, xy.data());
    glDrawArrays(mode, 0, (GLsizei)(xy.size() / 2));
}

std::string upper_fmt(const char* f, double v) {
    char b[48];
    if (std::isnan(v)) return "NAN";
    if (std::isinf(v)) return v > 0 ? "INF" : "-INF";
    std::snprintf(b, sizeof b, f, v);
    for (char* p = b; *p; ++p) *p = (char)std::toupper((unsigned char)*p);
    return b;
}

// Shortens from the left, keeping the end of a dotted module path.
std::string fit_left(const std::string& s, float px, float room) {
    if (text_width(s, px) <= room) return s;
    const int keep = std::max(1, (int)(room / (6.f * px)) - 2);
    return keep < (int)s.size() ? ".." + s.substr(s.size() - (size_t)keep) : s;
}

std::vector<std::string> wrap(const std::string& s, float px, float room) {
    std::vector<std::string> rows(1);
    size_t p = 0;
    while (p < s.size()) {
        size_t e = s.find(' ', p);
        if (e == std::string::npos) e = s.size();
        const std::string word = s.substr(p, e - p);
        const std::string joined = rows.back().empty() ? word : rows.back() + " " + word;
        if (!rows.back().empty() && text_width(joined, px) > room) rows.push_back(word);
        else rows.back() = joined;
        p = e + 1;
    }
    return rows;
}

const float kPanelBg[4] = {0.040f, 0.045f, 0.070f, 1.f};
const float kPanelLine[4] = {0.14f, 0.15f, 0.22f, 1.f};
const float kPlotBg[4] = {0.055f, 0.060f, 0.090f, 1.f};
const float kGrid[4] = {0.55f, 0.57f, 0.66f, 0.12f};
const float kIceRGBA[4] = {0.51f, 0.75f, 1.00f, 1.f};
const float kEmberRGBA[4] = {0.93f, 0.29f, 0.23f, 1.f};
const float kLilac[4] = {0.78f, 0.62f, 1.00f, 1.f};
const float kCritical[4] = {1.00f, 0.36f, 0.32f, 1.f};
const float kWarn[4] = {1.00f, 0.66f, 0.28f, 1.f};
const float kGood[4] = {0.45f, 0.82f, 0.56f, 1.f};

// One chart: a group of series sharing a chart key, e.g. loss and val_loss.
void draw_chart(const std::string& key, const std::vector<const Series*>& group, float x, float y, float w,
                float h, float lp, float sc) {
    std::vector<float> q;
    const float* palette[] = {kTextGold, kIceRGBA, kLilac, kEmberRGBA};

    // title: name, then the latest value of each series
    push_text(q, x, y, lp, key);
    flush_quads(q, kTextBright);
    float tx = x + text_width(key, lp) + 12.f * sc;
    for (size_t i = 0; i < group.size(); ++i) {
        const Series& s = *group[i];
        const double last = s.value.back();
        std::string t = (group.size() > 1 ? (is_validation(s.name) ? "VAL " : "TRAIN ") : std::string()) +
                        upper_fmt("%.4g", last);
        push_text(q, tx, y, lp, t);
        flush_quads(q, std::isfinite(last) ? palette[i % 4] : kCritical);
        tx += text_width(t, lp) + 12.f * sc;
    }

    const float px0 = x, py0 = y + 11.f * lp, pw = w, ph = h - 21.f * lp;
    if (ph < 10.f * sc) return;
    fill_rect(px0, py0, pw, ph, kPlotBg);

    double xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
    for (const Series* s : group)
        for (size_t i = 0; i < s->value.size(); ++i) {
            xmin = std::min(xmin, s->step[i]);
            xmax = std::max(xmax, s->step[i]);
            if (std::isfinite(s->value[i])) {
                ymin = std::min(ymin, s->value[i]);
                ymax = std::max(ymax, s->value[i]);
            }
        }
    if (!std::isfinite(ymin)) return;
    const bool logy = key.find("loss") != std::string::npos && ymin > 0 && ymax / ymin > 20;
    auto ty = [&](double v) { return logy ? std::log10(v) : v; };
    double lo = ty(ymin), hi = ty(ymax);
    if (hi - lo < 1e-12) { lo -= 0.5 * std::fabs(lo) + 1e-6; hi += 0.5 * std::fabs(hi) + 1e-6; }
    const double pad = (hi - lo) * 0.06;
    lo -= pad;
    hi += pad;
    if (xmax <= xmin) xmax = xmin + 1;

    std::vector<float> grid;
    for (int i = 1; i < 4; ++i) {
        const float gy = py0 + ph * (float)i / 4.f;
        grid.insert(grid.end(), {px0, gy, px0 + pw, gy});
    }
    draw_lines(grid, GL_LINES, kGrid);

    glEnable(GL_LINE_SMOOTH);
    for (size_t k = 0; k < group.size(); ++k) {
        const Series& s = *group[k];
        std::vector<float> line;
        float lx = 0, ly = 0;
        bool have = false;
        auto flush = [&]() {
            if (line.size() >= 4) draw_lines(line, GL_LINE_STRIP, palette[k % 4]);
            line.clear();
        };
        for (size_t i = 0; i < s.value.size(); ++i) {
            if (!std::isfinite(s.value[i])) { flush(); continue; }  // a gap marks NaN/inf
            lx = px0 + pw * (float)((s.step[i] - xmin) / (xmax - xmin));
            ly = py0 + ph * (float)(1.0 - (ty(s.value[i]) - lo) / (hi - lo));
            line.insert(line.end(), {lx, ly});
            have = true;
        }
        if (line.size() == 2) line.insert(line.end(), {lx + 1.f, ly});
        flush();
        if (have) fill_rect(lx - 2.5f * sc, ly - 2.5f * sc, 5.f * sc, 5.f * sc, palette[k % 4]);
    }

    const float sp = std::max(1.f, lp - 1.f);  // axis labels one size smaller
    auto yval = [&](double t) { return logy ? std::pow(10.0, t) : t; };
    push_text(q, px0 + 4.f * sc, py0 + 4.f * sc, sp, upper_fmt("%.3g", yval(hi)) + (logy ? " LOG" : ""));
    push_text(q, px0 + 4.f * sc, py0 + ph - 4.f * sc - 7.f * sp, sp, upper_fmt("%.3g", yval(lo)));
    push_text(q, px0, py0 + ph + 4.f * sc, sp, "STEP " + upper_fmt("%.0f", xmin));
    const std::string end = upper_fmt("%.0f", xmax);
    push_text(q, px0 + pw - text_width(end, sp), py0 + ph + 4.f * sc, sp, end);
    flush_quads(q, kText);
}

void draw_panel(const Scene& s, float x0, float w, float lp, float sc, float top) {
    const Training& t = *s.train;
    fill_rect(x0, 0, w, (float)s.fb_h, kPanelBg);
    if (x0 > 0) fill_rect(x0, 0, std::max(1.f, sc), (float)s.fb_h, kPanelLine);

    const float pad = 18.f * sc, cx = x0 + pad, cw = w - 2.f * pad;
    const float row = 10.f * lp;
    const float bottom = (float)s.fb_h - 16.f * sc - (x0 > 0 ? 0.f : 26.f * lp);  // room for the help rows
    float y = top;
    std::vector<float> q;

    // header
    double step = -1;
    for (const Series& ser : t.series)
        if (!ser.step.empty()) step = std::max(step, ser.step.back());
    push_text(q, cx, y, lp, "TRAINING");
    flush_quads(q, kTextGold);
    if (step >= 0) {
        push_text(q, cx + text_width("TRAINING", lp) + 14.f * sc, y, lp, "STEP " + upper_fmt("%.0f", step));
        flush_quads(q, kTextBright);
    }
    y += row * 1.6f;

    // checks are sized first so they always fit at the bottom
    struct Line {
        int level;
        std::string text;
        bool first;  // first row of an alert gets the colored marker
    };
    std::vector<Line> lines;
    for (const Alert& a : s.alerts) {
        const std::vector<std::string> rows = wrap(a.text, lp, cw - 14.f * sc);
        for (size_t i = 0; i < rows.size(); ++i) lines.push_back({a.level, rows[i], i == 0});
    }
    const size_t max_check_rows = 9;
    const bool cut = lines.size() > max_check_rows;
    if (cut) lines.resize(max_check_rows - 1);
    const float checks_h = row * (2.2f + (float)std::max<size_t>(1, lines.size() + (cut ? 1 : 0)));

    // charts
    std::vector<std::string> keys;
    std::vector<std::vector<const Series*>> groups;
    for (const Series& ser : t.series) {
        if (ser.value.empty()) continue;
        const std::string k = chart_key(ser.name);
        auto it = std::find(keys.begin(), keys.end(), k);
        if (it == keys.end()) { keys.push_back(k); groups.push_back({&ser}); }
        else groups[it - keys.begin()].push_back(&ser);
    }
    const float space = bottom - y - checks_h;
    const bool has_params = !t.params.empty();
    const float charts_h = has_params ? space * 0.58f : space;
    const int cols = cw > 900.f * sc && groups.size() > 1 ? 2 : 1;
    if (!groups.empty()) {
        const float min_h = 70.f * sc + 21.f * lp;
        const int max_rows = std::max(1, (int)(charts_h / min_h));
        const int shown = std::min((int)groups.size(), max_rows * cols);
        const int rows = (shown + cols - 1) / cols;
        const float gap = 14.f * sc;
        const float ch = std::min(220.f * sc, (charts_h - gap * (float)(rows - 1)) / (float)rows);
        const float colw = (cw - gap * (float)(cols - 1)) / (float)cols;
        for (int i = 0; i < shown; ++i)
            draw_chart(keys[i], groups[i], cx + (float)(i % cols) * (colw + gap), y + (float)(i / cols) * (ch + gap),
                       colw, ch, lp, sc);
        y += (float)rows * (ch + gap);
        if (shown < (int)groups.size()) {
            push_text(q, cx, y - gap + 2.f * sc, lp, "+" + std::to_string(groups.size() - shown) + " MORE CURVES (ENLARGE THE WINDOW)");
            flush_quads(q, kText);
            y += row;
        }
    } else {
        push_text(q, cx, y, lp, "WAITING FOR METRICS (MONITOR.STEP / VIEWER.METRIC)");
        flush_quads(q, kText);
        y += row * 2;
    }

    // layer health
    if (has_params) {
        y += row * 0.5f;
        const float c_bar = cx + cw * 0.36f, c_g = cx + cw * 0.64f, c_u = cx + cw * 0.82f;
        push_text(q, cx, y, lp, "LAYER HEALTH");
        flush_quads(q, kTextBright);
        push_text(q, c_bar, y, lp, "GRAD NORM");
        push_text(q, c_u, y, lp, "UPDATE/W");
        flush_quads(q, kText);
        y += row * 1.3f;
        const int fit = std::max(1, (int)((bottom - checks_h - y) / row));
        const int n = (int)t.params.size();
        const int show = n <= fit ? n : fit - 1;
        const float bw = c_g - c_bar - 10.f * sc;
        for (int i = 0; i < show; ++i) {
            const ParamStat& p = t.params[i];
            const float ry = y + (float)i * row;
            push_text(q, cx, ry, lp, fit_left(p.name, lp, c_bar - cx - 8.f * sc));
            flush_quads(q, kText);
            if (p.grad < 0) {
                push_text(q, c_bar, ry, lp, "NO GRAD");
                flush_quads(q, kText);
            } else {
                const double m = std::min(1.0, std::max(0.0, (std::log10(std::max(p.grad, 1e-12)) + 8.0) / 10.0));
                const float* col = p.grad > 100.0 ? kCritical : (p.grad < 1e-7 ? kIceRGBA : kTextGold);
                fill_rect(c_bar, ry + 2.f * lp, bw, 3.f * lp, kPlotBg);
                fill_rect(c_bar, ry + 2.f * lp, std::max(2.f * sc, bw * (float)m), 3.f * lp, col);
                push_text(q, c_g, ry, lp, upper_fmt("%.1e", p.grad));
                flush_quads(q, col);
            }
            if (p.update >= 0) {
                const float* col = p.update > 0.05 ? kWarn : (p.update < 1e-6 && p.grad > 0 ? kIceRGBA : kText);
                push_text(q, c_u, ry, lp, upper_fmt("%.1e", p.update));
                flush_quads(q, col);
            }
        }
        if (show < n) {
            push_text(q, cx, y + (float)show * row, lp, "+" + std::to_string(n - show) + " MORE LAYERS");
            flush_quads(q, kText);
        }
    }

    // checks
    float cy = bottom - checks_h + row * 0.6f;
    fill_rect(cx, cy - row * 0.5f, cw, std::max(1.f, sc), kPanelLine);
    push_text(q, cx, cy, lp, "CHECKS");
    flush_quads(q, kTextBright);
    cy += row * 1.3f;
    if (lines.empty()) {
        push_text(q, cx, cy, lp, t.series.empty() && t.params.empty() ? "NO DATA YET" : "NO PROBLEMS DETECTED");
        flush_quads(q, kGood);
    }
    for (const Line& l : lines) {
        const float* col = l.level >= 2 ? kCritical : kWarn;
        if (l.first) fill_rect(cx, cy + 1.f * lp, 4.f * sc, 5.f * lp, col);
        push_text(q, cx + 14.f * sc, cy, lp, l.text);
        flush_quads(q, col);
        cy += row;
    }
    if (cut) {
        push_text(q, cx + 14.f * sc, cy, lp, "MORE PROBLEMS NOT SHOWN");
        flush_quads(q, kText);
    }
}

float max_point_size() {
    static float cached = -1.f;
    if (cached < 0.f) {
        GLfloat range[2] = {1.f, 64.f};
        glGetFloatv(GL_POINT_SIZE_RANGE, range);
        cached = std::max(1.f, range[1]);
    }
    return cached;
}

}  // namespace

Hover draw_scene(const Scene& s) {
    Hover hover;
    glViewport(0, 0, s.fb_w, s.fb_h);
    glClearColor(kBg[0], kBg[1], kBg[2], 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glEnableClientState(GL_VERTEX_ARRAY);

    const Network& net = *s.net;
    const View& v = s.view;
    const int L = s.show_net ? net.layers() : 0;
    const float sc = s.scale;

    // The training panel takes the right side, or the whole window when there is no network.
    const bool panel = s.show_panel && s.train;
    const float panel_w = !panel ? 0.f
                        : !s.show_net ? (float)s.fb_w
                                      : std::min(std::max((float)s.fb_w * 0.4f, 380.f * sc), 640.f * sc);
    const int net_w = std::max(1, s.fb_w - (int)panel_w);
    glViewport(0, 0, net_w, s.fb_h);

    const float aspect = (float)net_w / (float)std::max(1, s.fb_h);
    const Mat4 P = perspective(45.f * 3.14159265f / 180.f, aspect, 0.05f, 50.f);
    const Mat4 V = mul(translate(0.f, 0.f, -v.dist), mul(rot_x(v.pitch), rot_y(v.yaw)));
    const Mat4 MVP = mul(P, V);

    // ---- positions, screen projections, per-layer scale ----
    std::vector<std::vector<int>> shown(L);
    std::vector<std::vector<V3>> pos(L);
    std::vector<std::vector<float>> sx(L), sy(L);
    std::vector<float> amax(L, 1e-6f);
    int maxn = 1;
    for (int l = 0; l < L; ++l) {
        shown[l] = shown_indices(net.sizes[l], v.max_show);
        const int n = (int)shown[l].size();
        maxn = std::max(maxn, n);
        pos[l].resize(n);
        sx[l].assign(n, -1e9f);
        sy[l].assign(n, -1e9f);
        const std::vector<float>& a = *s.acts[l];
        for (int k = 0; k < n; ++k) {
            pos[l][k] = node_pos(l, k, n, L, v.mix);
            project(MVP, pos[l][k], net_w, s.fb_h, sx[l][k], sy[l][k]);
            amax[l] = std::max(amax[l], std::fabs(a[shown[l][k]]));
        }
    }
    auto norm = [&](int l, float a) { return std::min(1.f, std::fabs(a) / amax[l]); };

    // ---- hover ----
    int hl = -1, hk = -1;
    if (s.mouse_x >= 0.f && s.mouse_x < (float)net_w) {
        float best = (14.f * sc) * (14.f * sc);
        for (int l = 0; l < L; ++l)
            for (size_t k = 0; k < shown[l].size(); ++k) {
                const float dx = sx[l][k] - s.mouse_x, dy = sy[l][k] - s.mouse_y;
                if (dx * dx + dy * dy < best) {
                    best = dx * dx + dy * dy;
                    hl = l;
                    hk = (int)k;
                }
            }
    }
    if (hl >= 0) hover = {hl, shown[hl][hk]};

    // ---- connections: brightness = |activation x weight|, normalized per layer ----
    std::vector<float> ev, ec;
    for (int l = 0; l + 1 < L; ++l) {
        const std::vector<float>& a = *s.acts[l];
        const std::vector<float>& w = net.W[l];
        const bool known = net.has_w[l];
        const int nin = net.sizes[l];
        const std::vector<int>& src = shown[l];
        const std::vector<int>& dst = shown[l + 1];
        auto signal = [&](int j, int i) { return known ? a[i] * w[(size_t)j * nin + i] : a[i]; };

        float smax = 1e-6f;
        for (int j : dst)
            for (int i : src) smax = std::max(smax, std::fabs(signal(j, i)));

        for (size_t kj = 0; kj < dst.size(); ++kj)
            for (size_t ki = 0; ki < src.size(); ++ki) {
                const float sig = signal(dst[kj], src[ki]);
                float alpha = (0.02f + 0.7f * std::pow(std::fabs(sig) / smax, 0.8f)) * v.gain;
                if (hl >= 0) {
                    const bool touch = (hl == l && hk == (int)ki) || (hl == l + 1 && hk == (int)kj);
                    alpha = touch ? std::min(1.f, 0.18f + alpha * 1.6f) : alpha * 0.15f;
                }
                alpha = std::min(alpha, 0.95f);
                if (alpha < 0.02f) continue;
                const float* c = sig >= 0.f ? kGold : kEmber;
                const V3& p1 = pos[l][ki];
                const V3& p2 = pos[l + 1][kj];
                ev.insert(ev.end(), {p1.x, p1.y, p1.z, p2.x, p2.y, p2.z});
                ec.insert(ec.end(), {c[0], c[1], c[2], alpha, c[0], c[1], c[2], alpha});
            }
    }

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(P.m);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(V.m);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE);  // additive: crossing lines glow
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.f);
    glEnableClientState(GL_COLOR_ARRAY);
    if (!ev.empty()) {
        glVertexPointer(3, GL_FLOAT, 0, ev.data());
        glColorPointer(4, GL_FLOAT, 0, ec.data());
        glDrawArrays(GL_LINES, 0, (GLsizei)(ev.size() / 3));
    }

    // ---- neurons ----
    const float R = std::min(std::max(s.fb_h * 0.8f / (float)maxn * 0.45f, 3.f * sc), 10.f * sc);
    std::vector<float> nv, nc;
    auto base_color = [&](int l, float a) { return l == 0 ? kIce : (a >= 0.f ? kGold : kEmber); };
    auto is_muted = [&](int l, int idx) {
        return s.muted && l < (int)s.muted->size() && idx < (int)(*s.muted)[l].size() && (*s.muted)[l][idx];
    };

    // glow
    for (int l = 0; l < L; ++l)
        for (size_t k = 0; k < shown[l].size(); ++k) {
            const int idx = shown[l][k];
            const float a = (*s.acts[l])[idx], m = norm(l, a);
            if (is_muted(l, idx) || m < 0.15f) continue;
            const float* c = base_color(l, a);
            nv.insert(nv.end(), {pos[l][k].x, pos[l][k].y, pos[l][k].z});
            nc.insert(nc.end(), {c[0], c[1], c[2], 0.28f * m});
        }
    glEnable(GL_POINT_SMOOTH);
    if (!nv.empty()) {
        glPointSize(std::min(R * 3.5f, max_point_size()));
        glVertexPointer(3, GL_FLOAT, 0, nv.data());
        glColorPointer(4, GL_FLOAT, 0, nc.data());
        glDrawArrays(GL_POINTS, 0, (GLsizei)(nv.size() / 3));
    }

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    if (hl >= 0) {
        const float ring[3] = {pos[hl][hk].x, pos[hl][hk].y, pos[hl][hk].z};
        const float white[4] = {1.f, 1.f, 1.f, 0.95f};
        glPointSize(std::min(R + 5.f * sc, max_point_size()));
        glVertexPointer(3, GL_FLOAT, 0, ring);
        glColorPointer(4, GL_FLOAT, 0, white);
        glDrawArrays(GL_POINTS, 0, 1);
    }

    nv.clear();
    nc.clear();
    for (int l = 0; l < L; ++l)
        for (size_t k = 0; k < shown[l].size(); ++k) {
            const int idx = shown[l][k];
            const float a = (*s.acts[l])[idx];
            nv.insert(nv.end(), {pos[l][k].x, pos[l][k].y, pos[l][k].z});
            if (is_muted(l, idx)) {
                nc.insert(nc.end(), kMutedRGBA, kMutedRGBA + 4);
            } else {
                const float* c = base_color(l, a);
                const float t = 0.25f + 0.75f * norm(l, a);
                nc.insert(nc.end(), {kDim[0] + (c[0] - kDim[0]) * t, kDim[1] + (c[1] - kDim[1]) * t,
                                     kDim[2] + (c[2] - kDim[2]) * t, 1.f});
            }
        }
    glPointSize(std::min(R, max_point_size()));
    glVertexPointer(3, GL_FLOAT, 0, nv.data());
    glColorPointer(4, GL_FLOAT, 0, nc.data());
    glDrawArrays(GL_POINTS, 0, (GLsizei)(nv.size() / 3));
    glDisableClientState(GL_COLOR_ARRAY);

    // ---- 2D overlay ----
    glViewport(0, 0, s.fb_w, s.fb_h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, s.fb_w, s.fb_h, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    const float lp = std::max(1.f, std::round(1.5f * sc));  // label pixel size
    const float hp = std::max(2.f, std::round(1.8f * sc));  // HUD pixel size
    std::vector<float> q, qb;
    char buf[96];

    if (panel) {
        const float hud_bottom = 16.f * sc + (float)s.hud.size() * 11.f * hp + 10.f * sc;
        draw_panel(s, (float)s.fb_w - panel_w, panel_w, lp, sc, s.show_net ? 16.f * sc : hud_bottom);
    }

    // layer captions under each column
    for (int l = 0; l < L; ++l) {
        float cx = 0.f, bottom = -1e9f;
        int cnt = 0;
        for (size_t k = 0; k < shown[l].size(); ++k)
            if (sy[l][k] > -1e8f) {
                cx += sx[l][k];
                bottom = std::max(bottom, sy[l][k]);
                ++cnt;
            }
        if (!cnt) continue;
        cx /= (float)cnt;
        const std::string name = layer_name(l, L);
        std::string info = std::to_string(net.sizes[l]);
        if (l > 0) info += std::string(" ") + act_name(net.acts[l - 1]);
        if ((int)shown[l].size() < net.sizes[l]) info += " (" + std::to_string(shown[l].size()) + " SHOWN)";
        const float y = std::min(bottom + R + 10.f * sc, (float)s.fb_h - 26.f * sc - 40.f * lp);  // clear of the help rows
        push_text(qb, cx - text_width(name, lp) / 2.f, y, lp, name);
        push_text(q, cx - text_width(info, lp) / 2.f, y + 10.f * lp, lp, info);
    }
    flush_quads(qb, kTextBright);
    flush_quads(q, kText);

    // input and output labels in the flat view
    if (v.mix < 0.5f && L >= 2) {
        if (shown[0].size() <= 48)
            for (size_t k = 0; k < shown[0].size(); ++k) {
                if (sy[0][k] < -1e8f) continue;
                const std::string t = "IN" + std::to_string(shown[0][k]);
                push_text(q, sx[0][k] - R - 8.f * sc - text_width(t, lp), sy[0][k] - 3.5f * lp, lp, t);
            }
        const int lo = L - 1;
        const bool prob = net.acts[lo - 1] == Act::Sigmoid || net.acts[lo - 1] == Act::Softmax;
        if (shown[lo].size() <= 48)
            for (size_t k = 0; k < shown[lo].size(); ++k) {
                if (sy[lo][k] < -1e8f) continue;
                const int idx = shown[lo][k];
                const float a = (*s.acts[lo])[idx];
                const float bar = prob ? std::min(1.f, std::max(0.f, a)) : norm(lo, a);
                const float x = sx[lo][k] + R + 8.f * sc, y = sy[lo][k];
                const float bw = 36.f * sc;
                fill_rect(x, y - 1.5f * sc, bw, 3.f * sc, kBarTrack);
                fill_rect(x, y - 1.5f * sc, bw * bar, 3.f * sc, kTextGold);
                const std::string name = s.labels && idx < (int)s.labels->size() ? (*s.labels)[idx] : "OUT" + std::to_string(idx);
                std::snprintf(buf, sizeof buf, " %.2f", a);
                push_text(q, x + bw + 6.f * sc, y - 3.5f * lp, lp, name + buf);
            }
        flush_quads(q, kText);
    }

    // tooltip for the hovered neuron
    if (hl >= 0) {
        const int idx = hover.index;
        const float a = (*s.acts[hl])[idx];
        std::string t = layer_name(hl, L) + " #" + std::to_string(idx);
        if (hl == L - 1 && s.labels && idx < (int)s.labels->size()) t += " " + (*s.labels)[idx];
        std::snprintf(buf, sizeof buf, "  %.4f", a);
        t += buf;
        if (is_muted(hl, idx)) t += "  MUTED";
        const float w = text_width(t, lp) + 16.f * sc, h = 7.f * lp + 12.f * sc;
        float x = s.mouse_x + 16.f * sc, y = s.mouse_y + 16.f * sc;
        if (x + w > (float)net_w) x = s.mouse_x - w - 8.f * sc;
        if (y + h > s.fb_h) y = s.mouse_y - h - 8.f * sc;
        fill_rect(x, y, w, h, kTooltipBg);
        push_text(q, x + 8.f * sc, y + 6.f * sc, lp, t);
        flush_quads(q, kTextBright);
    }

    // HUD, top left
    float y = 16.f * sc;
    for (size_t i = 0; i < s.hud.size(); ++i) {
        push_text(q, 16.f * sc, y, hp, s.hud[i]);
        flush_quads(q, i == 0 ? kTextGold : ((int)i == s.hud_highlight ? kTextBright : kText));
        y += 11.f * hp;
    }
    if (!s.help.empty()) {
        // wrap at the double spaces between key hints
        std::vector<std::string> rows(1);
        const float room = (float)(s.show_net ? net_w : s.fb_w) - 32.f * sc;
        size_t p = 0;
        while (p < s.help.size()) {
            size_t e = s.help.find("  ", p);
            if (e == std::string::npos) e = s.help.size();
            const std::string item = s.help.substr(p, e - p);
            const std::string joined = rows.back().empty() ? item : rows.back() + "  " + item;
            if (!rows.back().empty() && text_width(joined, lp) > room) rows.push_back(item);
            else rows.back() = joined;
            p = e;
            while (p < s.help.size() && s.help[p] == ' ') ++p;
        }
        float hy = (float)s.fb_h - 16.f * sc - 7.f * lp - (float)(rows.size() - 1) * 10.f * lp;
        for (const std::string& r : rows) {
            push_text(q, 16.f * sc, hy, lp, r);
            hy += 10.f * lp;
        }
        flush_quads(q, kText);
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    return hover;
}
