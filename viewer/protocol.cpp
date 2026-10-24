#include "protocol.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>

namespace {

struct Cursor {
    const char* p;

    void skip() {
        while (*p == ' ' || *p == '\t' || *p == '\r') ++p;
    }
    bool done() {
        skip();
        return *p == 0;
    }
    bool word(std::string& out) {
        skip();
        const char* s = p;
        while (*p && !std::isspace((unsigned char)*p)) ++p;
        out.assign(s, p);
        return !out.empty();
    }
    bool integer(int& v) {
        skip();
        char* end;
        long x = std::strtol(p, &end, 10);
        if (end == p || (*end && !std::isspace((unsigned char)*end))) return false;
        p = end;
        v = (int)x;
        return true;
    }
    bool number(float& v) {
        skip();
        if (!*p) return false;
        char* end;
        v = std::strtof(p, &end);
        if (end == p) return false;
        p = end;
        if (!std::isfinite(v)) v = 0.f;
        return true;
    }
    // Keeps nan and inf: a NaN loss is exactly what the viewer must report.
    bool raw_number(double& v) {
        skip();
        if (!*p) return false;
        char* end;
        v = std::strtod(p, &end);
        if (end == p) return false;
        p = end;
        return true;
    }
};

// Reads numbers to the end of the line; false if anything else is left.
bool read_floats(Cursor& c, std::vector<float>& out) {
    out.clear();
    float v;
    while (c.number(v)) out.push_back(v);
    return c.done();
}

std::string num(size_t n) { return std::to_string(n); }

}  // namespace

void set_topology(State& st, const std::vector<int>& sizes) {
    st.net.set_sizes(sizes);
    st.input.assign(sizes[0], 0.f);
    st.input_set = false;
    st.ext.assign(sizes.size(), {});
    for (size_t l = 0; l < sizes.size(); ++l) st.ext[l].assign(sizes[l], 0.f);
    st.ext_time.assign(sizes.size(), -1.0);
    st.labels.clear();
    ++st.version;
}

bool apply_line(State& st, const std::string& line, double now, std::string& err) {
    Cursor c{line.c_str()};
    std::string cmd;
    if (!c.word(cmd) || cmd[0] == '#') return true;

    if (cmd == "status") {
        c.skip();
        st.status = c.p;
        while (!st.status.empty() && std::isspace((unsigned char)st.status.back())) st.status.pop_back();
        return true;
    }
    // ---- training curves and layer health: these work without a network ----
    if (cmd == "metric") {
        std::string name;
        double value, step;
        if (!c.word(name) || !c.raw_number(value)) { err = "'metric' needs a name and a value"; return false; }
        Series& s = st.train.get(name);
        if (!c.raw_number(step)) step = s.step.empty() ? 0.0 : s.step.back() + 1.0;
        if (!c.done()) { err = "'metric' takes a name, a value and an optional step"; return false; }
        s.add(step, value);
        st.train.any = true;
        return true;
    }
    if (cmd == "pstats") {
        int n;
        if (!c.integer(n) || n < 0 || n > 512 || !c.done()) { err = "'pstats' needs a count from 0 to 512"; return false; }
        st.train.params.assign(n, {});
        st.train.any = true;
        return true;
    }
    if (cmd == "pstat") {
        int i;
        std::string name;
        double g, w, u;
        if (!c.integer(i) || !c.word(name) || !c.raw_number(g) || !c.raw_number(w) || !c.raw_number(u) || !c.done()) {
            err = "'pstat' needs: index name grad_norm weight_norm update_ratio";
            return false;
        }
        if (i < 0 || i >= (int)st.train.params.size()) { err = "'pstat' index is outside the last 'pstats' count"; return false; }
        st.train.params[i] = {name, g, w, u};
        return true;
    }
    if (cmd == "clearmetrics") {
        st.train.clear();
        return true;
    }

    if (cmd == "layers") {
        std::vector<int> sizes;
        int v;
        while (c.integer(v)) {
            if (v < 1) { err = "layer sizes must be at least 1"; return false; }
            sizes.push_back(v);
        }
        if (!c.done() || sizes.size() < 2) { err = "'layers' needs two or more whole-number sizes"; return false; }
        if (sizes != st.net.sizes) set_topology(st, sizes);
        return true;
    }

    const int L = st.net.layers();
    if (L < 2) { err = "'" + cmd + "' arrived before 'layers'"; return false; }

    if (cmd == "act") {
        std::vector<Act> acts;
        std::string w;
        while (c.word(w)) {
            Act a;
            if (!parse_act(w, a)) { err = "unknown activation '" + w + "' (use relu, tanh, sigmoid, softmax or linear)"; return false; }
            acts.push_back(a);
        }
        if ((int)acts.size() != L - 1) { err = "'act' needs " + num(L - 1) + " names, got " + num(acts.size()); return false; }
        st.net.acts = acts;
        return true;
    }
    if (cmd == "w" || cmd == "b" || cmd == "acts") {
        int l;
        if (!c.integer(l)) { err = "'" + cmd + "' needs a layer index first"; return false; }
        const int max_l = cmd == "acts" ? L - 1 : L - 2;
        if (l < 0 || l > max_l) { err = "'" + cmd + "' layer " + std::to_string(l) + " is outside 0.." + std::to_string(max_l); return false; }
        std::vector<float> v;
        if (!read_floats(c, v)) { err = "'" + cmd + "' values must all be numbers"; return false; }
        const size_t want = cmd == "w" ? (size_t)st.net.sizes[l + 1] * st.net.sizes[l]
                          : cmd == "b" ? (size_t)st.net.sizes[l + 1]
                                       : (size_t)st.net.sizes[l];
        if (v.size() != want) { err = "'" + cmd + " " + std::to_string(l) + "' needs " + num(want) + " values, got " + num(v.size()); return false; }
        if (cmd == "w") { st.net.W[l] = std::move(v); st.net.has_w[l] = true; }
        else if (cmd == "b") st.net.B[l] = std::move(v);
        else { st.ext[l] = std::move(v); st.ext_time[l] = now; }
        return true;
    }
    if (cmd == "input") {
        std::vector<float> v;
        if (!read_floats(c, v)) { err = "'input' values must all be numbers"; return false; }
        if ((int)v.size() != st.net.sizes[0]) { err = "'input' needs " + num(st.net.sizes[0]) + " values, got " + num(v.size()); return false; }
        st.input = std::move(v);
        st.input_set = true;
        return true;
    }
    if (cmd == "labels") {
        std::vector<std::string> labels;
        std::string w;
        while (c.word(w)) labels.push_back(w);
        st.labels = std::move(labels);
        return true;
    }
    if (cmd == "reset") {
        for (double& t : st.ext_time) t = -1.0;
        st.input_set = false;
        return true;
    }
    err = "unknown command '" + cmd + "'";
    return false;
}

bool load_file(State& st, const std::string& path, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    std::string line, e;
    int n = 0;
    while (std::getline(f, line)) {
        ++n;
        if (!apply_line(st, line, kPinned, e)) {
            err = path + ":" + std::to_string(n) + ": " + e;
            return false;
        }
    }
    if (st.net.layers() < 2) { err = path + ": no 'layers' line"; return false; }
    return true;
}
