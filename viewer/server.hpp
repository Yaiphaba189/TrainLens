#pragma once

#include <functional>
#include <string>
#include <vector>

// Non-blocking line sources: a TCP listener on 127.0.0.1 (any number of
// clients) and, optionally, stdin. Polled once per frame from the render loop.
class LineServer {
public:
    LineServer() = default;
    LineServer(const LineServer&) = delete;
    LineServer& operator=(const LineServer&) = delete;
    ~LineServer();

    bool listen(int port, std::string& err);
    void enable_stdin();

    // Calls fn for every complete line received since the last poll.
    void poll(const std::function<void(const std::string&)>& fn);

    bool listening() const { return listen_fd_ >= 0; }
    int port() const { return port_; }
    int clients() const { return (int)conns_.size(); }
    bool reading_stdin() const { return stdin_ && !stdin_eof_; }

private:
    struct Conn {
        int fd;
        std::string buf;
    };
    int listen_fd_ = -1;
    int port_ = 0;
    std::vector<Conn> conns_;
    bool stdin_ = false;
    bool stdin_eof_ = false;
    std::string stdin_buf_;
};
