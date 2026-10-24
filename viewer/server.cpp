#include "server.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstring>

namespace {

const size_t kMaxLine = 64u << 20;     // a client sending a longer line is dropped
const size_t kReadBudget = 16u << 20;  // per client per poll, so one client cannot stall a frame

bool set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

// Hands complete lines to fn and keeps the unfinished tail in buf.
void drain_lines(std::string& buf, const std::function<void(const std::string&)>& fn) {
    size_t start = 0, nl;
    while ((nl = buf.find('\n', start)) != std::string::npos) {
        size_t end = nl;
        if (end > start && buf[end - 1] == '\r') --end;
        fn(buf.substr(start, end - start));
        start = nl + 1;
    }
    buf.erase(0, start);
}

// Reads what is available without blocking. Returns false once the source
// is closed or broken.
bool read_available(int fd, std::string& buf, bool socket) {
    char tmp[65536];
    size_t got = 0;
    while (got < kReadBudget) {
        ssize_t n = socket ? recv(fd, tmp, sizeof tmp, 0) : read(fd, tmp, sizeof tmp);
        if (n > 0) {
            buf.append(tmp, (size_t)n);
            got += (size_t)n;
            continue;
        }
        if (n == 0) return false;
        if (errno == EINTR) continue;
        return errno == EAGAIN || errno == EWOULDBLOCK;
    }
    return true;
}

}  // namespace

LineServer::~LineServer() {
    for (Conn& c : conns_) close(c.fd);
    if (listen_fd_ >= 0) close(listen_fd_);
}

bool LineServer::listen(int port, std::string& err) {
    std::signal(SIGPIPE, SIG_IGN);
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        err = std::strerror(errno);
        return false;
    }
    int one = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // local only
    if (bind(listen_fd_, (sockaddr*)&addr, sizeof addr) < 0 || ::listen(listen_fd_, 8) < 0 ||
        !set_nonblocking(listen_fd_)) {
        err = "port " + std::to_string(port) + ": " + std::strerror(errno);
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    port_ = port;
    return true;
}

void LineServer::enable_stdin() {
    set_nonblocking(STDIN_FILENO);
    stdin_ = true;
}

void LineServer::poll(const std::function<void(const std::string&)>& fn) {
    if (listen_fd_ >= 0) {
        for (;;) {
            int fd = accept(listen_fd_, nullptr, nullptr);
            if (fd < 0) break;
            set_nonblocking(fd);
            conns_.push_back({fd, {}});
        }
    }
    for (size_t i = 0; i < conns_.size();) {
        bool alive = read_available(conns_[i].fd, conns_[i].buf, true);
        drain_lines(conns_[i].buf, fn);
        if (conns_[i].buf.size() > kMaxLine) alive = false;
        if (alive) {
            ++i;
        } else {
            close(conns_[i].fd);
            conns_.erase(conns_.begin() + (long)i);
        }
    }
    if (stdin_ && !stdin_eof_) {
        if (!read_available(STDIN_FILENO, stdin_buf_, false)) {
            stdin_eof_ = true;
            stdin_buf_ += '\n';  // the last line may lack a newline
        }
        drain_lines(stdin_buf_, fn);
    }
}
