#pragma once

#include <string>

// Saves the current OpenGL framebuffer (w x h) as an RGB PNG.
bool save_framebuffer_png(const std::string& path, int w, int h, std::string& err);
