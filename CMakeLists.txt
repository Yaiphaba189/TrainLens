cmake_minimum_required(VERSION 3.20)

# Version comes from pyproject.toml when built by pip (scikit-build-core).
if(DEFINED SKBUILD_PROJECT_VERSION)
  set(NEUROLENS_VERSION ${SKBUILD_PROJECT_VERSION})
else()
  set(NEUROLENS_VERSION "0.0.0.dev")
endif()

project(neurolens_viewer VERSION ${NEUROLENS_VERSION} LANGUAGES C CXX)

if(WIN32)
  message(FATAL_ERROR "The NeuroLens viewer supports macOS and Linux for now (it uses POSIX sockets). "
                      "Windows support is tracked in CONTRIBUTING.md.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Portable wheels build SDL2 from source and link it statically, so users need
# nothing installed. Local builds use a system SDL2 when there is one.
option(NEUROLENS_VENDOR_SDL "Build SDL2 from source and link it statically" OFF)

if(NOT NEUROLENS_VENDOR_SDL)
  find_package(SDL2 CONFIG QUIET)
endif()

if(TARGET SDL2::SDL2)
  set(NEUROLENS_SDL SDL2::SDL2)
  message(STATUS "NeuroLens: using system SDL2 ${SDL2_VERSION}")
else()
  message(STATUS "NeuroLens: building SDL2 from source (static)")
  include(FetchContent)
  set(SDL_SHARED OFF CACHE BOOL "" FORCE)
  set(SDL_STATIC ON CACHE BOOL "" FORCE)
  set(SDL_TEST OFF CACHE BOOL "" FORCE)
  set(SDL_STATIC_PIC ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    SDL2
    URL https://github.com/libsdl-org/SDL/releases/download/release-2.30.9/SDL2-2.30.9.tar.gz
    URL_HASH SHA256=24b574f71c87a763f50704bbb630cbe38298d544a1f890f099a4696b1d6beba4
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(SDL2)
  set(NEUROLENS_SDL SDL2::SDL2-static)
endif()

set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
find_package(ZLIB REQUIRED)

add_executable(neurolens-viewer
  viewer/main.cpp
  viewer/network.cpp
  viewer/png.cpp
  viewer/protocol.cpp
  viewer/render.cpp
  viewer/server.cpp
  viewer/training.cpp)
target_compile_definitions(neurolens-viewer PRIVATE NEUROLENS_VERSION="${NEUROLENS_VERSION}")
target_link_libraries(neurolens-viewer PRIVATE ${NEUROLENS_SDL} OpenGL::GL ZLIB::ZLIB)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(neurolens-viewer PRIVATE -Wall -Wextra)
endif()

# pip installs the viewer next to Python's other console scripts, so it lands
# on PATH (and works for editable installs). Plain CMake installs to bin/.
if(DEFINED SKBUILD_SCRIPTS_DIR)
  install(TARGETS neurolens-viewer RUNTIME DESTINATION ${SKBUILD_SCRIPTS_DIR})
else()
  install(TARGETS neurolens-viewer RUNTIME DESTINATION bin)
endif()
