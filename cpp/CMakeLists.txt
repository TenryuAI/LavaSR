cmake_minimum_required(VERSION 3.22)
project(LavaSR VERSION 1.0.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Platform & compiler settings
# ---------------------------------------------------------------------------
include(cmake/platform_settings.cmake)
include(FetchContent)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(BUILD_CLI        "Build the command-line lavasr tool"  ON)
option(BUILD_SHARED_LIB "Build lavasr_core as a shared lib"  OFF)

# ---------------------------------------------------------------------------
# Third-party: dr_wav (single-header WAV I/O, MIT)
# ---------------------------------------------------------------------------
FetchContent_Declare(
    dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(dr_libs)

# ---------------------------------------------------------------------------
# Third-party: nlohmann/json (header-only JSON, MIT)
# ---------------------------------------------------------------------------
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
# Disable nlohmann/json's own test/install targets to speed up configuration
set(JSON_BuildTests    OFF CACHE BOOL "" FORCE)
set(JSON_Install       OFF CACHE BOOL "" FORCE)
set(JSON_MultipleHeaders OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# nlohmann/json exposes nlohmann_json::nlohmann_json; alias for our usage
if(NOT TARGET nlohmann_json_iface)
    add_library(nlohmann_json_iface INTERFACE)
    target_link_libraries(nlohmann_json_iface INTERFACE nlohmann_json::nlohmann_json)
endif()

# ---------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------
include(cmake/onnxruntime.cmake)

# ---------------------------------------------------------------------------
# Core library: lavasr_core
# ---------------------------------------------------------------------------
set(LAVASR_SOURCES
    src/LavaSR.cpp
    src/AudioIO.cpp
    src/Resampler.cpp
    src/Stft.cpp
)

if(BUILD_SHARED_LIB)
    add_library(lavasr_core SHARED ${LAVASR_SOURCES})
else()
    add_library(lavasr_core STATIC ${LAVASR_SOURCES})
endif()

target_include_directories(lavasr_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${dr_libs_SOURCE_DIR}>      # dr_wav.h
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/third_party
)

target_link_libraries(lavasr_core
    PUBLIC
        onnxruntime
        nlohmann_json_iface
)

# Platform-specific link libraries
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(lavasr_core PUBLIC log)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(lavasr_core PUBLIC m pthread)
elseif(WIN32)
    target_link_libraries(lavasr_core PUBLIC winmm)
endif()

# Export C symbols from shared library on all platforms
if(BUILD_SHARED_LIB)
    target_compile_definitions(lavasr_core PRIVATE LAVASR_BUILDING_DLL)
endif()

# ---------------------------------------------------------------------------
# CLI executable
# ---------------------------------------------------------------------------
if(BUILD_CLI AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS"
              AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    add_executable(lavasr app/main.cpp)
    target_link_libraries(lavasr PRIVATE lavasr_core)

    # Copy ORT runtime library next to the binary (function defined in cmake/onnxruntime.cmake)
    ort_install_runtime(lavasr)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS lavasr_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/lavasr DESTINATION include)
