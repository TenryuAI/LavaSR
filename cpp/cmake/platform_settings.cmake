# ---------------------------------------------------------------------------
# platform_settings.cmake
#
# Applies compiler flags, ABI settings, and architecture-specific options
# for all four target platforms.  Included after project() declaration.
# ---------------------------------------------------------------------------

# ---- Silence C++17 deprecation warnings on macOS -------------------------
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "" FORCE)
endif()

# ---- Minimum C++ standard ------------------------------------------------
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# ---- Common optimisation flags -------------------------------------------
if(MSVC)
    add_compile_options(
        /O2
        /W3
        /wd4244  # narrowing conversions (float↔double)
        /wd4267  # size_t narrowing
    )
    # Enable AVX2 on Windows x64 (ORT also benefits)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "ARM")
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)

    # ARM NEON (Linux aarch64, Android arm64-v8a, Apple Silicon)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8")
        add_compile_options(-march=armv8-a+simd)
    elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
        # x86 host: SSE4.2 + AVX2
        add_compile_options(-msse4.2 -mavx2)
    endif()
endif()

# ---- Android-specific ----------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # Default ABI: arm64-v8a (override from CLI: -DANDROID_ABI=x86_64)
    if(NOT DEFINED ANDROID_ABI)
        set(ANDROID_ABI "arm64-v8a")
    endif()
    # STL
    if(NOT DEFINED ANDROID_STL)
        set(ANDROID_STL "c++_shared")
    endif()
    set(ANDROID_PLATFORM "android-24" CACHE STRING "")
    message(STATUS "[Platform] Android ${ANDROID_ABI}, platform ${ANDROID_PLATFORM}")
endif()

# ---- iOS-specific --------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "")
    # XCFramework embedding is handled by the Xcode project; see cmake/onnxruntime.cmake
    set(ENABLE_BITCODE OFF)
    message(STATUS "[Platform] iOS arm64")
endif()

# ---- Position-Independent Code (needed for shared libraries) -------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- rpath on Linux/macOS (find .so next to executable) -----------------
if(UNIX AND NOT APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
elseif(APPLE)
    set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
endif()
