# ---------------------------------------------------------------------------
# Toolchain: Android arm64-v8a via NDK
#
# Prerequisites:
#   export ANDROID_NDK=/path/to/android-ndk-r26c
#
# Usage:
#   cmake -B build-android \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/android.cmake \
#         [-DANDROID_ABI=arm64-v8a]         (default)
#         [-DANDROID_PLATFORM=android-24]   (default)
# ---------------------------------------------------------------------------

if(NOT DEFINED ENV{ANDROID_NDK} AND NOT DEFINED ANDROID_NDK)
    message(FATAL_ERROR
        "ANDROID_NDK is not set.\n"
        "Set the environment variable ANDROID_NDK or pass -DANDROID_NDK=<path>.")
endif()

if(NOT DEFINED ANDROID_NDK)
    set(ANDROID_NDK "$ENV{ANDROID_NDK}")
endif()

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_ANDROID_NDK "${ANDROID_NDK}")

if(NOT DEFINED ANDROID_ABI)
    set(ANDROID_ABI "arm64-v8a")
endif()
if(NOT DEFINED ANDROID_PLATFORM)
    set(ANDROID_PLATFORM "android-24")
endif()
if(NOT DEFINED ANDROID_STL)
    set(ANDROID_STL "c++_shared")
endif()

set(CMAKE_ANDROID_ARCH_ABI   "${ANDROID_ABI}")
set(CMAKE_ANDROID_STL_TYPE   "${ANDROID_STL}")

# Include the NDK toolchain file
include("${ANDROID_NDK}/build/cmake/android.toolchain.cmake")
