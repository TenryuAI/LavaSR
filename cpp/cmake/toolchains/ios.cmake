# ---------------------------------------------------------------------------
# Toolchain: iOS arm64 (device only)
#
# Usage:
#   cmake -B build-ios -G Xcode \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/ios.cmake \
#         -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
#
# After building the static library, embed it and the ORT XCFramework
# (path stored in ONNXRUNTIME_XCFRAMEWORK) into your Xcode project.
# ---------------------------------------------------------------------------

set(CMAKE_SYSTEM_NAME iOS)

set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0")
set(CMAKE_OSX_ARCHITECTURES     "arm64")

# Use Xcode toolchain
set(CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH YES)
set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE   NO)

# Required to avoid linker errors with arm64 slice
set(CMAKE_XCODE_ATTRIBUTE_VALID_ARCHS "arm64")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
