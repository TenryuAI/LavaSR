# ---------------------------------------------------------------------------
# Toolchain: cross-compile for Linux ARM64 (aarch64-linux-gnu)
#
# Usage:
#   cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux_aarch64.cmake
# ---------------------------------------------------------------------------

set(CMAKE_SYSTEM_NAME    Linux)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross compiler (install: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu)
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_STRIP        aarch64-linux-gnu-strip)

# Sysroot (optional; set if your cross-sysroot is not in the default search path)
# set(CMAKE_SYSROOT /path/to/aarch64-sysroot)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
