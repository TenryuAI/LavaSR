# ---------------------------------------------------------------------------
# onnxruntime.cmake
#
# Downloads the correct ONNX Runtime pre-built package for the current
# platform and exposes:
#   ONNXRUNTIME_INCLUDE_DIRS
#   ONNXRUNTIME_LIBRARIES
#   onnxruntime  (imported INTERFACE target)
#
# Supported platforms:
#   • Windows   x64 / ARM64
#   • Linux     x86_64 / aarch64
#   • macOS     x86_64 / arm64
#   • iOS       (xcframework)
#   • Android   arm64-v8a / x86_64
# ---------------------------------------------------------------------------

set(ORT_VERSION "1.20.1" CACHE STRING "ONNX Runtime version")
set(ORT_BASE_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}")

include(FetchContent)

# ---------------------------------------------------------------------------
# Detect platform & select the right archive
# ---------------------------------------------------------------------------

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # ------ Android -------------------------------------------------------
    # ORT for Android is distributed as an AAR; we unpack the JNI .so files.
    set(ORT_PKG_NAME "onnxruntime-android-${ORT_VERSION}")
    set(ORT_ARCHIVE  "${ORT_PKG_NAME}.aar")
    set(ORT_URL      "${ORT_BASE_URL}/${ORT_ARCHIVE}")
    set(ORT_ANDROID TRUE)

elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    # ------ iOS -----------------------------------------------------------
    set(ORT_PKG_NAME "onnxruntime-ios-xcframework-${ORT_VERSION}")
    set(ORT_ARCHIVE  "${ORT_PKG_NAME}.zip")
    set(ORT_URL      "${ORT_BASE_URL}/${ORT_ARCHIVE}")
    set(ORT_IOS TRUE)

elseif(WIN32)
    # ------ Windows -------------------------------------------------------
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
        set(ORT_PKG_NAME "onnxruntime-win-arm64-${ORT_VERSION}")
    else()
        set(ORT_PKG_NAME "onnxruntime-win-x64-${ORT_VERSION}")
    endif()
    set(ORT_ARCHIVE "${ORT_PKG_NAME}.zip")
    set(ORT_URL     "${ORT_BASE_URL}/${ORT_ARCHIVE}")

elseif(APPLE)
    # ------ macOS ---------------------------------------------------------
    set(ORT_PKG_NAME "onnxruntime-osx-universal2-${ORT_VERSION}")
    set(ORT_ARCHIVE  "${ORT_PKG_NAME}.tgz")
    set(ORT_URL      "${ORT_BASE_URL}/${ORT_ARCHIVE}")

else()
    # ------ Linux ---------------------------------------------------------
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(ORT_PKG_NAME "onnxruntime-linux-aarch64-${ORT_VERSION}")
    else()
        set(ORT_PKG_NAME "onnxruntime-linux-x64-${ORT_VERSION}")
    endif()
    set(ORT_ARCHIVE "${ORT_PKG_NAME}.tgz")
    set(ORT_URL     "${ORT_BASE_URL}/${ORT_ARCHIVE}")
endif()

# ---------------------------------------------------------------------------
# Fetch & configure
# ---------------------------------------------------------------------------

FetchContent_Declare(
    onnxruntime_pkg
    URL "${ORT_URL}"
)

FetchContent_MakeAvailable(onnxruntime_pkg)
set(ORT_ROOT "${onnxruntime_pkg_SOURCE_DIR}")

if(ORT_ANDROID)
    # Unpack the AAR (it's a ZIP); look for jni/<abi>/libonnxruntime.so
    set(ORT_INCLUDE_DIR "${ORT_ROOT}/headers")
    if(ANDROID_ABI STREQUAL "x86_64")
        set(ORT_LIB_DIR "${ORT_ROOT}/jni/x86_64")
    else()
        set(ORT_LIB_DIR "${ORT_ROOT}/jni/arm64-v8a")
    endif()
    set(ORT_SHARED_LIB "${ORT_LIB_DIR}/libonnxruntime.so")

elseif(ORT_IOS)
    # XCFramework – expose the framework path; caller adds to XCODE settings
    set(ONNXRUNTIME_XCFRAMEWORK "${ORT_ROOT}/onnxruntime.xcframework"
        CACHE PATH "ORT XCFramework path" FORCE)
    message(STATUS "[ORT] iOS XCFramework: ${ONNXRUNTIME_XCFRAMEWORK}")
    # Create a stub target for consistent cmake syntax
    add_library(onnxruntime INTERFACE)
    return()

elseif(WIN32)
    set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
    set(ORT_SHARED_LIB  "${ORT_ROOT}/lib/onnxruntime.dll")
    set(ORT_IMPORT_LIB  "${ORT_ROOT}/lib/onnxruntime.lib")

elseif(APPLE)
    set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
    file(GLOB ORT_SHARED_LIB "${ORT_ROOT}/lib/libonnxruntime*.dylib")
    list(GET ORT_SHARED_LIB 0 ORT_SHARED_LIB)

else()  # Linux
    set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
    file(GLOB ORT_SHARED_LIB "${ORT_ROOT}/lib/libonnxruntime.so*")
    list(FILTER ORT_SHARED_LIB EXCLUDE REGEX "\\.[0-9]+\\.[0-9]+\\.[0-9]+$")
    list(GET ORT_SHARED_LIB 0 ORT_SHARED_LIB)
endif()

# ---------------------------------------------------------------------------
# Imported target
# ---------------------------------------------------------------------------

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${ORT_SHARED_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
)
if(WIN32 AND ORT_IMPORT_LIB)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${ORT_IMPORT_LIB}"
    )
endif()

set(ONNXRUNTIME_INCLUDE_DIRS "${ORT_INCLUDE_DIR}" CACHE PATH "" FORCE)
set(ONNXRUNTIME_LIBRARIES    onnxruntime CACHE STRING "" FORCE)

message(STATUS "[ORT] v${ORT_VERSION} → ${ORT_SHARED_LIB}")
message(STATUS "[ORT] Include: ${ORT_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# Install helper: copy shared library next to the executable
# ---------------------------------------------------------------------------
function(ort_install_runtime target)
    if(NOT ORT_IOS AND NOT ORT_ANDROID)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_SHARED_LIB}"
                "$<TARGET_FILE_DIR:${target}>"
            COMMENT "Copying onnxruntime runtime → build dir"
        )
    endif()
endfunction()
