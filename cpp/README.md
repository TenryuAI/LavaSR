# LavaSR C++ 推理引擎

基于 **ONNX Runtime** 的跨平台 C++ 实现，支持：
- ARM Linux（aarch64）
- Windows（x64 / ARM64）
- iOS（arm64）
- Android（arm64-v8a / x86\_64）

---

## 目录结构

```
cpp/
├── export/                 # Python ONNX 导出脚本
│   ├── export_bwe.py
│   ├── export_denoiser.py
│   └── README.md
├── include/lavasr/
│   ├── LavaSR.h            # 公共 C++ / C API
│   ├── AudioIO.h
│   ├── Resampler.h
│   └── Stft.h
├── src/
│   ├── LavaSR.cpp
│   ├── AudioIO.cpp
│   ├── Resampler.cpp
│   └── Stft.cpp
├── app/main.cpp            # CLI 工具
├── cmake/
│   ├── onnxruntime.cmake   # ORT 自动下载
│   ├── platform_settings.cmake
│   └── toolchains/
│       ├── linux_aarch64.cmake
│       ├── ios.cmake
│       └── android.cmake
└── CMakeLists.txt
```

---

## Step 1：导出 ONNX 模型

```bash
cd cpp/export
pip install torch torchaudio vocos onnxruntime einops huggingface_hub
pip install -e ../..          # 安装 LavaSR Python 包

# 获取本地模型目录
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('YatharthS/LavaSR'))"

# 导出 BWE（自动 Mode A/B 选择）
python export_bwe.py <model_dir> ./models --cutoff 7500

# 导出去噪器
python export_denoiser.py <model_dir>/denoiser/denoiser.bin ./models/denoiser.onnx
```

产出：`models/bwe.onnx`、`models/bwe_config.json`、`models/denoiser.onnx`、`models/denoiser_config.json`

---

## Step 2：编译

### Windows x64（MSVC）

```bat
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### Windows ARM64（MSVC）

```bat
cmake -B build-arm64 -G "Visual Studio 17 2022" -A ARM64
cmake --build build-arm64 --config Release
```

### Linux x64

```bash
cmake -B build && cmake --build build -j$(nproc)
```

### Linux ARM64（交叉编译）

```bash
# 需要安装：sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
cmake -B build-aarch64 \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux_aarch64.cmake
cmake --build build-aarch64 -j$(nproc)
```

### Android arm64-v8a

```bash
export ANDROID_NDK=/path/to/ndk
cmake -B build-android \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/android.cmake \
      -DBUILD_CLI=OFF
cmake --build build-android -j$(nproc)
```

产出：`liblavasr_core.a`（静态库）供 JNI 调用。

### iOS（Xcode）

```bash
cmake -B build-ios -G Xcode \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/ios.cmake \
      -DBUILD_CLI=OFF
cmake --build build-ios --config Release
```

产出：`liblavasr_core.a`，ORT XCFramework 路径由 CMake 变量 `ONNXRUNTIME_XCFRAMEWORK` 给出，
需在 Xcode 项目的 **Frameworks, Libraries, and Embedded Content** 中手动添加。

---

## Step 3：使用 CLI

```bash
./build/lavasr \
    --bwe       models/bwe.onnx \
    --denoiser  models/denoiser.onnx \
    --cutoff    7500 \
    input.wav   output.wav
```

全部选项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cutoff <hz>` | 7500 | FastLRMerge 分频点（Hz） |
| `--denoise` | off | 开启 ULUNAS 去噪器 |
| `--batch` | off | 按 1.28 s 分块处理超长音频 |
| `--input-sr <hz>` | 从 WAV 头读取 | 覆盖输入采样率 |

输出始终为 **48 kHz**，声道数与输入相同（1 或 2）。

---

## C / C++ API 集成

### C++ 集成

```cpp
#include "lavasr/LavaSR.h"

lavasr::LavaSRConfig cfg;
cfg.cutoff_hz = 7500;
cfg.denoise   = false;

lavasr::LavaSR enhancer("bwe.onnx", "denoiser.onnx", cfg);

// pcm: interleaved float32 @ in_sr Hz
auto out = enhancer.enhance(pcm.data(), n_frames, in_sr, n_channels);
// out: interleaved float32 @ 48000 Hz
```

### C / FFI（Swift、Kotlin JNI、Python ctypes）

```c
LavaSRHandle h = lavasr_create("bwe.onnx", "denoiser.onnx", 7500, 0);
float* out; int out_frames;
lavasr_enhance(h, pcm, n_frames, 16000, 2, 0, &out, &out_frames);
// use out[0..out_frames*2]
lavasr_free_buffer(out);
lavasr_free(h);
```

---

## 依赖

| 依赖 | 获取方式 | 许可证 |
|------|---------|--------|
| ONNX Runtime | CMake FetchContent 自动下载 | MIT |
| dr_wav | CMake FetchContent（mackron/dr_libs） | MIT / Public Domain |
| nlohmann/json | CMake FetchContent | MIT |

所有依赖在 `cmake --build` 时自动获取，无需手动安装。

---

## 技术说明

| 模块 | 说明 |
|------|------|
| `Stft.cpp` | 自包含 radix-2 FFT（n\_fft 须为 2 的幂）；STFT/ISTFT；Vocos head 自定义 forward；FastLRMerge |
| `Resampler.cpp` | Kaiser 窗 sinc 重采样，支持任意整数比 |
| `AudioIO.cpp` | dr\_wav 单头文件 WAV 读写，支持 PCM/Float32 |
| `LavaSR.cpp` | 流水线：重采样 → 去噪（STFT→ONNX→ISTFT）→ 重采样 → BWE（ONNX）→ FastLRMerge |
