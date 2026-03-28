# LavaSR ONNX 导出与 C++ 编译经验总结

> 环境：Windows 10 (26200)、Python 3.11、PyTorch 2.11.0+cpu、CMake 4.0.1、Visual Studio 2022 Community

---

## 一、ONNX 模型导出

### 1.1 整体流程

```
HuggingFace 模型权重
        │
   export_bwe.py          export_denoiser.py
        │                         │
  bwe.onnx (+.data)         denoiser.onnx
  bwe_config.json           denoiser_config.json
        │                         │
        └──────────┬──────────────┘
                   │
          C++ 推理引擎读取
```

### 1.2 BWE 模型导出（export_bwe.py）

**导出策略：Mode A → Mode B 降级**

| 模式 | ONNX 图范围 | 状态 |
|---|---|---|
| Mode A "full" | wav_48k → enhanced_48k（含 ISTFT + FastLRMerge） | **失败** |
| Mode B "split" | wav_48k → raw_head_out（仅 Vocos 主干，不含 ISTFT） | **成功** |

**Mode A 失败原因：**  
Vocos `ISTFT` 头内有运行时断言：
```python
# vocos/spectral_ops.py line 72
assert (window_envelope > 1e-11).all()
```
PyTorch 2.5+ 默认使用 `torch.export.export`（dynamo 路径），该断言在符号形状追踪时触发 `GuardOnDataDependentSymNode`，无法生成静态计算图。

**Mode B 成功原因：**  
`BWESplit` 只导出 Vocos 的特征提取器 + 神经网络主干，返回 `raw_head_out` 张量（n_fft+2, T_frames），ISTFT 和 FastLRMerge 由 C++ 负责后处理，绕开了断言问题。

**最终产物：**
```
cpp/models/
├── bwe.onnx           (209 KB，模型结构图)
├── bwe.onnx.data      (53 MB，外部权重文件)
└── bwe_config.json    ({"mode":"split","cutoff_hz":7500,"n_fft":2048,"hop_length":512,...})
```

---

### 1.3 Denoiser 模型导出（export_denoiser.py）

**导出失败原因（初次）：**  
同样是 PyTorch 2.11 默认 dynamo 导出器与 `dynamic_axes` 的兼容性问题：
```
ValueError: Found the following conflicts between user-specified ranges and inferred ranges:
- Received user-specified dim hint Dim.DYNAMIC, but tracing inferred a static shape of 100
  for dimension inputs['spec'].shape[2].
```
dynamo 导出器将时间维 T_frames 静态固化为 dummy 输入的 100，即使显式指定了 `dynamic_axes`。

**修复方案：强制使用 legacy trace-based 导出器**
```python
torch.onnx.export(
    inner, (dummy,), output_path,
    dynamic_axes={"stft_2ch": {2: "T_frames"}, "stft_enh": {2: "T_frames"}},
    opset_version=17,
    dynamo=False,   # 关键：禁用 dynamo，使用旧的 TorchScript trace 路径
)
```

**最终产物：**
```
cpp/models/
├── denoiser.onnx          (1.1 MB)
└── denoiser_config.json   ({"n_fft":512,"hop_length":256,"win_length":512,"sample_rate":16000})
```

---

### 1.4 Windows 编码问题（cp932/GBK）

**现象：**  
PyTorch 2.x 的 ONNX 导出器内部会打印进度 emoji（✅ / ❌），在 Windows 日文/中文区域设置（代码页 932 / GBK）下触发：
```
UnicodeEncodeError: 'cp932' codec can't encode character '\u2705' in position 89
```
该异常在 `_capture_strategies.py` 的成功回调中抛出，导致脚本崩溃——即使 ONNX 文件已写入磁盘，后续的 config 保存和 ORT 验证也会被跳过。

**修复方案：双重保险**

1. 在每个导出脚本顶部重配置标准输出编码：
```python
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
```

2. 运行时设置环境变量：
```powershell
$env:PYTHONUTF8 = "1"
python cpp/export/export_bwe.py ...
```

> **经验：** 在 Windows 多语言环境下开发 Python 工具，凡涉及第三方库的 stdout 输出，都应在脚本入口处设置 UTF-8 编码，尤其是调用 PyTorch/HuggingFace 等会打印 emoji 的库。

---

## 二、C++ 编译

### 2.1 依赖管理（CMake FetchContent）

项目使用 CMake 的 `FetchContent` 自动下载三个依赖：

| 依赖 | 获取方式 | 用途 |
|---|---|---|
| dr_libs | GIT (master) | dr_wav.h — WAV 文件读写 |
| nlohmann/json | GIT (v3.11.3) | JSON 配置文件解析 |
| ONNX Runtime | URL (zip/tgz) | 神经网络推理引擎 |

### 2.2 nlohmann/json 获取方式问题

**初次方案（失败）：**  
直接下载单个 `json.hpp` 文件：
```cmake
FetchContent_Declare(
    nlohmann_json
    URL      https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
    URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406
    DOWNLOAD_NO_EXTRACT TRUE
)
```
问题一：SHA256 哈希值已过时（实际为 `9bea4c8...`），CMake 下载后验证失败，重试 5 次后报错。  
问题二：即使哈希修正，`${nlohmann_json_SOURCE_DIR}` 只是一个目录，文件名是 `json.hpp` 而非 `nlohmann/json.hpp`，源码中的 `#include <nlohmann/json.hpp>` 无法找到。

**最终方案（成功）：Git 拉取完整仓库**
```cmake
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests    OFF CACHE BOOL "" FORCE)
set(JSON_Install       OFF CACHE BOOL "" FORCE)
set(JSON_MultipleHeaders OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)
```
nlohmann/json 的 CMakeLists.txt 会自动暴露 `nlohmann_json::nlohmann_json` 目标，include 路径自动配置为 `single_include/`，`#include <nlohmann/json.hpp>` 可以正确解析。

> **经验：** 对有完整 CMake 支持的开源库，优先用 GIT 方式拉取并通过其自带目标链接；用 URL 下载单文件时需额外处理 include 路径结构，且哈希随内容更新而失效。

---

### 2.3 onnxruntime.cmake 二次 include 问题

**现象：**
```
CMake Error: add_library cannot create imported target "onnxruntime"
because another target with the same name already exists.
```

**原因：**  
`CMakeLists.txt` 中对 `cmake/onnxruntime.cmake` include 了两次：
```cmake
include(cmake/onnxruntime.cmake)   # 第 46 行（正确位置）
...
include(cmake/onnxruntime.cmake)   # 第 102 行（多余，为调用 ort_install_runtime 而误加）
```
第二次 include 时再次执行 `add_library(onnxruntime SHARED IMPORTED GLOBAL)`，导致目标重复。

**修复：** 删除多余的 include，`ort_install_runtime` 函数在第一次 include 时已经定义：
```cmake
# 正确写法：直接调用，无需再次 include
ort_install_runtime(lavasr)
```

> **经验：** CMake 中 `include()` 没有幂等保护（不同于 `find_package`），对于会创建目标的 `.cmake` 文件，要么在文件头部加 `include_guard()`，要么确保调用方只 include 一次。

---

### 2.4 最终编译结果

编译命令：
```powershell
# 配置（约 50 秒，主要是 FetchContent 下载）
cmake -S . -B build/win_x64 -G "Visual Studio 17 2022" -A x64 -Wno-dev

# 编译 Release（约 5 秒）
cmake --build build/win_x64 --config Release --parallel 8
```

输出文件：
```
cpp/build/win_x64/Release/
├── lavasr.exe          193 KB   — 命令行工具
├── lavasr_core.lib    1.4 MB   — 静态库（含所有 DSP + ORT 调用）
└── onnxruntime.dll     11 MB   — ORT 运行时（需随 exe 一同分发）
```

C4819 警告（源文件含无法在 cp932 下表示的字符）属于无害警告，不影响功能，可通过在 `platform_settings.cmake` 中添加 `/utf-8` 编译选项彻底消除：
```cmake
if(MSVC)
    add_compile_options(/utf-8)
endif()
```

---

## 三、端到端验证

```powershell
.\lavasr.exe `
    --bwe      "cpp/models/bwe.onnx" `
    --denoiser "cpp/models/denoiser.onnx" `
    --cutoff   7500 `
    "LavaSR/test16.wav" `
    "test16_cpp_enhanced.wav"
```

```
[LavaSR] Input : test16.wav  (16000 Hz, 2 ch, 807368 frames)
[LavaSR] Config: cutoff=7500 Hz, denoise=off, batch=off
[LavaSR] BWE      : cpp/models/bwe.onnx
[LavaSR] Denoiser : cpp/models/denoiser.onnx
[LavaSR] Output: 2422104 frames @ 48000 Hz, 2 ch
[LavaSR] Saved → test16_cpp_enhanced.wav
耗时：3.9 秒（含 ORT 模型加载）
```

输出帧数 = 807368 × 3 = 2422104，与 Python 版本完全一致，16 kHz → 48 kHz 3× 上采样正确。

---

## 四、问题速查表

| 问题 | 根因 | 解决方案 |
|---|---|---|
| BWE Mode A 导出失败 | Vocos ISTFT 内 `assert .all()` 在 dynamo 符号追踪时失败 | 改用 Mode B（split），ISTFT 由 C++ 实现 |
| Denoiser T_frames 维度静态化 | PyTorch 2.5+ 默认 dynamo 导出器不识别 `dynamic_axes` | 传入 `dynamo=False` 强制使用旧 TorchScript trace 导出器 |
| UnicodeEncodeError cp932 | torch.onnx 回调打印 emoji，Windows cp932 不支持 | `sys.stdout.reconfigure(encoding="utf-8")` + `$env:PYTHONUTF8=1` |
| nlohmann JSON SHA256 不匹配 | GitHub release 文件已更新，旧哈希过期 | 改用 `GIT_REPOSITORY` + `GIT_TAG` 方式获取 |
| `#include <nlohmann/json.hpp>` 找不到 | 只下载了裸 json.hpp，无目录结构 | 用 git 拉取完整仓库，使用官方 CMake 目标 |
| onnxruntime 目标重复 | `onnxruntime.cmake` 被 include 两次 | 删除多余 include，只保留第一处 |
| **C++ 输出高频缺失（截止 ~4kHz）** | `Resampler.cpp` 中 Kaiser-sinc 截止频率公式多除了 2，给出 4kHz 而非正确的 8kHz | 将 `1.0 / (2.0 * max(up,down))` 改为 `1.0 / max(up,down)` |
| **C++ 输出波形削顶（幅值 ≈ 2×）** | `build_kaiser_sinc` 滤波器公式 `2.0 * cutoff * sinc(cutoff*n)` 的 DC gain = 2（应为 1）。理想低通为 `cutoff * sinc(cutoff*n)`，多余的 `2.0×` 因子使重采样输出幅值翻倍，进而导致 BWE 输出超过 ±1 削顶 | 将 `h[i] = sinc(cutoff*n) * kaiser * 2.0 * cutoff` 改为 `h[i] = sinc(cutoff*n) * kaiser * cutoff` |

---

## 五、后续建议

1. **消除 C4819 警告**：在 `cmake/platform_settings.cmake` 中为 MSVC 添加 `/utf-8` 选项。
2. **BWE 重新导出可用 trace-based**：`bwe.onnx` 目前由 dynamo 导出，若需确保动态形状稳定可改用 `dynamo=False`（注意 `torchaudio.transforms.Spectrogram` 的 reshape 在 trace 下会有 TracerWarning，需验证）。
3. **ORT 预置模型路径**：可在 `lavasr.exe` 旁放置 `models/` 目录，实现免参数启动。
4. **Android/iOS 编译**：ONNX Runtime 已通过 `onnxruntime.cmake` 支持 AAR 和 XCFramework，参照 `cmake/toolchains/` 中的工具链文件可直接交叉编译。
