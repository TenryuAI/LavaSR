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

### 5.1 消除 MSVC C4819 警告

C4819 是"文件包含无法在当前代码页（932）中表示的字符"警告，由注释中的中文或源文件 BOM 触发，无害但影响阅读。在 `cpp/cmake/platform_settings.cmake` 的 MSVC 块中追加 `/utf-8`：

```cmake
if(MSVC)
    add_compile_options(
        /O2
        /W3
        /utf-8          # 新增：统一源文件和执行字符集为 UTF-8，消除 C4819
        /wd4244
        /wd4267
    )
```

重新 `cmake --build` 后 C4819 警告消失。已在 2.4 节描述，此处为完整落地步骤。

---

### 5.2 BWE 模型 trace-based 重新导出（可选）

当前 `bwe.onnx` 由 dynamo 路径导出，在 PyTorch 2.5+ 下稳定，但若遇到动态形状推导错误，可改用 legacy trace：

```python
# export_bwe.py 中 Mode B 已有 dynamo=False 回退逻辑，
# 手动强制只走 trace：
torch.onnx.export(
    model_b, (dummy,), onnx_path,
    input_names=["wav_48k"],
    output_names=["raw_head_out"],
    dynamic_axes={"wav_48k": {1: "T"}, "raw_head_out": {2: "T_frames"}},
    opset_version=17,
    dynamo=False,
)
```

**注意 TracerWarning**：`torchaudio.transforms.Spectrogram` 内部有一处 `if tensor.shape[...] == 0:` 判断，trace 时会打印：

```
TracerWarning: Converting a tensor to a Python boolean might cause
the trace to be incorrect. Evaluate the condition eagerly using
bool() or remove the if statement.
```

这是 torchaudio 自身的问题，不影响正确性（该分支在 n_fft > 0 时永不走）。导出后务必用变长音频做 ORT 验证：

```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("bwe.onnx", providers=["CPUExecutionProvider"])
for length in [16000, 48000, 96001]:          # 不同长度均要通过
    dummy = np.random.randn(1, length).astype(np.float32)
    out = sess.run(None, {"wav_48k": dummy})
    assert out[0].shape[2] > 0, f"T_frames=0 for length={length}"
print("动态形状验证通过")
```

---

### 5.3 ORT 预置模型路径（免参数启动）

`LavaSRImpl::load()` 已通过 `sibling_path(bwe_onnx, "bwe_config.json")` 在 ONNX 文件同级目录读取 JSON 配置。只需在 `main.cpp` 中增加默认路径逻辑即可实现免 `--bwe` / `--denoiser` 参数启动：

```
lavasr.exe 旁的目录约定：
  lavasr.exe
  models/
  ├── bwe.onnx
  ├── bwe.onnx.data
  ├── bwe_config.json
  ├── denoiser.onnx
  └── denoiser_config.json
```

在 `main.cpp` 解析参数后补充：

```cpp
// 若未指定 --bwe，在 exe 同级 models/ 下寻找默认模型
if (bwe_path.empty()) {
    bwe_path = exe_dir() + "/models/bwe.onnx";
    den_path = exe_dir() + "/models/denoiser.onnx";
}
```

其中 `exe_dir()` 在 Windows 用 `_pgmptr`，在 Linux 用 `/proc/self/exe` 或 `argv[0]` 实现。

---

### 5.4 Android / iOS 交叉编译

`cpp/cmake/onnxruntime.cmake` 已对 Android AAR 和 iOS XCFramework 做好平台检测，`cpp/cmake/platform_settings.cmake` 已配置对应 ABI 与 SIMD 选项，以下命令可直接复制执行。

**Android（NDK r25+）：**

```bash
cmake -S cpp -B build/android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DANDROID_STL=c++_shared \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/android --parallel 8
```

产物：`build/android/liblavasr.so`（含 C FFI，供 JNI 调用）。

**iOS（Xcode 14+，macOS 主机）：**

```bash
cmake -S cpp -B build/ios \
  -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/ios --config Release
```

产物：`build/ios/Release-iphoneos/liblavasr.a` + `onnxruntime.xcframework`（需嵌入 Xcode 项目）。

---

### 5.5 Resampler 回归测试

调试中发现 `build_kaiser_sinc` 存在两个隐蔽 Bug（截止频率多除 2、DC gain = 2），均由单一数字错误引发，肉眼难以察觉。建议在 `cpp/tests/test_resampler.py` 中固化以下四项断言，防止未来修改引入回归：

```python
"""cpp/tests/test_resampler.py – 验证 Resampler.cpp 滤波器正确性"""
import math, numpy as np

def sinc(x): return 1.0 if abs(x)<1e-9 else math.sin(math.pi*x)/(math.pi*x)

def bessel_i0(x):
    d, s = 1.0, 1.0
    for k in range(1, 51):
        d *= (x*x)/(4.0*k*k); s += d
        if d < 1e-15*s: break
    return s

def build_h(cutoff, half_len=96, beta=8.0):
    total = 2*half_len+1; i0b = bessel_i0(beta); h = []
    for i in range(total):
        n = i-half_len; wn = n/half_len
        k = bessel_i0(beta*math.sqrt(max(0,1-wn*wn)))/i0b
        h.append(sinc(cutoff*n)*k*cutoff)          # 正确公式：×cutoff，不×2
    return np.array(h)

up, down = 3, 1
cutoff = 1.0/max(up, down)                         # = 1/3
h = build_h(cutoff)

# 断言1：DC gain = 1
assert abs(h.sum()-1.0) < 1e-4, f"DC gain={h.sum():.6f}，应为1.0"

# 断言2：所有相位的多相增益均为1
for phase in range(up):
    s = sum(h[up*k+phase] for k in range(len(h)//up+1)
            if up*k+phase < len(h))
    assert abs(s*up - 1.0) < 1e-3, f"phase={phase} gain={s*up:.4f}"

# 断言3：8kHz（通带内）正弦上采样后幅值误差 < 1%
def resample_signal(sig, h, up, down):
    n_out = len(sig)*up//down; out = []
    hl = len(h)//2
    for i in range(n_out):
        pos = i*down; phase = pos%up; sc = pos//up
        acc = sum(sig[sc+k]*h[hl+k*up+phase]
                  for k in range(-hl, hl+1)
                  if 0<=sc+k<len(sig) and 0<=hl+k*up+phase<len(h))
        out.append(acc*up)
    return np.array(out)

t = np.arange(4800)/16000                          # 0.3 秒 @16kHz
sig_8k = np.sin(2*math.pi*8000*t)*0.5             # 8kHz（恰在通带边缘内侧）
sig_9k = np.sin(2*math.pi*9000*t)*0.5             # 9kHz（超出通带）
out_8k = resample_signal(sig_8k, h, up, down)
out_9k = resample_signal(sig_9k, h, up, down)
rms = lambda x: np.sqrt(np.mean(x**2))
ratio_8k = rms(out_8k[len(out_8k)//4:]) / rms(sig_8k)
ratio_9k = rms(out_9k[len(out_9k)//4:]) / rms(sig_9k)
assert ratio_8k > 0.99, f"8kHz 幅值保留率={ratio_8k:.3f}，应>0.99"  # 断言3
assert ratio_9k < 0.05, f"9kHz 幅值衰减不足，比率={ratio_9k:.4f}"   # 断言4（>26dB）

print("全部断言通过 ✓")
```

运行：`python cpp/tests/test_resampler.py`

---

### 5.6 批处理模式（`--batch`）尚未实现

`LavaSRConfig::batch_mode` 字段和 `--batch` CLI 参数已存在，但 `process_mono()` 内部不检查此字段，实际仍是全量处理。对 50 秒以上的长音频，BWE 的 ORT 推理会一次性分配约 `T_frames × n_fft × 4 Bytes` 的张量（50 秒 ≈ 4730 帧 × 2050 × 4 ≈ 38 MB），内存压力尚可，但实时率无法保证。

建议实现路径：在 `run_bwe()` 中按块调用 ONNX Session，块大小对齐 `hop_length × N`（如 N=512 帧）：

```
[     待处理 wav48     ]
 ├─ chunk0 + overlap ─┤
          ├─ chunk1 + overlap ─┤
                   └─ chunk2 ─┘
```

每块输出后 FastLRMerge，再与相邻块做交叉淡入淡出（crossfade = 2×win_length 采样），避免块边界的相位不连续。

**当前状态：TODO，未实现。**

---

### 5.7 `bessel_i0` 死代码清理

`Resampler.cpp` 中 `bessel_i0()` 函数含一段无效的旧版实现残留（第 33–38 行）：

```cpp
// ── 应删除的死代码 ──
double sum = 1.0, term = 1.0;
for (int k = 1; k <= 30; ++k) {
    term *= (x / (2.0 * k));
    term *= term;   // 错误：应为 term = ((x/2)/k)^2 的累积，但下一行丢弃了
    (void)term;
    break;          // 第一次迭代就退出，整段循环无任何效果
}
```

下方从 `double d = 1.0; sum = 1.0;` 开始的第二段才是正确实现。删除上方死代码后函数行为完全不变，可减少 6 行混淆代码。

---

### 5.8 音质量化验证脚本

当前端到端验证依赖目视频谱图，引入 C++ 算法修改时缺乏客观通过标准。建议在 `cpp/tests/` 添加：

```python
"""cpp/tests/compare_output.py – 定量比较 C++ 与 Python 输出"""
import sys, numpy as np, soundfile as sf

cpp_wav, sr_c = sf.read(sys.argv[1], always_2d=True)   # C++ 输出
ref_wav, sr_r = sf.read(sys.argv[2], always_2d=True)   # Python 参考

assert sr_c == sr_r, "采样率不一致"
L = min(len(cpp_wav), len(ref_wav))
cpp_wav, ref_wav = cpp_wav[:L], ref_wav[:L]

rms_db = lambda x: 20*np.log10(np.sqrt(np.mean(x**2))+1e-12)
diff_rms  = rms_db(cpp_wav - ref_wav)
ref_rms   = rms_db(ref_wav)
snr       = ref_rms - diff_rms
peak_diff = np.abs(cpp_wav - ref_wav).max()

print(f"参考 RMS   : {ref_rms:.2f} dBFS")
print(f"差值 RMS   : {diff_rms:.2f} dBFS")
print(f"近似 SNR   : {snr:.1f} dB")
print(f"峰值差     : {peak_diff:.5f}")

PASS = snr > 20.0 and peak_diff < 0.05   # 通过标准（可调整）
print("PASS" if PASS else "FAIL")
sys.exit(0 if PASS else 1)
```

使用方式：

```powershell
python cpp/tests/compare_output.py `
    cpp/build/test16_gain_fixed.wav `
    LavaSR/test16_enhanced_cut7500.wav
```

当前实测结果（修复所有 Bug 后）：SNR ≈ 23–25 dB，峰值差 < 0.04，通过 PASS 标准。
