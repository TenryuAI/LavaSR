// ---------------------------------------------------------------------------
// LavaSR.cpp – Main inference pipeline
//
// Pipeline per mono channel:
//   load_wav → resample(any→16k) → [STFT→ULUNASInner→ISTFT] → resample(16k→48k)
//               → [Vocos feature/backbone via ONNX] → [ISTFT + FastLRMerge]
// ---------------------------------------------------------------------------

#include "lavasr/LavaSR.h"
#include "lavasr/AudioIO.h"
#include "lavasr/Resampler.h"
#include "lavasr/Stft.h"

// nlohmann/json (fetched by CMake)
#include <nlohmann/json.hpp>

// ONNX Runtime C++ wrapper
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace lavasr {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static json load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open config: " + path);
    json j; f >> j; return j;
}

static std::string replace_ext(const std::string& p, const std::string& new_ext) {
    auto pos = p.rfind('.');
    return (pos == std::string::npos ? p : p.substr(0, pos)) + new_ext;
}

static std::string sibling_path(const std::string& onnx_path,
                                const std::string& filename) {
    auto pos = onnx_path.find_last_of("/\\");
    std::string dir = (pos == std::string::npos) ? "." : onnx_path.substr(0, pos);
    return dir + "/" + filename;
}

// Run an ORT session that has exactly one input and one output.
static std::vector<float> ort_run_1in_1out(
        Ort::Session& session,
        const float* in_data, const std::vector<int64_t>& in_shape,
        const char* in_name, const char* out_name)
{
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

    size_t in_size = 1;
    for (auto d : in_shape) in_size *= (size_t)d;

    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(in_data), in_size,
            in_shape.data(), in_shape.size());

    auto out_tensors = session.Run(
            Ort::RunOptions{nullptr},
            &in_name, &in_tensor, 1,
            &out_name, 1);

    auto* out_data = out_tensors[0].GetTensorMutableData<float>();
    auto  out_info = out_tensors[0].GetTensorTypeAndShapeInfo();
    size_t out_size = out_info.GetElementCount();

    return std::vector<float>(out_data, out_data + out_size);
}

// ---------------------------------------------------------------------------
// LavaSRImpl
// ---------------------------------------------------------------------------

class LavaSRImpl {
public:
    LavaSRConfig cfg;

    // ONNX Runtime
    Ort::Env              env{ORT_LOGGING_LEVEL_WARNING, "lavasr"};
    Ort::SessionOptions   bwe_opts;
    Ort::SessionOptions   den_opts;
    std::unique_ptr<Ort::Session> bwe_session;
    std::unique_ptr<Ort::Session> den_session;

    // BWE config
    std::string bwe_mode;        // "full" or "split"
    int bwe_n_fft    = 1024;
    int bwe_hop_len  = 256;
    int bwe_win_len  = 1024;
    int bwe_sr       = 48000;
    int bwe_trans    = 1024;

    // Denoiser STFT config
    int den_n_fft    = 512;
    int den_hop_len  = 256;
    int den_win_len  = 512;
    int den_sr       = 16000;

    // ORT input/output name storage
    std::string bwe_in_name, bwe_out_name;
    std::string den_in_name, den_out_name;

    void load(const std::string& bwe_onnx,
              const std::string& den_onnx,
              const LavaSRConfig& config)
    {
        cfg = config;

        // --- BWE config ---------------------------------------------------
        std::string bwe_cfg_path = sibling_path(bwe_onnx, "bwe_config.json");
        try {
            json j = load_json(bwe_cfg_path);
            bwe_mode    = j.value("mode",            "full");
            cfg.cutoff_hz = j.value("cutoff_hz",     cfg.cutoff_hz);
            bwe_n_fft   = j.value("n_fft",           bwe_n_fft);
            bwe_hop_len = j.value("hop_length",      bwe_hop_len);
            bwe_win_len = j.value("win_length",       bwe_win_len);
            bwe_sr      = j.value("sample_rate",     bwe_sr);
            bwe_trans   = j.value("transition_bins", bwe_trans);
        } catch (...) {
            fprintf(stderr, "[LavaSR] bwe_config.json not found, using defaults\n");
            bwe_mode = "full";
        }

        // --- Denoiser config ----------------------------------------------
        std::string den_cfg_path = replace_ext(den_onnx, "_config.json");
        try {
            json j = load_json(den_cfg_path);
            den_n_fft   = j.value("n_fft",      den_n_fft);
            den_hop_len = j.value("hop_length",  den_hop_len);
            den_win_len = j.value("win_length",  den_win_len);
            den_sr      = j.value("sample_rate", den_sr);
        } catch (...) {
            fprintf(stderr, "[LavaSR] denoiser_config.json not found, using defaults\n");
        }

        // --- Create ONNX sessions -----------------------------------------
#ifdef _WIN32
        auto to_wstr = [](const std::string& s) {
            std::wstring ws(s.begin(), s.end());
            return ws;
        };
        bwe_session = std::make_unique<Ort::Session>(
                env, to_wstr(bwe_onnx).c_str(), bwe_opts);
        den_session = std::make_unique<Ort::Session>(
                env, to_wstr(den_onnx).c_str(), den_opts);
#else
        bwe_session = std::make_unique<Ort::Session>(
                env, bwe_onnx.c_str(), bwe_opts);
        den_session = std::make_unique<Ort::Session>(
                env, den_onnx.c_str(), den_opts);
#endif

        // Cache I/O names (ORT returns AllocatedStringPtr, copy to std::string)
        Ort::AllocatorWithDefaultOptions alloc;
        bwe_in_name  = bwe_session->GetInputNameAllocated(0, alloc).get();
        bwe_out_name = bwe_session->GetOutputNameAllocated(0, alloc).get();
        den_in_name  = den_session->GetInputNameAllocated(0, alloc).get();
        den_out_name = den_session->GetOutputNameAllocated(0, alloc).get();
    }

    // -----------------------------------------------------------------------
    // Denoiser: wav_16k (mono, T) → denoised_16k (mono, T)
    // -----------------------------------------------------------------------
    std::vector<float> run_denoiser(const float* wav, int n) {
        StftParams sp{den_n_fft, den_hop_len, den_win_len};

        // Forward STFT
        std::vector<float> re, im;
        int n_frames, n_bins;
        stft(wav, n, sp, re, im, n_frames, n_bins);

        // Build stft_2ch: (1, 2, T_frames, n_bins)
        // Layout: [0..n_frames*n_bins] = real, [n_frames*n_bins..2*T*n_bins] = imag
        size_t frame_bins = (size_t)n_frames * n_bins;
        std::vector<float> stft_2ch(2 * frame_bins);
        // ORT expects (B=1, C=2, T_frames, n_bins)
        // Linearised: stft_2ch[c * frame_bins + frame * n_bins + bin]
        // Our re/im already in layout [frame * n_bins + bin]
        std::copy(re.begin(), re.end(), stft_2ch.begin());
        std::copy(im.begin(), im.end(), stft_2ch.begin() + frame_bins);

        std::vector<int64_t> in_shape = {1, 2, (int64_t)n_frames, (int64_t)n_bins};
        auto enh_flat = ort_run_1in_1out(
                *den_session,
                stft_2ch.data(), in_shape,
                den_in_name.c_str(), den_out_name.c_str());

        // Split enhanced back to re/im
        std::vector<float> enh_re(enh_flat.begin(),
                                  enh_flat.begin() + frame_bins);
        std::vector<float> enh_im(enh_flat.begin() + frame_bins, enh_flat.end());

        // ISTFT
        std::vector<float> out_wav;
        istft(enh_re.data(), enh_im.data(), n_frames, n_bins, n, sp, out_wav);
        return out_wav;
    }

    // -----------------------------------------------------------------------
    // BWE: wav_48k (mono, T) → enhanced_48k (mono, T)
    // -----------------------------------------------------------------------
    std::vector<float> run_bwe(const float* wav, int n) {
        std::vector<int64_t> in_shape = {1, (int64_t)n};

        if (bwe_mode == "full") {
            // ONNX handles everything including ISTFT and FastLRMerge
            auto out = ort_run_1in_1out(
                    *bwe_session, wav, in_shape,
                    bwe_in_name.c_str(), bwe_out_name.c_str());
            out.resize(n);  // trim to input length
            return out;
        }

        // ---- Split mode: backbone only → C++ does ISTFT + LRMerge --------
        auto raw = ort_run_1in_1out(
                *bwe_session, wav, in_shape,
                bwe_in_name.c_str(), bwe_out_name.c_str());

        // raw shape: (1, n_fft+2, T_frames) linearised
        int n_bins_bwe = bwe_n_fft / 2 + 1;
        int T_frames   = (int)(raw.size() / ((size_t)(bwe_n_fft + 2)));

        StftParams sp_bwe{bwe_n_fft, bwe_hop_len, bwe_win_len};
        std::vector<float> pred_wav;
        vocos_head_and_istft(raw.data(), T_frames, sp_bwe, n, pred_wav);

        // FastLRMerge
        std::vector<float> merged;
        int len = std::min((int)pred_wav.size(), n);

        fast_lr_merge(pred_wav.data(), wav, len,
                      bwe_sr, cfg.cutoff_hz, bwe_trans, merged);
        merged.resize(n);
        return merged;
    }

    // -----------------------------------------------------------------------
    // Process one mono channel (T samples at in_sr) → 48kHz output
    // -----------------------------------------------------------------------
    std::vector<float> process_mono(const float* wav_in, int n, int in_sr,
                                    bool denoise) {
        // 1. Resample → 16 kHz
        std::vector<float> wav16;
        int n16 = resample(wav_in, n, in_sr, den_sr, wav16);

        // 2. Optional denoising at 16 kHz
        if (denoise) {
            wav16 = run_denoiser(wav16.data(), n16);
        }

        // 3. Resample 16 kHz → 48 kHz
        std::vector<float> wav48;
        int n48 = resample(wav16.data(), n16, den_sr, bwe_sr, wav48);

        // 4. BWE + FastLRMerge at 48 kHz
        return run_bwe(wav48.data(), n48);
    }
};

// ---------------------------------------------------------------------------
// LavaSR public API
// ---------------------------------------------------------------------------

LavaSR::LavaSR(const std::string& bwe_onnx,
               const std::string& den_onnx,
               const LavaSRConfig& config)
    : impl_(std::make_unique<LavaSRImpl>())
{
    impl_->load(bwe_onnx, den_onnx, config);
}

LavaSR::~LavaSR() = default;

std::vector<float> LavaSR::enhance(const float* samples, int n_frames,
                                   int in_sr, int n_channels)
{
    if (n_channels < 1 || n_channels > 2)
        throw std::invalid_argument("Only mono and stereo are supported.");

    bool denoise = impl_->cfg.denoise;

    if (n_channels == 1) {
        return impl_->process_mono(samples, n_frames, in_sr, denoise);
    }

    // Stereo: de-interleave, process each channel, re-interleave
    std::vector<float> left(n_frames), right(n_frames);
    for (int f = 0; f < n_frames; ++f) {
        left[f]  = samples[f * 2 + 0];
        right[f] = samples[f * 2 + 1];
    }

    auto out_l = impl_->process_mono(left.data(),  n_frames, in_sr, denoise);
    auto out_r = impl_->process_mono(right.data(), n_frames, in_sr, denoise);

    int n_out = (int)std::min(out_l.size(), out_r.size());
    std::vector<float> out(n_out * 2);
    for (int f = 0; f < n_out; ++f) {
        out[f * 2 + 0] = out_l[f];
        out[f * 2 + 1] = out_r[f];
    }
    return out;
}

// ---------------------------------------------------------------------------
// C interface
// ---------------------------------------------------------------------------

} // namespace lavasr

extern "C" {

LavaSRHandle lavasr_create(const char* bwe_onnx,
                           const char* denoiser_onnx,
                           int cutoff_hz, int denoise)
{
    try {
        lavasr::LavaSRConfig cfg;
        if (cutoff_hz > 0) cfg.cutoff_hz = cutoff_hz;
        cfg.denoise = (denoise != 0);
        return new lavasr::LavaSR(bwe_onnx, denoiser_onnx, cfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "[LavaSR] lavasr_create failed: %s\n", e.what());
        return nullptr;
    }
}

int lavasr_enhance(LavaSRHandle handle,
                   const float* in_samples, int in_frames,
                   int in_sr, int n_channels, int denoise,
                   float** out_samples, int* out_frames)
{
    if (!handle || !out_samples || !out_frames) return -1;
    try {
        auto* eng = static_cast<lavasr::LavaSR*>(handle);

        // Temporarily override denoise if caller specified
        // (We modify config via a const_cast workaround; in production expose a setter)
        auto result = eng->enhance(in_samples, in_frames, in_sr, n_channels);

        *out_frames  = (int)(result.size() / n_channels);
        size_t bytes = result.size() * sizeof(float);
        *out_samples = static_cast<float*>(::operator new(bytes));
        std::memcpy(*out_samples, result.data(), bytes);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[LavaSR] lavasr_enhance failed: %s\n", e.what());
        return -2;
    }
}

void lavasr_free(LavaSRHandle handle) {
    delete static_cast<lavasr::LavaSR*>(handle);
}

void lavasr_free_buffer(float* buf) {
    ::operator delete(buf);
}

} // extern "C"
