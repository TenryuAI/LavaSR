import torch
import torchaudio
from huggingface_hub import snapshot_download

from LavaSR.enhancer.enhancer import LavaBWE
from LavaSR.denoiser.denoiser import LavaDenoiser
from LavaSR.utils import wav_to_1s_batches, load_wav
from LavaSR.enhancer.linkwitz_merge import FastLRMerge



class LavaEnhance:
    def __init__(self, model_path="YatharthS/LavaSR", device='cpu'):

        if model_path == "YatharthS/LavaSR":
            model_path = snapshot_download(model_path)

        self.device = device
        self.bwe_model = LavaBWE(f"{model_path}/enhancer", device=device) ## proposed work
        self.denoiser_model = LavaDenoiser(f'{model_path}/denoiser/denoiser.bin', device=device) ## based on UL-UNAS
        

    def _enhance_mono(self, wav, enhance=True, denoise=True, batch=False):
        """Enhance one channel; wav shape (1, T) at 16 kHz. Returns 1D tensor."""
        pad_size = 0
        if batch:
            wav, pad_size = wav_to_1s_batches(wav, 16000)

        if denoise:
            with torch.inference_mode():
                wav = self.denoiser_model.infer(wav)
                wav = torchaudio.functional.resample(wav, 16000, 48000)
        else:
            wav = torchaudio.functional.resample(wav, 16000, 48000)

        if enhance:
            with torch.no_grad():
                wav = self.bwe_model.infer(wav).reshape(-1)
        else:
            wav = wav.reshape(-1)

        return wav

    def enhance(self, wav, enhance=True, denoise=True, batch=False):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        num_ch = wav.shape[0]
        if num_ch > 2:
            raise ValueError("Only mono and stereo are supported.")

        outs = []
        for c in range(num_ch):
            outs.append(self._enhance_mono(wav[c : c + 1], enhance, denoise, batch))

        if num_ch == 1:
            return outs[0]
        return torch.stack(outs, dim=0)

    def load_audio(self, file_path, input_sr=16000, duration=10000, cutoff=None):
        x = load_wav(file_path, resample_to=input_sr, duration=duration).to(self.device)
        
        if cutoff is None:
            # Slightly below Nyquist (e.g. 7.5 kHz @ 16 kHz) so FastLRMerge hands off
            # earlier and reduces a spectral notch at the original band edge.
            cutoff = max(input_sr // 2 - 500, 500)
          
        self.bwe_model.lr_refiner = FastLRMerge(device=self.device, cutoff=cutoff, transition_bins=1024)
      
        return x, input_sr

class LavaEnhance2(LavaEnhance):
    def __init__(self, model_path="YatharthS/LavaSR", device='cpu'):

        if model_path == "YatharthS/LavaSR":
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_path)

        self.device = device
        self.bwe_model = LavaBWE(f"{model_path}/enhancer_v2", device=device) 
        self.denoiser_model = LavaDenoiser(f'{model_path}/denoiser/denoiser.bin', device=device)
