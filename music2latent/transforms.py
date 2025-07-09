from torchaudio.transforms import Spectrogram, InverseSpectrogram
import torch
class StreamableSTFT(torch.nn.Module):
    
    def __init__(self,
                 nfft=1024,
                 hop_size=256,
                 stream=False,
                 skip_features=None,
                 log1p=False,
                 alpha_rescale=1,
                 beta_rescale=1):
        super().__init__()
        self.nfft = nfft
        self.hop_size = hop_size
        self.stream = stream
        self.log1p = log1p
        self.skip_features = skip_features
        self.alpha_rescale = alpha_rescale
        self.beta_rescale = beta_rescale
        self.nskip = 1
        self.n_fade = 4  # in frames

        self.register_buffer('audio_buffer',
                             torch.zeros((1, 1, nfft - hop_size)))
        
        self.register_buffer("out_buffer", torch.zeros(4, 1, self.hop_size * self.n_fade))
        self.register_buffer("spec_buffer", torch.zeros(1, self.nfft // 2 + 1, self.n_fade, dtype=torch.complex64))

        

        self.transform = Spectrogram(n_fft=nfft,
                                     win_length=nfft,
                                     hop_length=hop_size,
                                     center=not stream,
                                     normalized=False,
                                     power=None)

        self.inverse_transform = ISTFT(n_fft=nfft,
                                       win_length=nfft,
                                       hop_length=hop_size,
                                       padding="same" if stream else "center")


    @property
    def device(self):
        return self.transform.window.device
    
    def normalize_complex(self, x):
        return (self.beta_rescale *
                (x.abs()**self.alpha_rescale).to(torch.complex64) *
                torch.exp(1j * torch.angle(x).to(torch.complex64)))

    def denormalize_complex(self, x):
        x = x / self.beta_rescale
        return (x.abs()**(1.0 / self.alpha_rescale)).to(
            torch.complex64) * torch.exp(
                1j * torch.angle(x).to(torch.complex64))

    @torch.jit.export
    def forward(self, x):
        # X : B x hop_size
        if self.stream == True:
            if self.audio_buffer.shape[0] != x.shape[0]:
                print(
                    "Resizing and resetting buffer - the batch size has changed"
                )
                self.audio_buffer = torch.zeros(
                    (x.shape[0], 1, self.nfft - self.hop_size)).to(x)

            x = torch.cat([self.audio_buffer, x], dim=-1)
            self.audio_buffer = x[..., -(self.nfft - self.hop_size):]

        spec = self.transform(x)

        spec = spec if self.stream else spec[..., :-1]
        spec = self.normalize_complex(spec)

        if self.skip_features is not None:
            spec = spec[:, :, self.skip_features:]  #Drop constant componnet

        return torch.cat((torch.real(spec), torch.imag(spec)), -3)

    def inverse(self, spec):
        n = spec.shape[0]
        real, imag = torch.chunk(spec, 2, -3)

        spec = torch.complex(real.squeeze(-3), imag.squeeze(-3))
        spec = self.denormalize_complex(spec)

        spec = spec.unsqueeze(1)

        if self.skip_features is not None:
            spec = torch.cat(
                (torch.zeros_like(spec)[:, :, :self.skip_features], spec), -2)

        spec = spec.squeeze(1)

        if self.stream == False:
            spec = torch.cat((spec, torch.zeros_like(spec)[:, :, :1]), -1)
        else:
            spec = torch.cat((self.spec_buffer[:n], spec), -1)
            self.spec_buffer[:n] = spec[:n, :, -self.n_fade:]
        
        x = self.inverse_transform(spec.squeeze(1)).unsqueeze(1)
        
        if self.stream:
            n = x.shape[0]  # batch size
            fade_len = self.hop_size * self.n_fade

            # Create fade window
            alpha = torch.linspace(0, 1, fade_len, device=x.device)[None, None, :]

            # Apply crossfade
            x[..., :fade_len] = (1 - alpha) * self.out_buffer[:n].to(x) + alpha * x[..., :fade_len]

            # Update output buffer
            self.out_buffer[:n] = x[..., -fade_len:].detach().clone()

            # Trim fade from output
            x = x[..., :-fade_len]
        return x
    
    

class ISTFT(torch.nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self,
                 n_fft: int,
                 hop_length: int,
                 win_length: int,
                 padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec,
                               self.n_fft,
                               self.hop_length,
                               self.win_length,
                               self.window,
                               center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y

