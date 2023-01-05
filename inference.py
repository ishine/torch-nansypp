import argparse
import os

import librosa
import numpy as np
import torch

from nansypp import Nansypp
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default=None)
parser.add_argument('--wav1', default=None)
parser.add_argument('--wav2', default=None)
parser.add_argument('--out-dir', default='./outputs')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
# load checkpoint
ckpt = torch.load(args.ckpt, map_location='cpu')
nansypp = Nansypp.load(ckpt)

device = torch.device('cuda:0')
nansypp.to(device)
nansypp.eval()

# load wav
# pack
SR = nansypp.config.sr
wav1, _ = librosa.load(args.wav1, sr=SR)
wav2, _ = librosa.load(args.wav2, sr=SR)
wavs = [wav1, wav2]
wavlen = np.array([len(w) for w in wavs])
# convert
wavlen = torch.tensor(wavlen, device=device)

wav1 = torch.tensor([wav1], device=device)
wav2 = torch.tensor([wav2], device=device)
with torch.no_grad():
    # [B, mel, T] reconstruction
    synth = nansypp.forward(wav1, wav1, wavlen[0,][None], wavlen[0,][None])
    print(f'[*] reconstruct {synth.shape}')
    HOP = 256
    for i, (w, l) in enumerate(zip(synth, wavlen)):
        sf.write(os.path.join(args.out_dir, f'rctor{i}.wav'), w.cpu().numpy()[:l.item() * HOP], SR, 'PCM_24')
    synth = nansypp.forward(wav2, wav2, wavlen[1,][None], wavlen[1,][None])
    print(f'[*] reconstruct {synth.shape}')

    HOP = 256
    for i, (w, l) in enumerate(zip(synth, wavlen)):
        sf.write(os.path.join(args.out_dir, f'rctor{1}.wav'), w.cpu().numpy()[:l.item() * HOP], SR, 'PCM_24')
        
    # [B, mel, T] reconstruction
    synth = nansypp.forward(wav1, wav2, wavlen[0,][None], wavlen[1,][None])
    print(f'[*] vc {synth.shape}')

    HOP = 256
    for i, (w, l) in enumerate(zip(synth, wavlen)):
        sf.write(os.path.join(args.out_dir, f'vc{i}.wav'), w.cpu().numpy()[:l.item() * HOP], SR, 'PCM_24')
    # [B, mel, T] reconstruction
    synth = nansypp.forward(wav2, wav1, wavlen[1,][None], wavlen[0,][None])
    print(f'[*] vc {synth.shape}')

    HOP = 256
    for i, (w, l) in enumerate(zip(synth, wavlen)):
        sf.write(os.path.join(args.out_dir, f'vc{1}.wav'), w.cpu().numpy()[:l.item() * HOP], SR, 'PCM_24')

