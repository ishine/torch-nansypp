import os
import pickle
from typing import Dict, List, Optional, Tuple

import json
import librosa
import numpy as np
from tqdm import tqdm

from speechset.datasets.reader import DataReader


class PPIN(DataReader):
    """AI-Hub, Pattern-pronounciation including numbers, default 16khz.
    """
    SR = 16000

    def __init__(self, data_dir: str, sr: Optional[int] = None, load: bool = True):
        """Initializer.
        Args:
            data_dir: dataset directory.
            sr: sampling rate.
        """
        self.sr = sr or PPIN.SR
        if load:
            self.speakers_, self.transcript = self.load_data(data_dir)

    def dump(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'sr': self.sr,
                'speakers': self.speakers_,
                'transcript': self.transcript}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            dumped = pickle.load(f)
        # construct without load
        reader = cls('', load=False)
        # loading
        reader.sr = dumped['sr']
        reader.speakers_ = dumped['speakers']
        reader.transcript = dumped['transcript']
        return reader

    def dataset(self) -> Dict[str, Tuple[int, str]]:
        """Return file reader.
        Returns:
            file-format datum reader.
        """
        return self.transcript

    def speakers(self) -> List[str]:
        """List of speakers.
        Returns:
            list of the speakers.
        """
        return self.speakers_

    def load_data(self, data_dir: str) \
            -> Tuple[List[str], Dict[str, Tuple[int, str]]]:
        """Load audio.
        Args:
            data_dir: dataset directory.
        Returns:
            list of speakers, file paths and transcripts.
        """
        wav_dir = os.path.join(data_dir, '원천데이터')
        meta_dir = os.path.join(data_dir, '라벨링데이터')

        trans, speakers = {}, set()
        for class_ in tqdm(os.listdir(meta_dir), desc='ppin'):
            class_dir = os.path.join(meta_dir, class_)
            if not os.path.isdir(class_dir):
                continue
            # transcript id
            for tid in tqdm(os.listdir(class_dir), leave=False):
                for filename in tqdm(os.listdir(os.path.join(class_dir, tid)), leave=False):
                    if not filename.endswith('.json'):
                        continue
                    try:
                        # read meta file
                        with open(os.path.join(class_dir, tid, filename), encoding='utf-8') as f:
                            meta = json.load(f)
                        # extract
                        script = meta.get('script', {}).get('scriptITN', '')
                        sid = meta.get('recordPerson', {}).get('recorderID', '')
                        # consistency check
                        audio_path = os.path.join(wav_dir, class_, tid, filename.replace('.json', '.pcm'))
                        if not os.path.exists(audio_path):
                            continue
                        # update info
                        speakers.add(sid)
                        trans[audio_path] = (sid, script)
                    except:
                        import warnings
                        warnings.warn(f'json open failed on `{filename}`')
        # post-processing
        speakers = list(speakers)
        trans = {
            name: (speakers.index(sid), script)
            for name, (sid, script) in trans.items()}
        # read audio
        return speakers, trans

    @classmethod
    def pcm_reader(cls, path: str) -> np.ndarray:
        with open(path, 'rb') as f:
            buf = f.read()
        # to array
        pcm = np.frombuffer(buf, dtype=np.int16)
        # 2byte buffer
        return librosa.util.buf_to_float(pcm, 2)

    def load_audio(self, path: str, sr: int) -> np.ndarray:
        """Load the audio from PCM file.
        Args:
            path: path to the PCM files.
            sr: sampling rate.
        """
        wav = PPIN.pcm_reader(path)
        # alias
        orig_sr = PPIN.SR
        return librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
