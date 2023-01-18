import pickle
from typing import Dict, List, Optional, Tuple

import datasets
import librosa
import numpy as np
from tqdm import tqdm

from speechset.datasets.reader import DataReader


class CommonVoice(DataReader):
    """Mozilla common voice.
    """
    SR = 48000
    REPO = 'mozilla-foundation/common_voice_11_0'

    def __init__(self,
                 data_dir: str,
                 sr: Optional[int] = None,
                 repo: Optional[str] = None,
                 langs: Optional[List[str]] = None,
                 load: bool = True):
        """Initializer.
        Args:
            data_dir: dataset directory.
            sr: sampling rate.
        """
        self.sr = sr or CommonVoice.SR
        self.repo = repo or CommonVoice.REPO
        self.langs = langs or datasets.get_dataset_config_names(self.repo)
        self.datasets, self.speakers_, self.transcript = self.load_data(data_dir, load=load)

    def dump(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'sr': self.sr,
                'repo': self.repo,
                'langs': self.langs,
                'speakers': self.speakers_,
                'transcript': self.transcript}, f)

    @classmethod
    def load(cls, data_dir: str, path: str):
        with open(path, 'rb') as f:
            dumped = pickle.load(f)
        # construct without load
        reader = cls(data_dir, dumped['sr'], dumped['repo'], dumped['langs'], load=False)
        # loading
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

    def load_data(self, data_dir: str, load: bool = True) \
            -> Tuple[List[datasets.Dataset], List[str], Dict[str, Tuple[int, str]]]:
        """Load audio.
        Args:
            data_dir: dataset directory.
        Returns:
            list of speakers, file paths and transcripts.
        """
        hfs, speakers, trans = [], set(), {}
        for i, lang in enumerate(tqdm(self.langs, desc='huggingface')):
            hf = datasets.load_dataset(
                self.repo, lang, split='train', cache_dir=data_dir)
            hfs.append(hf)
            if load:
                # transcripting
                for j, datum in enumerate(tqdm(hf, leave=False)):
                    cid = datum['client_id']
                    speakers.add(cid)
                    trans[f'{i}_{j}'] = (cid, datum['sentence'])
        # rearange
        speakers = list(speakers)
        trans = {i: (speakers.index(cid), sent) for i, (cid, sent) in trans.items()}
        return hfs, speakers, trans

    def load_audio(self, path: str, sr: int) -> np.ndarray:
        """Read the audio from cache.
        Args:
            path: index of the hugginface datasets.
            sr: sampling rate.
        Returns:
            audio sequence.
        """
        i, j = [int(u) for u in path.split('_')]
        term = self.datasets[i][j]['audio']
        return librosa.resample(
            term['array'], orig_sr=term['sampling_rate'], target_sr=sr)        
