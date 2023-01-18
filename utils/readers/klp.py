import os
import pickle
from typing import Callable, Dict, List, Optional, Tuple

import json
import numpy as np
from tqdm import tqdm

from speechset.datasets.reader import DataReader


class KLP(DataReader):
    """AI-Hub, korean loanword pronounciation, default 16khz.
    """
    SR = 16000

    def __init__(self, data_dir: str, sr: Optional[int] = None, load: bool = True):
        """Initializer.
        Args:
            data_dir: dataset directory.
            sr: sampling rate.
        """
        self.sr = sr or KLP.SR
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
        trans, speakers = {}, set()
        for class_ in tqdm(os.listdir(data_dir), desc='KLP'):
            class_dir = os.path.join(data_dir, class_)
            if not os.path.isdir(class_dir):
                continue

            for filename in tqdm(os.listdir(class_dir), leave=False):
                if not filename.endswith('.json'):
                    continue
                # read meta file
                with open(os.path.join(class_dir, filename)) as f:
                    meta = json.load(f)
                # extract
                script = meta.get('발화정보', {}).get('stt', '')
                sid = meta.get('녹음자정보', {}).get('recorderId', '')
                # consistency check
                audio_path = os.path.join(class_dir, filename.replace('.json', '.wav'))
                if not os.path.exists(audio_path):
                    continue
                # update info
                speakers.add(sid)
                trans[audio_path] = (sid, script)
        # post-processing
        speakers = list(speakers)
        trans = {
            name: (speakers.index(sid), script)
            for name, (sid, script) in trans.items()}
        # read audio
        return list(speakers), trans
