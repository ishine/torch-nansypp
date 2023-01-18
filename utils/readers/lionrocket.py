import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from speechset.datasets.reader import DataReader


class Lionrocket(DataReader):
    """Lionrocket dataset loader.
    Use other opensource settings, 16bit, sr: 16khz.
    """
    SR = 44100

    def __init__(self, data_dir: str, sr: Optional[int] = None):
        """Initializer.
        Args:
            data_dir: dataset directory.
            sr: sampling rate.
        """
        self.sr = sr or Lionrocket.SR
        self.speaker, self.transcript = self.load_data(data_dir)

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
        return [self.speaker]

    def load_data(self, data_dir: str) -> Tuple[str, Dict[str, Tuple[int, str]]]:
        """Load audio.
        Args:
            data_dir: dataset directory.
        Returns:
            loaded data, speaker list, file paths and transcripts.
        """
        if os.path.exists(os.path.join(data_dir, 'alignment.json')):
            # read filename-text pair
            with open(os.path.join(data_dir, 'alignment.json'), encoding='utf-8') as f:
                table = {}
                for name, (trans, _) in json.load(f).items():
                    name, _ = os.path.splitext(os.path.basename(name))
                    table[os.path.join(data_dir, 'audio', f'{name}.wav')] = (0, trans)
        else:
            # placeholder
            table = {
                os.path.join(data_dir, 'audio', filename): (0, '')
                for filename in os.listdir(os.path.join(data_dir, 'audio'))
                if filename.endswith('.wav')}
        # read audio
        return os.path.basename(data_dir), table
