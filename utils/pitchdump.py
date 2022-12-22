import os
from typing import Callable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import speechset
from speechset.datasets import DataReader


class PitchReader(DataReader):
    """Dumped pitch loader.
    """
    def __init__(self, reader: DataReader, pitch_dir: str):
        """Initializer.
        Args:
            reader: data reader.
            pitch_dir: path to the dumped pitch directory.
        """
        self.reader = reader
        self.pitch_dir = pitch_dir
        # cache
        self.preproc_ = reader.preproc()

    def dataset(self) -> List[str]:
        """Return the file reader, forwarding from `self.reader.dataset`.
        Returns:
            file-format datum reader.
        """
        return self.reader.dataset()

    def preproc(self) -> Callable:
        """Return data preprocessor.
        Returns:
            preprocessor.
        """
        return self.preprocessor

    def speakers(self) -> List[str]:
        """List of speakers, forwarding from `self.reader.speakers`.
        Returns:
            list of speakers.
        """
        return self.reader.speakers()

    def preprocessor(self, path: str) -> Tuple[int, str, Tuple[np.ndarray, np.ndarray]]:
        """Load dumped pitch.
        Args:
            path: str, path.
        Returns:
            tuple,
                (optional) sid: int, speaker id.
                text: str, text.
                audio, pitch: [np.float32; [T]], speech signal and pitch sequence.
        """
        # forwarding
        out = self.preproc_(path)
        filename, _ = os.path.splitext(os.path.basename(path))
        # [T]
        pitch = np.load(os.path.join(self.pitch_dir, f'{filename}.npy'))
        if len(out) == 2:
            text, audio = out
            return text, (audio, pitch)
        # if sid supports
        sid, text, audio = out
        return sid, text, (audio, pitch)
    
    @classmethod
    def dump(cls,
             reader: DataReader,
             out_dir: str,
             sr: int,
             numerator: float = 0.8,
             freq_min: float = 75,
             freq_max: float = 600,
             verbose: Optional[Callable] = tqdm):
        """Dump the pitch.
        Args:
            reader: data reader.
            out_dir: path to the output directory.
            sr: sampling rate.
            numerator: the `numerator / freq_min` number of the pitch will be computed in one sec.
            freq_min, freq_max: pitch frequency lowerbound and upperbound.
        """
        import parselmouth
        dataset, preproc = reader.dataset(), reader.preproc()
        if verbose:
            dataset = verbose(dataset)

        os.makedirs(out_dir, exist_ok=True)
        for path in dataset:
            out = preproc(path)
            # unpack
            if len(out) == 2:
                _, audio = out
            else:
                _, _, audio = out
            # compute pitch, using praat
            snd = parselmouth.Sound(audio, sampling_frequency=sr)
            pitch = parselmouth.praat.call(
                snd, 'To Pitch', numerator / freq_min, freq_min, freq_max)
            name, _ = os.path.splitext(os.path.basename(path))
            np.save(
                os.path.join(out_dir, f'{name}.npy'),
                pitch.selected_array['frequency'])

class WavPitchDataset(speechset.WavDataset):
    """Waveform-Pitch dataset.
    """
    def __init__(self, reader: DataReader, pitch_dir: str):
        """Caching dataset and preprocessor from reader.
        Args:
            reader: data reader.
            pitch_dir: path to the dumped pitch directory.
        """
        super().__init__(PitchReader(reader, pitch_dir))

    def normalize(self, _: str, term: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Normalize datum.
        Args:
            _: transcript.
            term: [np.float32; [T]], speech in range [-1, 1] and pitch sequence.
        Returns:
            speech and pitch.
        """
        return term

    def collate(self, bunch: List[Tuple[np.ndarray, np.ndarray]]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collate bunch of datum to the batch data.
        Args:
            bunch: B x [np.float32; [T]], speech signal and pitch sequences.
        Returns:
            batch data.
                speeches: [np.float32; [B, T]], speech signal.
                lengths: [np.long; [B]], speech lengths.
                pitches: [np.float32; [B, T]], pitch sequences.
                pitlen: [np.long; [B]], pitch lengths.
        """
        speeches, lengths = super().collate([speech for speech, _ in bunch])
        # reuse
        pitches, pitlen = super().collate([pitch for _, pitch in bunch])
        return speeches, lengths, pitches, pitlen


if __name__ == '__main__':
    def main():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--out-dir', required=True)
        parser.add_argument('--sr', default=22050, type=int)
        args = parser.parse_args()

        # hard code the reader
        reader = speechset.utils.DumpReader('./datasets/libri_test_clean')
        PitchReader.dump(reader, args.out_dir, args.sr)
        
    main()
