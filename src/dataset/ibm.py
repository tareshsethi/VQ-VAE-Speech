import os
from torch.utils.data import Dataset
import errno
import shutil
import random

import pathlib
AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_manifest(dir):
    audios = []
    dir = os.path.expanduser(dir)
    # print (dir)
    # dir = '/home/derekhuang/VQ-VAE-Speech/data/ibm/raw/IBM-Corpus'
    dir = '/home/taresh/VQ-VAE-Speech/data/ibm/raw/IBM-Corpus'

    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = path
                    audios.append(item)
    return audios

def load_txts(dir):
    """Create a dictionary with all the text of the audio transcriptions."""
    utterences = dict()
    txts = []
    dir = os.path.expanduser(dir)
    # dir = '/home/derekhuang/VQ-VAE-Speech/data/ibm/raw/IBM-Corpus'
    dir = '/home/taresh/VQ-VAE-Speech/data/ibm/raw/IBM-Corpus'

    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if fname.endswith(".txt"):
                    with open(os.path.join(root, fname), "r") as f:
                        fname_no_ext = os.path.basename(
                            fname).rsplit(".", 1)[0]
                        utterences[fname_no_ext] = f.readline()
    return utterences

class IBM(Dataset):
    dset_path = 'IBM-Corpus'

    def make_speaker_dic(self, root):
        speakers = [
            str(speaker.name) for speaker in pathlib.Path(root).glob('wav/*/')]
        speakers = sorted([speaker for speaker in speakers])
        speaker_dic = {speaker: i for i, speaker in enumerate(speakers)}
        return speaker_dic

    def __init__(self, root, downsample=False, transform=None, target_transform=None, download=True, dev_mode=False, ratio=0.8):
        super(IBM, self).__init__()

        self.root = os.path.expanduser(root)
        self.raw_folder = '../data/ibm/raw'
        if os.path.isdir('..' + os.sep + self.raw_folder):
            self.raw_folder = '..' + os.sep + self.raw_folder
        self.downsample = downsample
        self.transform = transform
        self.target_transform = target_transform
        self.dev_mode = dev_mode
        self.data = []
        self.labels = []
        self.chunk_size = 1000
        self.num_samples = 0
        self.max_len = 0
        self.cached_pt = 0

        if download:
            self.download()

        dset_abs_path = os.path.join(
            self.root, self.raw_folder, self.dset_path)

        self.audios = make_manifest(dset_abs_path)
        self.utterences = load_txts(dset_abs_path)
        self.speaker_dic = self.make_speaker_dic(dset_abs_path)

        random.shuffle(self.audios)
        split = int(len(self.audios)*ratio)

        self.audios_train = self.audios[:split]
        self.audios_val = self.audios[split:]

    def download(self):
        assert "not implemented"