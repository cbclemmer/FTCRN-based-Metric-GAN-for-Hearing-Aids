import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram

class AudioDataset(Dataset):
    def __init__(self, list_noisy_files, list_clean_files):
        self.noisy_files = list_noisy_files
        self.clean_files = list_clean_files

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_wav, _ = torchaudio.load(self.noisy_files[idx], num_frames=44100)  # 1 second clips for example
        clean_wav, _ = torchaudio.load(self.clean_files[idx], num_frames=44100)

        # Convert to spectrogram or any other representation
        spec_transform = Spectrogram()
        noisy_spec = spec_transform(noisy_wav)
        clean_spec = spec_transform(clean_wav)

        return noisy_spec, clean_spec

# noisy_files = ["path_to_noisy1.wav", "path_to_noisy2.wav", ...]
# clean_files = ["path_to_clean1.wav", "path_to_clean2.wav", ...]

# dataset = AudioDataset(noisy_files, clean_files)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)