"""CloudCast-like DataModule (folder-of-sequences)."""
import lightning as L
from torch.utils.data import DataLoader
from .sequence_folder import SequenceFolderDataset

class CloudCastDM(L.LightningDataModule):
    def __init__(self, root, T=6, H=256, W=256, batch_size=2, num_workers=4,
                 weather_keys=("temp","rh","wind","press"), frame_glob="*.*",
                 per_band_mean=None, per_band_std=None):
        super().__init__()
        self.root=root; self.T=T; self.H=H; self.W=W
        self.bs=batch_size; self.nw=num_workers
        self.weather_keys=list(weather_keys)
        self.frame_glob=frame_glob
        self.mean=per_band_mean; self.std=per_band_std

    def setup(self, stage=None):
        common = dict(T=self.T, H=self.H, W=self.W, weather_keys=self.weather_keys,
                      frame_glob=self.frame_glob, per_band_mean=self.mean, per_band_std=self.std)
        self.tr = SequenceFolderDataset(self.root, "train", **common)
        self.va = SequenceFolderDataset(self.root, "val", **common)

    def train_dataloader(self): return DataLoader(self.tr, batch_size=self.bs, shuffle=True,  num_workers=self.nw, pin_memory=True)
    def val_dataloader(self):   return DataLoader(self.va, batch_size=self.bs, shuffle=False, num_workers=self.nw, pin_memory=True)
