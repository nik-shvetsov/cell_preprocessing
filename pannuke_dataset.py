import shutil
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from data_utils import compute_hv_map, download_data_url, rm_n_mkdir


class PanNukeDataset(Dataset):
    def __init__(
        self,
        data_dir,
        selected_folds,
        kfold=3,
        kfold_seed=42,
    ):
        self.data_dir = Path(data_dir)
        self.kfold = kfold
        self.kfold_seed = kfold_seed
        self.labels = {
            0: "Background",
            1: "Neoplastic",  # Malignant cancer cells
            2: "Inflammatory",  # All types of immune cells
            3: "Connective",  # Connective/Soft tissue cells
            4: "Dead",  # Dead Cells
            5: "Epithelial",  # Benign cancer cells
        }

        img_paths = [x for x in Path(self.data_dir).rglob("*.png")]
        mask_paths = [x for x in Path(self.data_dir).rglob("*.npy")]
        mask_dict = {x.stem: x for x in mask_paths}
        mask_paths = [mask_dict[x.stem] for x in img_paths]
        assert len(img_paths) > 0, "No images found in the data_dir"
        assert len(mask_paths) > 0, "No masks found in the data_dir"
        assert [Path(x).stem for x in img_paths] == [
            Path(x).stem for x in mask_paths
        ], "Mismatch between image and mask files"

        fnames = [p.stem for p in img_paths]
        dataframe = pl.DataFrame(
            {
                "index": range(len(img_paths)),
                "img_path": [str(x) for x in img_paths],
                "mask_path": [str(x) for x in mask_paths],
            }
        )

        # Split the data into different k-folds
        if kfold == 3:
            dataframe_all_fold = dataframe.with_columns(
                pl.col("img_path")
                .map_elements(
                    lambda s: int(Path(s).name.split("_")[0][1:]), return_dtype=pl.Int32
                )
                .alias("fold")
            )
        else:
            assert kfold >= 1, "kfold must be equal or greater than 1"
            dataframe_all_fold = self.split_fold(dataframe, kfold, seed=kfold_seed)

        if isinstance(selected_folds, int):
            selected_folds = [selected_folds]
        assert all(
            [0 < x <= kfold for x in selected_folds]
        ), "Selected_fold must be in range [1, kfold]"

        # | index | img_path | mask_path | fold |
        self.dataframe = dataframe_all_fold.filter(pl.col("fold").is_in(selected_folds))
        self.img_paths = [x for x in self.dataframe["img_path"]]
        self.mask_paths = [x for x in self.dataframe["mask_path"]]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        fname = Path(self.img_paths[idx]).stem
        img_path = Path(self.img_paths[idx])
        mask_path = Path(self.mask_paths[idx])

        try:
            tissue_type = str(img_path.name).split(sep="_")[-1]
        except:
            tissue_type = "unknown"

        img_np = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).transpose(
            2, 0, 1
        )
        masks = np.load(str(mask_path))

        return img_np, masks

    def get_subset(self, splits):
        return PanNukeDataset(
            data_dir=self.data_dir,
            selected_folds=splits,
            kfold=self.kfold,
            kfold_seed=self.kfold_seed,
        )

    def split_fold(self, dataframe, kfold, seed=42):
        def assign_folds(df):
            shuffled_df = df.sample(fraction=1.0, seed=seed)
            fold_numbers = (pl.arange(0, shuffled_df.height) % kfold) + 1
            return shuffled_df.with_columns(fold_numbers.alias("fold"))

        out_dataframe = assign_folds(dataframe)
        return out_dataframe.sort("index")


class PanNukeDataModule:
    def __init__(
        self,
        data_dir,
        kfold=3,
        batch_size=8,
        kfold_seed=42,
    ):
        np.random.seed(42)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.kfold = kfold
        self.kfold_seed = kfold_seed

    @classmethod
    def download_and_process(self, raw_dir, proc_dir, verbose=False):
        for fold_idx in tqdm([1, 2, 3]):
            proc_img_dir = Path(proc_dir, "images")
            proc_img_dir.mkdir(exist_ok=True, parents=True)
            proc_mask_dir = Path(proc_dir, "masks")
            proc_mask_dir.mkdir(exist_ok=True, parents=True)

            url = f"https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{fold_idx}.zip"
            if verbose:
                print(f"Downloading fold{fold_idx} data")
            tmp_fold_dir = Path(raw_dir, f"tmp_fold_{fold_idx}")
            prep_dir = Path(raw_dir, f"fold_{fold_idx}")
            download_data_url(url, tmp_fold_dir)

            ###
            rm_n_mkdir(prep_dir)
            for directory in ["images", "masks"]:
                extracted_npy_files = Path(
                    raw_dir, f"tmp_fold_{fold_idx}", directory, f"fold{fold_idx}"
                ).glob("*.npy")
                for npy_file in extracted_npy_files:
                    shutil.move(npy_file, prep_dir)
            shutil.rmtree(tmp_fold_dir)

            ###
            imgs_meta = np.load(Path(prep_dir, "images.npy")).astype(np.uint8)
            masks_meta = np.load(Path(prep_dir, "masks.npy")).astype(np.int32)
            types_meta = np.load(Path(prep_dir, "types.npy"))

            # change from (B, H, W, C) to (B, C, H, W)
            masks_meta = np.transpose(masks_meta, (0, 3, 1, 2))

            for i in range(imgs_meta.shape[0]):
                cv2.imwrite(
                    str(
                        Path(proc_img_dir, f"f{fold_idx}_{i}_{str(types_meta[i])}.png")
                    ),
                    cv2.cvtColor(imgs_meta[i], cv2.COLOR_RGB2BGR),
                )
                # Do not include last channel, which is background vs cells
                np.save(
                    str(
                        Path(proc_mask_dir, f"f{fold_idx}_{i}_{str(types_meta[i])}.npy")
                    ),
                    masks_meta[i][:-1, :, :],
                )

        return None

    def get_dataset(self, selected_folds=None):
        if selected_folds is None:
            selected_folds = range(1, self.kfold + 1)
        return PanNukeDataset(
            data_dir=Path(self.data_dir),
            selected_folds=selected_folds,
            kfold=self.kfold,
            kfold_seed=self.kfold_seed,
        )

    @property
    def train_dataloader(self):
        return DataLoader(
            dataset=self.get_dataset(
                selected_folds=[1, 2],
            ),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    @property
    def valid_dataloader(self):
        return DataLoader(
            dataset=self.get_dataset(
                selected_folds=[3],
            ),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    @property
    def test_dataloader(self):
        return DataLoader(
            dataset=self.get_dataset(
                selected_folds=[3],
            ),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
