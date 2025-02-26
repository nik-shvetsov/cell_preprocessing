import shutil
from pathlib import Path
import sys
import cv2
from PIL import Image
import numpy as np
from scipy.ndimage import label
import openslide
import polars as pl

import torch
from torch.utils.data import Dataset, DataLoader
from patchify import patchify

from data_utils import (
    compute_hv_map,
    pad_array_to_patch_size,
    process_xml_annotations,
    rm_n_mkdir,
    download_data_url,
)
from utils import remove_small_objs
from skimage.morphology import remove_small_objects as sk_remove_small_objects

from lovely_numpy import lo
from tqdm import tqdm

import matplotlib.pyplot as plt


class MoNuSACDataset(Dataset):
    def __init__(self, data_dir, selected_folds, kfold=3, kfold_seed=42):
        self.data_dir = Path(data_dir)
        self.kfold = kfold
        self.kfold_seed = kfold_seed
        self.labels = {
            0: "Background",
            1: "Epithelial",  # All types of epithelial cells
            2: "Lymphocyte",  # Lymphocyte type of immune cells
            3: "Macrophage",  # Macrophages type of immune cells
            4: "Neutrophil",  # Neutrophils type of immune cells
            5: "Ambiguous",  # Only for the test set
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

        # img_np = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        img_np = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).transpose(
            2, 0, 1
        )
        masks = np.load(str(mask_path))

        return img_np, masks

    def get_subset(self, splits):
        return MoNuSACDataset(
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


class MoNuSACDataModule:
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
    def download_and_process(
        self,
        raw_dir,
        proc_dir,
        patch_size=256,
        diff_amb=False,
        remove_small_obj_size=30,
        verbose=False,
    ):
        modes = {
            "train": {
                "url": "https://drive.google.com/uc?id=1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq",
                "types_mapping": {
                    "Background": 0,
                    "Epithelial": 1,
                    "Lymphocyte": 2,
                    "Macrophage": 3,
                    "Neutrophil": 4,
                },
            },
            "test": {
                "url": "https://drive.google.com/uc?id=1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ",
                "types_mapping": {
                    "Background": 0,
                    "Epithelial": 1,
                    "Lymphocyte": 2,
                    "Macrophage": 3,
                    "Neutrophil": 4,
                    "Ambiguous": 5,
                },
            },
        }

        for mode in modes.keys():
            if verbose:
                print(f"Downloading {mode} data")
            download_data_url(modes[mode]["url"], Path(raw_dir, f"{mode}"))

            if verbose:
                print(f"Processing {mode} data")
            self._process_monusac(
                mode,
                raw_dir,
                proc_dir,
                modes[mode]["types_mapping"],
                patch_size=patch_size,
                diff_amb=diff_amb,
                remove_small_obj_size=remove_small_obj_size,
                verbose=verbose,
            )

    @staticmethod
    def _process_monusac(
        mode,
        raw_folder,
        out_folder,
        types_mapping,
        patch_size=256,
        diff_amb=False,
        remove_small_obj_size=None,
        verbose=False,
    ):
        types_mapping_no_bg = {
            k: v for k, v in types_mapping.items() if k != "Background"
        }

        patients_full_path = Path(raw_folder, mode).glob(
            "*"
        )  # [str(x) for x in Path(raw_folder).glob('*')]

        if diff_amb:
            Path(out_folder, "non-ambiguous", mode, "images").mkdir(
                parents=True, exist_ok=True
            )
            Path(out_folder, "non-ambiguous", mode, "masks").mkdir(
                parents=True, exist_ok=True
            )

            Path(out_folder, "ambiguous", "images").mkdir(parents=True, exist_ok=True)
            Path(out_folder, "ambiguous", "masks").mkdir(parents=True, exist_ok=True)

        else:
            # 'train_test'
            Path(out_folder, "images").mkdir(parents=True, exist_ok=True)
            Path(out_folder, "masks").mkdir(parents=True, exist_ok=True)

        for patient_path in tqdm(list(patients_full_path)):
            patient_name = Path(patient_path).stem
            patient_images_paths_svs = Path(patient_path).glob(
                "*.svs"
            )  # [str(x) for x in Path(patient_path).glob('*.svs')]
            for sub_image_path in patient_images_paths_svs:
                sub_image_name = Path(sub_image_path).stem
                img = openslide.OpenSlide(sub_image_path)
                np_img = np.array(
                    img.read_region((0, 0), 0, img.level_dimensions[0]).convert("RGB")
                )

                ### Process masks by available types from .xmls
                xml_file_path = Path(
                    Path(sub_image_path).parent, Path(sub_image_path).stem + ".xml"
                )
                inst_type_dict = process_xml_annotations(xml_file_path, img)

                for k, v in types_mapping_no_bg.items():
                    if k not in inst_type_dict.keys():
                        inst_type_dict[k] = np.zeros(np.shape(np_img)[0:2])

                ### Generate instance map and type map
                inst_map = np.zeros(np.shape(np_img)[0:2])
                type_map = np.zeros(np.shape(np_img)[0:2])

                for k, v in types_mapping_no_bg.items():
                    uniques = np.unique(inst_type_dict[k])[1:]  # exclude 0
                    for val in uniques:
                        inst_map[inst_type_dict[k] == val] = val
                        type_map[inst_type_dict[k] == val] = v

                np_dict = {
                    "inst_map": inst_map.astype(np.int16),
                    "type_map": type_map.astype(np.int8),
                }

                np_masks = np.zeros(
                    (len(types_mapping_no_bg), *np_dict["type_map"].shape)
                )
                for k, inst_labels in types_mapping_no_bg.items():
                    inst_mask = (np_dict["type_map"] == inst_labels).astype(int)
                    inst_mask = inst_mask * np_dict["inst_map"]
                    np_masks[inst_labels - 1] = (
                        inst_mask  # -1 as enumeration (types_mapping_no_bg) starts from 1 and mask indexes from 0
                    )

                # C x H x W -> H x W x C
                np_masks = np_masks.transpose(1, 2, 0)  # (H x W x 4)

                if (
                    np_img.shape[0] % patch_size != 0
                    or np_img.shape[1] % patch_size != 0
                ):
                    np_img = pad_array_to_patch_size(np_img, patch_size)
                    np_masks = pad_array_to_patch_size(np_masks, patch_size)

                assert (
                    np_img.shape[0] % patch_size == 0
                    and np_img.shape[1] % patch_size == 0
                ), f"Image shape: {np_img.shape}, patch_size: {patch_size}"

                if np_img.shape[0] > patch_size or np_img.shape[1] > patch_size:
                    # Patchify the image and the label, using padding and mirroring
                    np_img_patches = patchify(
                        np_img, (patch_size, patch_size, 3), step=patch_size
                    ).reshape(-1, patch_size, patch_size, 3)
                    np_masks_patches = patchify(
                        np_masks,
                        (patch_size, patch_size, len(types_mapping_no_bg)),
                        step=patch_size,
                    ).reshape(-1, patch_size, patch_size, len(types_mapping_no_bg))

                    assert (
                        np_img_patches.shape[0] == np_masks_patches.shape[0]
                    ), f"Image patches: {np_img_patches.shape}, Mask patches: {np_masks_patches.shape}, Image shape: {np_img.shape}, Mask shape: {np_masks.shape}"

                    for patch_idx in range(np_img_patches.shape[0]):
                        relabeled_maps = []
                        for map_idx in range(np_masks_patches.shape[-1]):
                            # np_masks_patches.shape[-1] == 4 (cell types) (class maps)

                            ### Relabeling. Not needed here, since it is done afterwards
                            # for i, val in enumerate(np.unique(np_masks_patches[patch_idx, :, :, map_idx])):
                            #     np_masks_patches[patch_idx, :, :, map_idx][np_masks_patches[patch_idx, :, :, map_idx] == val] = i

                            if remove_small_obj_size is not None:
                                orig_map = remove_small_objs(
                                    np_masks_patches[patch_idx, :, :, map_idx].astype(
                                        np.int32
                                    ),
                                    remove_small_obj_size,
                                )
                            else:
                                orig_map = np_masks_patches[
                                    patch_idx, :, :, map_idx
                                ].astype(np.int32)
                            # orig_map = np_masks_patches[patch_idx, :, :, map_idx]

                            ### Fix errors in label() results (when connected cells are labeled as one)
                            ### TODO: better to use watershed with precomputed gradients to get better results
                            relabeled_map = label(orig_map)[0]

                            prev_max = max(np.unique(relabeled_map))
                            for i, val in enumerate(np.unique(relabeled_map)[1:]):

                                cellregion_relabeled_map = np.zeros_like(relabeled_map)
                                cellregion_relabeled_map[relabeled_map == val] = val
                                cellregion_orig_map = np.ma.masked_array(
                                    orig_map, mask=(cellregion_relabeled_map == 0)
                                ).filled(0)

                                if len(np.unique(cellregion_orig_map)) > 2:
                                    for i, val in enumerate(
                                        np.unique(cellregion_orig_map)[1:]
                                    ):
                                        relabeled_map[cellregion_orig_map == val] = (
                                            prev_max + 1
                                        )
                                        prev_max += 1

                            ### Relabel the map, os labels are continuous
                            # print ("B", np.unique(relabeled_map)[1:])
                            for i, val in enumerate(np.unique(relabeled_map)[1:]):
                                relabeled_map[relabeled_map == val] = i + 1
                            # print ("A", np.unique(relabeled_map)[1:])

                            relabeled_maps.append(relabeled_map)

                        np_masks_patches[patch_idx, :, :, :] = np.array(
                            relabeled_maps
                        ).transpose(1, 2, 0)

                else:
                    np_img_patches = np.expand_dims(np_img, axis=0)
                    np_masks_patches = np.expand_dims(np_masks, axis=0)

                # # plot in a row of 5
                # test_idx = 0
                # fig, axs = plt.subplots(1, 5, figsize=(20, 20))
                # axs[0].imshow(np_img_patches[test_idx])
                # for j in range(4):
                #     axs[j+1].imshow(np_masks_patches[test_idx, :, :, j])
                # plt.show()
                # plt.imshow(np_masks_patches[test_idx,:,:,0])
                # plt.show()
                # exit()

                for i in range(np_img_patches.shape[0]):
                    sub_image_name_pn = f"{mode}_{sub_image_name}_{i}"
                    np_masks_patch = np_masks_patches[i, :, :, :].transpose(2, 0, 1)

                    if all(
                        len(np.unique(np_masks_patches[i, :, :, j])) == 1
                        for j in range(np_masks_patches[i, :, :, :].shape[-1])
                    ):
                        if verbose:
                            print(f"Warn: {sub_image_name_pn} is empty")
                        continue

                    if diff_amb:
                        if mode == "train":
                            out_dir = Path(out_folder, "non-ambiguous", mode)
                            np_patch_save = np_masks_patch

                        if mode == "test":
                            if (
                                np_masks_patch.shape[0] == 5
                                and len(np.unique(np_masks_patch[-1])) > 1
                            ):
                                out_dir = Path(out_folder, "ambiguous")
                                np_patch_save = np_masks_patch
                            elif (
                                np_masks_patch.shape[0] == 5
                                and len(np.unique(np_masks_patch[-1])) == 1
                            ):
                                out_dir = Path(out_folder, "non-ambiguous", mode)
                                np_patch_save = np_masks_patch[:-1]
                            else:
                                raise ValueError(
                                    f"! Error: {sub_image_name_pn} has {np_masks_patch.shape[0]} channels"
                                )

                    if not diff_amb:
                        # 'train_test'
                        out_dir = Path(out_folder)
                        np_patch_save = np_masks_patch
                        if mode == "test":
                            np_patch_save = np_masks_patch[:-1]

                    cv2.imwrite(
                        str(Path(out_dir, "images", f"{sub_image_name_pn}.png")),
                        np_img_patches[i, :, :, ::-1],
                    )
                    np.save(
                        Path(out_dir, "masks", f"{sub_image_name_pn}.npy"),
                        np_patch_save,
                    )

        # if mode == 'train' and self.train_val_size is not None:
        #     rm_n_mkdir(Path(out_folder, 'valid', 'images'))
        #     rm_n_mkdir(Path(out_folder, 'valid', 'masks'))
        #     imgs = list(Path(out_folder, mode, 'images').glob('*.png'))
        #     np.random.shuffle(imgs)
        #     split = int(len(imgs) * self.train_val_size)
        #     val_imgs = imgs[split:]
        #     for img in val_imgs:
        #         shutil.move(img, Path(out_folder, 'valid', 'images', img.name))
        #         shutil.move(Path(out_folder, mode, 'masks', f"{img.stem}.npy"), Path(out_folder, 'valid', 'masks', f"{img.stem}.npy"))

    def get_dataset(self, selected_folds=None):
        if selected_folds is None:
            selected_folds = range(1, self.kfold + 1)
        return MoNuSACDataset(
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
