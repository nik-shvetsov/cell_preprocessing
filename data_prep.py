from pathlib import Path
from pannuke_dataset import PanNukeDataModule
from monusac_dataset import MoNuSACDataModule

if __name__ == "__main__":
    process = False
    check = True

    if process:
        MoNuSACDataModule.download_and_process(
            raw_dir=Path("/path/to/data/monusac/raw"),
            proc_dir=Path("/path/to/data/monusac/processed"),
            patch_size=256,
            diff_amb=False,
            remove_small_obj_size=30,
            verbose=False,
        )

        PanNukeDataModule.download_and_process(
            raw_dir=Path("/path/to/data/pannuke/raw"),
            proc_dir=Path("/path/to/data/pannuke/processed"),
            verbose=False,
        )

    if check:
        data_modules = {
            "MoNuSAC": MoNuSACDataModule("/path/to/data/monusac/processed"),
            "PanNuke": PanNukeDataModule("/path/to/data/pannuke/processed"),
        }
        for dm_name, dm in data_modules.items():
            print()
            print(dm_name)
            print(dm.get_dataset(selected_folds=[1, 2, 3]).dataframe)

            for batch in dm.train_dataloader:
                batch_img, batch_mask = batch
                print(batch_img.shape, batch_mask.shape)
                break
            print()
