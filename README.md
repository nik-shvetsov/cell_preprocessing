# Preprocessing of PanNuke and MoNuSAC datasets for individual cell imgages

## data_prep.py
Preprocess and creates datamodules for further use for PanNuke and MoNuSAC datasets. MoNuSAC dataset is preprocessed with additional parameters, such as `patch_size`, `diff_amb` (whether to handle 'ambigious' class) and `remove_small_obj_size`.

## split_cells.ipynb
Notebook extracts individual cell images and transform them to get a unified cell image set. Images are saved for further use. From PanNuke we extract 'Epithelial', 'Neoplastic' and 'Inflammatory' cells. From MoNuSAC we extract 'Epithelial', 'Lymphocyte', 'Macrophage' and 'Neutrophil' cells.

