import os
from pathlib import Path
from glob import glob

import cv2
import numpy as np
from skimage import draw
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from xml.dom import minidom

import gdown
from zipfile import ZipFile
import shutil
import urllib


def rm_n_mkdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def bounding_box(image):
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def pad_array_to_patch_size(np_arr, patch_size):
    pad_x = (
        0
        if np_arr.shape[0] % patch_size == 0
        else patch_size - (np_arr.shape[0] % patch_size)
    )
    pad_y = (
        0
        if np_arr.shape[1] % patch_size == 0
        else patch_size - (np_arr.shape[1] % patch_size)
    )
    arr_padded = np.pad(np_arr, ((0, pad_x), (0, pad_y), (0, 0)), mode="reflect")
    return arr_padded


def download_from_url(url, download_dir, name=None):
    if name is None:
        name = os.path.basename(url)

    path = os.path.join(download_dir, name)

    if os.path.exists(path):
        return
    else:
        os.makedirs(download_dir, exist_ok=True)

        # Download the file from `url` and save it locally under `file_name`:
        with urllib.request.urlopen(url) as response, open(path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)


def download_data_url(url, rdir, wdir=Path("tmp")):
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    # if os.path.exists(rdir):
    #     shutil.rmtree(rdir)

    zip_path = Path(wdir, "tmpdata.zip")
    ext_path = Path(wdir, "tmpdata")

    gdown.download(url, str(zip_path), quiet=True, fuzzy=True)
    with ZipFile(zip_path, "r") as zipObj:
        zipObj.extractall(path=ext_path)

    for pdir in glob(f"{str(ext_path)}/*"):
        if os.path.isdir(pdir):
            shutil.move(pdir, rdir)
    shutil.rmtree(wdir)


def compute_hv_map(mask):
    """
    Compute center of mass for each nucleus, then compute distance of each nuclear pixel to its corresponding center
    of mass.
    Nuclear pixel distances are normalized to (-1, 1). Background pixels are left as 0.
    Operates on a single mask.

    Based on https://github.com/vqdang/hover_net/src/loader/augs.py#L192

    Args:
        mask (np.ndarray): Mask indicating individual nuclei. Array of shape (H, W),
            where each pixel is in {0, ..., n} with 0 indicating background pixels and {1, ..., n} indicating
            n unique nuclei.

    Returns:
        np.ndarray: array of hv maps of shape (2, H, W). First channel corresponds to horizontal and second vertical.
    """
    assert (
        mask.ndim == 2
    ), f"Input mask has shape {mask.shape}. Expecting a mask with 2 dimensions (H, W)"

    out = np.zeros((2, mask.shape[0], mask.shape[1]))
    # each individual nucleus is indexed with a different number
    inst_list = list(np.unique(mask))

    try:
        inst_list.remove(0)  # 0 is background
    except Exception:
        raise ValueError("No 0 pixels nuclei found in mask")

    for inst_id in inst_list:
        # get the mask for the nucleus
        inst_map = mask == inst_id
        inst_map = inst_map.astype(np.uint8)
        contours, _ = cv2.findContours(
            inst_map, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE
        )

        # get center of mass coords
        mom = cv2.moments(contours[0])
        com_x = mom["m10"] / (mom["m00"] + 1e-6)
        com_y = mom["m01"] / (mom["m00"] + 1e-6)
        inst_com = (int(com_y), int(com_x))

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        # add to output mask
        # this works assuming background is 0, and each pixel is assigned to only one nucleus.
        out[0, :, :] += inst_x
        out[1, :, :] += inst_y
    return out


def compute_summed_class_masks(masks):
    """
    Compute a summed mask and a detection class mask from a set of binary masks.
    The summed mask is the sum of all binary masks.
    The detection class mask assigns a unique class to each pixel in the summed mask, based on the index of the masks map, that has the highest value at that pixel.

    Args:
        masks (np.ndarray): Array of shape (n_masks, H, W) containing n binary masks. It includes only type cell masks without background.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Summed mask of shape (H, W)
            - np.ndarray: Detection class mask of shape (H, W)
    """
    summed_mask = np.sum(masks, axis=0)
    detection_class_mask = np.zeros_like(summed_mask.squeeze())
    for i in range(masks.shape[0]):
        detection_class_mask[masks[i, :, :] > 0] = i + 1
    return summed_mask, detection_class_mask


def process_xml_annotations(xml_file_path, img):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Generate n-ary mask for each cell-type
    count = 0
    # result = []
    result = {}
    for k in range(len(root)):
        label = [x.attrib["Name"] for x in root[k][0]]
        label = label[0]

        for child in root[k]:
            for x in child:
                r = x.tag
                if r == "Attribute":
                    label = x.attrib["Name"]
                    n_ary_mask = np.transpose(
                        np.zeros(
                            (img.read_region((0, 0), 0, img.level_dimensions[0]).size)
                        )
                    )

                if r == "Region":
                    regions = []
                    vertices = x[1]
                    coords = np.zeros((len(vertices), 2))
                    for i, vertex in enumerate(vertices):
                        coords[i][0] = vertex.attrib["X"]
                        coords[i][1] = vertex.attrib["Y"]
                    regions.append(coords)
                    try:
                        # may throw error if len(regions[0]) < 4
                        poly = Polygon(regions[0])
                        vertex_row_coords = regions[0][:, 0]
                        vertex_col_coords = regions[0][:, 1]
                        fill_row_coords, fill_col_coords = draw.polygon(
                            vertex_col_coords, vertex_row_coords, n_ary_mask.shape
                        )
                        # Keep track of giving unique values to each instance in an image
                        count = count + 1
                        n_ary_mask[fill_row_coords, fill_col_coords] = count

                    except Exception as e:
                        print(f"Error in {xml_file_path}, {label}, {e}")
                        print(
                            f"Wrong shape for {xml_file_path}, regions[0].shape: {regions[0].shape}"
                        )

        result[label] = n_ary_mask
    return result
