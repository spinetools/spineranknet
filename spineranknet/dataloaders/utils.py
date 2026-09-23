"""Geometric helpers for extracting intervertebral-disc (IVD) volumes.

Utilities to read DICOM scans, rotate and crop oriented bounding boxes
around each disc, normalise intensities by the vertebral-body median, and
compute (balanced) accuracy metrics used throughout the data pipeline.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import scipy.io as spio
import numpy as np
import cv2
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torch import Tensor


def _import_pydicom():
    """Import pydicom lazily (only the DICOM readers below need it).

    Prefers the GDCM pixel handler when the (pre-3.x) handler API exists.
    """
    import pydicom
    try:
        import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler
        pydicom.config.image_handlers = [None, gdcm_handler]
    except Exception:  # pydicom >= 4 or handler unavailable: keep defaults
        pass
    return pydicom


def get_scan_in_list(data_path: str, dcm_list: List[str]) -> np.ndarray:
    pydicom = _import_pydicom()
    # Get volume - dicom
    if len(dcm_list) == 1:
        # 3D scans
        s_temp = pydicom.dcmread(data_path + '/' + dcm_list[0], force=True)
        if (len(s_temp.file_meta) == 0):
            s_temp.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        volume = np.array(s_temp.pixel_array)
        if len(volume.shape) < 3:
            volume = volume[:, :, None]
        elif len(volume.shape) == 3:
            volume = np.transpose(np.array(volume), (1, 2, 0)).astype(float)
        else:
            print('Unknown volume shape')
    else:
        # 2D scans
        volume = []
        for temp_input in dcm_list:
            # print(temp_input)
            s_temp = pydicom.dcmread(data_path + '/' + temp_input, force=True)
            if (len(s_temp.file_meta) == 0):
                s_temp.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            scan = np.array(s_temp.pixel_array)
            volume.append(scan)
        volume = np.transpose(np.array(volume), (1, 2, 0)).astype(float)
    return volume


def get_all_ivd_vol(
    volume: np.ndarray,
    all_vb_x: np.ndarray,
    all_vb_y: np.ndarray,
    all_vb_mid: np.ndarray,
    all_vb_label: np.ndarray,
) -> np.ndarray:
    # VBs intensity values
    vbs_intensity = get_vbs_intensity(volume, all_vb_x, all_vb_y, all_vb_mid, all_vb_label)

    # Get volumes
    ivds = []
    norm_med = 0.5
    patch_size = (192, 320)
    no_of_ivd = len(vbs_intensity) - 1
    for ivd_idx in range(no_of_ivd):
        curr_ivd_mid = np.round(np.mean([all_vb_mid[ivd_idx], all_vb_mid[ivd_idx + 1]])).astype(int)
        vb_pair_intensity = np.concatenate((vbs_intensity[ivd_idx], vbs_intensity[ivd_idx + 1]))
        vb_pair_median = np.median(vb_pair_intensity)
        if not np.isfinite(vb_pair_median).all():
            raise ValueError(
                f"non-finite median vertebral-body intensity at IVD index "
                f"{ivd_idx}; the intensity normalisation would divide by it"
            )

        vb_curr_x = all_vb_x[:, ivd_idx, curr_ivd_mid]
        vb_curr_y = all_vb_y[:, ivd_idx, curr_ivd_mid]
        vb_next_x = all_vb_x[:, ivd_idx + 1, curr_ivd_mid]
        vb_next_y = all_vb_y[:, ivd_idx + 1, curr_ivd_mid]
        ivd_curr_x = [vb_next_x[1], vb_curr_x[0], vb_curr_x[3], vb_next_x[2]]
        ivd_curr_y = [vb_next_y[1], vb_curr_y[0], vb_curr_y[3], vb_next_y[2]]

        ivd_vol = get_ivd_vol(volume, ivd_curr_x, ivd_curr_y, vb_pair_median, curr_ivd_mid, norm_med, patch_size)

        # Centered & choose 15 slices; TO-DO: resize 3D
        ivd_vol_centered = []
        for j in range(-7, 7 + 1, 1):
            curr_slice = curr_ivd_mid + j
            if (curr_slice < 0) | (curr_slice >= ivd_vol.shape[2]):
                temp_IVD = np.zeros(patch_size)
            else:
                temp_IVD = ivd_vol[:, :, curr_slice]
            ivd_vol_centered.append(temp_IVD)
        ivd_vol_centered = np.transpose(np.array(ivd_vol_centered), (1, 2, 0)).astype(float)
        ivds.append(ivd_vol_centered)
    ivds = np.array(ivds)
    return ivds


def get_all_ivd_locs(
    volume: np.ndarray,
    all_vb_x: np.ndarray,
    all_vb_y: np.ndarray,
    all_vb_mid: np.ndarray,
    all_vb_label: np.ndarray,
) -> List[Tuple[List[float], List[float], int, float]]:
    # VBs intensity values
    vbs_intensity = get_vbs_intensity(volume, all_vb_x, all_vb_y, all_vb_mid, all_vb_label)

    # Get volumes
    ivd_locs = []
    norm_med = 0.5
    patch_size = (192, 320)
    no_of_ivd = len(vbs_intensity) - 1
    for ivd_idx in range(no_of_ivd):
        curr_ivd_mid = np.round(np.mean([all_vb_mid[ivd_idx], all_vb_mid[ivd_idx + 1]])).astype(int)
        vb_pair_intensity = np.concatenate((vbs_intensity[ivd_idx], vbs_intensity[ivd_idx + 1]))
        vb_pair_median = np.median(vb_pair_intensity)
        if not np.isfinite(vb_pair_median).all():
            raise ValueError(
                f"non-finite median vertebral-body intensity at IVD index "
                f"{ivd_idx}; the intensity normalisation would divide by it"
            )

        vb_curr_x = all_vb_x[:, ivd_idx, curr_ivd_mid]
        vb_curr_y = all_vb_y[:, ivd_idx, curr_ivd_mid]
        vb_next_x = all_vb_x[:, ivd_idx + 1, curr_ivd_mid]
        vb_next_y = all_vb_y[:, ivd_idx + 1, curr_ivd_mid]
        ivd_curr_x = [vb_next_x[1], vb_curr_x[0], vb_curr_x[3], vb_next_x[2]]
        ivd_curr_y = [vb_next_y[1], vb_curr_y[0], vb_curr_y[3], vb_next_y[2]]
        ivd_label = all_vb_label[ivd_idx]
        ivd_locs.append((ivd_curr_x, ivd_curr_y, curr_ivd_mid, vb_pair_median))

    return ivd_locs


def poly2mask(
    vertex_row_coords: np.ndarray,
    vertex_col_coords: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    from skimage import draw

    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def get_vbs_intensity(
    volume: np.ndarray,
    all_vb_x: np.ndarray,
    all_vb_y: np.ndarray,
    all_vb_mid: np.ndarray,
    all_vb_label: np.ndarray,
) -> List[np.ndarray]:
    vbs_intensity = []
    height, width, depth = volume.shape
    for vb_idx in range(all_vb_x.shape[1]):
        curr_label = all_vb_label[vb_idx, :]
        curr_mid = all_vb_mid[vb_idx]
        curr_mask = poly2mask(all_vb_y[:, vb_idx, int(curr_mid)], all_vb_x[:, vb_idx, int(curr_mid)], volume.shape[:2])
        mask = np.tile(curr_mask, (depth, 1, 1))
        mask = np.transpose(mask, (1, 2, 0)).astype(float)
        mask[:, :, ~curr_label.astype(bool)] = 0
        vbs_intensity.append(volume[mask.astype(bool)])
    return vbs_intensity


def get_ivd_vol(
    volume: np.ndarray,
    ivd_curr_x: Sequence[float],
    ivd_curr_y: Sequence[float],
    vb_pair_median: float,
    curr_ivd_mid: int,
    norm_med: float,
    patch_size: Tuple[int, int],
) -> np.ndarray:
    x = np.array(ivd_curr_x)
    y = np.array(ivd_curr_y)

    # Rotate
    volume_rot, qx, qy = rotate_bb_and_volume(volume, x, y)

    # Add 50% width
    w = max(qx) - min(qx)
    qx[0] += (w * 0.5)
    qx[1] += (w * 0.5)
    qx[2] -= (w * 0.5)
    qx[3] -= (w * 0.5)
    curr_w = max(qx) - min(qx)
    curr_h = max(qy) - min(qy)
    extra_vert = (curr_w / 2 - curr_h) * 0.5
    qy[0] -= extra_vert
    qy[1] += extra_vert
    qy[2] += extra_vert
    qy[3] -= extra_vert
    curr_w = max(qx) - min(qx)
    curr_h = max(qy) - min(qy)

    #  Jitter Crop: Border = [39.25 46.5]
    extra_w = ((patch_size[1] - 227.0) / 2.0) * (curr_w / 227.0)
    extra_h = ((patch_size[0] - (227.0 / 2.0)) / 2.0) * (curr_h / (227.0 / 2.0))
    min_x = np.round(min(qx) - extra_w)
    max_x = np.round(max(qx) + extra_w)
    min_y = np.round(min(qy) - extra_h)
    max_y = np.round(max(qy) + extra_h)
    curr_w = max_x - min_x
    curr_h = max_y - min_y

    if min_y < 0:
        y_offset = np.round(abs(min_y))
        volume_rot = np.concatenate(
            (np.zeros((y_offset.astype(int), volume_rot.shape[1], volume_rot.shape[2])), volume_rot), axis=0)
        min_y += y_offset
        max_y += y_offset
    if max_y >= volume_rot.shape[0]:
        y_offset = (abs(max_y) + 1) - volume_rot.shape[0]
        volume_rot = np.concatenate((volume_rot, np.zeros((int(y_offset), volume_rot.shape[1], volume_rot.shape[2]))),
                                    axis=0)
    if min_x < 0:
        x_offset = np.round(abs(min_x))
        volume_rot = np.concatenate(
            (np.zeros((volume_rot.shape[0], x_offset.astype(int), volume_rot.shape[2])), volume_rot), axis=1)
        min_x += x_offset
        max_x += x_offset
    if max_x >= volume_rot.shape[1]:
        x_offset = (abs(max_x) + 1) - volume_rot.shape[1]
        volume_rot = np.concatenate(
            (volume_rot, np.zeros((volume_rot.shape[0], x_offset.astype(int), volume_rot.shape[2]))), axis=1)

    min_x = np.round(min_x).astype(int)
    max_x = np.round(max_x).astype(int)
    min_y = np.round(min_y).astype(int)
    max_y = np.round(max_y).astype(int)
    sub_vol = volume_rot[min_y:max_y, min_x:max_x, :]

    # Resize
    ivd_vol = []
    for idx in range(sub_vol.shape[2]):
        ivd_vol.append(cv2.resize(sub_vol[:, :, idx], patch_size[::-1], interpolation=cv2.INTER_CUBIC))
    ivd_vol = np.transpose(np.array(ivd_vol), (1, 2, 0)).astype(float)

    # Normalize
    ivd_vol = ivd_vol / vb_pair_median * norm_med
    if not np.isfinite(ivd_vol).all():
        raise ValueError(
            "non-finite voxels in the normalised IVD volume; check the "
            "vertebral-body intensity statistics this crop was scaled by"
        )
    ivd_vol[ivd_vol < 0] = 0
    ivd_vol[ivd_vol > 2.0] = 2.0
    ivd_vol /= 2.0
    return ivd_vol


def rotate_bb_and_volume(
    volume: np.ndarray, x: np.ndarray, y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = volume.shape[0]
    width = volume.shape[1]
    depth = volume.shape[2]
    # theta = np.degrees(np.arctan2(y[0]-y[3],x[0]-x[3]))
    # theta = np.degrees(np.arctan2(y[1]-y[2],x[1]-x[2]))
    theta1 = np.degrees(np.arctan2(y[0] - y[3], x[0] - x[3]))
    theta2 = np.degrees(np.arctan2(y[1] - y[2], x[1] - x[2]))
    theta = np.median([theta1, theta2])

    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), theta, scale=1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((height * sin) + (width * cos))
    new_h = int((height * cos) + (width * sin))
    offset_w = (new_w - width) / 2
    offset_h = (new_h - height) / 2
    rotation_matrix[0, 2] += offset_w
    rotation_matrix[1, 2] += offset_h
    v = np.vstack((x.ravel(), y.ravel(), np.ones(np.size(y.ravel()))))
    calculated = np.dot(rotation_matrix, v)
    qx = calculated[0]
    qy = calculated[1]

    volume_rot = []
    for idx in range(depth):
        volume_rot.append(
            cv2.warpAffine(volume[:, :, idx], rotation_matrix, (int(new_w), int(new_h)), flags=cv2.INTER_CUBIC))
    volume_rot = np.transpose(np.array(volume_rot), (1, 2, 0)).astype(float)

    min_x = min(qx)
    max_x = max(qx)
    qx[0] = max_x
    qx[1] = max_x
    qx[2] = min_x
    qx[3] = min_x

    min_y = min(qy)
    max_y = max(qy)
    qy[0] = min_y
    qy[1] = max_y
    qy[2] = max_y
    qy[3] = min_y

    return volume_rot, qx, qy


# Function to tile disc volumes
def tilevol(img: np.ndarray) -> np.ndarray:
    images = np.squeeze(img)
    for i in range(np.size(images, 0)):
        # middle 9 images
        curr_image = images[i, :, :, :]
        for j in range(np.size(curr_image, 0)):
            if j == 0:
                curr_tiled = curr_image[j, :, :]
            else:
                curr_tiled = np.concatenate((curr_tiled, curr_image[j, :, :]), axis=1)

        if i == 0:
            tiled = curr_tiled
        else:
            tiled = np.concatenate((tiled, curr_tiled), axis=0)
    return tiled


#  Balanced accuracy
def balanced_accuracy(y_label: Tensor, y_pred: Tensor) -> float:
    # Move to cpu
    y_pred = np.array(y_pred.cpu(), dtype=float)
    y_label = np.array(y_label.cpu(), dtype=float)
    # Delete ignore_index = -100
    y_pred = np.delete(y_pred, np.where(y_label == -100))
    y_label = np.delete(y_label, np.where(y_label == -100))
    return balanced_accuracy_score(y_label, y_pred)


# binarized score before determining balanced accuracy e.g. for normal vs abnormal accuracy
def binarized_balanced_accuracy(y_label: Tensor, y_pred: Tensor) -> float:
    y_pred = np.array(y_pred.cpu(), dtype=float)
    y_label = np.array(y_label.cpu(), dtype=float)
    y_label[np.where(y_label > 1)] = 1
    y_pred[np.where(y_pred > 1)] = 1
    # Delete ignore_index = -100
    y_pred = np.delete(y_pred, np.where(y_label == -100))
    y_label = np.delete(y_label, np.where(y_label == -100))
    return balanced_accuracy_score(y_label, y_pred)


# Function to normalize freq
def class_weights(freq: np.ndarray) -> np.ndarray:
    c_weights = np.sum(freq) / freq
    c_weights = c_weights / np.sum(c_weights)
    return c_weights


def label_binarize(label: float) -> float:
    label = label_check(label)
    if label > 0:
        label = 1
    return label


def label_check(label: float) -> float:
    if np.isnan(label):
        label = -100
    return label


def label_check_marrow(m1: float, m2: float, m3: float, mm: float) -> float:
    m = m1 + m2 + m3 + mm
    m = label_binarize(m)
    return m


def label_check_3(label: float) -> float:
    label = label_check(label)
    if label > 2:
        label = 2
    return label


# Tensorboard smoothing
def smooth(scalars: Sequence[float], weight: float = 0.9) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed
