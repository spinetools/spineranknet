'''DataLoader - Adapted with automatic margin computation
'''
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
import logging
from spineranknet.dataloaders.utils import label_check, label_binarize, label_check_marrow, label_check_3
from collections import Counter

Sample = Dict[str, Any]
GenoDict = Dict[str, Dict[str, Any]]


class GenodiscDataset(Dataset):
    """Genodisc with automatic margin computation"""

    def __init__(self, genodict: GenoDict, scan_path: str, transform: bool = False,
                 slices: int = 9, height: int = 112, width: int = 224,
                 original_height: int = 192, original_width: int = 320,
                 original_slices: int = 15) -> None:
        """
        Args:
            genodict (dict): Dictionary of disc volumes & labels.
            scan_path (dict): Root path of image dir.
            transform (callable, optional): Optional transform to be applied on a sample.
            slices (int): Target number of slices (S dimension).
            height (int): Target height (H dimension) after cropping.
            width (int): Target width (W dimension) after cropping.
            original_height (int): Original height of IVD volumes (default: 192).
            original_width (int): Original width of IVD volumes (default: 320).
            original_slices (int): Original number of slices (default: 15).
        """
        self.genodict = genodict
        self.scan_path = scan_path
        self.transform = transform
        self.slices = slices
        self.height = height
        self.width = width
        self.original_height = original_height
        self.original_width = original_width
        self.original_slices = original_slices

        # Automatically compute margins
        self.margin_rows = (original_height - height) // 2
        self.margin_cols = (original_width - width) // 2

    def __len__(self) -> int:
        return len(self.genodict)

    def __getitem__(self, idx: int) -> Sample:
        ids = list(self.genodict.keys())
        curr_vb_name = ids[idx]
        return self.get_vb_by_name(curr_vb_name)

    # Maps vb_name suffixes to VertebralLevel class indices (0-5).
    _VB_LEVEL_MAP = {
        "T12_L1": 0, "L1_L2": 1, "L2_L3": 2,
        "L3_L4": 3, "L4_L5": 4, "L5_S1": 5,
    }

    def _extract_labels(self, score: Dict[str, Any]) -> Dict[str, int]:
        """Extract labels from a score dict.

        Override in subclasses to change binarization behaviour
        (see ``GenodiscOrdinalDataset``).
        """
        return {
            'Pfirrmann': label_check(score['Pfirrmann']) - 1,
            'Narrowing': label_check(score['Narrowing']),
            'UpperEndplateDefect': label_binarize(score['UpperEndplateDefect']),
            'LowerEndplateDefect': label_binarize(score['LowerEndplateDefect']),
            'UpperEndplateDefectFourClasses': label_check(score['UpperEndplateDefect']),
            'LowerEndplateDefectFourClasses': label_check(score['LowerEndplateDefect']),
            'UpperMarrow': label_check_marrow(
                score['UpperModic1'], score['UpperModic2'],
                score['UpperModic3'], score['UpperModicM']),
            'LowerMarrow': label_check_marrow(
                score['LowerModic1'], score['LowerModic2'],
                score['LowerModic3'], score['LowerModicM']),
            'Spondylolisthesis': label_check_3(score['Spondylolisthesis']),
            'SpondylolisthesisBinary': label_binarize(score['Spondylolisthesis']),
            'CentralCanalStenosis': label_check(score['CentralCanalStenosis']),
            'ForaminalStenosisLeft': label_binarize(score['ForaminalStenosisLeft']),
            'ForaminalStenosisRight': label_binarize(score['ForaminalStenosisRight']),
            'Herniation': label_binarize(score['Herniation']),
            'AnteriorBulging': label_binarize(score['AnteriorBulging']),
            'PosteriorBulging': label_binarize(score['PosteriorBulging']),
            'FacetJointArthropathyLeft': label_binarize(score['FacetJointArthropathyLeft']),
            'FacetJointArthropathyRight': label_binarize(score['FacetJointArthropathyRight']),
        }

    @classmethod
    def _vertebral_level_from_name(cls, vb_name: str) -> int:
        """Derive VertebralLevel class index (0-5) from the disc volume name.

        E.g. ``"GENODISC_001_L3_L4"`` → 3 (L3-L4).
        Returns -100 if the level cannot be determined.
        """
        for suffix, idx in cls._VB_LEVEL_MAP.items():
            if vb_name.endswith(suffix):
                return idx
        return -100

    @staticmethod
    def _invalidate_all(labels: Dict[str, int]) -> Dict[str, int]:
        """Set every label to -100 (missing)."""
        return {k: -100 for k in labels}

    def get_vb_by_name(self, curr_vb_name: str) -> Sample:
        curr_ = self.genodict[curr_vb_name]
        score = curr_['score']
        labels = self._extract_labels(score)

        if self.transform:
            seq_idx = random.randint(0, len(curr_['sequence'])-1)
            curr_seq = curr_['sequence'][seq_idx]
            image = np.load(self.scan_path + curr_vb_name + '_' + curr_seq + '.npy', allow_pickle=True)
            if curr_seq[:2] == 'T1':
                # Pfirrmann & Narrowing should only be trained using T2w
                labels['Pfirrmann'] = -100
                labels['Narrowing'] = -100
        else:
            curr_seq = 'T2_S1'
            if os.path.isfile(self.scan_path + curr_vb_name + '_' + curr_seq + '.npy'):
                image = np.load(self.scan_path + curr_vb_name + '_' + curr_seq + '.npy', allow_pickle=True)
            else:
                image = np.zeros((self.original_height, self.original_width, self.original_slices), np.float32)
                labels = self._invalidate_all(labels)

        if not np.isfinite(image).all():
            logging.error('Image contains NaN')

        # Get original dimensions
        num_rows, num_cols, num_slices = image.shape

        # Define cropping boundaries based on computed margins
        max_cols = num_cols - self.margin_cols
        min_cols = self.margin_cols
        max_rows = num_rows - self.margin_rows
        min_rows = self.margin_rows

        if self.transform:
            # Slice shift +/- 2
            shift_slice = random.randint(-2, 2)
            image = np.roll(image, shift_slice, axis=2)
            if shift_slice > 0:
               image[:,:,0:shift_slice] = 0
            elif shift_slice < 0:
               image[:,:,num_slices+shift_slice:num_slices] = 0

            # Slice flip
            if random.random() > 0.5:
                old_fsl = labels['ForaminalStenosisLeft']
                old_fjl = labels['FacetJointArthropathyLeft']
                labels['ForaminalStenosisLeft'] = labels['ForaminalStenosisRight']
                labels['ForaminalStenosisRight'] = old_fsl
                labels['FacetJointArthropathyLeft'] = labels['FacetJointArthropathyRight']
                labels['FacetJointArthropathyRight'] = old_fjl
                image = np.flip(image, axis=2).copy()

            # Find empty slices (before selecting subset)
            zero_slice = np.mean(image, axis=(0,1))
            # Select center slices for tracking
            slice_start = max(0, (num_slices - self.slices) // 2)
            slice_end = slice_start + self.slices
            zero_slice = zero_slice[slice_start:slice_end]

            # Intensity +/- 0.1
            image = image + random.uniform(-0.1,0.1)

            # Translation +/- 32x 24y pixels
            shift_cols = random.randint(-32, 22)
            max_cols += shift_cols
            min_cols += shift_cols
            shift_rows = random.randint(-24, 24)
            max_rows += shift_rows
            min_rows += shift_rows

            # Scale +/- 0.1
            col_range = max_cols-min_cols
            row_range = max_rows-min_rows
            bb_scale = np.array(random.uniform(0.9,1.1))
            max_cols = max_cols + col_range*(bb_scale - 1.0)
            min_cols = min_cols - col_range*(bb_scale - 1.0)
            max_rows = max_rows + row_range*(bb_scale - 1.0)
            min_rows = min_rows - row_range*(bb_scale - 1.0)

            # Sanity Check
            if max_cols >= num_cols:
                min_cols = min_cols + (max_cols - num_cols)
                max_cols = num_cols - 1
            if max_rows >= num_rows:
                min_rows = min_rows + (max_rows - num_rows)
                max_rows = num_rows - 1
            if min_cols < 0:
                max_cols = max_cols + abs(min_cols)
                min_cols = 0
            if min_rows < 0:
                max_rows = max_rows + abs(min_rows)
                min_rows = 0
            max_cols = int(np.round(max_cols))
            min_cols = int(np.round(min_cols))
            max_rows = int(np.round(max_rows))
            min_rows = int(np.round(min_rows))

            # Select center slices
            slice_start = max(0, (num_slices - self.slices) // 2)
            slice_end = slice_start + self.slices
            image = image[:, :, slice_start:slice_end]

            # Rotation +/- 15.0
            rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), random.uniform(-15.0,15.0), 1)
            # Rotate in chunks of 3 slices
            for i in range(0, self.slices, 3):
                end_idx = min(i + 3, self.slices)
                image[:,:,i:end_idx] = cv2.warpAffine(
                    image[:,:,i:end_idx],
                    rotation_matrix,
                    (num_cols, num_rows),
                    flags=cv2.INTER_CUBIC
                )

            # Resize to target dimensions (height x width)
            image = cv2.resize(image[min_rows:max_rows, min_cols:max_cols, :],
                             (self.width, self.height),
                             interpolation=cv2.INTER_CUBIC)

            # Zero out empty slices
            image[:,:,zero_slice==0] = 0
        else:
            # No augmentation - crop and select center slices
            slice_start = max(0, (num_slices - self.slices) // 2)
            slice_end = slice_start + self.slices
            image = image[min_rows:max_rows, min_cols:max_cols, slice_start:slice_end]

        # Transpose to (S, H, W) format and add batch dimension
        image = np.transpose(image, (2, 0, 1))[None,:,:,:]

        # Derive VertebralLevel from the volume name (e.g. "..._L3_L4" → 3).
        # If image was missing (all labels invalidated), keep VertebralLevel as -100 too.
        if labels.get('Pfirrmann', 0) == -100 and labels.get('Narrowing', 0) == -100:
            labels['VertebralLevel'] = -100
        else:
            labels['VertebralLevel'] = self._vertebral_level_from_name(curr_vb_name)

        sample = {'image': image, 'labels': labels, 'name': curr_vb_name}
        return sample

    @staticmethod
    def compute_class_weights(
        genodict: GenoDict, label_name: str = 'Pfirrmann', ignore_label: int = -100,
    ) -> Tuple[Dict[int, float], Tensor]:
        """
        Compute class weights for imbalanced datasets.

        Args:
            genodict (dict): Dictionary of disc volumes & labels.
            label_name (str): Name of the label to compute weights for.
            ignore_label (int): Label value to ignore (e.g., -100 for missing data).

        Returns:
            dict: Dictionary mapping class labels to weights.
            torch.Tensor: Tensor of class weights ordered by class index.
        """
        labels = []
        for vb_name in genodict:
            score = genodict[vb_name]['score']

            # Process the label based on its type
            if label_name == 'Pfirrmann':
                label = label_check(score['Pfirrmann']) - 1
            elif label_name in ['Narrowing', 'CentralCanalStenosis',
                               'AnnularTears', 'Herniation',
                               'ForaminalStenosisRight', 'ForaminalStenosisLeft',
                               'FacetJointArthropathyRight', 'FacetJointArthropathyLeft']:
                label = label_check(score[label_name])
            elif label_name in ['UpperEndplateDefect', 'LowerEndplateDefect',
                               'AnteriorBulging', 'PosteriorBulging']:
                label = label_binarize(score[label_name])
            elif label_name in ['UpperMarrow', 'LowerMarrow']:
                if label_name == 'UpperMarrow':
                    label = label_check_marrow(score['UpperModic1'], score['UpperModic2'],
                                              score['UpperModic3'], score['UpperModicM'])
                else:
                    label = label_check_marrow(score['LowerModic1'], score['LowerModic2'],
                                              score['LowerModic3'], score['LowerModicM'])
            elif label_name == 'Spondylolisthesis':
                label = label_check_3(score['Spondylolisthesis'])
            else:
                continue

            if label != ignore_label:
                labels.append(label)

        # Count occurrences
        label_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(label_counts)

        # Compute weights: inversely proportional to frequency
        class_weights = {}
        for label, count in label_counts.items():
            class_weights[label] = total_samples / (num_classes * count)

        # Create ordered tensor
        max_class = max(label_counts.keys())
        weight_tensor = torch.zeros(max_class + 1)
        for label, weight in class_weights.items():
            weight_tensor[label] = weight

        print(f"\nClass weights for {label_name}:")
        print(f"  Label counts: {dict(label_counts)}")
        print(f"  Weights: {class_weights}")

        return class_weights, weight_tensor

    @staticmethod
    def compute_all_class_weights(genodict: GenoDict) -> Dict[str, Dict[str, Any]]:
        """
        Compute class weights for all labels in the dataset.

        Args:
            genodict (dict): Dictionary of disc volumes & labels.

        Returns:
            dict: Dictionary mapping label names to (class_weights_dict, weight_tensor).
        """
        label_names = [
            'Pfirrmann', 'Narrowing', 'UpperEndplateDefect', 'LowerEndplateDefect',
            'UpperMarrow', 'LowerMarrow', 'Spondylolisthesis', 'CentralCanalStenosis',
            'ForaminalStenosisLeft', 'ForaminalStenosisRight', 'Herniation',
            'AnteriorBulging', 'PosteriorBulging',
            'FacetJointArthropathyLeft', 'FacetJointArthropathyRight'
        ]

        all_weights = {}
        for label_name in label_names:
            try:
                class_weights, weight_tensor = GenodiscDataset.compute_class_weights(
                    genodict, label_name=label_name
                )
                all_weights[label_name] = {
                    'class_weights': class_weights,
                    'weight_tensor': weight_tensor
                }
            except Exception as e:
                print(f"Error computing weights for {label_name}: {e}")

        return all_weights


    @staticmethod
    def compute_class_weights_for_tasks(
        genodict: GenoDict,
        task_definitions: Dict[str, Dict[str, Any]],
        tasks: List[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """Compute class weights for tasks.

        Data-driven: uses ``task_definitions`` to resolve the correct score
        key (via ``dataset_key``), number of classes, and binary flag.
        Supports all task variants including ``*Ordinal`` tasks, VertebralLevel,
        and SpondylolisthesisBinary.
        """
        from collections import Counter

        print("Computing class weights...\n")

        # ── Special-case tasks for label extraction ──────────────────────
        _MARROW_FIELDS = {
            'UpperMarrow': ('UpperModic1', 'UpperModic2', 'UpperModic3', 'UpperModicM'),
            'LowerMarrow': ('LowerModic1', 'LowerModic2', 'LowerModic3', 'LowerModicM'),
        }

        def get_label(score: Dict[str, Any], task: str, vb_name: str = None) -> int:
            td = task_definitions[task]
            num_classes = td["num_classes"]
            is_binary = td.get("binary", False)
            score_key = td.get("dataset_key", task)

            # VertebralLevel — derived from vb_name, not score dict
            if task == 'VertebralLevel':
                return GenodiscDataset._vertebral_level_from_name(vb_name)

            # Marrow composites
            if task in _MARROW_FIELDS:
                m1, m2, m3, mm = _MARROW_FIELDS[task]
                return label_check_marrow(score[m1], score[m2], score[m3], score[mm])

            # Pfirrmann — 0-indexed (raw value minus 1)
            if score_key == 'Pfirrmann':
                return label_check(score['Pfirrmann']) - 1

            # Spondylolisthesis 3-class
            if score_key == 'Spondylolisthesis' and num_classes == 3:
                return label_check_3(score['Spondylolisthesis'])

            # SpondylolisthesisBinary
            if task == 'SpondylolisthesisBinary':
                return label_binarize(score['Spondylolisthesis'])

            # Generic: binary tasks → binarize, ordinal tasks → label_check
            raw = label_check(score[score_key])
            if raw == -100:
                return -100
            if is_binary:
                return 0 if raw == 0 else 1
            # Ordinal / multiclass — keep full scale, clamp to [0, num_classes)
            return min(int(raw), num_classes - 1)

        # ── Collect labels ───────────────────────────────────────────────
        task_labels = {t: [] for t in tasks}
        for vb_name in genodict:
            for task in tasks:
                try:
                    label = get_label(genodict[vb_name]['score'], task, vb_name=vb_name)
                    if label >= 0:
                        task_labels[task].append(label)
                except Exception:
                    pass

        # ── Compute weights ──────────────────────────────────────────────
        class_weight = {}
        for task in tasks:
            if not task_labels[task]:
                class_weight[task] = np.ones(task_definitions[task]["num_classes"])
                continue

            counts = Counter(task_labels[task])
            total = len(task_labels[task])
            num_classes = task_definitions[task]["num_classes"]

            weights = np.ones(num_classes)
            for cls, count in counts.items():
                if cls < num_classes:
                    weights[cls] = total / (len(counts) * count)

            class_weight[task] = weights
            print(f"{task:30s}: {total:5d} samples → {weights.round(2)}")

        return class_weight, {t: i for i, t in enumerate(tasks)}

class GenodiscOrdinalDataset(GenodiscDataset):
    """Same as GenodiscDataset but keeps ALL labels ordinal (no binarization).

    ``label_check`` (NaN → -100, otherwise keep raw int) is used everywhere
    instead of ``label_binarize``.  This means tasks like EndplateDefect,
    ForaminalStenosis, Herniation, Bulging, and FacetJointArthropathy
    retain their full ordinal scales.

    Unchanged tasks (already ordinal in the parent):
      Pfirrmann (5 classes, 0-indexed), Narrowing (4), CentralCanalStenosis (4),
      UpperMarrow / LowerMarrow (binary Modic), Spondylolisthesis (3-class).

    Changed labels (binary → ordinal):
      UpperEndplateDefect, LowerEndplateDefect,
      ForaminalStenosisLeft, ForaminalStenosisRight, Herniation,
      AnteriorBulging, PosteriorBulging,
      FacetJointArthropathyLeft, FacetJointArthropathyRight
    """

    def _extract_labels(self, score: Dict[str, Any]) -> Dict[str, int]:
        return {
            'Pfirrmann': label_check(score['Pfirrmann']) - 1,
            'Narrowing': label_check(score['Narrowing']),
            'UpperEndplateDefect': label_check(score['UpperEndplateDefect']),
            'LowerEndplateDefect': label_check(score['LowerEndplateDefect']),
            'UpperEndplateDefectFourClasses': label_check(score['UpperEndplateDefect']),
            'LowerEndplateDefectFourClasses': label_check(score['LowerEndplateDefect']),
            'UpperMarrow': label_check_marrow(
                score['UpperModic1'], score['UpperModic2'],
                score['UpperModic3'], score['UpperModicM']),
            'LowerMarrow': label_check_marrow(
                score['LowerModic1'], score['LowerModic2'],
                score['LowerModic3'], score['LowerModicM']),
            'Spondylolisthesis': label_check_3(score['Spondylolisthesis']),
            'SpondylolisthesisBinary': label_binarize(score['Spondylolisthesis']),
            'CentralCanalStenosis': label_check(score['CentralCanalStenosis']),
            'ForaminalStenosisLeft': label_check(score['ForaminalStenosisLeft']),
            'ForaminalStenosisRight': label_check(score['ForaminalStenosisRight']),
            'Herniation': label_check(score['Herniation']),
            'AnteriorBulging': label_check(score['AnteriorBulging']),
            'PosteriorBulging': label_check(score['PosteriorBulging']),
            'FacetJointArthropathyLeft': label_check(score['FacetJointArthropathyLeft']),
            'FacetJointArthropathyRight': label_check(score['FacetJointArthropathyRight']),
        }


def collate_fn(batch: List[Sample]) -> Dict[str, Any]:
    """Custom collate that stacks images and label dicts."""
    images = torch.from_numpy(np.concatenate([s['image'] for s in batch], axis=0)).float()
    names = [s['name'] for s in batch]
    label_keys = batch[0]['labels'].keys()
    labels = {k: torch.tensor([s['labels'][k] for s in batch], dtype=torch.long)
              for k in label_keys}
    return {'image': images, 'labels': labels, 'name': names}

def compute_class_weights(
    genodict: GenoDict, tasks: List[str], task_defs: Dict[str, Dict[str, Any]],
) -> Dict[str, Tensor]:
    """Compute inverse-frequency class weights for each task.

    Supports ``dataset_key`` indirection in *task_defs*: if a task has a
    ``"dataset_key"`` field, that key is used to look up the raw score
    from the genodict instead of the task name itself.  This allows
    ``*Ordinal`` task variants to share the same underlying score field.
    """
    weights = {}
    for task in tasks:
        td = task_defs[task]
        num_classes = td["num_classes"]
        # Resolve the score key — e.g. "UpperEndplateDefectOrdinal" → "UpperEndplateDefect"
        score_key = td.get("dataset_key", task)
        is_binary = td.get("binary", False)
        counts = Counter()

        if task == 'VertebralLevel':
            # Derive from vb_name keys, not from score dict
            for vb_name in genodict:
                lbl = GenodiscDataset._vertebral_level_from_name(vb_name)
                if 0 <= lbl < num_classes:
                    counts[lbl] += 1
        else:
            for vb in genodict.values():
                score = vb['score']
                try:
                    if score_key == 'Pfirrmann':
                        lbl = label_check(score['Pfirrmann']) - 1
                    elif score_key in ['UpperMarrow', 'LowerMarrow']:
                        if score_key == 'UpperMarrow':
                            lbl = label_check_marrow(score['UpperModic1'], score['UpperModic2'],
                                                     score['UpperModic3'], score['UpperModicM'])
                        else:
                            lbl = label_check_marrow(score['LowerModic1'], score['LowerModic2'],
                                                     score['LowerModic3'], score['LowerModicM'])
                    elif score_key == 'Spondylolisthesis' and is_binary:
                        lbl = label_binarize(score['Spondylolisthesis'])
                    elif score_key == 'Spondylolisthesis':
                        lbl = label_check_3(score['Spondylolisthesis'])
                    elif is_binary:
                        lbl = label_binarize(score[score_key])
                    else:
                        lbl = label_check(score[score_key])
                    if 0 <= lbl < num_classes:
                        counts[lbl] += 1
                except (KeyError, TypeError):
                    continue

        total = sum(counts.values())
        w = np.ones(num_classes, dtype=np.float32)
        for c, cnt in counts.items():
            if cnt > 0 and c < num_classes:
                w[c] = total / (len(counts) * cnt)
        weights[task] = torch.from_numpy(w)
    return weights


class TestTimeAugmentationGenodiscDataset(GenodiscDataset):
    def get_vb_by_name(self, curr_vb_name: str) -> Sample:
        curr_ = self.genodict[curr_vb_name]
        score = curr_['score']
        labels = self._extract_labels(score)

        if self.transform:
            seq_idx = random.randint(0, len(curr_['sequence'])-1)
            curr_seq = curr_['sequence'][seq_idx]
            image = np.load(self.scan_path + curr_vb_name + '_' + curr_seq + '.npy', allow_pickle=True)
            if curr_seq[:2] == 'T1':
                labels['Pfirrmann'] = -100
                labels['Narrowing'] = -100
        else:
            curr_seq = 'T2_S1'
            if os.path.isfile(self.scan_path + curr_vb_name + '_' + curr_seq + '.npy'):
                image = np.load(self.scan_path + curr_vb_name + '_' + curr_seq + '.npy', allow_pickle=True)
            else:
                image = np.zeros((self.original_height, self.original_width, self.original_slices), np.float32)
                labels = self._invalidate_all(labels)

        if not np.isfinite(image).all():
            logging.error('Image contains NaN')

        # Augmentations
        if self.transform:
            images = []
            orig_image = image.copy()
            for use_flip in [False, True]:
                for delta_x in [-16,0,16]:
                    for delta_y in [-16,0,16]:
                        for delta_mid in [-1, 0, 1]:
                            image = orig_image.copy()
                            num_rows, num_cols, num_slices = image.shape
                            max_cols = num_cols - self.margin_cols
                            min_cols = self.margin_cols
                            max_rows = num_rows - self.margin_rows
                            min_rows = self.margin_rows

                            shift_slice = delta_mid
                            image = np.roll(image, shift_slice, axis=2)
                            if shift_slice > 0:
                                image[:,:,0:shift_slice] = 0
                            elif shift_slice < 0:
                                image[:,:,num_slices+shift_slice:num_slices] = 0

                            if use_flip:
                                image = np.flip(image, axis=2).copy()

                            # Find empty slices
                            slice_start = max(0, (num_slices - self.slices) // 2)
                            slice_end = slice_start + self.slices
                            zero_slice = np.mean(image, axis=(0,1))
                            zero_slice = zero_slice[slice_start:slice_end]

                            # Translation
                            shift_cols = delta_y
                            max_cols += shift_cols
                            min_cols += shift_cols
                            shift_rows = delta_x
                            max_rows += shift_rows
                            min_rows += shift_rows

                            # Sanity Check
                            if max_cols >= num_cols:
                                min_cols = min_cols + (max_cols - num_cols)
                                max_cols = num_cols - 1
                            if max_rows >= num_rows:
                                min_rows = min_rows + (max_rows - num_rows)
                                max_rows = num_rows - 1
                            if min_cols < 0:
                                max_cols = max_cols + abs(min_cols)
                                min_cols = 0
                            if min_rows < 0:
                                max_rows = max_rows + abs(min_rows)
                                min_rows = 0
                            max_cols = int(np.round(max_cols))
                            min_cols = int(np.round(min_cols))
                            max_rows = int(np.round(max_rows))
                            min_rows = int(np.round(min_rows))

                            # Select slices
                            image = image[:, :, slice_start:slice_end]

                            # Resize
                            image = cv2.resize(image[min_rows:max_rows,min_cols:max_cols,:],
                                             (self.width, self.height),
                                             interpolation=cv2.INTER_CUBIC)
                            image[:,:,zero_slice==0] = 0
                            image = np.transpose(image, (2, 0, 1))[None,:,:,:]
                            images.append(image)

            # Update labels if flip was used (for the last augmentation)
            if use_flip:
                old_fsl = labels['ForaminalStenosisLeft']
                old_fjl = labels['FacetJointArthropathyLeft']
                labels['ForaminalStenosisLeft'] = labels['ForaminalStenosisRight']
                labels['ForaminalStenosisRight'] = old_fsl
                labels['FacetJointArthropathyLeft'] = labels['FacetJointArthropathyRight']
                labels['FacetJointArthropathyRight'] = old_fjl
        else:
            num_rows, num_cols, num_slices = image.shape
            max_cols = num_cols - self.margin_cols
            min_cols = self.margin_cols
            max_rows = num_rows - self.margin_rows
            min_rows = self.margin_rows
            slice_start = max(0, (num_slices - self.slices) // 2)
            slice_end = slice_start + self.slices
            image = image[min_rows:max_rows, min_cols:max_cols, slice_start:slice_end]
            image = np.transpose(image, (2, 0, 1))[None,:,:,:]
            images = [image]

        # Derive VertebralLevel from the volume name
        if labels.get('Pfirrmann', 0) == -100 and labels.get('Narrowing', 0) == -100:
            labels['VertebralLevel'] = -100
        else:
            labels['VertebralLevel'] = self._vertebral_level_from_name(curr_vb_name)

        sample = {'images': images, 'labels': labels, 'name': curr_vb_name}
        return sample


class TestTimeAugmentationGenodiscOrdinalDataset(
    GenodiscOrdinalDataset, TestTimeAugmentationGenodiscDataset,
):
    """TTA variant of ``GenodiscOrdinalDataset``.

    MRO ensures ``_extract_labels`` comes from ``GenodiscOrdinalDataset``
    (ordinal, no binarization) and ``get_vb_by_name`` comes from
    ``TestTimeAugmentationGenodiscDataset`` (multi-augmentation TTA).
    """
    pass


if __name__ == '__main__':
    import argparse
    import os
    import pickle

    # Smoke test: load one training sample. The data location comes from the
    # GENODISC_ROOT environment variable (same layout and default as the
    # trainer, see README "Data") or from the command line.
    _root = os.environ.get("GENODISC_ROOT") or "data/GENODISCv2"
    parser = argparse.ArgumentParser(description="Load one training sample as a smoke test.")
    parser.add_argument("--ivd_path", default=os.path.join(_root, "IVDs-numpy", ""),
                        help="directory of the IVD .npy volumes (default: "
                             "$GENODISC_ROOT/IVDs-numpy/; GENODISC_ROOT defaults "
                             "to data/GENODISCv2)")
    parser.add_argument("--genodict_path", default=os.path.join(_root, "geno_ivd_v1.pkl"),
                        help="split pickle (default: $GENODISC_ROOT/geno_ivd_v1.pkl)")
    cli = parser.parse_args()

    Warning(f"Running {__file__} in testing mode.")
    # GenodiscDataset concatenates scan_path and file name: keep the trailing separator.
    IVD_PATH = os.path.join(os.path.expanduser(cli.ivd_path), "")
    GENODICT_PATH = os.path.expanduser(cli.genodict_path)

    with open(GENODICT_PATH, 'rb') as f:
        geno = pickle.load(f)

    # Create dataset with default dimensions (9, 112, 224)
    # Margins are computed automatically: rows=40, cols=48
    train_ds = GenodiscDataset(geno['Train'], IVD_PATH, transform=True)

    sample = train_ds[0]
    print(f"\nSample image shape: {sample['image'].shape}")  # Should be (1, 9, 112, 224)
    print(f"Sample labels: {list(sample['labels'].keys())}")

#     # Example: Compute class weights for all labels
#     print("\n" + "="*60)
#     print("Computing class weights for all labels...")
#     print("="*60)
#     all_weights = GenodiscDataset.compute_all_class_weights(geno['Train'])
#
#     # Example: Use with different dimensions
#     print("\n" + "="*60)
#     print("Testing with custom dimensions...")
#     print("="*60)
#     custom_ds = GenodiscDataset(geno['Train'], IVD_PATH, transform=False,
#                                 slices=13, height=128, width=256)
#     sample2 = custom_ds[0]
#     print(f"Custom sample shape: {sample2['image'].shape}")  # Should be (1, 7, 128, 256)
