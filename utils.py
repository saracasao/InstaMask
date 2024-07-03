import cv2
import numpy as np
import torch
from typing import Any, Dict, List

kernel = {'film': [np.ones((7, 7), np.uint8), 1],
          'basket': [np.ones((7, 7), np.uint8), 1],
          'carboard': [np.ones((7, 7), np.uint8), 1],
          'video_tape': [np.ones((1, 1), np.uint8), 2],
          'filament': [np.ones((1, 1), np.uint8), 2],
          'bag': [np.ones((7, 7), np.uint8), 1]}


def remove_small_regions(mask, label):
    """If it's needed, clean small regions inside the masks"""
    k = kernel[label][0]
    iter = kernel[label][1]

    clean_noise = cv2.erode(mask, k, iterations=iter)
    dilation = cv2.dilate(clean_noise, k, iterations=(2*iter))
    return dilation


def mask_to_rle(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = torch.nonzero(diff)

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    assert type(rle) == dict, "Error in rle type {}".format(rle)
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def binary_mask_to_bbox(binary_mask):
    binary_mask = np.asarray(binary_mask, dtype=np.uint8)

    segmentation = np.where(binary_mask == 255)
    xmin = int(np.min(segmentation[1]))
    xmax = int(np.max(segmentation[1]))
    ymin = int(np.min(segmentation[0]))
    ymax = int(np.max(segmentation[0]))

    width = xmax - xmin
    height = ymax - ymin

    return xmin, ymin, width, height
