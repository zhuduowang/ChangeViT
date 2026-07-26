import cv2
import os
from os.path import join as osp
from os.path import isfile
import numpy
import torch.utils.data


def _resolve_dir(parent, names):
    """Find a child directory by exact name first, then case-insensitively."""
    for name in names:
        candidate = osp(parent, name)
        if os.path.isdir(candidate):
            return candidate
    wanted = {name.lower() for name in names}
    if os.path.isdir(parent):
        for child in os.listdir(parent):
            if child.lower() in wanted:
                candidate = osp(parent, child)
                if os.path.isdir(candidate):
                    return candidate
    raise FileNotFoundError(
        "Could not find any of {} under '{}'.".format(", ".join(names), parent)
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_root='data/', mode='train', transform=None):
        self.file_root = os.path.abspath(os.path.expanduser(file_root))
        self.mode = mode
        mode_dir = _resolve_dir(self.file_root, [mode])
        a_dir = _resolve_dir(mode_dir, ["A", "t1", "T1"])
        b_dir = _resolve_dir(mode_dir, ["B", "t2", "T2"])
        label_dir = _resolve_dir(mode_dir, ["label", "labels", "gt", "mask", "masks"])

        self.file_list = sorted([x for x in os.listdir(a_dir) if isfile(osp(a_dir, x))])
        if not self.file_list:
            raise FileNotFoundError("No images found in '{}'.".format(a_dir))

        missing = [
            x for x in self.file_list
            if not isfile(osp(b_dir, x)) or not isfile(osp(label_dir, x))
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                "Found {} files in '{}', but matching files are missing in '{}' "
                "or '{}'. First missing: {}".format(
                    len(missing), a_dir, b_dir, label_dir, preview
                )
            )

        self.pre_images = [osp(a_dir, x) for x in self.file_list]
        self.post_images = [osp(b_dir, x) for x in self.file_list]
        self.gts = [osp(label_dir, x) for x in self.file_list]

        self.transform = transform

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image_name = self.pre_images[idx]
        label_name = self.gts[idx]
        post_image_name = self.post_images[idx]

        pre_image = cv2.imread(pre_image_name)
        label = cv2.imread(label_name, 0)
        post_image = cv2.imread(post_image_name)

        img = numpy.concatenate((pre_image, post_image), axis=2)

        if self.transform:
            [img, label] = self.transform(img, label)

        return img, label

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}
