import os

import albumentations as A
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .utils import polar_inv_transform, polar_transform


class ODOC(Dataset):
    """ODOC Dataset"""

    def __init__(self, base_dir=None, split="train", transform=None, polar_trans=False):
        self._base_dir = base_dir
        self.sample_list = []
        self.transform = transform
        self.polar_trans = polar_trans
        print(os.getcwd())
        train_path = self._base_dir + "/train.list"
        test_path = self._base_dir + "/test.list"
        valid_path = self._base_dir + "/valid.list"

        if split == "train":
            self.is_train = True
            with open(train_path, "r") as f:
                self.image_list = f.readlines()
        elif split == "test":
            self.is_train = False
            with open(test_path, "r") as f:
                self.image_list = f.readlines()
        elif split == "valid":
            self.is_train = False
            with open(valid_path, "r") as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace("\n", "") for item in self.image_list]

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((256, 256))]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.augmentation = A.Compose(
            [
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.OneOf(
                    [
                        A.ChannelShuffle(p=0.3),
                        A.FancyPCA(),
                        A.ColorJitter(),
                    ]
                ),
                # A.ShiftScaleRotate(),
                A.ToGray(p=0.3),
                A.CLAHE(),
            ]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/h5py_all" + "/" + image_name, "r")
        image = h5f["img"][:].astype(np.uint8)
        label_cup = h5f["mask"][:, :, 0].astype(np.uint8)
        label_disc = h5f["mask"][:, :, 1].astype(np.uint8)
        con_gau_cup = h5f["con_gau"][:, :, 0].astype(np.uint8)
        con_gau_disc = h5f["con_gau"][:, :, 1].astype(np.uint8)

        # debug
        # _image = polar_transform(image)
        # plt.imshow(_image)
        # plt.show()

        pre_process = A.Resize(256, 256)
        masks = [label_cup, label_disc, con_gau_cup, con_gau_disc]
        transformed = pre_process(image=image, masks=masks)
        image = transformed["image"]
        label_cup, label_disc, con_gau_cup, con_gau_disc = transformed["masks"]

        # 如果使用了极坐标变换，则应当使用最原始的gt，防止gt不在圆的范围内;因为用来监督的mask也应当是极坐标变换后的
        ori_label_cup, ori_label_disc, ori_con_gau_cup, ori_con_gau_disc = (
            label_cup,
            label_disc,
            con_gau_cup,
            con_gau_disc,
        )

        if self.is_train:
            if self.polar_trans is True:
                # plt.subplot(131)
                # plt.imshow(image)
                # image = polar_transform(image)

                # plt.subplot(132)
                # plt.imshow(image)

                # plt.subplot(133)
                # _image = polar_inv_transform(image)
                # plt.imshow(_image)
                # plt.show()

                image = polar_transform(image)
                label_cup = polar_transform(label_cup)
                label_disc = polar_transform(label_disc)
                con_gau_cup = polar_transform(con_gau_cup)
                con_gau_disc = polar_transform(con_gau_disc)
            else:
                masks = [label_cup, label_disc, con_gau_cup, con_gau_disc]
                transformed = self.augmentation(image=image, masks=masks)
                image = transformed["image"]
                label_cup, label_disc, con_gau_cup, con_gau_disc = transformed["masks"]

            # print("label_cup.shape:", label_cup.shape)
            # print("label_disc.shape:", label_disc.shape)
            # print("image.type:", type(image))
            # # print("label_cup.shape:", label_cup.size())
            # plt.subplot(311)
            # plt.imshow(image)
            # plt.subplot(312)
            # plt.imshow(label_cup)
            # plt.subplot(313)
            # plt.imshow(label_disc)
            # plt.show()

            image = self.test_transform(image)

            label_cup = self.transform(label_cup)
            label_disc = self.transform(label_disc)
            label = torch.cat((label_cup, label_disc), 0)

            con_gau_cup = self.transform(con_gau_cup)
            con_gau_disc = self.transform(con_gau_disc)
            con_gau = torch.cat((con_gau_cup, con_gau_disc), 0)

            ori_label_cup = self.transform(ori_label_cup)
            ori_label_disc = self.transform(ori_label_disc)
            ori_label = torch.cat((ori_label_cup, ori_label_disc), 0)

            ori_con_gau_cup = self.transform(ori_con_gau_cup)
            ori_con_gau_disc = self.transform(ori_con_gau_disc)
            ori_con_gau = torch.cat((ori_con_gau_cup, ori_con_gau_disc), 0)

            sample = {"img": image, "mask": label, "con_gau": con_gau}

            return sample
        else:
            # _img = h5f["img"][:]
            # plt.imshow(_img)
            # plt.show()
            image = h5f["img"][:]
            # image = A.clahe(image)
            image = self.test_transform(image)
            label = self.transform(h5f["mask"][:])
            con_gau = self.transform(h5f["con_gau"][:])
            sample = {"img": image, "ori_mask": ori_label, "ori_con_gau": ori_con_gau}
        return sample
