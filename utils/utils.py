import cv2
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from sklearn.metrics import balanced_accuracy_score, jaccard_score


# Optional ROI 极坐标变换
def polar_transform(image):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    _image = cv2.linearPolar(
        src=image, center=center, maxRadius=center[0], flags=cv2.WARP_FILL_OUTLIERS
    )
    return cv2.rotate(_image, cv2.ROTATE_90_CLOCKWISE)


def polar_inv_transform(image):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    _image = cv2.linearPolar(
        src=cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
        center=center,
        maxRadius=center[0],
        flags=cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP,
    )
    return _image


def calculate_metric_percase(pred, gt):
    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = np.sum(y_true * y_pred)
        return (2.0 * intersection + smooth) / (
            np.sum(y_true) + np.sum(y_pred) + smooth
        )

    dice = dice_coef(gt, pred)
    jc = jaccard_score(gt.flatten(), pred.flatten())
    bc = balanced_accuracy_score(gt.flatten(), pred.flatten())
    dice2 = metric.binary.dc(pred, gt)
    return dice, jc, dice2, bc


def model_test(model, test_loader):
    model.eval()
    with torch.no_grad():
        total_metric_cup = 0.0
        total_metric_disc = 0.0

        cup_list = []
        disc_list = []
        name_list = []

        for i_batch, sampled_batch in enumerate(test_loader):
            volume_batch, label_batch, edge_batch = (
                sampled_batch["img"],
                sampled_batch["mask"],
                sampled_batch["con_gau"],
            )
            volume_batch, label_batch, edge_batch = (
                volume_batch.type(torch.FloatTensor),
                label_batch.type(torch.FloatTensor),
                edge_batch.type(torch.FloatTensor),
            )
            volume_batch, label_batch, edge_batch = (
                volume_batch.cuda(),
                label_batch.cuda(),
                edge_batch.cuda(),
            )

            # 注意，这里需要转化一下label，因为在读入的时候进行了归一化，1会变得很小
            label_batch = (label_batch > 0).float()
            edge_batch = (edge_batch > 0).float()

            outputs1, edge_outputs1, graph_regulation_loss = model(volume_batch)

            pred_edge = F.interpolate(
                input=edge_outputs1, size=(256, 256), mode="bilinear"
            )
            pred_seg = F.interpolate(input=outputs1, size=(256, 256), mode="bilinear")

            # seg
            y_pre = pred_seg.cpu().data.numpy().squeeze()
            y_pre_gt = label_batch.cpu().data.numpy().squeeze()

            y_map_cup = (y_pre[0] > 0.5).astype(np.uint8)
            y_map_disc = (y_pre[1] > 0.5).astype(np.uint8)

            """uncomment below to see a smoothed boundary"""
            # image = Image.fromarray(y_map_cup)
            # filter_image = image.filter(ImageFilter.ModeFilter(size=10))
            # y_map_cup = np.asarray(filter_image)
            # y_map_cup = (y_map_cup > 0).astype(np.uint8)
            #
            # image = Image.fromarray(y_map_disc)
            # filter_image = image.filter(ImageFilter.ModeFilter(size=10))
            # y_map_disc = np.asarray(filter_image)
            # y_map_disc = (y_map_disc > 0).astype(np.uint8)

            y_map_gt_cup = y_pre_gt[0, ...].astype(np.uint8)
            y_map_gt_disc = y_pre_gt[1, ...].astype(np.uint8)

            # plt.figure()
            # plt.subplot(2, 2, 1)
            # plt.imshow(y_map_cup, cmap="gray")
            # plt.subplot(2, 2, 2)
            # plt.imshow(y_map_disc, cmap="gray")
            # plt.subplot(2, 2, 3)
            # plt.imshow(y_map_gt_cup, cmap="gray")
            # plt.subplot(2, 2, 4)
            # plt.imshow(y_map_gt_disc, cmap="gray")
            # plt.show()

            single_metric_cup = calculate_metric_percase(y_map_cup, y_map_gt_cup)
            total_metric_cup += np.asarray(single_metric_cup)
            single_metric_disc = calculate_metric_percase(y_map_disc, y_map_gt_disc)
            total_metric_disc += np.asarray(single_metric_disc)
            cup_list.append(single_metric_cup)
            disc_list.append(single_metric_disc)
            # name_list.append(sampled_name)

        cup_dice_mean = np.array(cup_list)[:, 0].mean()
        disc_dice_mean = np.array(disc_list)[:, 0].mean()

        BC_cup_mean = np.array(cup_list)[:, 3].mean()
        BC_disc_mean = np.array(disc_list)[:, 3].mean()

        print("cup_dice_mean: {:.4f}".format(cup_dice_mean))
        print("disc_dice_mean: {:.4f}".format(disc_dice_mean))

        print("BC_cup_mean: {:.4f}".format(BC_cup_mean))
        print("BC_disc_mean: {:.4f}".format(BC_disc_mean))

        CI_cup_dice = []
        CI_disc_dice = []
        CI_cup_BC = []
        CI_disc_BC = []
        n_bootstraps = 2000
        rng_seed = 42  # control reproducibility
        rng = np.random.RandomState(rng_seed)
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(test_loader), len(test_loader))
            cup_dice_CI = np.array(cup_list)[indices, 0].mean()
            disc_dice_CI = np.array(disc_list)[indices, 0].mean()
            BC_cup_CI = np.array(cup_list)[indices, 3].mean()
            BC_disc_CI = np.array(disc_list)[indices, 3].mean()
            CI_cup_dice.append(cup_dice_CI)
            CI_disc_dice.append(disc_dice_CI)
            CI_cup_BC.append(BC_cup_CI)
            CI_disc_BC.append(BC_disc_CI)
        # cup_dice_CI
        sorted_scores = np.sort(np.array(CI_cup_dice))
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
        print(
            "95CI_cup_dice, lower: {:.4f}, higher: {:.4f}".format(
                confidence_lower, confidence_upper
            )
        )

        cup_dice_up95 = confidence_upper
        cup_dice_low95 = confidence_lower

        # disc_dice_CI
        sorted_scores = np.sort(np.array(CI_disc_dice))
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
        print(
            "95CI_disc_dice, lower: {:.4f}, higher: {:.4f}".format(
                confidence_lower, confidence_upper
            )
        )
        # BC_cup_CI
        sorted_scores = np.sort(np.array(CI_cup_BC))
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
        print(
            "95CI_cup_BC, lower: {:.4f}, higher: {:.4f}".format(
                confidence_lower, confidence_upper
            )
        )

        # BC_disc_CI
        sorted_scores = np.sort(np.array(CI_disc_BC))
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
        print(
            "95CI_disc_BC, lower: {:.4f}, higher: {:.4f}".format(
                confidence_lower, confidence_upper
            )
        )

        return cup_dice_mean, cup_dice_up95, cup_dice_low95


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(
            torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0) :])
        )
