import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.ODOC_BMVC import ODOC_seg_edge
from utils.criterion import BinaryDiceLoss
from utils.Dataloader_ODOC import ODOC
from utils.utils import clip_gradient, model_test

# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path", type=str, default="../segmentation_data", help="Name of Experiment"
)
parser.add_argument("--exp", type=str, default="oc_od/ALL_DATA_AUG", help="model_name")
parser.add_argument(
    "--max_iterations", type=int, default=50000, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=48, help="batch_size per gpu")
parser.add_argument("--base_lr", type=float, default=0.006, help="learning rate")
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument(
    "--beta",
    type=float,
    default=0.1,
    help="balance factor to control edge and body loss",
)
parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping margin")
parser.add_argument(
    "--decay_rate", type=float, default=0.6, help="decay rate of learning rate"
)
parser.add_argument(
    "--decay_itetations",
    type=int,
    default=30000,
    help="every n itetations decay learning rate",
)
parser.add_argument(
    "--polar_transform",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="polar transform pre-process",
)
parser.add_argument("--postgnn", type=str, default="APPNP", help="PostGNN Type")
parser.add_argument(
    "--aggregation_mode",
    type=str,
    default="sum",
    help="adjacency matrix aggregation mode",
)
parser.add_argument("--postgnn_depth", type=int, default=3, help="Depths of PostGNN")
parser.add_argument("--prop_nums", type=int, default=3, help="Propagation numbers")

args = parser.parse_args()


train_data_path = args.root_path
snapshot_path = (
    "../model/"
    + args.exp
    + "{}polar_{}layer_{}_{}_props_{}_aggregate_{}_bs_beta_{}_base_lr_{}".format(
        int(args.polar_transform),
        args.postgnn_depth,
        args.postgnn,
        args.prop_nums,
        args.aggregation_mode,
        args.batch_size,
        args.beta,
        args.base_lr,
    )
)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(","))
max_iterations = args.max_iterations
base_lr = args.base_lr
postgnn = args.postgnn

"""reproducible"""

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

"""reproducible"""

num_classes = 2

if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    code_path = snapshot_path + "/code"
    if os.path.exists(code_path):
        shutil.rmtree(code_path)

    model_performance_log_path = os.path.join(snapshot_path, "performance_log.txt")

    if os.path.exists(model_performance_log_path):
        os.remove(model_performance_log_path)

    
    ignore = shutil.ignore_patterns(".git", "__pycache__")
    shutil.copytree(
        src=".", dst=code_path, ignore=ignore
    )

    log_path = snapshot_path + "/log.txt"
    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = ODOC_seg_edge(
        postgnn=args.postgnn,
        postgnn_depth=args.postgnn_depth,
        prop_nums=args.prop_nums,
        aggregation_mode=args.aggregation_mode,
    )
    model = model.cuda()

    # load_path = snapshot_path + '/start_checkpoint.pth'
    # model.load_state_dict(torch.load(load_path))

    apply_polar_transform = args.polar_transform
    if apply_polar_transform is True:
        print("Apply Polar Transform.")

    db_train = ODOC(
        base_dir=train_data_path, split="train"
    )
    db_valid = ODOC(
        base_dir=train_data_path, split="valid"
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        shuffle=True,
    )
    validloader = DataLoader(db_valid, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    dice_loss = BinaryDiceLoss()

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    best_model_iter = 0
    best_cup_dice_mean = 0

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            model.train()
            volume_batch, label_batch, edge_batch = (
                sampled_batch["img"],
                sampled_batch["mask"],
                sampled_batch["con_gau"],
            )

            # 注意，这里需要转化一下label，因为在读入的时候进行了归一化，1会变得很小
            label_batch = (label_batch > 0).float()
            edge_batch = (edge_batch > 0).float()

            edge_batch_com = edge_batch[:, 0, :, :] + edge_batch[:, 1, :, :]
            volume_batch, label_batch, edge_batch_com = (
                volume_batch.float().cuda(),
                label_batch.float().cuda(),
                edge_batch_com.float().cuda(),
            )

            outputs, edge_outputs, graph_regulation_loss = model(volume_batch)

            # upscale back to 256x256
            edge_outputs = F.interpolate(
                input=edge_outputs, size=(256, 256), mode="bilinear"
            )
            outputs = F.interpolate(input=outputs, size=(256, 256), mode="bilinear")

            cup_loss = dice_loss(outputs[:, 0, ...], label_batch[:, 0, ...].float())
            disc_loss = dice_loss(outputs[:, 1, ...], label_batch[:, 1, ...].float())
            region_loss = cup_loss + disc_loss

            edge_loss = dice_loss(edge_outputs.squeeze(), edge_batch_com.float())

            loss = region_loss + args.beta * edge_loss

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("loss/loss", loss, iter_num)
            writer.add_scalar("loss/cup_loss", cup_loss, iter_num)
            writer.add_scalar("loss/disc_loss", disc_loss, iter_num)
            writer.add_scalar("loss/edge_loss", edge_loss, iter_num)

            logging.info(
                "iteration %d : loss : %f, cup_loss: %f, disc_loss: %f, edge_loss: %f"
                % (
                    iter_num,
                    loss.item(),
                    cup_loss.item(),
                    disc_loss.item(),
                    edge_loss.item(),
                )
            )

            #  save every 1000 item_num
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "iter_" + str(iter_num) + ".pth"
                )
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

                print("Validation: ...")
                cup_dice_mean, cup_dice_up95, cup_dice_low95 = model_test(
                    model, validloader
                )
                with open(model_performance_log_path, "a") as f:
                    p_log = "iter: {}   cup_dice_mean: {:.4f}   cup_dice_up95: {:.4f}   cup_dice_low95: {:.4f}\n".format(
                        iter_num, cup_dice_mean, cup_dice_up95, cup_dice_low95
                    )
                    f.write(p_log)

                if cup_dice_mean > best_cup_dice_mean:
                    best_cup_dice_mean = cup_dice_mean
                    best_model_iter = iter_num

            if iter_num >= max_iterations:
                break

        # change lr, decay every 30000 iterations
        if (iter_num + 1) % args.decay_itetations == 0:
            lr_ = base_lr * args.decay_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

        if iter_num >= max_iterations:
            iterator.close()
            break

    with open(model_performance_log_path, "a") as f:
        p_log = "best_iter: {:.4f}  best_cup_dice_mean: {:.4f}\n".format(
            best_model_iter, best_cup_dice_mean
        )
        f.write(p_log)

    writer.close()
