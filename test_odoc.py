import argparse
import os

import torch
from torch.utils.data import DataLoader

from lib.ODOC_BMVC import ODOC_seg_edge
from utils.Dataloader_ODOC import ODOC
from utils.utils import model_test

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path", type=str, default="../segmentation_data", help="Name of Experiment"
)
parser.add_argument("--exp", type=str, default="oc_od/ALL_DATA_AUG", help="model_name")
parser.add_argument(
    "--max_iterations", type=int, default=50000, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=48, help="batch_size per gpu")
parser.add_argument(
    "--base_lr", type=float, default=0.006, help="maximum epoch number to train"
)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument(
    "--beta",
    type=float,
    default=0.1,
    help="balance factor to control edge and body loss",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0,
    help="balance factor to control consistency loss and body loss",
)
parser.add_argument("--postgnn", type=str, default="APPNP", help="PostGNN Type")
parser.add_argument(
    "--aggregation_mode",
    type=str,
    default="sum",
    help="adjacency matrix aggregation mode",
)

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = (
    "../model/"
    + args.exp
    + "_{}_{}_aggregate_{}_bs_beta_{}_base_lr_{}".format(
        args.postgnn, args.aggregation_mode, args.batch_size, args.beta, args.base_lr
    )
)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


saved_model_path = os.path.join(snapshot_path, "best_model.pth")


if __name__ == "__main__":
    model = ODOC_seg_edge(postgnn=args.postgnn, aggregation_mode=args.aggregation_mode)
    model = model.cuda()

    db_test = ODOC(base_dir=train_data_path, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    best_performance = 0.0
    model.load_state_dict(torch.load(saved_model_path))
    print("Test Start...")
    model_test(model=model, test_loader=testloader)
