# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import cv2
from PIL import Image

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from core.inference import get_final_preds_wo_c_s
from utils.utils import create_logger
from utils.vis import save_demo_images
import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--modelDir", help="model directory", type=str, default="")
    parser.add_argument("--logDir", help="log directory", type=str, default="")
    parser.add_argument("--dataDir", help="data directory", type=str, default="")
    parser.add_argument("--imFile", help="input image file", type=str, default="")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, "valid")

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(final_output_dir, "final_state.pth")
        logger.info("=> loading model from {}".format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    device = torch.device("cpu")
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    basewidth = 384

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    print(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cap.get(cv2.CAP_PROP_FPS),
    )

    while True:
        _, frame = cap.read()
        img = frame
        # im_name = args.imFile
        # img = Image.open(im_name).convert("RGB")
        # print("Image original shape = ", img.size)
        print(img.shape)
        wpercent = float(basewidth) / float(img.shape[1])
        hsize = float(img.shape[0]) * float(wpercent)
        # img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        img = cv2.resize(img, (384, 288))

        img = test_transform(img)
        img = torch.unsqueeze(img, 0)

        # print("Input image final shape = ", img.shape)
        print(img.shape[-1])
        # evaluate on validation set
        with torch.no_grad():

            outputs = model(img)

        image_to_pred_scale = img.shape[-1] / outputs.shape[-1]
        # print("Predicted heatmap size = ", outputs.shape)
        # print(outputs)

        # print("Image to prediction scale = ", image_to_pred_scale)

        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        preds, maxvals = get_final_preds_wo_c_s(output.clone().cpu().numpy())

        # 結果画像を表示する
        # dispFps.disp(vis_result)
        # cv2.imshow("frame", vis_result)

        # waitKey()...1msの間キー入力を待つ関数
        keyboard_input = cv2.waitKey(100)  # キー操作取得。64ビットマシンの場合,& 0xFFが必要
        prop_val = cv2.getWindowProperty("frame", cv2.WND_PROP_ASPECT_RATIO)  # ウィンドウが閉じられたかを検知する用

        # qが押されるか、ウィンドウが閉じられたら終了
        if keyboard_input in (27, ord("q"), ord("Q")):
            break
        save_demo_images(
            batch_image=img,
            batch_joints=preds * image_to_pred_scale,
            batch_joints_vis=maxvals,
            # file_name="out.png",
        )


if __name__ == "__main__":
    main()
