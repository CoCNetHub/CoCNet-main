import argparse
from loguru import logger
import cv2
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import pandas as pd
import ast
import shutil
import time
from typing import List, Dict, Union, Tuple
from PIL import Image, ImageDraw, ImageFilter
import hashlib
import os
import torch
import torchvision.transforms as transforms
from transformers import BertTokenizer
import ruamel.yaml as yaml
from albef.model import ALBEF
from albef.utils import *
from albef.vit import interpolate_pos_embed
import matplotlib.pyplot as plt
from nltk import pos_tag
import numpy as np
from math import exp, log, sqrt, ceil

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="image"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_inf


def image_demo(predictor):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    albef_path = "/home/CoCNet-main/albef/"
    box_representation_method = "crop,blur"
    box_method_aggregator = 'sum'
    device = 0
    albef_mode_list = ['itm', 'itc']
    for albef_mode in albef_mode_list:
        blur_std_dev = 100
        cache_path = None
        executor = AlbefExecutor(config_path=os.path.join(albef_path, "config.yaml"),
                                 checkpoint_path=os.path.join(albef_path, "checkpoint.pth"),
                                 box_representation_method=box_representation_method,
                                 method_aggregator=box_method_aggregator, device=device, mode=albef_mode,
                                 blur_std_dev=blur_std_dev, cache_path=cache_path)

        dataset_path = "/home/CoCNet-main/data/"
        files = os.listdir(dataset_path)
        for file in files:
            subFileDirPath = dataset_path + file
            subFileDir = os.listdir(subFileDirPath)

                for i in range(len(data)):
                    annotation_list = data.iloc[i]
                    img_file_item, x1, y1, x2, y2, queries = annotation_list
                    annot = [x1, y1, x2, y2]
                    img_file = "/root/autodl-tmp/train2014/" + img_file_item
                    outputs, img_info = predictor.inference(img_file)
                    objects, objectName, max_object = predictor.visual(outputs[0], img_info)
                    # subject clue module
                    '''
                    To address the semantic confusion in category inference, we exploit the subject clue through the potent ability of LLMs in textual semantic understanding to retain the proposals with the true category.
                    '''

                    # attribute clue module
                    '''
                    RIPS scheme, inspired by the keyphrase extraction task in natural language processing (Zhang et al., 2021). Under RIPS, the matching score between a query and an image decreases significantly if the corresponding proposal is blurry or cropped. 
                    '''

                    # spatial clue module
                    '''
                    the Gaussian-based soft heuristic rules to model the relationship between the location words in expression and spatial features in the image.
                    '''



def read_annotations(trn_file):
    trn_data = pd.read_csv(trn_file)
    trn_data['bbox'] = trn_data.bbox.apply(
        lambda x: ast.literal_eval(x))
    sample = trn_data['query'].iloc[0]
    if sample[0] == '[':
        trn_data['query'] = trn_data['query'].apply(
            lambda x: ast.literal_eval(x))
    trn_data['x1'] = trn_data.bbox.apply(lambda x: x[0])
    trn_data['y1'] = trn_data.bbox.apply(lambda x: x[1])
    trn_data['x2'] = trn_data.bbox.apply(lambda x: x[2])
    trn_data['y2'] = trn_data.bbox.apply(lambda x: x[3])
    trn_df = trn_data[['img_id', 'x1', 'y1', 'x2', 'y2', 'query']]
    return trn_df


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    if args.demo == "image":
        image_demo(predictor)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
