import argparse
import os
import time
from loguru import logger

import cv2

import torch

# import nltk
# from nltk.parse.corenlp import CoreNLPParser
import re
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import pandas as pd
import ast
import numpy as np

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import clip
import shutil
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
import abc
import gc
import time
from typing import Iterable, Dict
import logging

import webcolors

from typing import List, Dict, Union, Tuple
from PIL import Image, ImageDraw, ImageFilter
import hashlib
import os

import torch
import torchvision.transforms as transforms
from transformers import BertTokenizer
import ruamel.yaml as yaml

from interpreter import Box
from albef.model import ALBEF
from albef.utils import *
from albef.vit import interpolate_pos_embed
import matplotlib.pyplot as plt

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length
from nltk import pos_tag
import nltk
from nltk.parse.corenlp import CoreNLPParser
import webcolors
import numpy as np
from math import exp, log, sqrt, ceil

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def prepare_logits_processor(
        temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.inference_mode()
def generate_stream(
        model,
        tokenizer,
        params: Dict,
        device: str,
        context_len: int,
        stream_interval: int = 2,
        judge_sent_end: bool = False,
):
    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    past_key_values = out = None
    sent_interrupt = False
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""


class SimpleChatIO(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:

        return "how about you?"

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)


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


def plot_bbx(img_info, bboxes, scores, cls, cls_conf, cls_names):
    colorName = ['red', 'purple', 'blue', 'white', 'green', 'yellow', 'brown', 'gray', 'black',
                 'deeppink', 'skyblue', 'cyan', 'gold', 'orange', 'magenta', 'auqamarin', 'olive', 'pink']
    colorRGB = [[255, 0, 0], [128, 0, 128], [0, 0, 255], [255, 255, 255], [0, 128, 0], [255, 255, 0], [165, 42, 42],
                [128, 128, 128], [0, 0, 0],
                [255, 20, 147], [0, 191, 255], [0, 255, 255], [255, 215, 0], [255, 165, 0], [255, 0, 255],
                [127, 255, 170], [128, 128, 0], [255, 192, 203]]
    image = cv2.imread(img_info)
    bboxes = bboxes.tolist()
    masks = np.zeros((image.shape), dtype=np.uint8)
    colorList = [None] * len(colorName)
    classList = [None] * 80
    objectList = list(range(len(bboxes)))
    i = -1
    for index in range(len(bboxes)):
        zeros = np.zeros((image.shape), dtype=np.uint8)
        classindex = int(cls[index])
        if (classList[classindex] == None):
            i = i + 1
            classList[classindex] = i
        colorIndexItem = classList[classindex]
        if (i < len(colorName)):
            colorIndex = colorIndexItem
            colorList[colorIndex] = {'colorName': colorName[colorIndex], 'colorRGB': colorRGB[colorIndex]}
            shape = 'rectangle'
            zeros_mask = cv2.rectangle(zeros, (int(bboxes[index][0]), int(bboxes[index][1])),
                                       (int(bboxes[index][2]), int(bboxes[index][3])), color=(
                colorRGB[colorIndex][2], colorRGB[colorIndex][1], colorRGB[colorIndex][0]), thickness=-1)
        elif (i < 2 * len(colorName)):
            colorIndex = colorIndexItem - len(colorName)
            colorList[colorIndex] = {'colorName': colorName[colorIndex], 'colorRGB': colorRGB[colorIndex]}
            shape = 'circle'
            zeros_mask = cv2.circle(zeros, (
                int((bboxes[index][0] + bboxes[index][1]) / 2), int((bboxes[index][1] + bboxes[index][1]) / 3)), (
                                    int(bboxes[index][2] - bboxes[index][0]), int(bboxes[index][3] - bboxes[index][1])),
                                    0, 0, 360,
                                    (colorRGB[colorIndex][2], colorRGB[colorIndex][1], colorRGB[colorIndex][0]), 3)
        else:
            colorIndex = colorIndexItem - 2 * len(colorName)
            colorList[colorIndex] = {'colorName': colorName[colorIndex], 'colorRGB': colorRGB[colorIndex]}
            shape = 'triangle'
            pt1 = (int((bboxes[index][0] + bboxes[index][1]) / 2), int(bboxes[index][1]))
            pt2 = (int(bboxes[index][0]), int(bboxes[index][3]))
            pt3 = (int(bboxes[index][2]), int(bboxes[index][3]))
            triangle_cnt = np.array([pt1, pt2, pt3])
            cv2.drawContours(image, [triangle_cnt], 0,
                             (colorRGB[colorIndex][2], colorRGB[colorIndex][1], colorRGB[colorIndex][0]), -1)
        objectList[index] = {'colorName': colorName[colorIndex], 'colorRGB': colorRGB[colorIndex], 'boxArea': bboxes[i],
                             'shape': shape}
        masks = np.array((zeros_mask + masks))

    alpha = 1
    # 第二张图片透明度
    beta = 0.7
    gamma = 0
    # cv2.addWeighted 将原始图片 与 mask 融合
    mask_img = cv2.addWeighted(image, alpha, masks, beta, gamma)
    img_name = img_info[int(img_info.rfind('/')) + 1:]
    cv2.imwrite(os.path.join('/root/autodl-tmp/attributeandsubjectandspatialalbef', img_name), mask_img)


def iou(box1, box2):
    h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
    inter = w * h
    union = area_box1 + area_box2 - inter
    iou = inter / union
    return iou


def subject_module(objectName):
    options = 'Options:' + ' '
    clsName = []
    for i in range(len(objectName)):
        option = '(' + str(chr(int(97) + i)) + ')' + str(objectName[i]) + ' '
        options = options + option
        clsName.append(objectName[i])
    return options, clsName


def relation(cropList, img_file):
    colorName = ['red', 'green', 'blue', 'purple', 'white', 'yellow', 'cyan']
    colorRGB = [[255, 0, 0], [0, 128, 0], [0, 0, 255], [155, 50, 210], [255, 255, 255], [255, 255, 0], [0, 255, 255]]
    image = cv2.imread(img_file)
    masks = np.zeros((image.shape), dtype=np.uint8)
    colorList = [None] * len(colorName)
    objectList = []
    colorOptions = []
    options = "Options: "
    ones = np.ones((image.shape), dtype=np.uint8)
    white = ones * 255
    for index in range(len(cropList)):
        zeros = np.zeros((image.shape), dtype=np.uint8)
        bbox = cropList[index]["bbox"]
        crop = cv2.imread(img_file)[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        white[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = crop
        if (index < len(colorName)):
            colorIndex = index
            shape = 'rectangle'
            colorOptions.append(colorName[index])
            zeros_mask = cv2.rectangle(zeros, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=
            (colorRGB[colorIndex][2], colorRGB[colorIndex][1], colorRGB[colorIndex][0]), thickness=-1)
            # zeros_mask = cv2.putText(zeros_mask, str(index), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        elif (index < 2 * len(colorName)):
            colorIndex = index - len(colorName)
            shape = 'circle'
            zeros_mask = cv2.ellipse(zeros, (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                                     , (int((bbox[2] - bbox[0]) / 2), int((bbox[3] - bbox[1]) / 2)), 0, 0, 360
                                     , (colorRGB[colorIndex][2], colorRGB[colorIndex][1], colorRGB[colorIndex][0]),
                                     thickness=-1)
        else:
            colorIndex = index - 2 * len(colorName)
            colorList[colorIndex] = {'colorName': colorName[colorIndex], 'colorRGB': colorRGB[colorIndex]}
            shape = 'triangle'
            pt1 = (int((bbox[0] + bbox[1]) / 2), int(bbox[1]))
            pt2 = (int(bbox[0]), int(bbox[3]))
            pt3 = (int(bbox[2]), int(bbox[3]))
            triangle_cnt = np.array([pt1, pt2, pt3])
            cv2.drawContours(image, [triangle_cnt], 0
                             , (colorRGB[colorIndex][2], colorRGB[colorIndex][1], colorRGB[colorIndex][0]), -1)
        option = '(' + str(chr(int(97) + index)) + ')' + str(colorName[colorIndex]) + ' ' + str(shape) + ' '
        options = options + option
        objectList.append \
            ({'colorName': colorName[colorIndex], 'colorRGB': colorRGB[colorIndex],
              'index': cropList[index]["index"], 'shape': shape})
        masks = np.array((zeros_mask + masks))

    alpha = 1
    # 第二张图片透明度
    beta = 0.8
    gamma = 0
    # cv2.addWeighted 将原始图片 与 mask 融合
    mask_img = cv2.addWeighted(image, alpha, masks, beta, gamma)
    mask_path = '/root/autodl-tmp/attributeandsubjectandspatialalbef/' + img_file[int(img_file.rfind('/')) + 1:]
    yuanben_path = '/root/autodl-tmp/attributeandsubjectandspatialalbef/' + "yuanben" + img_file[int(img_file.rfind('/')) + 1:]
    cv2.imwrite(os.path.join(mask_path), mask_img)
    cv2.imwrite(os.path.join(yuanben_path), image)
    return options, objectList, mask_path


def extract_center_word(text):
    # 设置CoreNLP服务器的地址和端口
    parser = CoreNLPParser(url='http://localhost:9000')

    # 对文本进行依存句法分析
    parse_tree = next(parser.raw_parse(text))

    # 查找中心词
    center_word = ''
    for subtree in parse_tree.subtrees():
        if subtree.label() == 'NP':
            for index in range(len(subtree)):
                if isinstance(subtree[index], nltk.Tree) and subtree[index].label() == 'NN':
                    if (index + 1 != len(subtree)) and (
                            isinstance(subtree[index + 1], nltk.Tree) and subtree[index + 1].label() == 'NN'):
                        center_word = center_word + subtree[index][0] + ' '
                    else:
                        center_word = center_word + subtree[index][0] + ' '
            if center_word:
                break

    return center_word


def softmax(x):
    # 计算每行的最大值
    row_max = np.max(x)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x = x - row_max
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp)
    s = x_exp / x_sum
    return s


#
# from scipy.stats import multivariate_normal

def sum_recu(tree):
    text = ''
    for subtree in tree:
        if isinstance(subtree, str):
            if (text == ''):
                text = subtree
            else:
                text = text + ' ' + subtree
        elif (len(subtree) == 1):
            if isinstance(subtree[0], str):
                text = text + ' ' + subtree[0]
            else:
                text = text + ' ' + sum_recu(subtree[0])
        else:
            text = text + ' ' + sum_recu(subtree)
    return text


def blur_demo(image, mask_path):  # 均值模糊（去噪）
    dst = cv2.blur(image, (20, 20))  # 卷积层 5行5列
    # cv2.imshow("blur_demo", dst)  # 生成目标
    cv2.imwrite(os.path.join(mask_path), dst)


def extract_center_word(text):
    # 设置CoreNLP服务器的地址和端口
    parser = CoreNLPParser(url='http://localhost:9000')
    # 对文本进行依存句法分析
    parse_tree = next(parser.raw_parse(text))
    center_word = ''
    for subtree in parse_tree.subtrees():
        if subtree.label() == 'NP':
            center_word = sum_recu(subtree)
            # for index in range(len(subtree)):
            #     if isinstance(subtree[index], nltk.Tree) and subtree[index].label() == 'NN':
            #         if(index+1!=len(subtree)) and (isinstance(subtree[index+1], nltk.Tree) and subtree[index+1].label() == 'NN'):
            #             center_word = center_word + subtree[index][0] + ' '
            #         else:
            #             center_word = center_word + subtree[index][0]
            #             return center_word
    return center_word


def extract_center_phrase(text):
    # 设置CoreNLP服务器的地址和端口
    parser = CoreNLPParser(url='http://localhost:9000')
    # 对文本进行依存句法分析
    parse_tree = next(parser.raw_parse(text))

    # 查找中心词
    subtreeList = []
    for subtree in parse_tree.subtrees():
        if (len(subtree) > 1):
            for index in range(len(subtree)):
                subtreeList.append(subtree[index])
            break

    textList = []
    for subtree in subtreeList:
        text_item = sum_recu(subtree)
        text_item = re.sub(r"\s+", " ", text_item)
        text_item = text_item.strip()
        textList.append(text_item)

    center_word_list = []
    sentence_list = []
    for textItem in textList:
        center_word = extract_center_word(textItem)
        center_word = re.sub(r"\s+", " ", center_word)
        center_word = center_word.strip()
        if (center_word != ''):
            center_word_list.append(center_word)
            sentence_list.append(textItem)

    nounnum = len(sentence_list)
    return center_word_list, nounnum


def cal_sigma(dmax, edge_value):
    return sqrt(- pow(dmax, 2) / log(edge_value))


def gaussian(array_like_hm, mean, sigma):
    """modifyed version normal distribution pdf, vector version"""
    array_like_hm -= mean
    x_term = array_like_hm[:, 0] ** 2
    y_term = array_like_hm[:, 1] ** 2
    exp_value = - (x_term + y_term) / 2 / pow(sigma, 2)
    return np.exp(exp_value)


def draw_heatmap(width, height, x, y, sigma, array_like_hm):
    m1 = (x, y)
    s1 = np.eye(2) * pow(sigma, 2)
    zz = gaussian(array_like_hm, m1, sigma)
    img = zz.reshape((height, width))
    return img


def test(width, height, x, y, array_like_hm):
    if (width < height):
        dmax = width
    else:
        dmax = height
    # dmax = dmax * 0.75
    edge_value = 0.01
    sigma = cal_sigma(dmax, edge_value)

    return draw_heatmap(width, height, x, y, sigma, array_like_hm)

class Executor:
    def __init__(self, device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None) -> None:
        IMPLEMENTED_METHODS = ["crop", "blur", "shade"]
        if any(m not in IMPLEMENTED_METHODS for m in box_representation_method.split(",")):
            raise NotImplementedError
        IMPLEMENTED_AGGREGATORS = ["max", "sum"]
        if method_aggregator not in IMPLEMENTED_AGGREGATORS:
            raise NotImplementedError
        self.box_representation_method = box_representation_method
        self.method_aggregator = method_aggregator
        self.enlarge_boxes = enlarge_boxes
        self.device = device
        self.expand_position_embedding = expand_position_embedding
        self.square_size = square_size
        self.blur_std_dev = blur_std_dev
        self.cache_path = cache_path

    def preprocess_image(self, image: Image) -> List[torch.Tensor]:
        return [preprocess(image) for preprocess in self.preprocesses]

    def preprocess_text(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def tensorize_inputs(self, caption: str, cropList: Image, type: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        images = []
        for preprocess in self.preprocesses:
            images.append([])
        if(type == 'crop'):
            for index in range(len(cropList)):
                crop_path = cropList[index]["crop_path"]
                crop_images = self.preprocess_image(Image.open(crop_path))
                for j, img in enumerate(crop_images):
                    images[j].append(img.to(self.device))
            imgs = [torch.stack(image_list) for image_list in images]
        if (type == 'reverse_mask'):
            for index in range(len(cropList)):
                crop_path = cropList[index]["mask_path"]
                crop_images = self.preprocess_image(Image.open(crop_path))
                for j, img in enumerate(crop_images):
                    images[j].append(img.to(self.device))
            imgs = [torch.stack(image_list) for image_list in images]

        text_tensor = self.preprocess_text(caption.lower()).to(self.device)
        return imgs, text_tensor

    @torch.no_grad()
    def __call__(self, caption: str, cropList: Image, type: str) -> torch.Tensor:
        images, text_tensor = self.tensorize_inputs(caption, cropList, type)
        all_logits_per_image = []
        all_logits_per_text = []
        box_representation_methods = self.box_representation_method.split(',')
        caption_hash = hashlib.md5(caption.encode('utf-8')).hexdigest()
        for model, images_t, model_name in zip(self.models, images, self.model_names):
            if self.cache_path is not None:
                text_cache_path = os.path.join(self.cache_path, model_name, "text"+("_shade" if self.box_representation_method == "shade" else ""))
            image_features = None
            text_features = None
            if self.cache_path is not None and os.path.exists(os.path.join(self.cache_path, model_name)):
                if os.path.exists(os.path.join(text_cache_path, caption_hash+".pt")):
                    text_features = torch.load(os.path.join(text_cache_path, caption_hash+".pt"), map_location=self.device)
                if os.path.exists(os.path.join(self.cache_path, model_name, image_name)):
                    if all([os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")) for method_name in box_representation_methods]):
                        image_features = []
                        for method_name in box_representation_methods:
                            features = torch.load(os.path.join(self.cache_path, model_name, image_name, method_name+".pt"), map_location=self.device)
                            image_features.append(torch.stack([
                                features[(box.x, box.y, box.w, box.h)]
                                for box in boxes
                            ]))
                        image_features = torch.stack(image_features)
                        image_features = image_features.view(-1, image_features.shape[-1])
            logits_per_image, logits_per_text, image_features, text_features = self.call_model(model, images_t, text_tensor, image_features=image_features, text_features=text_features)
            all_logits_per_image.append(logits_per_image)
            all_logits_per_text.append(logits_per_text)
            if self.cache_path is not None and image_name is not None and image_features is not None:
                image_features = image_features.view(len(box_representation_methods), len(boxes), image_features.shape[-1])
                if not os.path.exists(os.path.join(self.cache_path, model_name, image_name)):
                    os.makedirs(os.path.join(self.cache_path, model_name, image_name))
                for i in range(image_features.shape[0]):
                    method_name = box_representation_methods[i]
                    if not os.path.exists(os.path.join(self.cache_path, model_name, image_name, method_name+".pt")):
                        image_features_dict = {(box.x, box.y, box.w, box.h): image_features[i,j,:].cpu() for j, box in enumerate(boxes)}
                        torch.save(image_features_dict, os.path.join(self.cache_path, model_name, image_name, method_name+".pt"))
            if self.cache_path is not None and not os.path.exists(os.path.join(text_cache_path, caption_hash+".pt")) and text_features is not None:
                assert text_features.shape[0] == 1
                if not os.path.exists(text_cache_path):
                    os.makedirs(text_cache_path)
                torch.save(text_features.cpu(), os.path.join(text_cache_path, caption_hash+".pt"))

        all_logits_per_image = torch.stack(all_logits_per_image).sum(0)
        all_logits_per_text = torch.stack(all_logits_per_text).sum(0)
        if self.method_aggregator == "max":
            all_logits_per_text = all_logits_per_text.view(-1, len(cropList)).max(dim=0, keepdim=True)[0]
        elif self.method_aggregator == "sum":
            all_logits_per_text = all_logits_per_text.view(-1, len(cropList)).sum(dim=0, keepdim=True)

        result_partial = all_logits_per_text.view(-1)
        result_partial = result_partial.float()
        candidate_indices = [i for i in range(len(cropList))]
        result = torch.zeros((len(cropList))).to(result_partial.device)
        result[candidate_indices] = result_partial.softmax(dim=-1)
        return result

class AlbefExecutor(Executor):
    def __init__(self, checkpoint_path: str, config_path: str, max_words: int = 30, device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", mode: str = "itm", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None) -> None:
        super().__init__(device, box_representation_method, method_aggregator, enlarge_boxes, expand_position_embedding, square_size, blur_std_dev, cache_path)
        if device == "cpu":
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            checkpoint = torch.load(checkpoint_path)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        self.image_res = config["image_res"]

        bert_model_name = "./bert"
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model_names = ["albef_"+mode]


        model = ALBEF(config=config, text_encoder=bert_model_name, tokenizer=self.tokenizer)
        model = model.to(self.device)

        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        if 'visual_encoder_m.pos_embed':
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

        model.eval()
        model.logit_scale = 1./model.temp
        self.models = torch.nn.ModuleList(
            [
                model
            ]
        )
        self.image_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]
        )
        self.preprocesses = [self.image_transform]
        self.max_words = max_words
        self.mode = mode

    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        if "shade" in self.box_representation_method:
            modified_text = pre_caption(text+" is in red color.", self.max_words)
        else:
            modified_text = pre_caption(text, self.max_words)
        text_input = self.tokenizer(modified_text, padding='longest', return_tensors="pt")
        sep_mask = text_input.input_ids == self.tokenizer.sep_token_id
        text_input.attention_mask[sep_mask] = 0
        return text_input

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: Dict[str, torch.Tensor], image_features: torch.Tensor = None, text_features: torch.Tensor = None) -> torch.Tensor:
        image_feat = image_features
        text_feat = text_features
        if self.mode == "itm":
            image_embeds = model.visual_encoder(images)
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(images.device)
            output = model.text_encoder(
                text.input_ids,
                attention_mask = text.attention_mask,
                encoder_hidden_states = image_embeds,
                encoder_attention_mask = image_atts,
                return_dict = True,
            )
            vl_embeddings = output.last_hidden_state[:,0,:]
            vl_output = model.itm_head(vl_embeddings)
            logits_per_image = vl_output[:,1:2]
            logits_per_text = logits_per_image.permute(1, 0)
            image_feat = None
            text_feat = None
        else:
            if image_feat is None:
                image_embeds = model.visual_encoder(images, register_blk=-1)
                image_feat = torch.nn.functional.normalize(model.vision_proj(image_embeds[:,0,:]),dim=-1)
            if text_feat is None:
                text_output = model.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                                 return_dict = True, mode = 'text')
                text_embeds = text_output.last_hidden_state
                text_feat = torch.nn.functional.normalize(model.text_proj(text_embeds[:,0,:]),dim=-1)
            sim = image_feat@text_feat.t()/model.temp
            logits_per_image = sim
            logits_per_text = sim.t()
        return logits_per_image, logits_per_text, image_feat, text_feat

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
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.5):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        objects = []
        objectName = []
        max_score = 0
        for i in range(len(bboxes)):
            score = scores[i]
            if (max_score < score):
                max_score = score
                max_object = [
                    {"bbox": bboxes[i], "cls": cls[i], "cls_name": self.cls_names[int(cls[i])], "cls_conf": score}]
            if score < cls_conf:
                continue
            objectName.append(self.cls_names[int(cls[i])])
            objects.append(
                {"bbox": bboxes[i], "cls": cls[i], "cls_name": self.cls_names[int(cls[i])], "cls_conf": score})
        objectName = list(set(objectName))
        return objects, objectName, max_object

    def attribute(self, objects, img_file, subject='', subject1='', subject2=''):
        cropList = []
        img = cv2.imread(img_file)
        for index in range(len(objects)):
            bbox = objects[index]["bbox"]
            clsNameItem = objects[index]["cls_name"]
            if int(bbox[0] < 0):
                bbox[0] = 0
            if int(bbox[1] < 0):
                bbox[1] = 0
            if int(bbox[2] < 0):
                bbox[2] = 0
            if int(bbox[3] < 0):
                bbox[3] = 0

            if (subject != ''):
                if (clsNameItem == subject):
                    # crop picture
                    crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    crop_path = '/root/autodl-tmp/attributeandsubjectandspatialalbef/' + str(index) + '.png'
                    cv2.imwrite(os.path.join(crop_path), crop)
                    # Reverse blur
                    mask_img = cv2.imread(img_file)
                    ROI = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    ROIBlur = cv2.blur(ROI, (40, 40))
                    mask_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = ROIBlur
                    mask_path = '/root/autodl-tmp/attributeandsubjectandspatialalbef/' + "blur" + str(index) + '.png'
                    cv2.imwrite(os.path.join(mask_path), mask_img)
                    cropList.append({"bbox": bbox, "index": index, "crop_path": crop_path, "cls_name": clsNameItem,
                                     "mask_path": mask_path})
            else:
                # crop picture
                crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                crop_path = '/root/autodl-tmp/attributeandsubjectandspatialalbef/' + str(index) + '.png'
                cv2.imwrite(os.path.join(crop_path), crop)
                # Reverse blur
                mask_img = cv2.imread(img_file)
                ROI = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                ROIBlur = cv2.blur(ROI, (40, 40))
                mask_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = ROIBlur
                mask_path = '/root/autodl-tmp/attributeandsubjectandspatialalbef/' + "blur" + str(index) + '.png'
                cv2.imwrite(os.path.join(mask_path), mask_img)
                cropList.append({"bbox": bbox, "index": index, "crop_path": crop_path, "cls_name": clsNameItem,
                                 "mask_path": mask_path})
        return cropList

    def allobjects(self, objects, img_file):
        cropList = []
        img = cv2.imread(img_file)
        for index in range(len(objects)):
            bbox = objects[index]["bbox"]
            clsNameItem = objects[index]["cls_name"]
            if int(bbox[0] < 0):
                bbox[0] = 0
            if int(bbox[1] < 0):
                bbox[1] = 0
            if int(bbox[2] < 0):
                bbox[2] = 0
            if int(bbox[3] < 0):
                bbox[3] = 0
            # crop picture
            crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            crop_path = '/root/autodl-tmp/attributeandsubjectandspatialalbef/' + str(index) + '.png'
            cv2.imwrite(os.path.join(crop_path), crop)
            # Reverse blur
            mask_img = cv2.imread(img_file)
            ROI = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            ROIBlur = cv2.blur(ROI, (40, 40))
            mask_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = ROIBlur
            mask_path = '/root/autodl-tmp/attributeandsubjectandspatialalbef/' + "blur" + str(index) + '.png'
            cv2.imwrite(os.path.join(mask_path), mask_img)
            cropList.append({"bbox": bbox, "index": index, "crop_path": crop_path, "cls_name": clsNameItem,
                             "mask_path": mask_path})

        return cropList

    def position(self, objects, img_file, subject='', subject1_and_subject2=0, subject1='', subject2=''):
        positionList = []
        img = cv2.imread(img_file)
        for index in range(len(objects)):
            bbox = objects[index]["bbox"]
            clsNameItem = objects[index]["cls_name"]
            if int(bbox[0] < 0):
                bbox[0] = 0
            if int(bbox[1] < 0):
                bbox[1] = 0
            if int(bbox[2] < 0):
                bbox[2] = 0
            if int(bbox[3] < 0):
                bbox[3] = 0
            if (subject != ''):
                if (clsNameItem == subject):
                    positionList.append({"bbox": bbox, "index": index, "cls_name": clsNameItem})
            else:
                positionList.append({"bbox": bbox, "index": index, "cls_name": clsNameItem})
        return positionList


def image_demo(predictor):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    albef_path = "/home/instructREC/albef/"
    box_representation_method = "crop,blur"
    box_method_aggregator = 'sum'
    device = 0
    albef_mode = 'itm'
    blur_std_dev = 100
    cache_path = None
    executor = AlbefExecutor(config_path=os.path.join(albef_path, "config.yaml"),
                             checkpoint_path=os.path.join(albef_path, "checkpoint.pth"),
                             box_representation_method=box_representation_method,
                             method_aggregator=box_method_aggregator, device=device, mode=albef_mode,
                             blur_std_dev=blur_std_dev, cache_path=cache_path)
    clip_model_list_item = "albef"
    dataset_path = "/home/instructREC/data/"
    files = os.listdir(dataset_path)
    for file in files:
            subFileDirPath = dataset_path + file
            subFileDir = os.listdir(subFileDirPath)
            for subFile in subFileDir:
                subFileSecondDirPath = subFileDirPath + "/" + subFile

                annotationFilePath = subFileSecondDirPath + "/annotation/"
                annotationFileName = os.listdir(annotationFilePath)[0]
                annotationFileNamePath = annotationFilePath + annotationFileName
                data = read_annotations(annotationFileNamePath)

                globalPromptPath = subFileSecondDirPath + "/global_prompt/"
                globalPromptPathName = os.listdir(globalPromptPath)[0]
                globalPromptPathNamePath = globalPromptPath + globalPromptPathName
                subject_data1 = read_annotations_subject1(globalPromptPathNamePath)

                centerPromptPath = subFileSecondDirPath + "/center_prompt/"
                centerPromptPathName = os.listdir(centerPromptPath)[0]
                centerPromptPathNamePath = centerPromptPath + centerPromptPathName
                subject_data2 = read_annotations_subject2(centerPromptPathNamePath)

                j = 0
                for i in range(len(data)):
                    shutil.rmtree('/root/autodl-tmp/attributeandsubjectandspatialalbef/')
                    os.mkdir('/root/autodl-tmp/attributeandsubjectandspatialalbef/')
                    annotation_list = data.iloc[i]
                    img_file_item, x1, y1, x2, y2, queries = annotation_list
                    annot = [x1, y1, x2, y2]
                    img_file = "/root/autodl-tmp/train2014/" + img_file_item
                    outputs, img_info = predictor.inference(img_file)
                    objects, objectName, max_object = predictor.visual(outputs[0], img_info)
                    # 主题模块
                    subject_list1 = subject_data1.iloc[i]
                    subject_list2 = subject_data2.iloc[i]
                    _, subject1 = subject_list1
                    _, _, subject2 = subject_list2
                    # 位置模块
                    # 位置词
                    position_list = ["left", "right", "middle", "front", "behind", "bottom", "upper", "next", "near",
                                     "between"]
                    queries_list = queries.strip().split(" ")
                    # 介词词
                    preposition = ([word for word, tag in pos_tag(queries_list) if tag in ["IN", "TO", "JJR", "JJS"]])
                    position_text = queries.strip().split(" ")
                    options, cls = subject_module(objectName)
                    for queryitem in queries_list:
                        if (queryitem in position_list) and (queryitem in position_text):
                            position_flag = 1
                            position_text.remove(queryitem)
                        if (queryitem in cls) and (queryitem in position_text):
                            position_text.remove(queryitem)
                        if (queryitem in preposition) and (queryitem in position_text):
                            position_text.remove(queryitem)

                    if (pd.isna(subject1)) or (pd.isna(subject2)):
                        subject = ''
                    elif (subject1 != subject2):
                        subject = ''
                    else:
                        subject = subject1

                    if (len(objects) == 0):
                        objects = max_object

                    cropList = predictor.attribute(objects, img_file, subject, subject1, subject2)
                    objectList = predictor.allobjects(objects, img_file)

                    attribute_flag = 0
                    if (len(cropList) > 1):
                        # attribute_flag = 1
                        similarities = executor(queries, cropList, "crop")
                        similarities = similarities.tolist()
                        similarity_list = []
                        for index in range(len(similarities)):
                            similarity_list.append((similarities[index]) * 0.5)
                        mask_similarities = executor(queries, cropList, "reverse_mask")
                        mask_similarities = mask_similarities.tolist()
                        mask_similarity_list = []
                        for index in range(len(mask_similarities)):
                            mask_similarity_list.append((1 - mask_similarities[index]) * 0.5)
                        similarity_list = np.sum([similarity_list, mask_similarity_list], axis=0).tolist()
                        max = 0
                        for index in range(len(similarity_list)):
                            if (similarity_list[index] > max):
                                max = similarity_list[index]
                                max_predict = cropList[index]["bbox"]
                    else:
                        if (len(objects) == 1):
                            max_predict = max_object[0]["bbox"]
                        else:
                            max_predict = cropList[0]["bbox"]

                    # 进入位置模块的条件
                    postionflag = 0
                    flag_conbination_words = 0
                    xflag = 0
                    yflag = 0
                    if (position_flag == 1):
                        location_objects = objectList
                        positionList = predictor.position(objects, img_file, subject, subject1, subject2)
                        if (len(positionList) > 1):
                            img = cv2.imread(img_file)
                            height, width = img.shape[:2]
                            if (queries.find('left') != -1) and (queries.find('right') == -1) and (
                                    queries.find('middle') == -1) and (queries.find('front') == -1) and (
                                    queries.find('behind') == -1) and (queries.find('bottom') == -1) and (
                                    queries.find('upper') == -1):
                                postionflag = 1
                                xflag = 1
                                x_direction = int(0)
                                y_direction = int(height / 2)
                            if (queries.find('left') != -1) and (queries.find('bottom') != -1) and (
                                    queries.find('right') == -1) and (queries.find('middle') == -1) and (
                                    queries.find('front') == -1) and (queries.find('behind') == -1) and (
                                    queries.find('upper') == -1):
                                postionflag = 1
                                x_direction = int(0)
                                y_direction = int(height)
                            if (queries.find('left') != -1) and (queries.find('upper') != -1) and (
                                    queries.find('right') == -1) and (queries.find('middle') == -1) and (
                                    queries.find('front') == -1) and (queries.find('behind') == -1) and (
                                    queries.find('bottom') == -1):
                                postionflag = 1
                                x_direction = int(0)
                                y_direction = int(height)
                            if (queries.find('right') != -1) and (queries.find('left') == -1) and (
                                    queries.find('middle') == -1) and (queries.find('front') == -1) and (
                                    queries.find('behind') == -1) and (queries.find('bottom') == -1) and (
                                    queries.find('upper') == -1):
                                postionflag = 1
                                xflag = 1
                                x_direction = int(width)
                                y_direction = int(height / 2)
                            if (queries.find('right') != -1) and (queries.find('bottom') != -1) and (
                                    queries.find('left') == -1) and (queries.find('middle') == -1) and (
                                    queries.find('front') == -1) and (queries.find('behind') == -1) and (
                                    queries.find('upper') == -1):
                                postionflag = 1
                                x_direction = int(width)
                                y_direction = int(height)
                            if (queries.find('right') != -1) and (queries.find('upper') != -1) and (
                                    queries.find('right') == -1) and (queries.find('middle') == -1) and (
                                    queries.find('front') == -1) and (queries.find('behind') == -1) and (
                                    queries.find('bottom') == -1):
                                postionflag = 1
                                x_direction = int(width)
                                y_direction = int(0)
                            if (queries.find('middle') != -1) and (queries.find('right') == -1) and (
                                    queries.find('left') == -1) and (queries.find('front') == -1) and (
                                    queries.find('behind') == -1) and (queries.find('bottom') == -1) and (
                                    queries.find('upper') == -1):
                                postionflag = 1
                                x_direction = int(width / 2)
                                y_direction = int(height / 2)
                            if (queries.find('next to') != -1) and (queries.find('middle') == -1) and (
                                    queries.find('right') == -1) and (
                                    queries.find('left') == -1) and (queries.find('front') == -1) and (
                                    queries.find('behind') == -1) and (queries.find('bottom') == -1) and (
                                    queries.find('upper') == -1):
                                postionflag = 1
                                location_query = queries[queries.rfind('next to') + 8: len(queries)]
                                location_similarities = executor(location_query, objectList, "crop")
                                location_similarities = location_similarities.tolist()
                                location_max = 0
                                for index in range(len(location_similarities)):
                                    if (location_similarities[index] > location_max):
                                        flag_conbination_words = 1
                                        location_max = location_similarities[index]
                                        location_max_predict = location_objects[index]["bbox"]
                                        x_direction = int((location_max_predict[0] + location_max_predict[2]) / 2)
                                        y_direction = int((location_max_predict[1] + location_max_predict[3]) / 2)
                            if (queries.find('near') != -1) and (queries.find('middle') == -1) and (
                                    queries.find('right') == -1) and (
                                    queries.find('left') == -1) and (queries.find('front') == -1) and (
                                    queries.find('behind') == -1) and (queries.find('bottom') == -1) and (
                                    queries.find('upper') == -1 and queries.find('next') == -1):
                                postionflag = 1
                                location_query = queries[queries.rfind('near') + 8: len(queries)]
                                location_similarities = executor(location_query, objectList, "crop")
                                location_similarities = location_similarities.tolist()
                                location_max = 0
                                for index in range(len(location_similarities)):
                                    if (location_similarities[index] > location_max):
                                        flag_conbination_words = 1
                                        location_max = location_similarities[index]
                                        location_max_predict = location_objects[index]["bbox"]
                                        x_direction = int((location_max_predict[0] + location_max_predict[2]) / 2)
                                        y_direction = int((location_max_predict[1] + location_max_predict[3]) / 2)
                            if (queries.find('front') != -1) and (queries.find('right') == -1) and (
                                    queries.find('left') == -1) and (queries.find('middle') == -1) and (
                                    queries.find('behind') == -1) and (queries.find('bottom') == -1) and (
                                    queries.find('upper') == -1):
                                yflag = 1
                                postionflag = 1
                                x_direction = int(width / 2)
                                y_direction = int(height)
                            if (queries.find('bottom') != -1) and (queries.find('right') == -1) and (
                                    queries.find('left') == -1) and (queries.find('middle') == -1) and (
                                    queries.find('behind') == -1) and (queries.find('front') == -1) and (
                                    queries.find('upper') == -1):
                                yflag = 1
                                postionflag = 1
                                x_direction = int(width / 2)
                                y_direction = int(height)
                            if (queries.find('between') != -1) and (queries.find('and') != -1) and (
                                    queries.find('right') == -1) and (
                                    queries.find('left') == -1) and (queries.find('middle') == -1) and (
                                    queries.find('front') == -1) and (queries.find('bottom') == -1) and (
                                    queries.find('upper') == -1) and (queries.find('bottom') == -1):
                                location_query = queries[queries.rfind('between') + 8: len(queries)]
                                if (location_query.find('and') != -1):
                                    location_sub_query1 = location_query[0:location_query.rfind('and')]
                                    location_sub_query2 = location_query[location_query.rfind('and') + 4: len(queries)]
                                    postionflag = 1
                                    location_similarities = executor(location_sub_query1, objectList, "crop")
                                    location_similarities = location_similarities.tolist()
                                    location_max = 0
                                    for index in range(len(location_similarities)):
                                        if (location_similarities[index] > location_max):
                                            location_max = location_similarities[index]
                                            location_max_predict = location_objects[index]["bbox"]
                                            x_direction1 = int((location_max_predict[0] + location_max_predict[2]) / 2)
                                            y_direction1 = int((location_max_predict[1] + location_max_predict[3]) / 2)
                                    location_objects = predictor.attribute(objects, img_file, '')
                                    location_similarities = executor(location_sub_query2, objectList, "crop")
                                    location_similarities = location_similarities.tolist()
                                    location_max = 0
                                    for index in range(len(location_similarities)):
                                        if (location_similarities[index] > location_max):
                                            location_max = location_similarities[index]
                                            location_max_predict = location_objects[index]["bbox"]
                                            x_direction2 = int((location_max_predict[0] + location_max_predict[2]) / 2)
                                            y_direction2 = int((location_max_predict[1] + location_max_predict[3]) / 2)
                                    x_direction = int(x_direction1 + x_direction2) / 2
                                    y_direction = int(y_direction1 + y_direction2) / 2

                            if (queries.find('behind') != -1) and (queries.find('right') == -1) and (
                                    queries.find('left') == -1) and (queries.find('middle') == -1) and (
                                    queries.find('front') == -1) and (queries.find('bottom') == -1) and (
                                    queries.find('upper') == -1):
                                yflag = 1
                                postionflag = 1
                                x_direction = int(width) / 2
                                y_direction = int(0)

                            if (postionflag == 1):
                                x = np.arange(width, dtype=float)
                                y = np.arange(height, dtype=float)
                                xx, yy = np.meshgrid(x, y)
                                xxyy = np.c_[xx.ravel(), yy.ravel()]
                                img = test(width, height, x_direction, y_direction, xxyy.copy())
                                position_posibility_list = []
                                max_position_posibility = 0
                                for index in range(len(positionList)):
                                    box = positionList[index]["bbox"]
                                    box_x = int((box[0] + box[2]) / 2)
                                    box_y = int((box[1] + box[3]) / 2)
                                    if (xflag == 1):
                                        box_y = int(y_direction)
                                    if (yflag == 1):
                                        box_x = int(x_direction)
                                    position_posibility = img[box_y][box_x]
                                    if (flag_conbination_words == 1):
                                        location_max_predict_iou = iou(location_max_predict, box)
                                        if (location_max_predict_iou > 0.6):
                                            position_posibility = 0
                                    if (position_posibility > max_position_posibility):
                                        max_position_posibility = position_posibility
                                        max_predict = positionList[index]["bbox"]
                                    position_posibility_list.append(position_posibility)
                                position_posibility_list = softmax(position_posibility_list)
                                position_posibility_list = position_posibility_list.tolist()
                                max_value = 0
                                if (attribute_flag == 1):
                                    position_posibility_list = [i * 0.5 for i in position_posibility_list]
                                    similarity_list = [i * 0.5 for i in similarity_list]
                                    sum = np.sum([position_posibility_list, similarity_list], axis=0).tolist()
                                else:
                                    sum = position_posibility_list
                                for index in range(len(sum)):
                                    if (sum[index] > max_value):
                                        max_value = sum[index]
                                        max_predict = positionList[index]["bbox"]

                    iouValue = iou(max_predict, annot)
                    flag = "错误"
                    if iouValue > 0.5:
                        j = j + 1
                        flag = "正确"
                    print(j, i, j / (i + 1))
                    pre = j / (i + 1)
                    path = "/root/autodl-tmp/result/attributeandsubjectandspatialalbef/" + file + "_attributeandsubjectandspatialalbef_" + clip_model_list_item + '' + annotationFileName
                    fo = open(path, "a")
                    fo.write(str(queries) + ',' + str(flag) + ',' + str(pre) + "\n")

                attribute_result_path = "/root/autodl-tmp/result/attributeandsubjectandspatialalbef/"+"attributeandsubjectandspatialalbef"+clip_model_list_item+".csv"
                attribute_result = open(attribute_result_path,"a")
                attribute_result.write(str(annotationFileNamePath) + ',' + str(pre) + "\n")



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


def read_annotations_subject1(trn_file):
    trn_data = pd.read_csv(trn_file)
    trn_data['query'] = trn_data['query']
    trn_data['category'] = trn_data['category']
    trn_df = trn_data[['query', 'category']]
    return trn_df


def read_annotations_subject2(trn_file):
    trn_data = pd.read_csv(trn_file)
    trn_data['query'] = trn_data['query']
    trn_data['subject'] = trn_data['subject']
    trn_data['category'] = trn_data['category']
    trn_df = trn_data[['query', 'subject', 'category']]
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
