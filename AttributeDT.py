import clip
import json
import os
import shutil
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFilter
import cv2

# def attribute(objects, img_file):
#     cropList = []
#     img = Image.open(img_file)
#     for index in range(len(objects)):
#         bbox = objects[index]["bbox"]
#         if int(bbox[0]) < 0:
#             bbox[0] = 0
#         if int(bbox[1]) < 0:
#             bbox[1] = 0
#         if int(bbox[2]) < 0:
#             bbox[2] = 0
#         if int(bbox[3]) < 0:
#             bbox[3] = 0
#         # crop picture
#         crop = img.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
#
#         # Reverse blur
#         mask_img = Image.open(img_file)
#         ROI = crop.filter(ImageFilter.GaussianBlur(radius=100))
#         mask_img.paste(ROI, (int(bbox[0]), int(bbox[1])))
#         cropList.append({"bbox": bbox, "index": index, "crop_path": crop, "mask_path": mask_img})
#     return cropList

def iou(box1,box2):
    h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
    inter = w * h
    union = area_box1 + area_box2 - inter
    iou = inter / union
    return iou

def attribute(objects, img_file):
        cropList = []
        img = cv2.imread(img_file)
        for index in range(len(objects)):
            bbox = objects[index]["bbox"]
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
            crop_path = '/root/autodl-tmp/cropdt/' + str(index) + '.png'
            cv2.imwrite(os.path.join(crop_path), crop)
            cropList.append({"bbox": bbox, "index": index, "crop_path": crop_path})

        return cropList
def visual(bboxes):
    objects = []
    for i in range(len(bboxes)):
        bboxes[i][0] = int(float(bboxes[i][0]))
        bboxes[i][1] = int(float(bboxes[i][1]))
        bboxes[i][2] = int(float(bboxes[i][2]))
        bboxes[i][3] = int(float(bboxes[i][3]))
        objects.append({"bbox": bboxes[i]})
    return objects

def main():
    input_file = "/home/instructREC/datadt/refcoco/refcoco_testa.jsonl"
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # loads CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    with open(input_file) as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    j=0
    i=0
    detector_file = open("/home/instructREC/datadt/refcoco/refcoco_dets_dict.json")
    detections_list = json.load(detector_file)
    if isinstance(detections_list, dict):
        detections_map = {int(image_id): detections_list[image_id] for image_id in detections_list}
    else:
        detections_map = defaultdict(list)
        for detection in detections_list:
            detections_map[detection["image_id"]].append(detection["box"])
    for datum in tqdm(data):
        # objectList = [[ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]]
        #               for ann in datum["anns"]]
        objectList = [[box[0], box[1], box[0]+box[2], box[1]+box[3]] for box in detections_map[int(datum["image_id"])]]
        if len(objectList) == 0:
            objectList = [[0,0,0,0]]
        cls_id = [ann["category_id"] for ann in datum["anns"]]
        gold_index = [i for i in range(len(datum["anns"])) if str(datum["anns"][i]["id"]) in str(datum["ann_id"])]
        img_file_item = "_".join(datum["file_name"].split("_")[:-1]) + ".jpg"
        for sentence in datum["sentences"]:
            shutil.rmtree('/root/autodl-tmp/cropdt/')
            os.mkdir('/root/autodl-tmp/cropdt/')
            queries = sentence["raw"].lower()
            img_file = "/root/autodl-tmp/train2014/" + img_file_item
            objects = visual(objectList)
            cropList = attribute(objects, img_file)
            attribute_query = queries
            # 进入属性模块的条件
            if (len(cropList) > 1):
                crop_images = preprocess(Image.open(cropList[0]["crop_path"])).unsqueeze(0).to(device)
                for index in range(1, len(cropList)):
                    crop_path = cropList[index]["crop_path"]
                    crop_image = preprocess(Image.open(crop_path)).unsqueeze(0).to(device)
                    crop_images = torch.cat([crop_images, crop_image], dim=0)
                text = clip.tokenize([attribute_query]).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(crop_images)
                    text_features = clip_model.encode_text(text)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarities = (100.0 * image_features @ text_features.T).softmax(dim=0)
                similarities = similarities.tolist()
                similarity_list = []
                for index in range(len(similarities)):
                    similarity_list.append(similarities[index][0])
                similarity_list = np.sum([similarity_list], axis=0).tolist()
                max = 0
                for index in range(len(similarity_list)):
                    if (similarity_list[index] > max):
                        max = similarity_list[index]
                        max_predict = cropList[index]["bbox"]
            else:
                if (len(objects) == 1):
                    max_predict = cropList[0]["bbox"]
                else:
                    max_predict = cropList[0]["bbox"]

            # iouValue = iou(max_predict, annot)
            # flag = "错误"
            # for g_index in gold_index:
            #     if iou(max_predict, objectList[g_index]) > 0.5:
            #         j = j + 1
            #         flag = "正确"
            #         break
            argmax_ious = []
            max_ious = []
            for g_index in gold_index:
                ious = [iou(box, objectList[g_index]) for box in boxes]
                argmax_iou = -1
                max_iou = 0
                if max(ious) >= 0.5:
                    for index, value in enumerate(ious):
                        if value > max_iou:
                            max_iou = value
                            argmax_iou = index
                argmax_ious.append(argmax_iou)
                max_ious.append(max_iou)
            argmax_iou = -1
            max_iou = 0
            if max(max_ious) >= 0.5:
                for index, value in zip(argmax_ious, max_ious):
                    if value > max_iou:
                        max_iou = value
                        argmax_iou = index
            # if iouValue > 0.5:
            #     j = j + 1
            #     flag = "正确"
            print(j, i, j / (i + 1))
            i=i+1
            pre = j / (i + 1)
            path = "/root/autodl-tmp/result/cropdt/" + "refcoco" + "_attribute_11_19_" + "refcocotestagtt"
            fo = open(path, "a")
            fo.write(str(queries) + ',' + str(flag) + ',' + str(pre) + "\n")

if __name__ == "__main__":
    main()