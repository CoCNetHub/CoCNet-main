import json
from tqdm import tqdm

def main():
    input_file = "/home/instructREC/datadt/refcoco/refcoco_testa.jsonl"
    with open(input_file) as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    for datum in tqdm(data):
        gold_boxes = [[ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]]
                      for ann in datum["anns"]]
        cls_id = [ann["category_id"] for ann in datum["anns"]]
        gold_index = [i for i in range(len(datum["anns"])) if datum["anns"][i]["id"] in datum["ann_id"]]
        for sentence in datum["sentences"]:
            shutil.rmtree('/root/autodl-tmp/attributeandsubjectandspatial/')
            os.mkdir('/root/autodl-tmp/attributeandsubjectandspatial/')
            query = sentence["raw"].lower()
            ile_name = "_".join(datum["file_name"].split("_")[:-1]) + ".jpg"
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

            attribute_query = queries
            # 进入属性模块的条件
            attribute_flag = 0
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
                    similarity_list.append(similarities[index][0] * 0.5)

                mask_images = preprocess(Image.open(cropList[0]["mask_path"])).unsqueeze(0).to(device)
                for index in range(1, len(cropList)):
                    mask_path = cropList[index]["mask_path"]
                    mask_image = preprocess(Image.open(mask_path)).unsqueeze(0).to(device)
                    mask_images = torch.cat([mask_images, mask_image], dim=0)
                mask_query = attribute_query
                mask_query = clip.tokenize([mask_query]).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(mask_images)
                    text_features = clip_model.encode_text(mask_query)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    mask_similarities = (100.0 * image_features @ text_features.T).softmax(dim=0)
                    mask_similarities = mask_similarities.tolist()
                    mask_similarity_list = []
                    for index in range(len(mask_similarities)):
                        mask_similarity_list.append((1 - mask_similarities[index][0]) * 0.5)

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
                        location_objects = predictor.attribute(objects, img_file, '')
                        crop_images = preprocess(Image.open(location_objects[0]["crop_path"])).unsqueeze(0).to(
                            device)
                        for index in range(1, len(objectList)):
                            crop_path = location_objects[index]["crop_path"]
                            crop_image = preprocess(Image.open(crop_path)).unsqueeze(0).to(device)
                            crop_images = torch.cat([crop_images, crop_image], dim=0)
                        text = clip.tokenize([location_query]).to(device)
                        with torch.no_grad():
                            image_features = clip_model.encode_image(crop_images)
                            text_features = clip_model.encode_text(text)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                            location_similarities = (100.0 * image_features @ text_features.T).softmax(dim=0)
                        location_similarities = location_similarities.tolist()
                        location_max = 0
                        for index in range(len(location_similarities)):
                            if (location_similarities[index][0] > location_max):
                                flag_conbination_words = 1
                                location_max = location_similarities[index][0]
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
                        location_objects = predictor.attribute(objects, img_file, '')
                        crop_images = preprocess(Image.open(location_objects[0]["crop_path"])).unsqueeze(0).to(
                            device)
                        for index in range(1, len(objectList)):
                            crop_path = location_objects[index]["crop_path"]
                            crop_image = preprocess(Image.open(crop_path)).unsqueeze(0).to(device)
                            crop_images = torch.cat([crop_images, crop_image], dim=0)
                        text = clip.tokenize([location_query]).to(device)
                        with torch.no_grad():
                            image_features = clip_model.encode_image(crop_images)
                            text_features = clip_model.encode_text(text)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                            location_similarities = (100.0 * image_features @ text_features.T).softmax(dim=0)
                        location_similarities = location_similarities.tolist()
                        location_max = 0
                        for index in range(len(location_similarities)):
                            if (location_similarities[index][0] > location_max):
                                flag_conbination_words = 1
                                location_max = location_similarities[index][0]
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
                            location_objects = predictor.attribute(objects, img_file, '')
                            crop_images = preprocess(Image.open(location_objects[0]["crop_path"])).unsqueeze(
                                0).to(
                                device)
                            for index in range(1, len(objectList)):
                                crop_path = location_objects[index]["crop_path"]
                                crop_image = preprocess(Image.open(crop_path)).unsqueeze(0).to(device)
                                crop_images = torch.cat([crop_images, crop_image], dim=0)
                            text = clip.tokenize([location_sub_query1]).to(device)
                            with torch.no_grad():
                                image_features = clip_model.encode_image(crop_images)
                                text_features = clip_model.encode_text(text)
                                image_features /= image_features.norm(dim=-1, keepdim=True)
                                text_features /= text_features.norm(dim=-1, keepdim=True)
                                location_similarities = (100.0 * image_features @ text_features.T).softmax(
                                    dim=0)
                            location_similarities = location_similarities.tolist()
                            location_max = 0
                            for index in range(len(location_similarities)):
                                if (location_similarities[index][0] > location_max):
                                    location_max = location_similarities[index][0]
                                    location_max_predict = location_objects[index]["bbox"]
                                    x_direction1 = int((location_max_predict[0] + location_max_predict[2]) / 2)
                                    y_direction1 = int((location_max_predict[1] + location_max_predict[3]) / 2)
                            location_objects = predictor.attribute(objects, img_file, '')
                            crop_images = preprocess(Image.open(location_objects[0]["crop_path"])).unsqueeze(
                                0).to(
                                device)
                            for index in range(1, len(objectList)):
                                crop_path = location_objects[index]["crop_path"]
                                crop_image = preprocess(Image.open(crop_path)).unsqueeze(0).to(device)
                                crop_images = torch.cat([crop_images, crop_image], dim=0)
                            text = clip.tokenize([location_sub_query2]).to(device)
                            with torch.no_grad():
                                image_features = clip_model.encode_image(crop_images)
                                text_features = clip_model.encode_text(text)
                                image_features /= image_features.norm(dim=-1, keepdim=True)
                                text_features /= text_features.norm(dim=-1, keepdim=True)
                                location_similarities = (100.0 * image_features @ text_features.T).softmax(
                                    dim=0)
                            location_similarities = location_similarities.tolist()
                            location_max = 0
                            for index in range(len(location_similarities)):
                                if (location_similarities[index][0] > location_max):
                                    location_max = location_similarities[index][0]
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
                            position_posibility_list = [i * 0.7 for i in position_posibility_list]
                            similarity_list = [i * 0.3 for i in similarity_list]
                            sum = np.sum([position_posibility_list, similarity_list], axis=0).tolist()
                        else:
                            sum = position_posibility_list
                        for index in range(len(sum)):
                            if (sum[index] > max_value):
                                max_value = sum[index]
                                max_predict = positionList[index]["bbox"]
            # iouValue = iou(max_predict, annot)
            flag = "错误"
            # if iouValue > 0.5:
            #     j = j + 1
            #     flag = "正确"
            for g_index in gold_index:
                if iou(max_predict, gold_boxes[g_index]) > 0.5:
                    j = j + 1
                    flag = "正确"
                    break
            print(j, i, j / (i + 1))
            pre = j / (i + 1)
            path = "/root/autodl-tmp/result/attributeandsubjectandspatial/" + file + "_attributeandsubjectandspatialRN5016_" + annotationFileName
            fo = open(path, "a")
            fo.write(str(queries) + ',' + str(flag) + ',' + str(pre) + "\n")

        attribute_result = open(
            "/root/autodl-tmp/result/attributeandsubjectandspatial/attributeandsubjectandspatialRN5016.csv", "a")
        attribute_result.write(str(annotationFileNamePath) + ',' + str(pre) + "\n")

if __name__ == "__main__":
    main()