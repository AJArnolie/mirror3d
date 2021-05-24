import sys
import os
import time
import json
import random
import time
import numpy as np
import cv2
import math
from PIL import Image
import matplotlib.pyplot as plt
from utils.algorithm import *
import shutil

def read_plane_json(json_path):
    with open(json_path, 'r') as j:
        info = json.loads(j.read())
    plane_info = dict()
    for item in info:
        plane_info[item["mask_id"]] = dict()
        plane_info[item["mask_id"]]["plane_parameter"] = item["plane"]
        plane_info[item["mask_id"]]["mirror_normal"] = item["normal"]
    return plane_info

def check_converge(score_list=[], check_freq=2, change_ratio_threshold=0.03, logger=None):
    if logger:
        logger.info("######################### check_converge {} #########################".format(len(score_list)))
    print("######################### check_converge {} #########################".format(len(score_list)))
    if len(score_list) < check_freq*2:
        return False

    check_back_loss = score_list[-check_freq*2:-check_freq]
    check_forward_loss = score_list[-check_freq:]
    change_ratio =(np.abs(np.average(check_forward_loss) - np.average(check_back_loss)))/np.average(check_back_loss) 
    print("######################### change_ratio {} #########################".format(change_ratio))
    if logger:
        logger.info("######################### change_ratio {} #########################".format(change_ratio))
    if change_ratio <= change_ratio_threshold: 
        return True
    else:
        return False


def list_diff(list1, list2):
    """
    Get a list exist in list1 but don't exisit in list2
    """
    out = []
    for ele in list1:
        if not ele in list2:
            out.append(ele)
    return out

def center_crop_image(input_folder="", output_folder = "", new_w=608, new_h=456, ori_w=640, ori_h=480):
    """
    Center crop the image
    Args:
        input_folder : folder path; folder contains the image that needs to be cropped
        output_folder : folder path to saved the cropped result
    """
    os.makedirs(output_folder, exist_ok=True)
    w_border = int((ori_w - new_w)/2)
    h_border = int((ori_h - new_h)/2)
    for one_img in os.listdir(input_folder):
        one_img_path = os.path.join(input_folder, one_img)
        one_img_save_path = os.path.join(output_folder, one_img)
        h, w, _ = cv2.imread(one_img_path).shape
        if h == new_h and w == new_w:
            shutil.copy(one_img_path, one_img_save_path)
            continue
        try:
            if one_img_path.find("depth") > 0:
                ori_img = cv2.imread(one_img_path, cv2.IMREAD_ANYDEPTH) 
                ori_img = ori_img[h_border:h_border+new_h, w_border:w_border+new_w]
            else:
                ori_img = cv2.imread(one_img_path)
                ori_img = ori_img[h_border:h_border+new_h, w_border:w_border+new_w]
        except:
            print(print("error: ", one_img_path))
            continue
        cv2.imwrite(one_img_save_path, ori_img)
    print("corpped image saved to {}".format(output_folder))


def save_html(save_path, content):
    with open(save_path, "w") as outf:
        outf.write(str(content))
    print("html saved to {}".format(save_path))

def save_plane_parameter_2_json(plane_parameter, one_plane_para_save_path, instance_index):
    sample_id = '%02x%02x%02x' % (instance_index[0],instance_index[1],instance_index[2]) # BGR
    if os.path.exists(one_plane_para_save_path):
        with open(one_plane_para_save_path, 'r') as j:
            img_info = json.loads(j.read())
    else:
        img_info = []
    one_info = dict()
    one_info["plane"] = list(plane_parameter)
    one_info["normal"] = list(unit_vector(list(plane_parameter[:-1])))
    one_info["mask_id"] = sample_id
    img_info.append(one_info)
    save_json(one_plane_para_save_path,img_info)

def get_all_fileAbsPath_under_folder(folder_path):
    file_path_list = []
    for root, dirs, files in os.walk(os.path.abspath(folder_path)):
        for file in files:
            file_path_list.append(os.path.join(root, file))
    return file_path_list

def read_json(json_path):
    with open(json_path, 'r') as j:
        info = json.loads(j.read())
    return info

def read_txt(txt_path):
    with open(txt_path, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def save_txt(save_path, data):
    with open(save_path, "w") as file:
        for info in data:
            file.write(str(info))
            file.write("\n")
    print("txt saved to : ", save_path, len(data))

def save_json(save_path,data):
    out_json = json.dumps(data, sort_keys=False, indent=4, separators=(',', ':'),
                          ensure_ascii=False)
    with open(save_path, "w") as fo:
        fo.write(out_json)
        fo.close()
        print("json file saved to : ",save_path )

def rreplace(s, old, new):
    li = s.rsplit(old, 1)
    return new.join(li)


def get_compose_image(output_save_path, img_list, mini_img_w=320, mini_img_h=240, mini_image_per_row=9):
    """
    Args:
        img_list : Image Array
        output_save_path : composed image saved path
    """

    def add_margin(pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    image_col = math.ceil(len(img_list)/mini_image_per_row) 
    to_image = Image.new('RGB', (mini_image_per_row * mini_img_w, image_col * mini_img_h)) 

    for y in range(1, image_col + 1):
        for x in range(1, mini_image_per_row + 1):
            img_index = mini_image_per_row * (y - 1) + x - 1
            from_image = img_list[img_index].resize((mini_img_w, mini_img_h),Image.ANTIALIAS)
            from_image = add_margin(from_image, 20,20,20,20,(255,255,255))
            to_image.paste(from_image, ((x - 1) * mini_img_w, (y - 1) * mini_img_h))
    to_image.save(output_save_path) 
    print("image saved to :", output_save_path)


def nth_replace(s, sub, repl, n):
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find+len(sub):]
    return s

def save_heatmap_no_border(image, save_path=""):
    """ 
    Save heatmap with no border
    Args:
        image : M * N image 
    """
    plt.figure()
    fig = plt.imshow(image, cmap=plt.get_cmap("magma"))
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    figure = plt.gcf()
    plt.savefig(save_path, bbox_inches='tight', pad_inches = 0, dpi=100)
    print("image saved to : {}".format(save_path))



def get_filtered_percantage(dataset="m3d"):
    from tqdm import tqdm
    data_main_path = "/project/3dlg-hcvc/mirrors/www/Mirror3D_final/{}".format(dataset)
    test_json = "/project/3dlg-hcvc/mirrors/www/Mirror3D_final/{}/with_mirror/precise/network_input_json/test_10_normal_mirror.json".format(dataset)
    test_info = read_json(test_json)

    ref_mirror_area_filtered = []
    raw_mirror_area_filtered = []

    ref_none_mirror_filtered = []
    raw_none_mirror_filtered = []

    for item in tqdm(test_info["images"]):
        ref_depth = cv2.imread(os.path.join(data_main_path, item["mesh_refined_path"]), cv2.IMREAD_ANYDEPTH)
        raw_depth = cv2.imread(os.path.join(data_main_path, item["hole_raw_path"]), cv2.IMREAD_ANYDEPTH)

        instance_mask = cv2.imread(os.path.join(data_main_path, item["img_path"].replace("raw", "instance_mask")), cv2.IMREAD_ANYDEPTH) > 0
        none_mirror_mask =  (instance_mask==False)
        ref_mirror_area_filtered.append(((ref_depth < 1e-3)*instance_mask).sum() / instance_mask.sum() )
        raw_mirror_area_filtered.append(((raw_depth < 1e-3)*instance_mask).sum() / instance_mask.sum() )

        # if ((raw_depth < 1e-3)*instance_mask).sum() / instance_mask.sum() > 0.5:
        #     sample_name = item["mesh_raw_path"].split("/")[-1]
        #     colored_depth_path = "/project/3dlg-hcvc/mirrors/www/cr_vis/scannet_result_vis/sensor-D/colored_pred_depth/{}".format(sample_name)
        #     color_img_path = os.path.join(data_main_path, item["img_path"])
        #     print(colored_depth_path, color_img_path)

        ref_none_mirror_filtered.append(((ref_depth < 1e-3)*none_mirror_mask).sum() / none_mirror_mask.sum() )
        raw_none_mirror_filtered.append(((raw_depth < 1e-3)*none_mirror_mask).sum() / none_mirror_mask.sum() )

    
    print("ref mirror area filtered : {}".format(np.array(ref_mirror_area_filtered).mean()))
    print("raw mirror area filtered : {}".format(np.array(raw_mirror_area_filtered).mean()))


    print("ref none-mirror area filtered : {}".format(np.array(ref_none_mirror_filtered).mean()))
    print("raw none-mirror area filtered : {}".format(np.array(raw_none_mirror_filtered).mean()))


def AP_estimate():
    import random
    AP_list = []
    for run_time in range(10):
        # initialize 
        all_TP_num = 15
        all_detect_num = 29
        all_TP_num = int(all_detect_num*0.1)
        all_GT_num = 34
        recall_max = all_TP_num/ all_GT_num

        # random spread the GT 
        GT_index_list = random.sample(range(all_detect_num), all_TP_num)
        recall_list = []
        precision_list = []
        TP_count = 0

        # get precesion list and recall list
        for index in range(all_detect_num):
            if index in GT_index_list:
                TP_count += 1
            precision_list.append(TP_count/ (index + 1))
            recall_list.append(TP_count/ all_GT_num) 
        
        start = 0
        end = 0
        AP = 0
        last_recall = 0
        # get AP; after VOC 2010; 
        # assume we have n unique recall value, rank it from low to high (r1, r2, r3 ... rn=0.438)
        # relevant precesion list (p1, p2, p3, ... pn) 
        #          p1 is a precesion list which have k element, k is the index of the last occurance of r1 in recall list
        # AP = (r2 - r1) * max(p1) + (r3 - r2) * max(p2) + ... (0.438 - rn-1) * max(pn)
        for recall_threshold in np.unique(recall_list):
            end = max(np.where(np.array(recall_list)==recall_threshold)[0])
            precision_sub_list = precision_list[start:end+1]
            AP += max(precision_sub_list)*(recall_threshold-last_recall)
            last_recall = recall_threshold
            start = max(np.where(np.array(recall_list)==recall_threshold)[0])
        AP_list.append(AP)
        print("estimate AP : ",AP)
    print("avg estimate AP : ",np.mean(AP_list))
