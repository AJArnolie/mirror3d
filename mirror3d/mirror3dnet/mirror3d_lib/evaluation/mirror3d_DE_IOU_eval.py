#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import torch
from mirror3d.utils.mirror3d_metrics import *
from mirror3d.utils.general_utils import *
from mirror3d.utils.plane_pcd_utils import *

from detectron2.data import (
    MetadataCatalog,
)

import logging
import time
import numpy as np
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.events import get_event_storage

class Mirror3DNet_Eval:

    def __init__(self, output_list, cfg):
        self.output_list = output_list
        self.cfg = cfg
        self.mean_IOU = 0
        print("########## cfg.SEED ##########", cfg.SEED)
        log_file_save_path = os.path.join(self.cfg.OUTPUT_DIR, "eval_result.log")
        logging.basicConfig(filename=log_file_save_path, filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=logging.INFO)
        self.logger = logging

        if self.cfg.TRAIN_COCO_JSON.find("nyu") or self.cfg.VAL_COCO_JSON.find("nyu") > 0:
            self.dataset_name = "nyu"
        elif self.cfg.TRAIN_COCO_JSON.find("mp3d") or self.cfg.VAL_COCO_JSON.find("mp3d") > 0:
            self.dataset_name = "mp3d"
        elif self.cfg.TRAIN_COCO_JSON.find("scannet") or self.cfg.VAL_COCO_JSON.find("scannet") > 0:
            self.dataset_name = "scannet"

    def eval_main(self):
        self.save_masked_image(self.output_list)

        # ----------- evaluate REF_DEPTH_TO_REFINE (output from init_depth_generator/ \mnet'DE branch) + Mirror3d -----------
        if self.cfg.EVAL_INPUT_REF_DEPTH and "raw" not in self.cfg.REF_MODE:
            print("eval input {} + refine ...".format(self.cfg.REF_DEPTH_TO_REFINE))
            self.refine_input_txtD_and_eval(self.output_list)
   
         # ----------- evaluate \mnet/ PlaneRCNN model's predicted depth + Mirror3d -----------
        if self.cfg.EVAL_BRANCH_REF_DEPTH:
            print("eval DE_pred + refine ...")
            self.refine_DEbranch_predD_and_eval(self.output_list) 

    def refine_input_txtD_and_eval(self, output_list):
        anchor_normal = np.load(self.cfg.ANCHOR_NORMAL_NYP)
        refine_depth_fun = RefineDepth(self.cfg.FOCAL_LENGTH, self.cfg.REF_BORDER_WIDTH, self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)
        if self.cfg.REF_DEPTH_TO_REFINE.find("SAIC") > 0:
            input_tag = "RGBD"
            method_tag = "saic + \mnet"
        elif self.cfg.REF_DEPTH_TO_REFINE.find("BTS") > 0:
            input_tag = "RGB"
            method_tag = "BTS + \mnet"
        elif self.cfg.REF_DEPTH_TO_REFINE.find("VNL") > 0:
            input_tag = "RGB"
            method_tag = "VNL + \mnet"
        elif not self.cfg.OBJECT_CLS:
            input_tag = "RGB"
            method_tag = "PlaneRCNN"
        else:
            input_tag = "RGB"
            method_tag = "\mnet"
        
        train_with_refD=None 
        if self.cfg.REF_DEPTH_TO_REFINE.find("ref") > 0:
            train_with_refD = True
        else:
            train_with_refD = False

        depth_shift = np.array(self.cfg.DEPTH_SHIFT)
        for i, item in enumerate(output_list):
            one_output, one_input = item
            instances = one_output[0][0]["instances"]
            color_img_path = one_input[0]["mirror_color_image_path"]

            pred_mask = np.zeros(instances.image_size)
            pred_mask = pred_mask.astype(bool)

            other_predD_path = one_input[0]["raw_sensorD_path"]
            depth_to_ref = cv2.imread(other_predD_path, cv2.IMREAD_ANYDEPTH)

            if instances.to("cpu").has("pred_masks"):
                for index, one_pred_mask in enumerate(instances.to("cpu").pred_masks):
                    to_refine_area = one_pred_mask.numpy().astype(bool)
                    to_refine_area = np.logical_and(pred_mask==False, to_refine_area)
                    if to_refine_area.sum() == 0:
                        continue
                    pred_mask = np.logical_or(pred_mask , one_pred_mask)
                    if instances.to("cpu").pred_anchor_classes[index] >= anchor_normal.shape[0]:
                        continue
                    
                    pred_normal = anchor_normal[instances.to("cpu").pred_anchor_classes[index]] +  instances.to("cpu").pred_residuals[index].numpy()
                    pred_normal = unit_vector(pred_normal)

                    if "border" in self.cfg.REF_MODE:
                        depth_to_ref = refine_depth_fun.refine_depth_by_mirror_border(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, depth_to_ref)
                    else:
                        depth_to_ref = refine_depth_fun.refine_depth_by_mirror_area(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, depth_to_ref)
            depth_to_ref[depth_to_ref<0] = 0
            if self.cfg.EVAL_SAVE_DEPTH:
                refined_input_txt_output_folder = os.path.join(self.cfg.VAL_IMG_ROOT, "refined_depth", "refined_input_txt_pred_depth")
                os.makedirs(refined_input_txt_output_folder, exist_ok=True)
                pred_depth_scaled = (np.array(depth_to_ref/self.cfg.DEPTH_SHIFT) * depth_shift).astype(np.uint16)
                depth_np_save_path = refined_input_txt_output_folder + "/" + color_img_path.split("/")[-1]
                cv2.imwrite(depth_np_save_path[:-4] + ".png", pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if self.cfg.EVAL_SAVE_DEPTH:
            print("##### result saved to ##### {}".format(os.path.join(self.cfg.OUTPUT_DIR,"color_mask_gtD_predD.txt")))

    def refine_raw_inputD_and_eval(self, output_list):
        anchor_normal = np.load(self.cfg.ANCHOR_NORMAL_NYP)
        refine_depth_fun = RefineDepth(self.cfg.FOCAL_LENGTH, self.cfg.REF_BORDER_WIDTH, self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)

        mirror3d_eval_sensorD = Mirror3dEval(dataset_root=self.cfg.VAL_IMG_ROOT,train_with_ref_d=None, logger=self.logger,input_tag="sensor-D", method_tag="*",width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)
        mirror3d_eval_meshD = Mirror3dEval(dataset_root=self.cfg.VAL_IMG_ROOT,train_with_ref_d=None, logger=self.logger,input_tag="mesh-D", method_tag="*",width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)
        mirror3d_eval_hole = Mirror3dEval(dataset_root=self.cfg.VAL_IMG_ROOT,train_with_ref_d=None, logger=self.logger,input_tag="sensor-D", method_tag="\mnet",width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)
        mirror3d_eval_mesh = Mirror3dEval(dataset_root=self.cfg.VAL_IMG_ROOT,train_with_ref_d=None, logger=self.logger,input_tag="mesh-D", method_tag="\mnet",width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)

        have_mesh_D = False
        imgPath_info = dict()

        input_json = read_json(self.cfg.VAL_COCO_JSON)
        for item in input_json["images"]:
            img_path = os.path.join(self.cfg.VAL_IMG_ROOT, item["mirror_color_image_path"])
            
            imgPath_info[img_path] = item
        
        for i, item in enumerate(output_list):
            one_output, one_input = item
            instances = one_output[0][0]["instances"]
            color_img_path = one_input[0]["mirror_color_image_path"]
            hole_raw_depth_path = os.path.join(self.cfg.VAL_IMG_ROOT, imgPath_info[color_img_path]["raw_sensorD_path"])
            mesh_raw_depth_path = os.path.join(self.cfg.VAL_IMG_ROOT, imgPath_info[color_img_path]["raw_meshD_path"])

            pred_mask = np.zeros(instances.image_size)
            pred_mask = pred_mask.astype(bool)

            hole_depth_to_ref = cv2.imread(hole_raw_depth_path, cv2.IMREAD_ANYDEPTH)
            if mesh_raw_depth_path != hole_raw_depth_path:
                mesh_depth_to_ref = cv2.imread(mesh_raw_depth_path, cv2.IMREAD_ANYDEPTH)

            hole_depth_to_ref_output_folder = os.path.join(self.cfg.OUTPUT_DIR, "hole_raw_depth_mirror3d_refine")
            os.makedirs(hole_depth_to_ref_output_folder, exist_ok=True)
            mesh_depth_to_ref_output_folder = os.path.join(self.cfg.OUTPUT_DIR, "mesh_raw_depth_mirror3d_refine")
            os.makedirs(mesh_depth_to_ref_output_folder, exist_ok=True)


            if instances.to("cpu").has("pred_masks"):
                for index, one_pred_mask in enumerate(instances.to("cpu").pred_masks):
                    
                    to_refine_area = one_pred_mask.numpy().astype(bool)
                    to_refine_area = np.logical_and(pred_mask==False, to_refine_area)
                    if to_refine_area.sum() == 0:
                        continue
                    pred_mask = np.logical_or(pred_mask , one_pred_mask)
                    if instances.to("cpu").pred_anchor_classes[index] >= anchor_normal.shape[0]:
                        continue
                    pred_normal = anchor_normal[instances.to("cpu").pred_anchor_classes[index]] +  instances.to("cpu").pred_residuals[index].numpy()
                    pred_normal = unit_vector(pred_normal)

                    if mesh_raw_depth_path != hole_raw_depth_path:
                        if "border" in self.cfg.REF_MODE :
                            mesh_depth_to_ref = refine_depth_fun.refine_depth_by_mirror_border(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, mesh_depth_to_ref)
                        else:
                            mesh_depth_to_ref = refine_depth_fun.refine_depth_by_mirror_area(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, mesh_depth_to_ref)

                    if "border" in self.cfg.REF_MODE :
                        hole_depth_to_ref = refine_depth_fun.refine_depth_by_mirror_border(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, hole_depth_to_ref)
                    else:
                        hole_depth_to_ref = refine_depth_fun.refine_depth_by_mirror_area(one_pred_mask.numpy().astype(bool).squeeze(), pred_normal, hole_depth_to_ref)
            
            hole_depth_to_ref[hole_depth_to_ref<0] = 0
            
            mirror3d_eval_hole.compute_and_update_mirror3D_metrics(hole_depth_to_ref/self.cfg.DEPTH_SHIFT,  self.cfg.DEPTH_SHIFT, color_img_path,one_input[0]["raw_meshD_path"], one_input[0]["refined_meshD_path"], one_input[0]["mirror_instance_mask_path"])
            mirror3d_eval_sensorD.compute_and_update_mirror3D_metrics(cv2.imread(hole_raw_depth_path, cv2.IMREAD_ANYDEPTH)/self.cfg.DEPTH_SHIFT,  self.cfg.DEPTH_SHIFT, color_img_path,one_input[0]["raw_meshD_path"], one_input[0]["refined_meshD_path"], one_input[0]["mirror_instance_mask_path"])
            if self.cfg.EVAL_SAVE_DEPTH:
                mirror3d_eval_hole.save_result(hole_depth_to_ref_output_folder, hole_depth_to_ref/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, color_img_path,one_input[0]["raw_meshD_path"], one_input[0]["refined_meshD_path"], one_input[0]["mirror_instance_mask_path"])

            if mesh_raw_depth_path != hole_raw_depth_path:
                have_mesh_D = True
                mesh_depth_to_ref[mesh_depth_to_ref<0] = 0
                mirror3d_eval_mesh.compute_and_update_mirror3D_metrics(mesh_depth_to_ref/self.cfg.DEPTH_SHIFT,  self.cfg.DEPTH_SHIFT, color_img_path,one_input[0]["raw_meshD_path"], one_input[0]["refined_meshD_path"], one_input[0]["mirror_instance_mask_path"])
                mirror3d_eval_meshD.compute_and_update_mirror3D_metrics(cv2.imread(mesh_raw_depth_path, cv2.IMREAD_ANYDEPTH)/self.cfg.DEPTH_SHIFT,  self.cfg.DEPTH_SHIFT, color_img_path,one_input[0]["raw_meshD_path"], one_input[0]["refined_meshD_path"], one_input[0]["mirror_instance_mask_path"])
                if self.cfg.EVAL_SAVE_DEPTH:
                    mirror3d_eval_mesh.save_result(mesh_depth_to_ref_output_folder, mesh_depth_to_ref/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, color_img_path,one_input[0]["raw_meshD_path"], one_input[0]["refined_meshD_path"], one_input[0]["mirror_instance_mask_path"])
            else:
                os.rmdir(mesh_depth_to_ref_output_folder)
        
        print("############# hole raw depth + Mirror3dNet result #############")
        mirror3d_eval_hole.print_mirror3D_score()
        print("############# hole raw (sensor) depth result #############")
        mirror3d_eval_sensorD.print_mirror3D_score()
        if have_mesh_D:
            print("############# mesh raw depth + Mirror3dNet result #############")
            mirror3d_eval_mesh.print_mirror3D_score()
            print("############# mesh raw (meshD) depth refine result #############")
            mirror3d_eval_meshD.print_mirror3D_score()

    def eval_raw_DEbranch_predD(self, output_list):
        print("PREDICTING REFINED DEPTH!")
        refine_depth_fun = RefineDepth(self.cfg.FOCAL_LENGTH, self.cfg.REF_BORDER_WIDTH, self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)

        if not self.cfg.OBJECT_CLS:
            input_tag = "RGB"
            method_tag = "PlaneRCNN-DE"
        else:
            input_tag = "RGB"
            method_tag = "\mnet-DE"
        
        if self.cfg.REFINED_DEPTH:
            train_with_refD = True
        else:
            train_with_refD = False

        mirror3d_eval = Mirror3dEval(dataset_root=self.cfg.VAL_IMG_ROOT,train_with_ref_d=train_with_refD, logger=self.logger,input_tag=input_tag, method_tag=method_tag,width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)

        for one_output, one_input in output_list:
            pred_depth = one_output[1][0].detach().cpu().numpy()
            gt_depth = cv2.imread(one_input[0]["refined_meshD_path"], cv2.IMREAD_ANYDEPTH)

            np_pred_depth = pred_depth.astype(np.uint16)
              
            #mirror3d_eval.compute_and_update_mirror3D_metrics(np_pred_depth/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, one_input[0]["mirror_color_image_path"],one_input[0]["raw_meshD_path"], one_input[0]["refined_meshD_path"], one_input[0]["mirror_instance_mask_path"])

            if self.cfg.EVAL_SAVE_DEPTH:
                raw_branch_output_folder = os.path.join(self.cfg.OUTPUT_DIR, "DE_branch_pred_depth")
                os.makedirs(raw_branch_output_folder, exist_ok=True)
                mirror3d_eval.save_result(raw_branch_output_folder, np_pred_depth/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, one_input[0]["mirror_color_image_path"],one_input[0]["raw_meshD_path"], one_input[0]["refined_meshD_path"], one_input[0]["mirror_instance_mask_path"])
            
        print("evaluate DE result for {}".format(method_tag))
        self.logger.info("evaluate DE result for {}".format(method_tag))
        mirror3d_eval.print_mirror3D_score()

    def refine_DEbranch_predD_and_eval(self, output_list):

        refine_depth_fun = RefineDepth(self.cfg.FOCAL_LENGTH, self.cfg.REF_BORDER_WIDTH, self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)
        if not self.cfg.OBJECT_CLS:
            input_tag = "RGB"
            method_tag = "PlaneRCNN"
        else:
            input_tag = "RGB"
            method_tag = "\mnet"

        train_with_refD=None
        if self.cfg.MODEL.WEIGHTS.find("ref") > 0:
            train_with_refD = True
        else:
            train_with_refD = False
        mirror3d_eval = Mirror3dEval(dataset_root=self.cfg.VAL_IMG_ROOT,train_with_ref_d=train_with_refD, logger=self.logger,input_tag=input_tag, method_tag=method_tag,width=self.cfg.EVAL_WIDTH, height=self.cfg.EVAL_HEIGHT, dataset=self.dataset_name)

        for one_output, one_input in output_list:
            pred_depth = one_output[1][0].detach().cpu().numpy()
            np_pred_depth = pred_depth.copy()
            depth_p = pred_depth.copy()
            
            
            # -------------- refine depth with predict anchor normal ------------
            instances = one_output[0][0]["instances"]
            anchor_normals = np.load(self.cfg.ANCHOR_NORMAL_NYP)
            
            for instance_idx, pred_anchor_normal_class in enumerate(instances.pred_anchor_classes):
                instance_mask = instances.pred_masks[instance_idx].detach().cpu().numpy()
                
                if pred_anchor_normal_class >= anchor_normals.shape[0]:
                    continue
                else:
                    if self.cfg.ANCHOR_REG:
                        plane_normal = anchor_normals[pred_anchor_normal_class] + instances.pred_residuals[instance_idx].detach().cpu().numpy()
                    else:
                        plane_normal = anchor_normals[pred_anchor_normal_class]
                a, b, c = unit_vector(plane_normal)
                if "border" in self.cfg.REF_MODE:
                    depth_p = refine_depth_fun.refine_depth_by_mirror_border(instance_mask, [a, b, c], pred_depth)
                else:
                    depth_p = refine_depth_fun.refine_depth_by_mirror_area(instance_mask, [a, b, c], pred_depth)

            np_pred_depth = np_pred_depth.astype(np.uint16)
            depth_p = depth_p.astype(np.uint16)

            if self.cfg.EVAL_SAVE_DEPTH:
                refined_DE_branch_output_folder = os.path.join(self.cfg.OUTPUT_DIR, "refined_DE_branch_pred_depth")
                os.makedirs(refined_DE_branch_output_folder, exist_ok=True)
                mirror3d_eval.save_result(refined_DE_branch_output_folder, depth_p/self.cfg.DEPTH_SHIFT, self.cfg.DEPTH_SHIFT, one_input[0]["mirror_color_image_path"],one_input[0]["raw_meshD_path"], one_input[0]["refined_meshD_path"], one_input[0]["mirror_instance_mask_path"])
            
        print("eval refined depth from DE branch : {}".format(method_tag))
        mirror3d_eval.print_mirror3D_score()

    def save_masked_image(self, output_list):
        import shutil
        masked_img_save_folder = os.path.join(self.cfg.VAL_IMG_ROOT, "masked_img")
        os.makedirs(masked_img_save_folder, exist_ok=True)

        for one_output, one_input in output_list:
            instances = one_output[0][0]["instances"]
            img_path = one_input[0]["mirror_color_image_path"]
            color_img_folder = os.path.join(self.cfg.VAL_IMG_ROOT, img_path.replace(self.cfg.VAL_IMG_ROOT,"").split("/")[1])
            masked_img_save_path = img_path.replace(color_img_folder, masked_img_save_folder)
            masked_img_save_sub_folder = os.path.split(masked_img_save_path)[0]
            os.makedirs(masked_img_save_sub_folder, exist_ok=True)
            if instances.pred_boxes.tensor.shape[0] <= 0:
                shutil.copy(img_path, masked_img_save_path)
                print("######## no detection :", img_path)
                continue
            img = cv2.imread(img_path)
            v = Visualizer(img[:, :, ::-1], 
                metadata=MetadataCatalog.get("test_10_precise_normal_mirror"), 
                scale=0.5, 
                instance_mode=ColorMode.IMAGE_BW  
                )
            v = v.draw_instance_predictions(instances.to("cpu")) 
            output_img = v.get_image()[:, :, ::-1]
            cv2.imwrite(masked_img_save_path, output_img)
        print("masked image saved to : ", masked_img_save_folder)
            

    def eval_seg(self, output_list):

        eval_seg_fun = MirrorSegEval(self.cfg.EVAL_WIDTH, self.cfg.EVAL_HEIGHT)

        for i, item in enumerate(output_list):
            one_output, one_input = item
            instances = one_output[0][0]["instances"]
            mask_path = one_input[0]["mirror_instance_mask_path"]
            if not os.path.exists(mask_path) or "no_mirror" in one_input[0]["mirror_color_image_path"]:
                continue
            GT_mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
            GT_mask = GT_mask > 0
            pred_mask = np.zeros_like(GT_mask)
            pred_mask = pred_mask.astype(bool)
            if instances.to("cpu").has("pred_masks"):
                for one_pred_mask in instances.to("cpu").pred_masks:
                    pred_mask = np.logical_or(pred_mask , one_pred_mask)
                    pred_mask = pred_mask.numpy().astype(bool)
                
            eval_seg_fun.compute_and_update_seg_metrics(pred_mask, GT_mask)

        eval_seg_fun.print_seg_score()
        IOU_list, f_measure_list, MAE_list = eval_seg_fun.get_results()
        self.mean_IOU = np.mean(IOU_list)
        if not self.cfg.EVAL:
            storage = get_event_storage()
            storage.put_scalar("mean IOU",np.mean(IOU_list))
            storage.put_scalar("mean f measure",np.mean(f_measure_list))
            storage.put_scalar("mean MAE",np.mean(MAE_list))
