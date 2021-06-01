import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
from skimage import io
import sys
import bs4
from utils.algorithm import *
from utils.general_utils import *
from utils.plane_pcd_utils import *
import json
import shutil
from annotation.plane_annotation.plane_annotation_tool import *
from visualization.dataset_visualization import Dataset_visulization
from tqdm import tqdm




class Dataset_visulization(Dataset_visulization):

    def __init__(self, pred_w=480, pred_h=640, dataset_main_folder=None, test_json="", method_tag="mirror3D", process_index=0, multi_processing=False, 
                f=519, output_folder=None, overwrite=True, window_w=800, window_h=800, view_mode="topdown"):
        """
        Initilization

        Args:
            dataset_main_folder : Folder raw, hole_raw_depth/ mesh_raw_depth, instance_mask saved folder.
            output_folder(optional) : Inital pcd, img_info, border_vis saved forder (default : dataset_main_folder).
            process_index : The process index of multi_processing.
            multi_processing : Use multi_processing or not (bool).
            border_width : Half of mirror 2D border width (half of cv2.dilate kernel size; 
                           default kernel anchor is at the center); default : 50 --> actualy border width = 25.
            f : Camera focal length of current input data.
        """

        self.dataset_main_folder = dataset_main_folder
        self.method_tag = method_tag
        assert os.path.exists(dataset_main_folder), "please input a valid folder path"
        self.process_index = process_index
        self.multi_processing = multi_processing
        self.overwrite = overwrite
        self.window_w = window_w
        self.window_h = window_h
        self.view_mode = view_mode
        self.pred_w = pred_w
        self.pred_h = pred_h
        
        if "m3d" not in self.dataset_main_folder:
            self.is_matterport3d = False
        else:
            self.is_matterport3d = True
        self.color_img_list = []
        input_images = read_json(test_json)["images"]
        for one_info in input_images: 
            self.color_img_list.append(os.path.join(dataset_main_folder, one_info["mirror_color_image_path"]))
        self.color_img_list.sort()

        if multi_processing:
            self.color_img_list = self.color_img_list[process_index:process_index+1]
        self.f = f
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.error_info_path = os.path.join(self.output_folder, "error_img_list.txt")
    
    def generate_method_pred(self, info_txt_path):
        self.method_predFolder = dict()
        for one_line in read_txt(info_txt_path):
            one_info = one_line.strip().split()
            method_tag = one_info[0]
            pred_folder = one_info[1]
            self.method_predFolder[method_tag] = pred_folder
    
    def generate_color_depth_for_all_pred(self):

        for one_color_img_path in self.color_img_list:
            self.generate_color_depth_for_one_pred(one_color_img_path)

    def generate_color_depth_for_one_pred(self, color_img_path):
        """
        Generate point cloud for specific prediction

        Args:
            color_img_path : The color image absolute path for the specific sample.
        
        Output:
            "point cloud" : Saved under self.output_folder.
        """
        import open3d as o3d
        for item in self.method_predFolder.items():
            method_tag = item[0]
            prediction_output_folder = item[1]

            one_colored_pred_depth_folder =  os.path.join(self.output_folder, method_tag, "colored_pred_depth")
            os.makedirs(one_colored_pred_depth_folder, exist_ok=True)
            one_colored_pred_error_map_folder =  os.path.join(self.output_folder, method_tag, "colored_pred_error_map")
            os.makedirs(one_colored_pred_error_map_folder, exist_ok=True)

            one_info_folder =  os.path.join(self.output_folder, method_tag, "info")
            os.makedirs(one_info_folder, exist_ok=True)

            sample_name = color_img_path.split("/")[-1]

            if self.is_matterport3d:
                pred_depth_img_path = os.path.join(prediction_output_folder, rreplace(sample_name, "i", "d"))
                colored_pred_depth_save_path = os.path.join(one_colored_pred_depth_folder,  rreplace(sample_name, "i", "d"))
                colored_pred_error_map_save_path = os.path.join(one_colored_pred_error_map_folder,  rreplace(sample_name, "i", "d"))
                gt_depth_img_path = rreplace(color_img_path.replace("raw","mesh_refined_depth"),"i", "d")
            else:
                pred_depth_img_path =  os.path.join(prediction_output_folder, sample_name)
                colored_pred_depth_save_path = os.path.join(one_colored_pred_depth_folder,  sample_name)
                colored_pred_error_map_save_path = os.path.join(one_colored_pred_error_map_folder,  sample_name)
                gt_depth_img_path = color_img_path.replace("raw","hole_refined_depth")
            
            info_save_path = os.path.join(one_info_folder, "{}.json".format(sample_name.split(".")[0]))
            gt_depth = cv2.imread(gt_depth_img_path, cv2.IMREAD_ANYDEPTH)
            pred_depth = cv2.imread(pred_depth_img_path, cv2.IMREAD_ANYDEPTH)
            if self.is_matterport3d:
                depth_shift = 4000
            else:
                depth_shift = 1000
            gt_depth = np.asarray(cv2.resize(gt_depth, dsize=(self.pred_w, self.pred_h), interpolation=cv2.INTER_NEAREST), dtype=np.float32) / depth_shift
            pred_depth = np.asarray(cv2.resize(pred_depth, dsize=(self.pred_w, self.pred_h), interpolation=cv2.INTER_NEAREST), dtype=np.float32) / depth_shift
            
            rmse = (gt_depth - pred_depth) ** 2
            score = float(np.mean(rmse))
            if os.path.exists(info_save_path):
                info = read_json(info_save_path)
            else:
                info = dict()

            info["RMSE"] = score


            save_json(info_save_path, info)
            save_heatmap_no_border(pred_depth, colored_pred_depth_save_path)
            save_heatmap_no_border(rmse, colored_pred_error_map_save_path)

            print("colored_pred_depth_save_path to :", os.path.abspath(colored_pred_depth_save_path))
            print("colored_pred_error_map_save_path to :", os.path.abspath(colored_pred_error_map_save_path))


    def generate_pcd_for_whole_dataset(self):
        """
        Call function self.generate_pcd_for_one_prediction 
            to generate mesh.ply and pcd.ply for all sample under self.dataset_main_folders
        """
        for one_color_img_path in self.color_img_list:
            self.generate_pcd_for_one_prediction(one_color_img_path)

    def generate_pcd_for_one_prediction(self, color_img_path):
        """
        Generate point cloud for specific prediction

        Args:
            color_img_path : The color image absolute path for the specific sample.
        
        Output:
            "point cloud" : Saved under self.output_folder.
        """
        import open3d as o3d

        def save_topdown(pcd, topdown_img_save_path):

            mesh_center = pcd.get_center()
            rotation_step_degree = 10
            start_rotation = get_extrinsic(90,0,0,[0,0,0])
            pcd_points = np.array(pcd.points)
            min_h = min([np.abs(pcd_points[:,1].max()), np.abs(pcd_points[:,1].min())])
            stage_tranlation = get_extrinsic(0,0,0,[-mesh_center[0],-mesh_center[1] + min_h*3,-mesh_center[2]])
            start_position = np.dot(start_rotation, stage_tranlation)
            def rotate_view(vis):
                T_rotate = get_extrinsic(0,rotation_step_degree*(1),0,[0,0,0])
                cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
                cam.extrinsic = np.dot(np.dot(start_rotation, T_rotate), stage_tranlation)
                vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
                
                vis.capture_screen_image(filename=topdown_img_save_path, do_render=True)
                print("image saved to {}".format(topdown_img_save_path))
                vis.destroy_window()

            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.register_animation_callback(rotate_view)
            vis.create_window(width=self.window_w,height=self.window_h)
            vis.get_render_option().point_size = 1.0
            vis.add_geometry(pcd)
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            cam.extrinsic = start_position
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
            vis.run()


        def save_front(pcd, front_img_save_path):
            
            mesh_center = pcd.get_center()
            rotation_step_degree = 10
            start_position = get_extrinsic(0,0,0,[0,0,3000])

            def rotate_view(vis):
                T_to_center = get_extrinsic(0,0,0,mesh_center)
                T_rotate = get_extrinsic(0,rotation_step_degree*(1),0,[0,0,0])
                T_to_mesh = get_extrinsic(0,0,0,-mesh_center)
                cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
                cam.extrinsic = np.dot(start_position, np.dot(np.dot(T_to_center, T_rotate),T_to_mesh))
                vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
                
                vis.capture_screen_image(filename=front_img_save_path, do_render=True)
                print("image saved to {}".format(front_img_save_path))
                vis.destroy_window()
                return False

            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.register_animation_callback(rotate_view)
            vis.create_window(width=self.window_w,height=self.window_h)
            vis.get_render_option().point_size = 1.0
            vis.add_geometry(pcd)
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            cam.extrinsic = start_position
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
            vis.run()

        for item in self.method_predFolder.items():
            method_tag = item[0]
            prediction_output_folder = item[1]

            one_method_pcd_save_folder =  os.path.join(self.output_folder, method_tag, "pred_depth_ply")
            os.makedirs(one_method_pcd_save_folder, exist_ok=True)

            sample_name = color_img_path.split("/")[-1]
            pcd_save_path = os.path.join(one_method_pcd_save_folder,  sample_name.replace("png","ply"))
            topdown_img_save_path = os.path.join(one_method_pcd_save_folder,  "topdown_{}".format(sample_name))
            front_img_save_path = os.path.join(one_method_pcd_save_folder,  "front_{}".format(sample_name))


            if self.is_matterport3d:
                pred_depth_img_path = os.path.join(prediction_output_folder, rreplace(sample_name, "i", "d"))
            else:
                pred_depth_img_path =  os.path.join(prediction_output_folder, sample_name)
            # Get and save pcd for the instance
            pcd = get_pcd_from_rgbd_depthPath(f=self.f, depth_img_path=pred_depth_img_path, color_img_path=color_img_path, w=self.pred_w, h=self.pred_h)
            o3d.io.write_point_cloud(pcd_save_path, pcd)
            save_front(pcd, front_img_save_path)
            save_topdown(pcd, topdown_img_save_path)
            print("point cloud saved  to :", os.path.abspath(pcd_save_path))


    def set_view_mode(self, view_mode):
        """Function to save the view mode"""
        self.view_mode = view_mode

                
    def generate_html(self, vis_saved_main_folder="", sample_num_per_page=50, template_path=""):
        """
        (1) under vis_saved_folder there should only be the vislization output sub-folders and nothing else

        Each line will show:
            1. Sample ID & method name 
            2. Color image
            3. Colored RMSE map
            4. Colored predict depth
            5. Front view
            6. Topdown view
        """
        if self.is_matterport3d:
            method_folder_list = ["VNL+Mirror3DNet","VNL-raw","BTS+Mirror3DNet","BTS-raw","VNL-ref","BTS-ref","Mirror3DNet-raw","Mirror3DNet-DE-raw","PlaneRCNN-raw","PlaneRCNN-DE-raw","saic+Mirror3DNet","saic-raw","Mirror3DNet-ref","Mirror3DNet-DE-ref","PlaneRCNN-ref","PlaneRCNN-DE-ref","saic-ref","mesh-D+Mirror3DNet","sensor-D+Mirror3DNet","mesh-D","sensor-D"]
            # method_folder_list = ["GT", "meshD", "meshD_Mirror3DNet", "BTS_refD", "BTS_rawD", "BTS_Mirror3DNet", "VNL_refD", "VNL_rawD", "VNL_Mirror3DNet", "SAIC_refD", "SAIC_rawD", "SAIC_Mirror3DNet"]
        else:
             method_folder_list = ["sensor-D+Mirror3DNet-m3d","saic+Mirror3DNet-m3d","VNL+Mirror3DNet-m3d","BTS+Mirror3DNet-m3d","saic-ref","saic-raw","VNL-ref","VNL-raw","BTS-ref","BTS-raw","sensor-D","GT"]
            # method_folder_list = ["GT", "sensorD", "sensorD_Mirror3DNet", "BTS_refD", "BTS_rawD", "BTS_Mirror3DNet", "VNL_refD", "VNL_rawD", "VNL_Mirror3DNet", "SAIC_refD", "SAIC_rawD", "SAIC_Mirror3DNet"]
        colorImgSubset_list = [self.color_img_list[x:x+sample_num_per_page] for x in range(0, len(self.color_img_list), sample_num_per_page)]

        for html_index, one_colorSubset in enumerate(colorImgSubset_list):
            with open(template_path) as inf:
                txt = inf.read()
                soup = bs4.BeautifulSoup(txt, features="html.parser")
            for one_color_img_path in self.color_img_list:
                sample_name = os.path.split(one_color_img_path)[-1]
                sample_id = sample_name.split(".")[0]
                if self.is_matterport3d:
                    one_depth_sample_name = rreplace(sample_name, "i", "d")
                else:
                    one_depth_sample_name = sample_name
                for one_method_name in method_folder_list:
                    one_method_folder_path = os.path.join(vis_saved_main_folder, one_method_name)
                    one_RMSE_map = os.path.join(one_method_folder_path, "colored_pred_error_map", one_depth_sample_name)
                    one_predD_map = os.path.join(one_method_folder_path, "colored_pred_depth", one_depth_sample_name)
                    one_front_view_img = os.path.join(one_method_folder_path, "pred_depth_ply", "front_{}".format(sample_name))
                    one_topdown_view_img = os.path.join(one_method_folder_path, "pred_depth_ply", "topdown_{}".format(sample_name))
                    
                    new_div = soup.new_tag("div")
                    new_div['class'] = "one-sample"
                    
                    soup.body.append(new_div)

                    one_text = soup.new_tag("div")
                    one_text["class"] = "one-item"
                    if not self.is_matterport3d:
                        one_text['style'] = "padding-top: 100px;font-size: 25pt;"
                    else:
                        one_text['style'] = "padding-top: 100px;"
                    one_text.string = sample_id
                    new_div.append(one_text)

                    one_text = soup.new_tag("div")
                    one_text["class"] = "one-item"
                    if not self.is_matterport3d:
                        one_text['style'] = "padding-top: 100px;font-size: 25pt;"
                    else:
                        one_text['style'] = "padding-top: 100px;"
                    one_text.string = one_method_name
                    new_div.append(one_text)

                    # Append color image to one line in HTML
                    one_color_img = soup.new_tag("div")
                    color_img = soup.new_tag("div")
                    color_img["class"] = "one-item"
                    color_img_path = os.path.relpath(one_color_img_path, self.output_folder)
                    color_img.append(soup.new_tag('img', src=color_img_path))
                    one_color_img.append(color_img)
                    new_div.append(one_color_img)

                    # Append color image to one line in HTML
                    one_color_img = soup.new_tag("div")
                    color_img = soup.new_tag("div")
                    color_img["class"] = "one-item"
                    one_RMSE_map = os.path.relpath(one_RMSE_map, self.output_folder)
                    color_img.append(soup.new_tag('img', src=one_RMSE_map))
                    one_color_img.append(color_img)
                    new_div.append(one_color_img)

                    # Append color image to one line in HTML
                    one_color_img = soup.new_tag("div")
                    color_img = soup.new_tag("div")
                    color_img["class"] = "one-item"
                    one_predD_map = os.path.relpath(one_predD_map, self.output_folder)
                    color_img.append(soup.new_tag('img', src=one_predD_map))
                    one_color_img.append(color_img)
                    new_div.append(one_color_img)

                    # Append color image to one line in HTML
                    one_color_img = soup.new_tag("div")
                    color_img = soup.new_tag("div")
                    color_img["class"] = "one-item"
                    one_front_view_img = os.path.relpath(one_front_view_img, self.output_folder)
                    color_img.append(soup.new_tag('img', src=one_front_view_img))
                    one_color_img.append(color_img)
                    new_div.append(one_color_img)

                    # Append color image to one line in HTML
                    one_color_img = soup.new_tag("div")
                    color_img = soup.new_tag("div")
                    color_img["class"] = "one-item"
                    one_topdown_view_img = os.path.relpath(one_topdown_view_img, self.output_folder)
                    color_img.append(soup.new_tag('img', src=one_topdown_view_img))
                    one_color_img.append(color_img)
                    new_div.append(one_color_img)
                
            html_path = os.path.join(self.output_folder, "{}.html".format(html_index))
            save_html(html_path, soup)
            
            print("result visulisation saved in link {}".format(html_path.replace("/project/3dlg-hcvc/mirrors/www","http://aspis.cmpt.sfu.ca/projects/mirrors")))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--stage', default="6")
    parser.add_argument(
        '--test_json', default="/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu/with_mirror/precise/network_input_json/test_10_normal_mirror.json")
    parser.add_argument(
        '--method_predFolder_txt', default="/project/3dlg-hcvc/mirrors/www/notes/nyu_vis_0418.txt")
    parser.add_argument(
        '--dataset_main_folder', default="/project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu")
    parser.add_argument(
        '--process_index', default=0, type=int, help="process index")
    parser.add_argument('--multi_processing', help='do multi-process or not',action='store_true')
    parser.add_argument('--compare_with_raw', help='do multi-process or not',action='store_true')
    parser.add_argument('--overwrite', help='overwrite files under --output_folder or not',action='store_true')
    parser.add_argument(
        '--f', default=537, type=int, help="camera focal length")
    parser.add_argument(
        '--pred_w', default=640, type=int, help="width of the visilization window")
    parser.add_argument(
        '--pred_h', default=512, type=int, help="width of the visilization window")
    parser.add_argument(
        '--window_w', default=800, type=int, help="width of the visilization window")
    parser.add_argument(
        '--window_h', default=800, type=int, help="height of the visilization window")
    parser.add_argument(
        '--sample_num_per_page', default=100, type=int, help="height of the visilization window")
    parser.add_argument(
        '--vis_saved_folder', default="/project/3dlg-hcvc/mirrors/www/cr_vis/nyu_result_vis")
    parser.add_argument(
        '--output_folder', default="/project/3dlg-hcvc/mirrors/www/cr_vis/nyu_html")
    parser.add_argument(
        '--method_folder_list', nargs='+', default="", type=str)
    parser.add_argument("--midrule_index", nargs="+", type=int, default=[2,5], help="add /midrule in after these liens; index start from 1") 
    parser.add_argument(
        '--template_path', default="visualization/template/result_vis_template.html", type=str)
    parser.add_argument(
        '--view_mode', default="topdown", help="object view angle : (1) topdown (2) front")
    parser.add_argument(
        '--method_order_txt', default="", type=str)
    parser.add_argument(
        '--dataset_tag', default="NYUv2", type=str)
    parser.add_argument(
        '--all_info_json', default="output/ref_m3d_result.json", type=str)
    parser.add_argument('--min_threshold_filter', help='',action='store_true')
    parser.add_argument('--add_rmse', help='add rmse to result visulization or not',action='store_true')
    args = parser.parse_args()

    vis_tool = Dataset_visulization(pred_w = args.pred_w, pred_h = args.pred_h, dataset_main_folder=args.dataset_main_folder, process_index=args.process_index, \
                                    multi_processing=args.multi_processing, f=args.f, test_json=args.test_json, \
                                    output_folder=args.output_folder, overwrite=args.overwrite, \
                                    window_w=args.window_w, window_h=args.window_h, view_mode=args.view_mode)
    if args.stage == "1":
        vis_tool.generate_method_pred(args.method_predFolder_txt)
        vis_tool.generate_pcd_for_whole_dataset()
        vis_tool.generate_method_pred(args.method_predFolder_txt)
        vis_tool.generate_color_depth_for_all_pred()
    elif args.stage == "2":
        vis_tool.generate_html(vis_saved_main_folder=args.vis_saved_folder, sample_num_per_page=args.sample_num_per_page, template_path=args.template_path)
    