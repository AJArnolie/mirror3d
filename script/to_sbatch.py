from utils.general_utlis import *
import os
import argparse

def sh_to_sbatch(train_sh_path, log_config, sh_output_foler, sbatch_config, env_config):
    """
    Args:
        train_sh_path : the .sh file
        log_config : folder to store the sbatch .sh file
        sh_output_foler : folder to store the .out sbatch output
    """

    def get_config_from_pythonSetting(python_command):
        depth_tag = "rawD"
        method_name = "method"
        resume_tag = "scratch"
        have_mirror_tag = "all"
        for item in python_command:
            if item.find("-refined_depth") > 0:
                depth_tag = "refD"
            if item.find("bts") > 0:
                method_name = "bts"
            if item.find("_normal_config") > 0:
                method_name = "m3n_normal"
            if item.find("3dnet_config") > 0:
                method_name = "m3n_full"
            if item.find("rcnn_config") > 0:
                method_name = "planercnn"
            if item.find("saic") > 0:
                method_name = "saic"
            if item.find("vnl") > 0:
                method_name = "vnl"
            if item.find("-resume") > 0:
                resume_tag = "resume"
            if item.find("-coco_train") > 0 and item.find("_mirror.json") > 0:
                have_mirror_tag = "mirror"
        
        job_name = "{}_{}_{}_{}".format(method_name, depth_tag, resume_tag, have_mirror_tag)
        return job_name

    os.makedirs(sh_output_foler, exist_ok=True)
    sh_lines = read_txt(train_sh_path)
    one_python_command = []
    sbatch_lines = sbatch_config.copy()
    for line_index in range(len(sh_lines)):
        to_break = False
        if (line_index+1) != len(sh_lines) and len(sh_lines[line_index+1]) > 0:
            to_break = (sh_lines[line_index+1][0] == "#")

        if (line_index+1) == len(sh_lines)  or to_break:
            # Update config
            job_name = get_config_from_pythonSetting(one_python_command)
            sbatch_job_name = "#SBATCH --job-name={}".format(job_name)
            sbatch_lines.append(sbatch_job_name)
            sbatch_output = "#SBATCH --output={}".format(log_config)
            sbatch_lines.append(sbatch_output)

            # Save the sbatch output
            output_sh_path = os.path.join(sh_output_foler, "{}.sh".format(job_name))
            output_sh_line = sbatch_lines + env_config +  one_python_command[1:]
            save_txt(output_sh_path, output_sh_line)
            print("sh file saved to : {}".format(output_sh_path))
            one_python_command = []
            sbatch_lines = sbatch_config.copy()
            
        else:
            if len(sh_lines[line_index]) > 0:
                one_python_command.append(sh_lines[line_index].strip())
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--train_sh_path', default="", type=str) 
    parser.add_argument(
        '--log_config', default="/home/jiaqit/projects/rrg-msavva/jiaqit/cr_result/log/%x-%j.out", type=str) 
    parser.add_argument(
        '--sh_output_foler', default="", type=str) 

    args = parser.parse_args()


    # TODO change some config here
    sbatch_config = ["#!/bin/bash", \
                    "#SBATCH --account=rrg-msavva", \
                    "#SBATCH --gres=gpu:v100l:1", \
                    "#SBATCH --mem=48000", \
                    "#SBATCH --time=0-2:45"]
    # TODO 
    env_config = ["source /home/jiaqit/projects/rrg-msavva/jiaqit/setup/anaconda3/bin/activate", \
                  "conda activate mirror3d", \
                  'export PYTHONPATH="/home/jiaqit/projects/rrg-msavva/jiaqit/exp/Mirror3D"'] 

    sh_to_sbatch(args.train_sh_path, args.log_config, args.sh_output_foler, sbatch_config, env_config)