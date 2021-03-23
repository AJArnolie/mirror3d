#!/bin/zsh
python /local-scratch/jiaqit/exp/Mirror3D/utils/input_generation.py \
--mirror_data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise \
--no_mirror_data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/no_mirror \
--dataset_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet \
--split_info_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/split_info \
--split train \
--anchor_normal_path /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d_kmeans_normal_10.npy \
--contain_no_mirror & python /local-scratch/jiaqit/exp/Mirror3D/utils/input_generation.py \
--mirror_data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise \
--no_mirror_data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/no_mirror \
--dataset_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet \
--split_info_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/split_info \
--split train \
--anchor_normal_path /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d_kmeans_normal_10.npy & python /local-scratch/jiaqit/exp/Mirror3D/utils/input_generation.py \
--mirror_data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise \
--no_mirror_data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/no_mirror \
--dataset_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet \
--split_info_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/split_info \
--split test \
--anchor_normal_path /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d_kmeans_normal_10.npy \
--contain_no_mirror & python /local-scratch/jiaqit/exp/Mirror3D/utils/input_generation.py \
--mirror_data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise \
--no_mirror_data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/no_mirror \
--dataset_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet \
--split_info_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/split_info \
--split test \
--anchor_normal_path /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d_kmeans_normal_10.npy & python /local-scratch/jiaqit/exp/Mirror3D/utils/input_generation.py \
--mirror_data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise \
--no_mirror_data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/no_mirror \
--dataset_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet \
--split_info_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/split_info \
--split val \
--anchor_normal_path /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d_kmeans_normal_10.npy \
--contain_no_mirror & python /local-scratch/jiaqit/exp/Mirror3D/utils/input_generation.py \
--mirror_data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/with_mirror/precise \
--no_mirror_data_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet/no_mirror \
--dataset_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/scannet \
--split_info_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/split_info \
--split val \
--anchor_normal_path /project/3dlg-hcvc/mirrors/www/Mirror3D_final/mirror_normal/m3d_kmeans_normal_10.npy