# BTS Depth Prediction on RGB Images
python mirror3d/init_depth_generator/bts/pytorch/init_depth_gen_infer_custom.py \
--resume_checkpoint_path checkpoint/nyu/bts_nyu_v2_pytorch_densenet161/model \
--coco_val /vision2/u/ajarno/cs231a/$1 \
--coco_focal_len 491 \
--depth_shift 800 \
--input_height 480 \
--input_width 640 \

# VNL infer on raw sensor depth
python mirror3d/init_depth_generator/VNL_Monocular_Depth_Prediction/test_any_images.py \
--resume_checkpoint_path checkpoint/nyu/nyu_rawdata2.pth \
--coco_val /vision2/u/ajarno/cs231a/$1 \
--input_height 480 \
--input_width 640 \
--depth_shift 2000 \
--coco_focal_len 491

# GLPDepth Infer on RGB Images
python mirror3d/init_depth_generator/GLPDepth/code/test.py --data_path /vision2/u/ajarno/cs231a/$1 \
--ckpt_dir checkpoint/nyu/best_model_nyu.ckpt \
--save_eval_pngs \
--max_depth 10.0 \
--max_depth_eval 10.0 \

# DPT Depth Inference on RGB Images
python mirror3d/init_depth_generator/DPT/run_monodepth.py \
--input_path /vision2/u/ajarno/cs231a/$1/downsampled_images \
--output_path /vision2/u/ajarno/cs231a/$1/DPT_depth \
--model_weights checkpoint/nyu/dpt_large-midas-2f21e586.pt \
--model_type dpt_large

#python test_depth.py
# Mirror3DNet on Predicted Depth (produces predicted mirror planes and refined depth) 
python mirror3d/mirror3dnet/run_mirror3dnet_custom.py \
--eval \
--config mirror3d/mirror3dnet/config/mirror3dnet_config.yml \
--resume_checkpoint_path checkpoint/mp3d/mirror3dnet_rawD.pth \
--coco_val_root /vision2/u/ajarno/cs231a/$1 \
--coco_focal_len 491 \
--depth_shift 2500 \
--input_height 480 \
--input_width 640 \
--anchor_normal_npy mirror3d/mirror3dnet/config/mp3d_kmeans_normal_10.npy \

## mirror3dnet only normal 10 anchor normal
#python mirror3d/mirror3dnet/run_mirror3dnet.py \
#--eval \
#--resume_checkpoint_path checkpoint/mp3d/mirror3dnet_normal_10.pth \
#--config mirror3d/mirror3dnet/config/mirror3dnet_normal_config.yml \
#--coco_val mirror3d_input/nyu/test_10_precise_normal_mirror.json \
#--coco_val_root mirror3d/dataset/nyu \
#--coco_focal_len 519 \
#--mesh_depth \
#--depth_shift 1000 \
#--input_height 480 \
#--input_width 640 \
#--anchor_normal_npy mirror3d/mirror3dnet/config/mp3d_kmeans_normal_10.npy \
#--log_directory ../output2/nyu
