# bts train on refined sensor depth
python3 mirror3d/init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--refined_depth \
--coco_train mirror3d_input/nyu/train_10_precise_normal_all.json \
--coco_val mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_train_root mirror3d/dataset/nyu \
--coco_val_root mirror3d/dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory ../output/nyu

# bts train on raw sensor depth
python3 mirror3d/init_depth_generator/bts/pytorch/init_depth_gen_train.py \
--coco_train mirror3d_input/nyu/train_10_precise_normal_all.json \
--coco_val mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_train_root mirror3d/dataset/nyu \
--coco_val_root mirror3d/dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory ../output/nyu

# vnl train on refined sensor depth
python3 mirror3d/init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--refined_depth \
--coco_train mirror3d_input/nyu/train_10_precise_normal_all.json \
--coco_val mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_train_root mirror3d/dataset/nyu \
--coco_val_root mirror3d/dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 4 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory ../output/nyu


# vnl train on raw sensor depth
python3 mirror3d/init_depth_generator/VNL_Monocular_Depth_Prediction/init_depth_gen_train.py \
--coco_train mirror3d_input/nyu/train_10_precise_normal_all.json \
--coco_val mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_train_root mirror3d/dataset/nyu \
--coco_val_root mirror3d/dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 4 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory ../output/nyu

# saic train on refined sensor depth
python3 mirror3d/init_depth_generator/saic_depth_completion/init_depth_gen_train.py \
--refined_depth \
--coco_train mirror3d_input/nyu/train_10_precise_normal_all.json \
--coco_val mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_train_root mirror3d/dataset/nyu \
--coco_val_root mirror3d/dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 2 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory ../output/nyu


# saic train on raw sensor depth
python3 mirror3d/init_depth_generator/saic_depth_completion/init_depth_gen_train.py \
--coco_train mirror3d_input/nyu/train_10_precise_normal_all.json \
--coco_val mirror3d_input/nyu/test_10_precise_normal_mirror.json \
--coco_train_root mirror3d/dataset/nyu \
--coco_val_root mirror3d/dataset/nyu \
--coco_focal_len 519 \
--depth_shift 1000 \
--input_height 480 \
--input_width 640 \
--batch_size 8 \
--checkpoint_save_freq 1500 \
--num_epochs 100000 \
--learning_rate 1e-4 \
--log_directory ../output/nyu
