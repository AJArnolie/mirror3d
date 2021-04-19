#!/bin/zsh
log_folder=/project/3dlg-hcvc/jiaqit/waste/log/nyu_vis_0408
mkdir -p $log_folder
echo $log_folder

parallel -j 20 --eta "python visualization/result_visualization.py \
--stage 1 \
--multi_processing --process_index {1} \
--method_predFolder_txt /project/3dlg-hcvc/mirrors/www/notes/nyu_vis_0418.txt \
--dataset_main_folder /project/3dlg-hcvc/mirrors/www/Mirror3D_final/nyu \
--f 519 \
--pred_w 640 \
--pred_h 480 \
--output_folder /project/3dlg-hcvc/mirrors/www/cr_vis/debug_0418 >& ${log_folder}/nyu_vis_0408.log" ::: {0..70}