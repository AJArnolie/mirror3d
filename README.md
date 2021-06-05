# [Mirror3D](https://github.com/3dlg-hcvc/mirror3d): Depth Refinement for Mirror Surfaces
                            

## Environment Setup

- python 3.7.4

```
### Install packages 
pip install -e .
### Setup Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git
### Pull submodules
git submodule update --init --recursive
```


## Preparation for all implementations

```shell
$ mkdir workspace
$ cd workspace
### Put data under dataset folder
$ mkdir dataset
### Clone this repo
$ git clone https://github.com/3dlg-hcvc/Mirror3D.git

```

## Dataset

Please refer to [Mirror3D Dataset](https://github.com/3dlg-hcvc/mirror3d/blob/main/docs/Mirror3D_dataset.md) for instructions on how to prepare mirror data. Please visit our [project website](https://github.com/3dlg-hcvc/mirror3d) for updates and to browse the data.

### Mirror annotation tool
Please refer to [User Instruction](https://github.com/3dlg-hcvc/mirror3d/blob/main/docs/user_instruction.md) for instructions on how to annotate mirror data. 


## Models

### Mirror3DNet PyTorch Implementation

Mirror3DNet architecture can be used for either an RGB image or an RGBD image input. For an RGB input, we refine the depth of the predicted depth map D<sub>pred</sub> output by a depth estimation module. For RGBD input, we refine a noisy input depth D<sub>noisy</sub>.
![network-arch](docs/figure/network-arch-cr-new.png)

Please check [Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d/tree/main/mirror3dnet) for our network's pytorch implementation. 

### Initial Depth Generator Implementation

We test three methods on our dataset:

- [BTS: From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation](https://github.com/cogaplex-bts/bts)
- [VNL: Enforcing geometric constraints of virtual normal for depth prediction](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)
- [saic : Decoder Modulation for Indoor Depth Completion](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37)

We updated the dataloader and the main train/test script in the orginal repository to support our input format. 

## Network input

Our network input are JSON files stored based on [coco annotation format](https://cocodataset.org/#home). Please download [network input json](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/network_input_json.zip) to train and test our models. 

## Training

To train our models please run:

```shell
### Train on NYUv2 mirror data
bash script/nyu_train.sh
### Train on Matterport3D mirror data
bash script/mp3d_train.sh
```

By default, we put the unzipped data and network input packages under `../dataset`. Please change the relevant configuration if you store the data in different directories. Output checkpoints and tensorboard log files are saved under `--log_directory`.

## Inference

```shell
### Inferece on NYUv2 mirror data
script/nyu_infer.sh
### Inferece on Matterport3D mirror data
script/mp3d_infer.sh
```

Output depth maps are saved under folder named `pred_depth`. If you want to view the inference result, please run all steps in [mirror3d/visualization/result_visualization.py](https://github.com/3dlg-hcvc/mirror3d/blob/main/visualization/result_visualization.py).  


## Model Zoo



| **Source Dataset** | **Input** | **Train**            | **Method**                                                                                              | **Model Download**                                                                                                      |
|--------------------|-----------|----------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| NYUv2              | RGBD      | raw sensor depth     | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) | [saic_rawD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/nyu/saic_rawD.zip)                |
| NYUv2              | RGBD      | refined sensor depth | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) | [saic_refD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/nyu/saic_refD.zip)                |
| NYUv2              | RGB       | refined sensor depth | [BTS](https://github.com/cogaplex-bts/bts)                                                              | [bts_refD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/nyu/bts_refD.zip)                  |
| NYUv2              | RGB       | refined sensor depth | [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)                                        | [vnl_refD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/nyu/vnl_refD.zip)                  |
| Matterport3D       | RGBD      | raw mesh depth     | [Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d/tree/main/mirror3dnet)                              | [mirror3dnet_rawD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/mirror3dnet_rawD.zip) |
| Matterport3D       | RGBD      | refined mesh depth | [Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d/tree/main/mirror3dnet)                              | [mirror3dnet_refD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/mirror3dnet_refD.zip) |
| Matterport3D       | RGBD      | raw mesh depth     | [PlaneRCNN](https://github.com/NVlabs/planercnn/tree/01e03fe5a97b7afc4c5c4c3090ddc9da41c071bd)          | [planercnn_rawD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/planercnn_rawD.zip)     |
| Matterport3D       | RGBD      | refined mesh depth | [PlaneRCNN](https://github.com/NVlabs/planercnn/tree/01e03fe5a97b7afc4c5c4c3090ddc9da41c071bd)          | [planercnn_refD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/planercnn_refD.zip)     |
| Matterport3D       | RGBD      | raw mesh depth     | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) | [saic_rawD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/saic_rawD.zip)               |
| Matterport3D       | RGBD      | refined mesh depth | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) | [saic_refD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/saic_refD.zip)               |
| Matterport3D       | RGB       | *                    | [Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d/tree/main/mirror3dnet)                              | [mirror3dnet](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/mirror3dnet_normal_10.zip) |
| Matterport3D       | RGB       | raw mesh depth     | [BTS](https://github.com/cogaplex-bts/bts)                                                              | [bts_rawD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/bts_rawD.zip)                 |
| Matterport3D       | RGB       | refined mesh depth | [BTS](https://github.com/cogaplex-bts/bts)                                                              | [bts_refD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/bts_refD.zip)                 |
| Matterport3D       | RGB       | raw mesh depth     | [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)                                        | [vnl_rawD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/vnl_rawD.zip)                 |
| Matterport3D       | RGB       | refined mesh depth | [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)                                        | [vnl_refD](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/vnl_refD.zip)                 |

For [BTS](https://github.com/cogaplex-bts/bts) and [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction), please visit their official websites to download the model trained on NYUv2 raw sensor depth.

