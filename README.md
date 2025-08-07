# License Plate Object Detection

This project uses NVIDIA TAO Toolkit v.6.0.0 and RT-DETR for license plate detection. 

## Purpose for this project

So I was really interested in Nvidia's pretrained models such as the LDPNet in NGC catalog and challenged myself to see if I could train a custom ML model using RT-DETR (Real-Time Detection Transformer) using the latest version of TAO toolkit. 

## Dataset
 The model weights for the pretrained backbone was extracted from https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth. The dataset uses COCO format and was found in https://huggingface.co/datasets/keremberke/license-plate-object-detection.

### Note ###

This was the original classes that RTDETR pretrained Nvidia TAO model had over 80+ classes. For the ***_annotations.coco.json*** we only had 2 classes, classId=0 for VRP and classId=1 for License_Plate. This will be used for ***labels.txt*** You can reference all the model architecture, training, evaluation, inferencing, exporting to .onnx, and generating .engine within ***LP.spec.yaml***

## Model

This project uses a ResNet-50 backbone integrated within a Transformer-based object detection model inspired by the RT-DETR framework trained on my own singular RTX 4070 GPU.

- Backbone: ResNet-50 pretrained on DETR weights with intermediate feature maps extracted from layers [1, 2, 3].

- Transformer: 6 decoder layers and 1 encoder layer, using GELU activation for the encoder and SiLU for other activations.

- Model specifics:

  - 300 object queries

  - Multi-scale feature levels with strides [8, 16, 32] and 256 channels each

  - 8 attention heads and feedforward dimension of 1024

  - Uses Variational Focal Loss (VFL) with increased weighting for classification (vfl_loss_coef=2.0) and adjusted bounding box and GIoU loss coefficients.

The training is configured with the AdamW optimizer, applying different learning rates for the backbone (0.00002) and the rest of the model (0.0001). Learning rate scheduling uses a multi-step decay at steps [50000, 100000, 150000] with warmup over 1000 steps.

Check ***LPspec.yaml*** for how training and model was done.

## Evaluation for best epoch: "epoch 6"

### TAO Toolkit Evaluation (val set)

| Metric        | Value   |
| ------------- | ------- |
| **mAP**       | 0.615   |
| **mAP\@50**   | 0.917   |
| **val\_loss** | 1827.90 |

### TAO Toolkit Evaluation (val set)

| Metric                  | Value |
| ----------------------- | ----- |
| **AP (0.50:0.95)**      | 0.626 |
| **AP50**                | 0.937 |
| **AP75**                | 0.765 |
| **AP<sub>large</sub>**  | 0.646 |
| **AP<sub>medium</sub>** | 0.663 |
| **AP<sub>small</sub>**  | 0.423 |
| **AR<sub>large</sub>**  | 0.710 |
| **AR<sub>medium</sub>** | 0.746 |
| **AR<sub>small</sub>**  | 0.554 |
| **AR\@1**               | 0.676 |
| **AR\@10**              | 0.698 |
| **AR\@100**             | 0.710 |

## Inference

`tao model rtdetr inference -e /workspace/tao-experiments/specs/LPspec.yaml`

## Exporting to .onnx

`tao model rtdetr export /e /workspace/tao-experiments/specs/LPspec.yaml`

## Creating .engine from .onnx

`tao deploy rtdetr gen_trt_engine -e /workspace/tao-experiments/specs/LPspec.yaml`

### Evaluating with TensorRT Engine
`tao deploy rtdetr evaluate -e /workspace/tao-experiments/specs/LPspec.yaml`

## Setup

Used WSL2, specifically Ubuntu-22.04 LTS for compatibility in Nvidia's software. Had to integrate Docker-Desktop from Windows 11 machine for WSL integration to run docker containers.

`docker pull <repository:tag>`

| **REPOSITORY**                 | **TAG**              | **IMAGE ID** | **CREATED**  | **SIZE** |
| ------------------------------ | -------------------- | ------------ | ------------ | -------- |
| custom-deepstream-7.1-rtdetr_lpd |   debug            | 5350d9b6791f | 5 hours ago  | 32.1GB   |
| nvcr.io/nvidia/tao/tao-toolkit | 6.0.0-deploy         | d6d7f77609ee | 7 days ago   | 15.5GB   |
| nvcr.io/nvidia/tao/tao-toolkit | 6.0.0-pyt            | 53c6580161ac | 7 days ago   | 30.3GB   |
| nvcr.io/nvidia/deepstream      | 7.1-triton-multiarch | 79ae634e62e9 | 9 months ago | 20.4GB   |

Within the Ubuntu-22.04 environment, I had to setup the .tao_mounts.json to configure for the docker containers and mounting from the WSL environment to my Windows path. The source and destination were symlinked from my Ubuntu environment to my Windows machine, so I can reference for the directory.

***Note***
- I also configured a custom docker container that was modified for my specific setup. For instance, I had debug statements within nvdsinfer_custombboxparser.cpp for the custom parser within NvDsInferParseCustomDDETRTAO to check for confidence thresholds, classIds, number of boxes, and more. The container also was specific to the deepstream_app_config.txt and nvdsinfer_config.yaml as well. Also, it was important to build .engine for the .onnx (using trtexec within the container to cache within) as opposed to copying from host machine as running deepstream complains about difference in .engine that's allowed within the container.    

***.tao_mounts.json (adjust based on your needs)***
```
{
  "Mounts": [
    {
      "source": "/mnt/c/Users/Sammy/GitHubProjects/License-Plate-Object-Detection/data",
      "destination": "/workspace/tao-experiments/data"
    },
    {
      "source": "/mnt/c/Users/Sammy/GitHubProjects/License-Plate-Object-Detection/results",
      "destination": "/workspace/tao-experiments/results"
    },
    {
      "source": "/mnt/c/Users/Sammy/GitHubProjects/License-Plate-Object-Detection/specs",
      "destination": "/workspace/tao-experiments/specs"
    },
    {
      "source": "/mnt/c/Users/Sammy/GitHubProjects/License-Plate-Object-Detection/models",
      "destination": "/workspace/tao-experiments/models"
    },
    {
      "source": "/mnt/c/Users/Sammy/GitHubProjects/License-Plate-Object-Detection/configs",
      "destination": "/workspace/tao-experiments/configs"
    },
    {
      "source": "/mnt/c/Users/Sammy/GitHubProjects/License-Plate-Object-Detection/engines",
      "destination": "/workspace/tao-experiments/engines"
    },
    {
      "source": "/mnt/c/Users/Sammy/GitHubProjects/License-Plate-Object-Detection/export",
      "destination": "/workspace/tao-experiments/export"
    }
],
  "Envs": [
    {
      "variable": "CUDA_DEVICE_ORDER",
      "value": "PCI_BUS_ID"
    },
    {
      "variable": "NVIDIA_VISIBLE_DEVICES",
      "value": "all"
    }
  ],
  "DockerOptions": {
    "shm_size": "16G",
    "ulimits": {
      "memlock": -1,
      "stack": 67108864
    },
    "user": "1000:1000",
    "ports": {
      "8888": 8888,
      "6006": 6006
    }
  }
}
```
***To get the Docker container***
- `docker pull samloveswater/custom-deepstream-7.1-rtdetr_lpd:latest .`

***To run the Deepstream container with display***
- `xhost +`
- `docker run -it --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/c/Users/Sammy/GitHubProjects/License-Plate-Object-Detection:/workspace/tao-experiments custom-deepstream-lpr:latest bash`

***During training to view different metrics***
- `tensorboard --logdir /mnt/c/Users/Sammy/GitHubProjects/License-Plate-Object-Detection/results/ --port 6007`

Here are the specific Nvidia docs used to consult for this setup:
- ***NGC CLI***: https://docs.ngc.nvidia.com/cli/cmd.html
- ***TAO Launcher CLI***: https://docs.nvidia.com/tao/tao-toolkit/text/tao_launcher.html
- ***TAO for Containers***: https://docs.nvidia.com/tao/tao-toolkit/text/quick_start_guide/running_from_containers.html
- ***RT-DETR***: https://docs.nvidia.com/tao/tao-toolkit/text/cv_finetuning/pytorch/object_detection/rt_detr.html#
- ***TAO Deploy for RT-DETR***: https://docs.nvidia.com/tao/tao-toolkit/text/tao_deploy/rtdetr.html#rtdetr-with-tao-deploy
- ***COCO Format for Object Detection***: https://docs.nvidia.com/tao/tao-toolkit/text/data_annotation_format.html#object-detection-coco-format
- ***Deepstream for Docker Containers***: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html
- ***TAO integration with Deepstream***: https://docs.nvidia.com/tao/tao-toolkit/text/ds_tao/deepstream_tao_integration.html
- ***Deploy Deepstream for DETR***: https://docs.nvidia.com/tao/tao-toolkit/text/ds_tao/deepstream_tao_integration.html
- ***Reference for LPD for Nvidia***: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/lpdnet
- ***More reference*** : https://developer.nvidia.com/blog/creating-a-real-time-license-plate-detection-and-recognition-app/

## Conclusion

Although comparing to the performance for my model and Nvidia's pretrained TAO, my model definitely does not go head-to-head with Nvidia's well-developed models. In the future I would love to explore more datasets like the CCPD either as an addition or separate training cycle. From my experience for re-training from scratch to experimenting with augmentation for the model, there are a few considerations I would love to explore later on. I could train for more epochs, or at least finetune more of the parameters. I even consider maybe training Nvidia's LDPNet model with more data and fine-tuning to increase their accuracy and evaluation. However, this was a challenge I was enthuastic about because I wanted to learn how Nvidia's advance software in AI worked in production systems and the great community/tooling goes into building these sophisticated models.

- ***CCPD***: https://github.com/detectRecog/CCPD
