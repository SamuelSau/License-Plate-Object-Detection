# License Plate Object Detection

This project uses NVIDIA TAO Toolkit v.6.0.0 and RT-DETR for license plate detection. 

![License Plate Detection](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExdnFlZHl5emN2NHNtaHlkdnRkY3lzNGRkY3loa3JyNm1nZG5kNTZrdiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/iOGP7u1U2CmhGFw2Xg/giphy.gif)

## Purpose for this project

So I was really interested in Nvidia's pretrained models such as the LDPNet in NGC catalog and challenged myself to see if I could train a custom ML model using RT-DETR (Real-Time Detection Transformer) using the latest version of TAO toolkit. 

## Dataset
 The model weights for the pretrained backbone was extracted from https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth. The dataset uses COCO format and was found in https://huggingface.co/datasets/keremberke/license-plate-object-detection.

### DETR R50 COCO BBox Detection Val5k Evaluation Results 

```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.624
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.205
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.611
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.312
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.805
```

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

`tao model rtdetr inference -e /linux/path/to/LPspec.yaml`

## Exporting to .onnx

`tao model rtdetr export -e /linux/path/to/specs/LPspec.yaml`

## Creating .engine from .onnx

`tao deploy rtdetr gen_trt_engine -e /linux/path/to/LPspec.yaml`

### Evaluating with TensorRT Engine
`tao deploy rtdetr evaluate -e /linux/path/to/LPspec.yaml`

## Setup

Used WSL2, specifically Ubuntu-22.04 LTS for compatibility in Nvidia's software. Had to integrate Docker-Desktop from Windows 11 machine for WSL integration to run docker containers.

`docker pull <repository:tag>`

| **REPOSITORY**                 | **TAG**              | **IMAGE ID** | **CREATED**  | **SIZE** |
| ------------------------------ | -------------------- | ------------ | ------------ | -------- |
| custom-deepstream-7.1-rtdetr_lpd |   latest           | 5350d9b6791f | 5 hours ago  | 32.1GB   |
| nvcr.io/nvidia/tao/tao-toolkit | 6.0.0-deploy         | d6d7f77609ee | 7 days ago   | 15.5GB   |
| nvcr.io/nvidia/tao/tao-toolkit | 6.0.0-pyt            | 53c6580161ac | 7 days ago   | 30.3GB   |
| nvcr.io/nvidia/deepstream      | 7.1-triton-multiarch | 79ae634e62e9 | 9 months ago | 20.4GB   |

Within the Ubuntu-22.04 environment, I had to setup the .tao_mounts.json to configure for the docker containers and mounting from the WSL environment to my Windows path. The source and destination were symlinked from my Ubuntu environment to my Windows machine, so I can reference for the directory.

***Note***
- I also configured a custom docker container that was modified for my specific setup. I used `docker commit nvcr.io/nvidia/deepstream-7.1-triton-multiarch:latest` to create it. For instance, I had debug statements within nvdsinfer_custombboxparser.cpp for the custom parser within NvDsInferParseCustomDDETRTAO to check for confidence thresholds, classIds, number of boxes, and more. The container also was specific to the deepstream_app_config.txt and nvdsinfer_config.yaml as well. Also, it was important to build .engine for the .onnx (using trtexec within the container to cache within) as opposed to copying from host machine as running deepstream complains about difference in .engine (such as GPU architecture, CUDA version, etc...) that's allowed within the container. Although evaluations from the docker container and TAO Launcher CLI are identical despite different .engine generations.     

***.tao_mounts.json (adjust based on your needs)***
``` 
{
  "Mounts": [
    {
      "source": "/windows/path/to/data",
      "destination": "/linux/path/to/data"
    },
    {
      "source": "/windows/path/to/results",
      "destination": "/linux/path/to/results"
    },
    {
      "source": "/windows/path/to/specs",
      "destination": "/linux/path/to/specs"
    },
    {
      "source": "/windows/path/to/models",
      "destination": "/linux/path/to/models"
    },
    {
      "source": "/windows/path/to/configs",
      "destination": "/linux/path/to/configs"
    },
    {
      "source": "/windows/path/to/engines",
      "destination": "/workspace/tao-experiments/engines"
    },
    {
      "source": "/windows/path/to/export",
      "destination": "/linux/path/to/export"
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
***To get the custom Docker container***
- `docker pull samloveswater/custom-deepstream-7.1-rtdetr_lpd:latest .`

***To run the Deepstream container with display***
- `xhost +`
- `docker run -it --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /windows/path/to/:/linux/path/to/ custom-deepstream-lpr:latest bash`

***During training to view different metrics***
- `tensorboard --logdir /windows/path/to/results/ --port 6007`

Here are the specific Nvidia docs used to consult for this setup:
- ***NGC CLI***: https://docs.ngc.nvidia.com/cli/cmd.html
- ***TAO Launcher CLI***: https://docs.nvidia.com/tao/tao-toolkit/text/tao_launcher.html
- ***TAO for Containers***: https://docs.nvidia.com/tao/tao-toolkit/text/quick_start_guide/running_from_containers.html
- ***RT-DETR***: https://docs.nvidia.com/tao/tao-toolkit/text/cv_finetuning/pytorch/object_detection/rt_detr.html#
- ***TAO Deploy for RT-DETR***: https://docs.nvidia.com/tao/tao-toolkit/text/tao_deploy/rtdetr.html#rtdetr-with-tao-deploy
- ***COCO for Object Detection***: https://docs.nvidia.com/tao/tao-toolkit/text/data_annotation_format.html#object-detection-coco-format
- ***Deepstream for Docker Containers***: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html
- ***TAO integration with Deepstream***: https://docs.nvidia.com/tao/tao-toolkit/text/ds_tao/deepstream_tao_integration.html
- ***Deploy Deepstream for DETR***: https://docs.nvidia.com/tao/tao-toolkit/text/ds_tao/deepstream_tao_integration.html
- ***Reference for LPD for Nvidia***: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/lpdnet
- ***More reference*** : https://developer.nvidia.com/blog/creating-a-real-time-license-plate-detection-and-recognition-app/

## Conclusion

While my model doesn’t match the performance of NVIDIA’s pretrained TAO models, this project was a valuable learning experience. I explored training from scratch, data augmentation, and model tuning. In the future, I plan to experiment with additional datasets like CCPD and potentially fine-tune models like NVIDIA’s LPDNet to further improve performance. This challenge gave me hands-on insight into how advanced AI systems are built and deployed using NVIDIA’s powerful tools and community support.

- ***CCPD***: https://github.com/detectRecog/CCPD
