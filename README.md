# frigate-detector-edgetpu-yolo9

This repository provides a custom detector plugin for Frigate, specifically designed to enable the use of **YOLOv9** models with Google Coral Edge TPUs. This plugin handles the necessary post-processing for YOLO model outputs, making them compatible with Frigate's detection pipeline without modifying Frigate's core source code.

## Why YOLOv9 and this Plugin?

Frigate, an open-source NVR system, supports various detector hardware, including the Google Coral Edge TPU. While the default SSD MobileDet model works well in many case, in some cases YOLOv9 models can often offer improved detection accuracy and reduced false positives.

YOLOv9 can run almost entirely on the TPU of the Google Coral device. This detection method is both high-speed and low-energy, handling approximately 90 detections per second and using only a few Watts of power, while leaving the CPU available for other tasks. Another consideration is minimizing heat produced by the CPU, which can challenge my system's small fan.

There are other YOLO models with incompatible software license terms (eg Ultralytics is licensed with the AGPL). This approach is meant to be used to start with the GPL-licensed YOLOv9 model, which IS compatible with Frigate (also MIT-licensed). For more information, see [this discussion in Frigate's issues](https://github.com/blakeblackshear/frigate/discussions/15630#discussioncomment-11639733).

## Features

*   **Improved Accuracy:** Fewer false positive detections than the default SSD MobileDet object detection model.
*   **YOLO Compatibility:** Processes output from YOLO models exported for Edge TPU.
*   **Google Coral Edge TPU Support:** Optimized for efficient inference on Coral devices. ~10ms inference time (vs 7ms for the MobileDet model)
*   **Simple Integration:** Adds as a Frigate plugin via a Docker volume mount, no core Frigate code changes needed.

## Caveat

The plugin code does some post-processing on the CPU, so this approach will use the CPU more than the default SSD MobileDet model. However with careful selection of the YOLO model generation, type, and resolution the amount of CPU involved does not appear to be a bottleneck (on a very old machine, Intel 3rd generation i7).

## Example System Performance

* Google Coral mini-PCIe card
* average 25 detections per second
* zero skipped frames
* 12ms detection speed
* Detector CPU usage varies between 10%
* CPU is Intel 3rd Generation i7, 8GB RAM, circa 2012


## Prerequisites

Before you begin, ensure you have:

* **A running Frigate installation:** This plugin has been tested with Frigate **v0.15 and v0.16**.
* **Google Coral Edge TPU:** Properly configured and accessible by your Frigate Docker container. Should work with any variation of Google Coral hardware (USB, M2, mPCIe)

## Installation

Follow these steps to integrate the replacement `edgetpu_tfl.py` plugin into your Frigate setup.

### 1. Get a YOLO Model File for Edge TPU

* Use this Google Colab notebook to generate a MIT-licensed yolo v9 "tiny" model with resolution 192x192.

[https://colab.research.google.com/drive/1n3sCrsVWJKu2H5KjgUyl3akIUYykSYaP?usp=sharing](https://colab.research.google.com/drive/1n3sCrsVWJKu2H5KjgUyl3akIUYykSYaP?usp=sharing)


* **IMPORTANT** Look at the output of the conversion step and see what it says about the operations being supported by the Edge TPU. Any steps that are not "Mapped to the Edge TPU" will be run on the CPU, which is slower+hotter than if they were to run on the TPU. If a large portion (25+% ?) of the operations do not get "Mapped to Edge TPU" then the model will probably run slowly, 30 ms or more per detection. Try a smaller or less complex or older model.  **LOOK FOR THIS INDICATION OF A PROBLEM**:
```
Model successfully compiled but not all operations are supported by the Edge TPU. A percentage of the model will instead run on the CPU, which is slower. If possible, consider updating your model to use only operations supported by the Edge TPU. For details, visit g.co/coral/model-reqs.
Number of operations that will run on Edge TPU: 719
Number of operations that will run on CPU: 31

Operator                       Count      Status

SOFTMAX                        3          Mapped to Edge TPU
STRIDED_SLICE                  16         Mapped to Edge TPU
PAD                            2          Mapped to Edge TPU
TRANSPOSE                      2          Operation is otherwise supported, but not mapped due to some unspecified limitation
TRANSPOSE                      1          More than one subgraph is not supported
TRANSPOSE                      14         Mapped to Edge TPU
SPLIT                          6          Mapped to Edge TPU
ADD                            48         Mapped to Edge TPU
ADD                            6          More than one subgraph is not supported
SUB                            6          More than one subgraph is not supported
QUANTIZE                       5          Mapped to Edge TPU
MAXIMUM                        3          Mapped to Edge TPU
AVERAGE_POOL_2D                5          Mapped to Edge TPU
GATHER                         6          Operation not supported
MINIMUM                        3          Mapped to Edge TPU
CONCATENATION                  33         Mapped to Edge TPU
CONCATENATION                  6          More than one subgraph is not supported
MAX_POOL_2D                    3          Mapped to Edge TPU
RESHAPE                        9          Mapped to Edge TPU
FULLY_CONNECTED                3          Mapped to Edge TPU
LOGISTIC                       182        Mapped to Edge TPU
CONV_2D                        203        Mapped to Edge TPU
MUL                            179        Mapped to Edge TPU
MUL                            4          More than one subgraph is not supported
RESIZE_NEAREST_NEIGHBOR        2          Mapped to Edge TPU
Compilation child process completed within timeout period.
Compilation succeeded! 


✓ Edge TPU model created: yolov9-t-converted_onnx2tf_quant_tflite/yolov9-t-converted_full_integer_quant_edgetpu.tflite
  Size: 3588.8 KB
```
   * If you see something that indicates a significant number of operations will run on the CPU **re-run the export step again with new settings**. For example, reduce the value for the imgsz parameter to one of these: 320, 288, 256, 224, 192, 160. Or try again with a different .pt file for a different size of the model (instead of "small", try "tiny").
* When you have a model that minimizes CPU operations, download it and copy it to your docker host, to somewhere like /opt/frigate-plugins/

### 2. Download the Plugin File and Class Label File

Create a directory on your host system to store the plugin file and the labels file. For example, you might create `/opt/frigate-plugins/`.

```bash
sudo mkdir -p /opt/frigate-plugins
cd /opt/frigate-plugins
sudo wget https://raw.githubusercontent.com/dbro/frigate-detector-edgetpu-yolo9/main/edgetpu_tfl.py
sudo wget https://raw.githubusercontent.com/dbro/frigate-detector-edgetpu-yolo9/main/coco-labels.txt
# Or, if you cloned the repo:
# sudo cp path/to/cloned/repo/edgetpu_tfl.py /opt/frigate-plugins/
# sudo cp path/to/cloned/repo/coco-labels.txt /opt/frigate-plugins/
```

### 3. Update docker-compose.yml

You need to add a volume mount to your Frigate service in your docker-compose.yml file. This mounts the plugin file into Frigate's detector plugins directory.

Locate your Frigate service definition and add the following two lines under the volumes: section. Adjust /opt/frigate-plugins/edgetpu_tfl.py if you stored the file elsewhere.

The second line to add will make the COCO class labels accessible.

The third line to add will make your YOLO model accessible.

```yaml
# ... other services ...
frigate:
  # ... other frigate configurations ...
  volumes:
    # ... existing volumes ...
    - /opt/frigate-plugins/edgetpu_tfl.py:/opt/frigate/frigate/detectors/plugins/edgetpu_tfl.py:ro
    - /opt/frigate-plugins/coco-labels.txt:/opt/frigate/frigate/models/coco-labels.txt:ro
    - /opt/frigate-plugins/yolov9t_full_integer_quant_edgetpu.tflite:/opt/frigate/models/yolov9t_full_integer_quant_edgetpu.tflite:ro
  # ... rest of frigate service ...
```

After modifying docker-compose.yml, restart your Frigate container:

```bash
docker-compose down
docker-compose up -d
```

### 4. Configure Frigate's config.yml

Now, you need to tell Frigate to use this new detector plugin and your YOLO model.

In your config.yml, under the detectors: section, add the model_type parameter as shown below and update the model path to point to your YOLO model file that you mounted in the previous step.

```yaml
detectors:
  coral:
    type: edgetpu
    # ... other detector settings ...
  model:
      model_type: yolo-generic
      labelmap_path: /opt/frigate-plugins/coco-labels.txt
      path: /opt/frigate-plugins/yolov9t_full_integer_quant_edgetpu.tflite # Update this to your model's path
      # Optionally, if your model has specific input dimensions (eg 192x192), uncomment these lines:
      # width: 192
      # height: 192
```

### 5. Restart Frigate and Check Performance

Save the Frigate configuration and rexstart Frigate.

* Are there any error messages in Frigate logs during startup?
* What is the inference speed reported in Frigate's System Metrics page?
* What is the level of Detector CPU Usage?
* Are there any skipped frames?
* How many detection errors do you observe? Look for both false positives and false negatives.

## How it Works

This modified version of edgetpu_tfl.py replaces the standard version of the plugin and acts as an intermediary. When Frigate requests a detection:

1. Frigate's standard Edge TPU handler passes the image frame to the plugin.
2. The plugin loads your specified YOLO model.
3. It performs inference on the Edge TPU.
4. It then post-processes the raw output from the YOLO model to transform it into the format expected by Frigate (e.g., converting bounding box coordinates, handling confidence scores).
5. The processed detections are returned to Frigate.

## Contributing

This project is open-source. If you find issues or have improvements, feel free to open an issue or submit a pull request.

## References

* [https://github.com/blakeblackshear/frigate](https://github.com/blakeblackshear/frigate)
* [https://github.com/WongKinYiu/yolov9/tree/main](https://github.com/WongKinYiu/yolov9/tree/main)

## License
This project is licensed under the MIT License. See the LICENSE file for details.
