# frigate-detector-edgetpu-yolo9

This repository provides a custom detector plugin for [Frigate](https://github.com/blakeblackshear/frigate), specifically designed to enable the Google Coral Edge TPU to run a **YOLOv9** model to detect objects visible in camera feeds. The plugin handles the necessary post-processing for YOLO model outputs, making them compatible with Frigate's detection pipeline without modifying Frigate's core source code. YOLOv9 model weights can be downloaded here as well, see the release notes and installation instructions.

![abbeyroad-detections](https://github.com/user-attachments/assets/a4dfbe6f-2ed3-43f1-9697-6ea119e463fe)

## Why YOLOv9 and this Plugin?

Frigate is an open-source NVR system that supports various detector acceleration hardware, including the Google Coral Edge TPU. While the default SSD MobileDet model is an effective choice in many systems, running a YOLOv9 model may improve detection accuracy and reduce false positives.

YOLOv9 can run efficiently on the EdgeTPU, and requires a small amount of post-processing by the CPU. This detection method is both high-speed and low-energy, handling approximately 100 detections per second and using only a few Watts of power, while leaving the CPU available for other tasks. Another consideration is minimizing heat produced by the CPU, which can challenge my system's small fan.

There are other YOLO models with incompatible software license terms (eg Ultralytics is licensed with the AGPL). This approach is meant to be used to start with the GPL-licensed YOLOv9 model, which IS compatible with Frigate (also MIT-licensed). For more information, see [this discussion in Frigate's issues](https://github.com/blakeblackshear/frigate/discussions/15630#discussioncomment-11639733).

## Features

*   **Improved Accuracy:** Fewer false positive detections than the default SSD MobileDet object detection model.
*   **YOLO Compatibility:** Processes output from YOLO models exported for Edge TPU.
*   **Google Coral Edge TPU Support:** Optimized for efficient inference on Coral devices. 10ms inference time at 320x320 image size (vs 7ms for the MobileDet model)
*   **Simple Integration:** Adds as a Frigate plugin via a Docker volume mount, no core Frigate code changes needed.

## Example System Performance

* CPU is Intel 3rd Generation i7, 8GB RAM, circa 2012 (the built in GPU is not supported by OpenVINO)
* Google Coral mini-PCIe card
* 10ms detection speed when running a load test averaging 40 detections per second for images having size 320x320 pixels
* zero skipped frames
* Detector CPU usage varies between 15% and 20%

Regarding CPU usage, all YOLO models (pre YOLO v10) use the CPU during post-processing to decode and de-duplicate the raw results.

## Model Input Size, Detection Sensitivity, and Run Time

There are currently two versions of YOLO v9 model weights available for download that can be used with this plugin:

* [YOLO v9 "small" 320x320 input size](https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v1.5/yolov9-s-relu6-tpumax_320_int8_edgetpu.tflite)

This version takes 10ms to run detections on my old system, and should run a bit faster on newer systems. This translates to a capacity of 100 detections per second. Note that there can be multiple detections run for a single frame from the video feed, if there are multiple separate areas where motion gets detected. According to the EdgeTPU compiler, all 334 operations that are part of the model run on the Google Coral TPU.

* [YOLO v9 "small" 512x512 input size](https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v1.5/yolov9-s-relu6-tpumax_512_int8_edgetpu.tflite)

This model runs in about 21ms. It can detect smaller objects relative to the full frame size when compared with the 320x320 size version of the model. It might be preferable in situations where it is important to detect small or faraway objects within a large region of motion. All but 2 of the 334 operations in the model run on the Google Coral TPU, and the 2 that do not run at the end of the model's sequence which means the slowdown is minimal. The main reason for the slower operation is the larger tensor sizes.

Other versions of YOLO may run with this plugin, but they are not supported or offered for download here.

## Prerequisites

Before you begin, ensure you have:

* **A running Frigate installation:** This plugin has been tested with Frigate **v0.15 and v0.16**.
* **Google Coral Edge TPU:** Properly configured and accessible by your Frigate Docker container. Should work with any variation of Google Coral hardware (USB, M2, mPCIe)

## Installation

Follow these steps to integrate the replacement `edgetpu_tfl.py` plugin into your Frigate setup.

### 1. Download the Model Weights, Plugin File and Class Label File

Create a directory on your host system to store the plugin file. For example, you might create `/opt/frigate-plugins/`.

```bash
sudo mkdir -p /opt/frigate-plugins
cd /opt/frigate-plugins
# download weights
sudo wget https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v1.5/yolov9-s-relu6-tpumax_320_int8_edgetpu.tflite
# download plugin
sudo wget https://raw.githubusercontent.com/dbro/frigate-detector-edgetpu-yolo9/main/edgetpu_tfl.py
# download labels
sudo wget https://raw.githubusercontent.com/dbro/frigate-detector-edgetpu-yolo9/main/labels-coco17.txt
```

### 2. Update docker-compose.yml

You need to add a volume mount to your Frigate service in your docker-compose.yml file. This mounts the plugin file into Frigate's detector plugins directory.

Locate your Frigate service definition and add the following two lines in the volumes: section. Adjust /opt/frigate-plugins/edgetpu_tfl.py if you stored the file elsewhere.

The second line to add will make your YOLO model accessible.

```yaml
# ... other services ...
frigate:
  # ... other frigate configurations ...
  volumes:
    # ... existing volumes ...
    - /opt/frigate-plugins/edgetpu_tfl.py:/opt/frigate/frigate/detectors/plugins/edgetpu_tfl.py:ro
    - /opt/frigate-plugins/labels-coco17.txt:/opt/frigate/models/labels-coco17.txt:ro
    - /opt/frigate-plugins/yolov9-s-relu6-tpumax_320_int8_edgetpu.tflite:/opt/frigate/models/yolov9-s-relu6-tpumax_320_int8_edgetpu.tflite:ro
  # ... rest of frigate service ...
```

After modifying docker-compose.yml, restart your Frigate container:

```bash
docker-compose down
docker-compose up -d
```

### 3. Configure Frigate's config.yml

Now, you need to tell Frigate to use this new detector plugin and your YOLO model.

In your config.yml, in the model: section, add the model_type parameter as shown below and update the model path to point to your YOLO model file that you mounted in the previous step.

```yaml
model:
  model_type: yolo-generic
  labelmap_path: /opt/frigate/models/labels-coco17.txt
  path: /opt/frigate/models/yolov9-s-relu6-tpumax_320_int8_edgetpu.tflite
  # Optionally specify the model dimensions (these are the same as Frigate's default 320x320)
  width: 320
  height: 320
detectors:
  coral:
    type: edgetpu
    # ... other detector settings ...
```

### 4. Restart Frigate and Check Performance

Save the Frigate configuration and rexstart Frigate.

* Are there any error messages in Frigate logs during startup?
* What is the inference speed reported in Frigate's System Metrics page?
* What is the level of Detector CPU Usage?
* Are there any skipped frames?
* How many detection errors do you observe? Look for both false positives and false negatives.

## How it Works With Frigate

This modified version of edgetpu_tfl.py replaces the standard version of the plugin and acts as an intermediary. When Frigate requests a detection:

1. Frigate's standard Edge TPU handler passes the image frame to the plugin.
2. The plugin loads your specified YOLO model.
3. It performs inference on the Edge TPU.
4. It then post-processes the raw output from the YOLO model to transform it into the format expected by Frigate (e.g., converting bounding box coordinates, handling confidence scores).
5. The processed detections are returned to Frigate.

Note that Frigate does not automatically send the entire image from the camera to the detector. Instead, it looks for areas in the image that changed from previous frames, and sends only these regions with motion to the detector. There can be more than one region sent per frame from the camera. For example, on a still day, a region might be tightly cropped around a cat walking across a lawn. On a windy autumn day with many leaves or rain creating motion in the overall image, the region of motion could be the entire image or there could be multiple regions sent to the detector. Frigate next resizes the region to match the detection model's input size, then sends it to the detector.

To understand how Frigate uses motion to trigger object detection processing, see [this explanation in the Frigate FAQ](https://docs.frigate.video/configuration/masks/#further-clarification) and a live view of motion regions is in Frigate's "Debug" view in the Settings page..

## Lessons Learned Getting YOLO v9 to Run on Google Coral

It is possible to convert the pre-trained weights for YOLO from PyTorch format to ONNX and then to TFLite and then compile for EdgeTPU, which runs on Google Coral. However, simple conversions of the models available for download perform poorly on the Coral, because it uses operations that are not appropriate for the 8-bit limitation of Google Coral. This leads to a total loss of information and zero detections. The following changes to the YOLO model structure enable the information to be preserved accurately through the sequence of operations run on Google Coral. The goal is to let the Coral perform all the convolution operations, and then send the data for post-processing on the CPU.

* **ReLU6 activation** should be used instead of SiLU activation. It is smaller and faster than SiLU, and quantizes better.
* **Send Logit scores** to the CPU for post-processing to transform them into probabilities. The Coral cannot do the sigmoid() transformation.
* **Decode boxes and run Non-Maximum Supression (NMS) on the CPU** because the Coral cannot run these operations efficiently.
* **Separate tensors for each quantization scale** to preserve precision.
* **Data size limits are tricky.** For example, using a non-standard order of dimensions (BNC) in the boxes tensor enables it to run entirely on the TPU, but with the standard order (BCN) some operations must either run on the CPU or be done repeatedly on smaller chunks. Coral has a limited amount of on-chip RAM for caching the intermediate results of the operations, and if it needs more it will use the (slower) system RAM.
* **Post-processing takes time** Even after optimization, I estimate one-third of the 10ms detection time involves post-processing the output from the model running on Google Coral.

Applying those lessons lets the Google Coral run all of the convolutiuon operations on its TPU for a YOLO v9 "s" model with size 320x320 pixels. The resulting model uses 385kB of off-chip RAM, which slows down the process. It is possible to do some of the post-processing decoding on the TPU, but it is slower than the CPU for those tasks.

## Contributing

This project is open-source. If you find issues or have improvements, feel free to open an issue or submit a pull request.

## References

* [https://github.com/blakeblackshear/frigate](https://github.com/blakeblackshear/frigate)
* [https://github.com/WongKinYiu/yolov9/tree/main](https://github.com/WongKinYiu/yolov9/tree/main)
* [https://engineering.cloudflight.io/quantized-yolo-for-edge-solutions](https://engineering.cloudflight.io/quantized-yolo-for-edge-solutions)

## License
This project is licensed under the MIT License. See the LICENSE file for details.
