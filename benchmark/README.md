# Benchmarks for frigate-detector-edgetpu-yolov9

This is a method for measuring the accuracy and speed of the models available to Frigate for Google Coral devices.

These performance benchmarks were measured using the models available for download here, running on Coral TPU hardware. The threshold for minimum score was 0.4, and the NMS threshold was 0.4. mAP50 is the industry-standard accuracy score that considers a detection "correct" only if the predicted bounding box overlaps the real object by at least 50%.

## Requirements

In addition to a system with docker installed, download the COCO validation images and labels here:

* [COCO validation images](http://images.cocodataset.org/zips/val2017.zip)
* [labels](https://huggingface.co/datasets/merve/coco/blob/main/annotations/instances_train2017.json)

## Example commands

Make sure there are no other processes running that use the Coral device, ie. stop the Frigate container.

```
docker build -t benchmark .

docker run -it --rm \
  --privileged \
  --device /dev/apex_0 \
  -v $(pwd):/app \
  -v /path/to/your/models:/models \
  -v /path/to/coco:/coco \
  benchmark \
  --model /models/yolov9-s-relu6_320_edgetpu.tflite \
  --size 320 \
  --coco_ann /coco/instances_val2017.json \
  --coco_img_dir /coco/val2017/
```
