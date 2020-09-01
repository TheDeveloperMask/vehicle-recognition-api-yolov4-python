# Vehicle Recognition API

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Python server for [Spectrico's vehicle make and color classification](http://spectrico.com/car-make-model-recognition.html). The Flask server exposes REST API for car brand&color recognition. It consists of an object detector for finding the cars, and two classifiers to recognize the makes and the colors of the detected cars. The object detector is an implementation of YOLOv4 (OpenCV DNN backend). YOLOv4 weights were downloaded from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights). The classifiers are based on MobileNet v3 (Alibaba MNN backend). 

The full version recognize make, model and color of the vehicles. Here is a web demo to test it: [Vehicle Make and Model Recognition](http://spectrico.com/demo-car-mmr.html)

The API is simple: make a HTTP POST request to local host on port 6000.
The input image must be send using multipart/form-data encoding. It has to be jpg or png. Tested on Windows 10 and Ubuntu Linux.


![image](https://github.com/spectrico/vehicle-recognition-api-yolov4/raw/master/car-make-model.png?raw=true)

---

#### Usage
The server is started using:
```
$ python car_recognition_server.py
```
The request format using curl is:
```
curl "http://127.0.0.1:6000" -H "Content-Type: multipart/form-data" --form "image=@cars.jpg"
```
Python client example:
```
python api_client.py
```
The response is in JSON format:
```
{
  "cars": [
    {
      "make": "Porsche",
      "color": "yellow",
      "make_prob": "0.9999634027481079",
      "color_prob": "0.9446729421615601",
      "obj_prob": "0.99820864",
      "rect": {
        "left": "31",
        "top": "273",
        "width": "370",
        "height": "194"
      }
    },
    {
      "make": "Dodge",
      "color": "green",
      "make_prob": "0.9417884945869446",
      "color_prob": "0.7946781516075134",
      "obj_prob": "0.9469202",
      "rect": {
        "left": "466",
        "top": "302",
        "width": "383",
        "height": "170"
      }
    },
    {
      "make": "Tesla",
      "color": "red",
      "make_prob": "0.9999299049377441",
      "color_prob": "0.9288212060928345",
      "obj_prob": "0.75779015",
      "rect": {
        "left": "469",
        "top": "31",
        "width": "360",
        "height": "186"
      }
    },
    {
      "make": "Volvo",
      "color": "grey",
      "make_prob": "0.9970195889472961",
      "color_prob": "0.9284484386444092",
      "obj_prob": "0.69096345",
      "rect": {
        "left": "22",
        "top": "13",
        "width": "379",
        "height": "218"
      }
    }
  ]
}

```
---
## Dependencies
  - pip install numpy
  - pip install opencv-python
  - pip install MNN
  - yolov4.weights must be downloaded from [https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) and saved in folder yolov4

  If you use Windows, the OpenCV package is recommended to be installed from: https://www.lfd.uci.edu/~gohlke/pythonlibs/

---
## Credits
The YOLOv4 object detector is from: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
```
@article{bochkovskiy2020yolov4,
  title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Bochkovskiy, Alexey and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2004.10934},
  year={2020}
}
```
The car classifiers are based on MobileNetV3 mobile architecture: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
```
@inproceedings{howard2019searching,
  title={Searching for mobilenetv3},
  author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and others},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1314--1324},
  year={2019}
}
```

The runtime library of the classifier is [MNN](https://github.com/alibaba/MNN)
```
@inproceedings{alibaba2020mnn,
  author = {Jiang, Xiaotang and Wang, Huan and Chen, Yiliu and Wu, Ziqi and Wang, Lichuan and Zou, Bin and Yang, Yafeng and Cui, Zongyang and Cai, Yu and Yu, Tianhang and Lv, Chengfei and Wu, Zhihua},
  title = {MNN: A Universal and Efficient Inference Engine},
  booktitle = {MLSys},
  year = {2020}
}
```
