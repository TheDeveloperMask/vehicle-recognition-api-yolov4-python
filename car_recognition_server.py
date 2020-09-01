# Copyright © 2020 by Spectrico

from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np
import classifier
import traceback
import json
import io
import cv2

app = Flask(__name__)
CORS(app)

net = cv2.dnn_DetectionModel('yolov4/yolov4.cfg', 'yolov4/yolov4.weights')
net.setInputSize(608, 608)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)
car_make_classifier = classifier.Classifier('model-weights-spectrico-car-brand-recognition-mobilenet_v3-224x224-170620.mnn', 'labels-makes.txt')
car_color_classifier = classifier.Classifier('model-weights-spectrico-car-colors-recognition-mobilenet_v3-224x224-180420.mnn', 'labels-colors.txt')

@app.route("/", methods = ['POST'])
@cross_origin()
def objectDetect():
    if request.headers['Content-Type'].startswith('multipart/form-data'):
        objects = []
        try:
            import numpy as np
            data = np.frombuffer(request.files['image'].read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                response = app.response_class(
                    response='415 Bad image',
                    status=415,
                    mimetype='text/plain'
                )
                return response

            classes, confidences, boxes = net.detect(img, confThreshold=0.1, nmsThreshold=0.4)
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                if classId in [2, 5, 7]:
                    if confidence > 0.3:
                        left, top, width, height = box
                        x1 = left
                        x2 = left + width
                        y1 = top
                        y2 = top + height
                        car_img = img[y1:y2, x1:x2]
                        (make, make_confidence) = car_make_classifier.predict(car_img)
                        (color, color_confidence) = car_color_classifier.predict(car_img)
                        rect = {"left": str(x1), "top": str(y1), "width": str(x2-x1), "height": str(y2-y1)}
                        objects.append({"make": make, "color": color, "make_prob": str(make_confidence), "color_prob": str(color_confidence), "obj_prob": str(confidence), "rect": rect})
        except:
            traceback.print_exc()
            response = app.response_class(
                response='415 Unsupported Media Type',
                status=415,
                mimetype='text/plain'
            )
            return response
        response = app.response_class(
            response=json.dumps({'cars': objects}),
            status=200,
            mimetype='application/json'
        )
        return response
    else:
        return "415 Unsupported Media Type"

@app.route("/", methods = ['GET'])
@cross_origin()
def version():
    response = app.response_class(
        response='{"version":"car make and color recognition 1.0"}',
        status=200,
        mimetype='application/json'
    )
    return response

@app.before_request
def option_autoreply():
    """ Always reply 200 on OPTIONS request """
    if request.method == 'OPTIONS':
        resp = app.make_default_options_response()

        headers = None
        if 'ACCESS_CONTROL_REQUEST_HEADERS' in request.headers:
            headers = request.headers['ACCESS_CONTROL_REQUEST_HEADERS']

        h = resp.headers

        # Allow the origin which made the XHR
        h['Access-Control-Allow-Origin'] = request.headers['Origin']
        # Allow the actual method
        h['Access-Control-Allow-Methods'] = request.headers['Access-Control-Request-Method']
        # Allow for 10 seconds
        h['Access-Control-Max-Age'] = "10"

        # We also keep current headers
        if headers is not None:
            h['Access-Control-Allow-Headers'] = headers

        return resp


@app.after_request
def set_allow_origin(resp):
    """ Set origin for GET, POST, PUT, DELETE requests """

    h = resp.headers

    # Allow crossdomain for other HTTP Verbs
    if request.method != 'OPTIONS' and 'Origin' in request.headers:
        h['Access-Control-Allow-Origin'] = request.headers['Origin']

    return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000, debug=False, use_reloader=False, threaded=False)
