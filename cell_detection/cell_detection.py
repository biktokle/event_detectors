import json
import socket
import sys
import time
import tensorflow as tf
from skimage import io
from centroidtracker import CentroidTracker
import numpy as np
from communication import protocol, server

DETECT = 'DETECT'
port = int(sys.argv[1])
class Detector(server.Server):
    def __init__(self, port, chosen_protocol, threshold=0.5):
        self.model = load_model()
        self.ct = CentroidTracker(maxDisappeared=3)
        self.threshold = threshold
        super().__init__(port, chosen_protocol)



    def handle_request(self, data):
        image_path, region, b_boxes = data
        image = io.imread(image_path)
        image = image[region['ymin']:region['ymax'], region['xmin']:region['xmax']]
        image = image_to_8bit_equalized(image)
        objects, rects = self.detect(image)
        detection_dict = {}
        for did in objects:
            xcenter, ycenter = objects[did]
            xmin, ymin, xmax, ymax = rects[did]
            did_dict = {
                'xcenter': int(xcenter),
                'ycenter': int(ycenter),
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax)
            }
            detection_dict[str(did)] = did_dict
        return None, detection_dict






    def detect(self, img):

        (H, W) = img.shape[:2]
        image_np = np.array(img)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = self.model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        rects = []
        for (ymin, xmin, ymax, xmax), score in zip(detections['detection_boxes'], detections['detection_scores']):
            if score > self.threshold:
                rects += [(xmin, ymin, xmax, ymax) * np.array([W, H, W, H])]

        objects, rects = self.ct.update(rects)

        return objects, rects



# Saved model path
PATH_TO_SAVED_MODEL = r'.\cell_detector_igraph\saved_model'


def get_bounding_boxes(image,
                           ymin,
                           xmin,
                           ymax,
                           xmax):
  im_height, im_width, _ = image.shape
  return (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)


def load_model():
    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    return detect_fn


def image_to_8bit_equalized(image):
    ratio = np.amax(image) / 256
    img8 = (image / ratio).astype('uint8')

    return img8



# class Server:
#     def __init__(self, port):
#         self.conn = setup_connection('localhost', port)
#
#     def get_request(self):
#         return str(self.conn.recv(1024).decode('utf-8'))
#
#     def answer(self, data):
#         self.conn.sendall(bytes(data, encoding="utf-8"))



# def setup_connection(host, port):
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.bind((host, port))
#     s.listen()
#     conn, addr = s.accept()
#     return conn
#
#
#
# def parse_request(request):
#     data = json.loads(request)
#     if data['type'] == DETECT:
#         return data['data']['image_path'], data['data']['region']
#     else:
#         raise Exception('Unknown request from client')
#
# def create_detection_response(detections):
#     response = {'type': DETECT, 'data': {'detections': detections}}
#     return json.dumps(response)




if __name__=='__main__':
    Detector(port, protocol.DETECT)
