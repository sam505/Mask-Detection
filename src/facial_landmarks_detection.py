'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import pprint
import cv2
import logging
import time
import numpy as np
import argparse
import tensorflow as tf
import time
import logging
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class InputFeeder:
    def __init__(self, input_type, input_file):
        self.input_type = input_type
        if input_type == 'video' or input_type == 'image':
            self.input_file = input_file

    def load_data(self):
        if self.input_type == 'video':
            self.cap = cv2.VideoCapture(self.input_file)
        elif self.input_type == 'cam':
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.imread(self.input_file)

    def next_batch(self):
        if args.input_type == 'image':
            return self.cap
        else:
            while self.cap.isOpened():
                _, frame = self.cap.read()
                return frame

    def close(self):
        if not self.input_type == 'image':
            self.cap.release()


class FaceDetection:

    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights = model_name + '.xml'
        self.model_structure = model_name + '.bin'
        self.device = device
        self.net = None
        self.count = 0

        return

        raise NotImplementedError

    def load_model(self):
        core = IECore()
        start = time.time()
        model = core.read_network(self.model_weights, self.model_structure)
        logger.info("Loading the Face Detection Model...")
        self.net = core.load_network(network=model, device_name='CPU', num_requests=1)
        logger.info('Time taken to load the model is: {:.4f} seconds'.format(time.time() - start))

        return

        raise NotImplementedError

    def predict(self, image):
        processed_image = self.preprocess_input(image)
        input_name, input_shape, output_name, output_shape = self.check_model()
        input_dict = {input_name: processed_image}
        start = time.time()
        infer = self.net.start_async(request_id=0, inputs=input_dict)
        self.count += 1
        if self.net.requests[0].wait(-1) == 0:
            results = self.net.requests[0].outputs[output_name]
            logger.info('Face Detection Model Inference speed is: {:.3f} fps'.format(1 / (time.time() - start)))

        return results

        raise NotImplementedError

    def check_model(self):
        input_name = next(iter(self.net.inputs))
        input_shape = self.net.inputs[input_name].shape
        output_name = next(iter(self.net.outputs))
        output_shape = self.net.outputs[output_name].shape

        return input_name, input_shape, output_name, output_shape

        raise NotImplementedError

    def preprocess_input(self, image):
        input_name, input_shape, output_name, output_shape = self.check_model()
        image = cv2.resize(image, (input_shape[3], input_shape[2]), interpolation=cv2.INTER_AREA)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)

        return image

        raise NotImplementedError

    def preprocess_output(self, image):
        outputs = self.predict(image)
        h, w, c = image.shape
        for character in (outputs[0][0]):
            if character[2] > 0.3:
                x_min = int(w * character[3])
                y_min = int(h * character[4])
                x_max = int(w * character[5])
                y_max = int(h * character[6])
                crop = image[y_min - 40:y_max + 40, x_min - 50:x_max + 50]
                if args.input_type == 'image':
                    cv2.imwrite('cropped.jpg', crop)
                image = cv2.resize(image, (720, 480), interpolation=cv2.INTER_AREA)
                coords = [int(character[5]*720), int(character[4]*480)]

                return crop, image, coords

                raise NotImplementedError


def mask_detection(image, new_model):
    image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
    image = np.array(image).reshape(-1, 300, 300, 3)
    start = time.time()
    results = new_model.predict([image])
    logger.info('Mask Detection Model Inference speed is: {:.3f} fps'.format(1 / (time.time() - start)))
    if results[0] == 1:
        text = 'No mask'
    if results[0] == 0:
        text = 'Mask Detected'
    return text


def main(args):
    fd = FaceDetection('models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001', 'CPU')
    feed = InputFeeder(input_type=args.input_type, input_file=args.input_file)
    fd.load_model()
    feed.load_data()
    new_model = tf.keras.models.load_model('/home/sammie/Jupyter_notebook/Mask_detection/mask_detection.model')
    logger.info("Mask detection model loaded...")

    if args.input_type == 'image':
        batch = feed.next_batch()
        try:
            image, crop, coords = fd.preprocess_output(batch)
            if len(crop.shape) == 3:
                result = mask_detection(crop, new_model)
                if result == 'No mask':
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                image = cv2.putText(image, result, (100, 30), cv2.FONT_ITALIC, 1.0, color, 1)
                cv2.imwrite('Results.png', image)

        except TypeError:
            logger.info('No face detected in frame')

    else:
        while True:
            batch = feed.next_batch()
            try:
                crop, image, coords = fd.preprocess_output(batch)
                if len(crop.shape) == 3:
                    result = mask_detection(crop, new_model)

                if result == 'No mask':
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                image = cv2.putText(image, result, (coords[0], coords[1]), cv2.FONT_ITALIC, 0.5, color, 1)
                cv2.imshow('Results', image)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

            except TypeError:
                logger.info("No face detected!!!")
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_type', required=True, help='Enter the type of input either video, cam or image')
    parser.add_argument('--input_file', default='bin/demo.mp4', help='Enter the directory path for the input file')
    parser.add_argument('--device', default='CPU', help='Enter the name of the device to perform inference on')
    parser.add_argument('--show_results', default='no', help='Enter yes to show and no to hide performance results')
    parser.add_argument('--model_path', required=True, help='Add the path to the directory containing the four models')
    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        logger.info('Program cancelled by user')
