from openvino.inference_engine import IENetwork, IECore
from logo.image_iterator import ImageIterator
from config import *
import numpy as np
import cv2


class Logo2Vec:

    def __init__(self):
        self.ie = IECore()
        self.net = IENetwork(model=logo2vec_model_xml, weights=logo2vec_model_bin)
        self.net.batch_size = 1
        self.exec_net = self.ie.load_network(network=self.net, num_requests=2, device_name='CPU')

    def encode_images(self, images: np.array, is_async_mode=False) -> list:
        images = ImageIterator(images=images)
        embeddings = list()
        cur_request_id = 0
        next_request_id = 1

        while images.isOpened():
            if is_async_mode:
                ret, next_image = images.read()
            else:
                ret, image = images.read()

            if not ret:
                break

            if is_async_mode:
                request_id = next_request_id
                custom_image = cv2.resize(next_image, (299, 299), interpolation=cv2.INTER_LINEAR)
            else:
                request_id = cur_request_id
                custom_image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_LINEAR)
            custom_image = custom_image.transpose(2, 0, 1)
            custom_image = custom_image.reshape(1, 3, 299, 299)
            self.exec_net.start_async(request_id=request_id, inputs={'input_1': custom_image})

            if self.exec_net.requests[cur_request_id].wait(-1) == 0:
                embedding = self.exec_net.requests[cur_request_id].outputs['embedding/Relu'][0]
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)

            if is_async_mode:
                cur_request_id, next_request_id = next_request_id, cur_request_id
                image = next_image

        return embeddings
