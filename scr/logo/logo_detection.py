from openvino.inference_engine import IENetwork, IECore
from logo.yolo_openvino import detect as detect_vino  # openvino
from config import *


class LogoDetector:

    def __init__(self):
        self.ie = IECore()
        self.net = IENetwork(model=model_xml, weights=model_bin)
        self.net.batch_size = 1
        self.exec_net = self.ie.load_network(network=self.net, num_requests=2, device_name='CPU')

    def predict(self, images, thresh, is_async_mode=True):
        result = detect_vino(exec_net=self.exec_net, net=self.net, images=images, conf_threshold=thresh,
                             is_async_mode=is_async_mode)
        return result
