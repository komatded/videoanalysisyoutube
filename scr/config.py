import os
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCE_DIR = os.getenv(
    'RESOURCESDIR',
    os.path.abspath(os.path.join(MAIN_DIR, 'resources'))
)
TEMPFILEPATH = os.path.join(RESOURCE_DIR, 'temp_file')
TEST_FILE = os.path.join(RESOURCE_DIR, '28.jpg')

model_xml = os.path.join(RESOURCE_DIR, 'frozen_darknet_yolov3_model.xml')
model_bin = os.path.join(RESOURCE_DIR, 'frozen_darknet_yolov3_model.bin')

logo2vec_model_xml = os.path.join(RESOURCE_DIR, 'frozen_logovec_model.xml')
logo2vec_model_bin = os.path.join(RESOURCE_DIR, 'frozen_logovec_model.bin')

output_layers = ['yolo_82', 'yolo_94', 'yolo_106']
nms_threshold = 0.4
