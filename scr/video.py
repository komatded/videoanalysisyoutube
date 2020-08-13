import cv2
import time
from config import logger
from download import Download
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager


class Video:

    def __init__(self):
        self.download = Download()

    def __call__(self, video_url, threshold=15):
        start = time.time()
        video_path = self.download(video_url=video_url)
        logger.info('Video download {0} sec.'.format(round(time.time() - start), 1))
        start = time.time()
        if video_path:
            return self.get_frames(video_path=video_path, threshold=threshold)
        logger.info('Frames extraction {0} sec.'.format(round(time.time() - start), 1))
        return

    def get_frames(self, video_path, threshold):
        video = cv2.VideoCapture(video_path)
        start = time.time()
        plans = self.get_plans(video_path=video_path, threshold=threshold)
        logger.info('Plans extraction {0} sec.'.format(round(time.time() - start), 1))
        frames_ids = set()
        for ft0, ft1 in plans:
            frames_ids.add(ft1.frame_num)
            frames_ids.add((ft1.frame_num + ft0.frame_num) // 2)
        frame_id = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                video.release()
                break
            if frame_id in frames_ids:
                yield frame
            frame_id += 1

    @staticmethod
    def get_plans(video_path, threshold):
        video_manager = VideoManager([video_path])
        base_timecode = video_manager.get_base_timecode()
        video_manager.set_downscale_factor()
        video_manager.start()
        stats_manager = StatsManager()
        plan_manager = SceneManager(stats_manager)
        plan_manager.add_detector(ContentDetector(threshold=threshold))
        plan_manager.detect_scenes(frame_source=video_manager)
        plan_list = plan_manager.get_scene_list(base_timecode)
        return plan_list
