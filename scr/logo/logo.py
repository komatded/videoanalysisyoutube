from logo.logo2vec import Logo2Vec
from logo.logo_detection import LogoDetector


class LogoProcessor:

    def __init__(self):
        self.logo2vec = Logo2Vec()
        self.logo_detector = LogoDetector()

    def process(self, images, thresh):
        logos = self.logo_detector.predict(images=images, thresh=thresh)
        embeddings = self.logo2vec.encode_images(images=logos, is_async_mode=True)
        return {'embeddings': embeddings, 'images': logos}
