import tqdm


class ImageIterator:

    def __init__(self, images: list):
        self.images = iter(images)
        self.is_opened = True
        self.bar = tqdm.tqdm()
        self.bar.total = len(images)

    def isOpened(self):
        return self.is_opened

    def read(self):
        try:
            image = next(self.images)
            self.bar.update(1)
            self.bar.refresh()
            return True, image
        except StopIteration:
            self.is_opened = False
            return False, None
