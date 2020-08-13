from config import *
import youtube_dl
import glob


class Download:

    def __init__(self):
        self.ydl = youtube_dl.YoutubeDL({'outtmpl': TEMPFILEPATH, 'format': 'worst'})

    def __call__(self, video_url):
        output_file_path = None
        try:
            self.ydl.download([video_url])
            output_file_path = glob.glob(TEMPFILEPATH + '*')[0]
        except Exception as ex:
            logger.error(ex)
        return output_file_path
