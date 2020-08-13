from video import *
from logo.logo import LogoProcessor

import json
import time
from aiohttp import web
from mrrest import RESTApi

LP = LogoProcessor()
VIDEO = Video()


async def video_analysis(request):
    try:
        request = await request.json()
    except json.JSONDecodeError:
        raise web.HTTPBadRequest(text='wrong json format')

    result = list()
    video_url = request['url']
    frames = list(VIDEO(video_url=video_url))
    logger.info('Frames count {0}'.format(len(frames)))

    if frames:
        start = time.time()
        result = LP.process(frames, thresh=.2)
        logger.info('Logo processing {0} sec.'.format(round(time.time() - start), 1))
    return result


api = RESTApi(
    host='0.0.0.0',
    port=8000,
    routes=[web.post('/parse', video_analysis)])

api.run()


# import pickle
# from config import TEST_FILE
# import cv2
#
# frames = [cv2.imread(TEST_FILE)] * 10
# start = time.time()
# vectors = LP.process(frames)
# logger.info('Logo processing {0} sec.'.format(round(time.time() - start), 1))
# print(len(vectors))
# pickle.dump(vectors, open('test.pickle', 'wb'))
