#!/bin/bash
docker build -t youtube_video_new .
docker stop youtube_video
docker rm youtube_video
docker run -p 8555:8000 -d --name youtube_video youtube_video_new
docker rmi $(docker images -qa -f 'dangling=true')
exit 0
