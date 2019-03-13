import os
from os.path import join
from util.video_maker import VideoMaker

input_videos = '/Users/aleksandardjuric/Desktop/test/videos'
results_files = '/Users/aleksandardjuric/Desktop/test/results_videos'

maker = VideoMaker()

for file in os.listdir(input_videos):
    name, ext = os.path.splitext(file)
    if ext.lower() == '.mp4':
        maker.process_video(join(input_videos, file),
                      join(results_files, name + '.csv'))
