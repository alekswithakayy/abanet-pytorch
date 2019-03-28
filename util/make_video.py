import os
from os.path import join
from video_maker import VideoMaker

def main():
    input_videos = '/Main/projects/animal_behaviour_analysis/test/videos'
    results_files = '/Main/projects/animal_behaviour_analysis/test/results'

    maker = VideoMaker()

    for file in os.listdir(input_videos):
        name, ext = os.path.splitext(file)
        if ext.lower() == '.mp4':
            maker.process_video(join(input_videos, file),
                          join(results_files, name + '.csv'))

if __name__ == '__main__':
    main()
