import cv2
from tqdm import tqdm
from os.path import splitext, dirname, join, basename

class VideoMaker(object):

    def __init__(self):
        pass

    def process_video(self, video_file, results_file):
        results = [l.strip().split(',') for l in open(results_file, 'r').readlines()]

        video_cap = cv2.VideoCapture(video_file)
        length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))

        path = dirname(results_file)
        base = basename(video_file)
        name, ext = splitext(base)

        new_video_file = join(path, name + '_result' + ext.lower())
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        new_video = cv2.VideoWriter(new_video_file, fourcc, fps, (width, height))

        pbar = tqdm(total=length)
        print('Processing frames...')
        while video_cap.isOpened():
            isvalid, frame = video_cap.read()
            if isvalid and len(results) > 0:
                # _, species, count = results.pop(0)
                # text = 'Species=%s, Count=%s' % (species, count)
                _, species = results.pop(0)
                text = 'Species=%s' % (species)
                location = (20,60)
                fontScale = 1
                fontColor = (0,0,255)
                lineType = 2
                cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, fontColor, lineType)
            else:
                break
            pbar.update()
            new_video.write(frame)

        pbar.close()
        video_cap.release()
        new_video.release()
        cv2.destroyAllWindows()
