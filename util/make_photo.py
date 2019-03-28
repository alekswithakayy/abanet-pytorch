import os
import cv2

inference_dir = '/Users/aleksandardjuric/Desktop/test/photos'
results_dir = '/Users/aleksandardjuric/Desktop/test/results_photos'

for file in os.listdir(inference_dir):
    name, ext = os.path.splitext(file)
    if ext == '.jpg' or ext == '.png':
        results_file = os.path.join(results_dir, name + '.csv')
        results = [l.strip().split(',') for l in open(results_file, 'r').readlines()]
        species, count = results[0]

        frame = cv2.imread(os.path.join(inference_dir, file))

        text = 'Species=%s, Count=%s' % (species, count)
        location = (20,60)
        fontScale = 1
        fontColor = (0,0,255)
        lineType = 2
        cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX,
            fontScale, fontColor, lineType)

        cv2.imwrite(os.path.join(results_dir, file), frame)
