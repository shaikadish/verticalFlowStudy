"""Videos to frames conversion

This script converts input videos for each regime class to the set of image frames
which make up that video.
"""

# Imports
import cv2
import os
import glob
import argparse

# Args. By default, will extract training data unless test argument specified
parser = argparse.ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
args = parser.parse_args()
if args.test:
    test_train = 'test'
else:
    test_train = 'train'


def video2frames(video_path, test_train):
    # Load in and loop through each video frame by frame
    capture = cv2.VideoCapture(video_path)
    im_count = 0
    while True:
        ret, image = capture.read()
        if not ret:
            break

        # Create save path
        video_info = video_path.split('/')[-2:]
        image_name = video_info[-1][:-4] + f'_{im_count}' + video_info[-1][-4:]
        video_info[-1] = image_name
        save_path = f'FlowNet/data/regime_images/{test_train}/' + '/'.join(
            video_info)

        # Save image
        cv2.imwrite(save_path, image)


def main():
    video_root = f'FlowNet/data/regime_videos/{test_train}'
    labels = sorted(os.listdir(os.path.join(os.getcwd(), '')))
    for label in labels:
        # List of paths to all videos in regime
        video_list = [
            mp4 for mp4 in glob.iglob(
                video_root +
                '/{}/*.mp4'.format(label),
                recursive=True)]
        for video_path in video_list:
            video2frames(video_path, test_train)


if __name__ == "__main__":
    main()
