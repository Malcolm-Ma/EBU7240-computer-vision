# -*- coding: utf-8 -*-
# ******************************************************************************
#  Author:        Mingze Ma
#  Create Date:   2020/11/29
#  Description:   Exercise 1 (a)
# ******************************************************************************

# Import
import cv2 as cv

# Constants
VIDEO_SOURCE_NAME = 'ebu7240_hand.mp4'
VIDEO_SOURCE_PATH = '../../inputs/' + VIDEO_SOURCE_NAME
RESULT_NAME = 'ex1_a_hand_rgbtest.mp4'
RESULT_PATH = '../../results/' + RESULT_NAME


def generate_file_path(file_id):
    return '../../results/' + 'ex1_a/hand_frames' + str(file_id) + '.png'


def get_video_info(video_source: cv.VideoCapture):
    fps = video_source.get(cv.CAP_PROP_FPS)
    size = (video_source.get(cv.CAP_PROP_FRAME_WIDTH), video_source.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_source.get(cv.CAP_PROP_FRAME_COUNT))
    print('Video name:', VIDEO_SOURCE_NAME)
    print('[FPS]  {}\n[SIZE] {}'.format(fps, size))
    print('[INFO] {} total frames'.format(total_frames))
    return {
        'video_name': VIDEO_SOURCE_NAME,
        'video_fps': fps,
        'video_size': size,
        'total_frames': total_frames,
    }


def generate_frame_from_video(video_source: cv.VideoCapture, frame_size=1):
    result = []
    frame_id = 0
    open_state = video_source.isOpened()
    while open_state:
        frame_id = frame_id + 1
        state, frame = video_source.read()
        if frame_id < frame_size and state:
            print('Processing {}/{} frame'.format(frame_id, frame_size))
            result.append(frame)
            # cv.imwrite(generate_file_path(frame_id), frame)
            cv.waitKey(1)
        else:
            break
    return result


def resize_frames(img_list: []):
    result = []
    for item in img_list:
        resize_item = cv.resize(item, (640, 360))
        result.append(resize_item)
    return result


def generate_result(img_list):
    print('save video into path:', RESULT_PATH)
    sample_img = img_list[0]
    img_info = sample_img.shape
    size = (img_info[1], img_info[0])
    print(size)
    video_write = cv.VideoWriter(RESULT_PATH, cv.VideoWriter_fourcc(*'mp4v'), 30, size)
    img_id = 1
    for item in img_list:
        video_write.write(item)
        print('Writing {}/{} frame'.format(img_id, len(img_list) + 1))
        img_id = img_id + 1
    video_write.release()
    print('-----DONE-----')


def main():
    input_capture = cv.VideoCapture(VIDEO_SOURCE_PATH)
    video_info = get_video_info(input_capture)
    frame_list = generate_frame_from_video(input_capture, 90)
    input_capture.release()

    resize_frame_list = resize_frames(frame_list)

    generate_result(resize_frame_list)


if __name__ == '__main__':
    main()
