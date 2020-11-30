# -*- coding: utf-8 -*-
# ******************************************************************************
#  Author:        Mingze Ma
#  Create Date:   2020/11/29
#  Description:   Exercise 1 (b)
# ******************************************************************************

# Import
import cv2 as cv

# Constants
VIDEO_SOURCE_NAME = 'ebu7240_hand.mp4'
VIDEO_SOURCE_PATH = '../../inputs/' + VIDEO_SOURCE_NAME
RESULT_A_NAME = 'ex1_a_hand_rgbtest'
RESULT_B_NAME = 'ex1_b_hand_composition'
RESULT_PATH = '../../results/'
MY_NAME_PATH = '../../inputs/my_name.png'


def generate_file_path(file_id, ex_name='A'):
    if ex_name == 'A':
        return '../../results/' + 'ex1_a_hand_frames' + str(file_id) + '.png'
    else:
        return '../../results/' + 'ex1_b_hand_frames' + str(file_id) + '.png'


def generate_frame_from_video(video_source: cv.VideoCapture, frame_size=1):
    result = []
    frame_id = 0
    open_state = video_source.isOpened()
    while open_state:
        state, frame = video_source.read()
        if frame_id < frame_size and state:
            print('Processing {}/{} frame'.format(frame_id, frame_size))
            result.append(frame)
            # cv.imwrite(generate_file_path(frame_id), frame)
            cv.waitKey(1)
            frame_id = frame_id + 1
        else:
            break
    return result


def resize_frames(img_list: []):
    result = []
    for item in img_list:
        resize_item = cv.resize(item, (640, 360))
        result.append(resize_item)
    return result


def generate_video_with_img(img_list):
    my_name_img = cv.imread(MY_NAME_PATH)
    small_pic_size = (90, 180)
    result = []
    for epoch, pic in enumerate(img_list):
        pic[270:270 + small_pic_size[0], epoch * 2:epoch * 2 + small_pic_size[1], :] = my_name_img
        result.append(cv.cvtColor(pic, cv.COLOR_BGR2RGB))
    return result


def generate_result(img_list, extra_name='', ex_name='A'):
    if ex_name == 'A':
        save_path = RESULT_PATH + RESULT_A_NAME + extra_name + '.mp4'
    else:
        save_path = RESULT_PATH + RESULT_B_NAME + extra_name + '.mp4'
    print('save video into path:', save_path)
    sample_img = img_list[0]
    img_info = sample_img.shape
    size = (img_info[1], img_info[0])
    video_write = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*'mp4v'), 30, size)
    img_id = 1
    for item in img_list:
        video_write.write(item)
        print('Writing {}/{} frame'.format(img_id, len(img_list) + 1))
        img_id = img_id + 1
    video_write.release()
    print('-----DONE-----')


def get_specific_frame(img_list, start_num, target_frame, ex_name='A'):
    print('The {} frame is in the path: {}'.format(target_frame, generate_file_path(target_frame, ex_name)))
    cv.imwrite(generate_file_path(target_frame, ex_name), img_list[target_frame - start_num])


def main():
    # exercise_1 b
    # Task 1, 2
    input_capture = cv.VideoCapture(VIDEO_SOURCE_PATH)
    frame_list = generate_frame_from_video(input_capture, 90)
    input_capture.release()
    resize_frame_list = resize_frames(frame_list)

    res = generate_video_with_img(resize_frame_list)
    generate_result(res, ex_name='B')
    for item in [1, 21, 31, 61, 90]:
        get_specific_frame(res, 1, item, ex_name='B')


if __name__ == '__main__':
    main()
