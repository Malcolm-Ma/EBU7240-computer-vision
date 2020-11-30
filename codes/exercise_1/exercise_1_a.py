# -*- coding: utf-8 -*-
# ******************************************************************************
#  Author:        Mingze Ma
#  Create Date:   2020/11/29
#  Description:   Exercise 1 (a)
# ******************************************************************************

# Import
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

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


def separate_color_channels(img, no_value_channel_list=None):
    if no_value_channel_list is None:
        no_value_channel_list = []
    if len(no_value_channel_list) == 0:
        return img
    rgb_map = {
        'b': 0,
        'g': 1,
        'r': 2,
    }
    channel_1 = rgb_map[no_value_channel_list[0]]
    channel_2 = rgb_map[no_value_channel_list[1]]
    img_copy = img.copy()
    img_copy[:, :, channel_1] = 0
    img_copy[:, :, channel_2] = 0

    return img_copy


def generate_part_video(img_list, start=0, duration=0, no_value_channel_list=None):
    if no_value_channel_list is None:
        no_value_channel_list = []
    new_list = img_list[start - 1:start + duration - 1]
    result_list = []
    for item in new_list:
        resp = separate_color_channels(item, no_value_channel_list)
        result_list.append(resp)

    return result_list


def generate_video_with_img(img_list):
    my_name_img = cv.imread(MY_NAME_PATH)
    small_pic_size = (90, 180)
    result = []
    for epoch, pic in enumerate(img_list):
        pic[270:270 + small_pic_size[0], epoch * 2:epoch * 2 + small_pic_size[1], :] = my_name_img
        result.append(cv.cvtColor(pic, cv.COLOR_BGR2RGB))
    return result


def main():
    # Task 1
    input_capture = cv.VideoCapture(VIDEO_SOURCE_PATH)
    video_info = get_video_info(input_capture)
    frame_list = generate_frame_from_video(input_capture, 90)
    input_capture.release()

    resize_frame_list = resize_frames(frame_list)

    generate_result(resize_frame_list)

    # Task 2
    # Full color
    frame_1_to_30_res = generate_part_video(resize_frame_list, 1, 30, [])
    generate_result(frame_1_to_30_res, '_frame_1_to_30')
    get_specific_frame(frame_1_to_30_res, 1, 1)
    get_specific_frame(frame_1_to_30_res, 1, 21)
    # Zero values to G, B channel
    frame_31_to_50_res = generate_part_video(resize_frame_list, 31, 50, ['g', 'b'])
    generate_result(frame_31_to_50_res, '_frame_31_to_50')
    get_specific_frame(frame_31_to_50_res, 31, 31)
    # Zero values to R, B channel
    frame_51_to_70_res = generate_part_video(resize_frame_list, 51, 70, ['r', 'b'])
    generate_result(frame_51_to_70_res, '_frame_51_to_70')
    get_specific_frame(frame_51_to_70_res, 51, 61)
    # Zero values to R, G channel
    frame_71_to_90_res = generate_part_video(resize_frame_list, 71, 90, ['r', 'g'])
    generate_result(frame_71_to_90_res, '_frame_71_to_90')
    get_specific_frame(frame_71_to_90_res, 71, 90)


if __name__ == '__main__':
    main()
