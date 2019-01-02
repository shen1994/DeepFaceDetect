# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:41:49 2018

@author: shen1994
"""

import os
import pickle
import numpy as np
from PIL import Image

image_shape = [720.0, 720.0]

def scene_counter(path):
    
    return len(os.listdir(path))
    
def class_from_name(name):
    
    class_string = ''
    for index in range(len(name)):
        class_string += name[index]
        if name[index + 1] == '-' and name[index + 2] == '-':
            break
    class_int = int(class_string)
    
    return class_int
    
def class_one_hot(class_n, class_g):
    
    n_array = np.array([class_n])
    n_array_length = len(n_array)
    one_hot_g = np.zeros((n_array_length, class_g))
    one_hot_g[:, n_array] = 1
    
    if n_array_length == 1:
        return one_hot_g[0]
    else:
        return one_hot_g

def txt_to_dict(path, path_prex, scene_number):

    user_dict = {}                                                                                      
    with open(path, "r") as infile:
        context = infile.readlines()
        skip_number = 0
        for index in range(len(context)):
            if index < skip_number:
                continue
            name = context[index].replace('\n', '')
            # 场景类别添加
            # class_number = class_from_name(name)
            # class_array = class_one_hot(class_number, scene_number)
            box_number = int(context[index + 1].replace('\n', ''))

            image = Image.open(path_prex + os.sep + name)
            width, height = image.size
            width_scale = width / image_shape[0]
            height_scale = height / image_shape[1]

            box_lists = []
            for box_index in range(box_number):
                one_box = context[index + 2 + box_index].replace('\n', '').strip().split()
                one_box = [int(elem) for elem in one_box]
                if one_box[4] != 2 and one_box[7] != 2 and one_box[9] == 0:
                    box_size = []
                    box_size.append(one_box[0] / width_scale / image_shape[0])
                    box_size.append(one_box[1] / height_scale / image_shape[1])
                    box_size.append((one_box[0] + one_box[2] - 1) / width_scale / image_shape[0])
                    box_size.append((one_box[1] + one_box[3] - 1) / height_scale / image_shape[1])
                    # box_size.extend(class_array)
                    box_size.append(1)
                    
                    box_lists.append(np.array(box_size))
             
            box_lists = np.array(box_lists)  
            user_dict[path_prex + os.sep + name] = box_lists

            print(name + '--->OK!')

            skip_number = index + box_number + 2
    
    return user_dict

if __name__ == "__main__":
    
    image_path = 'images'
    train_path = 'train'
    valid_path = 'valid'
    face_split_path = 'face_split'
    train_bbx_path = 'wider_face_train.txt'
    valid_bbx_path = 'wider_face_valid.txt'
    train_gt_path = image_path + os.sep + face_split_path + os.sep + train_bbx_path
    valid_gt_path = image_path + os.sep + face_split_path + os.sep + valid_bbx_path
    train_full_path = image_path + os.sep + train_path
    valid_full_path = image_path + os.sep + valid_path
    
    train_scene_number = scene_counter(train_full_path)
    train_dict = txt_to_dict(train_gt_path, train_full_path, train_scene_number)
    with open('images/train_bbx_gt.pkl', 'wb') as infile:
        pickle.dump(train_dict, infile, pickle.HIGHEST_PROTOCOL)
    
    valid_scene_number = scene_counter(valid_full_path)
    valid_dict = txt_to_dict(valid_gt_path, valid_full_path, valid_scene_number)
    with open('images/valid_bbx_gt.pkl', 'wb') as infile:
        pickle.dump(valid_dict, infile, pickle.HIGHEST_PROTOCOL)
        
    print('path and box has been pointed ---> OK')           
        