# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 20:58:33 2018

@author: shen1994
"""

import os
import cv2
import pickle
from keras.preprocessing import image
from scipy.misc import imread
from scipy.misc import imresize

from ssd import SSD300
from ssd_utils import BBoxUtility
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    inputs = []
    images = []
    
    image_path = 'images/test/0--Parade/0_Parade_marchingband_1_32.jpg' # 'timg.jpg'# 'images/test/0--Parade/0_Parade_marchingband_1_32.jpg'
    img = image.load_img(image_path, target_size=(720, 720))
    # img = cv2.imread(image_path, 1)
    img = image.img_to_array(img)
    images.append(imread(image_path))
    inputs.append(img.copy())
    inputs = np.array(inputs)
    
    model = SSD300([720, 720, 3], num_classes=2)
    model.load_weights('model/weights.30-1.81.hdf5', by_name=True)
    
    preds = model.predict(inputs, batch_size=1, verbose=0)
    priors = pickle.load(open('model/prior_boxes.pkl', 'rb'))
    bbox_util = BBoxUtility(2, priors, is_decoder=True)
    results = bbox_util.detection_out(preds)
    
    for i, img in enumerate(images):
    # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.68]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

        plt.imshow(img / 255.)
        currentAxis = plt.gca()

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            # label_name = voc_classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
   
        plt.show()
        