# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 09:36:44 2018

@author: shen1994
"""

import os
import cv2
import tensorflow as tf

from tssd_utils import BBoxUtility

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # face detect model
    detect_graph_def = tf.GraphDef()
    detect_graph_def.ParseFromString(open("model/pico_FaceDetect_model.pb", "rb").read())
    detect_tensors = tf.import_graph_def(detect_graph_def, name="")
    detect_sess = tf.Session()
    detect_opt = detect_sess.graph.get_operations()
    detect_x = detect_sess.graph.get_tensor_by_name("input_1:0")
    detect_y = detect_sess.graph.get_tensor_by_name("predictions/concat:0")
    detect_util = BBoxUtility(detect_sess, 2, top_k=8)

    cv2.namedWindow("DeepFace", cv2.WINDOW_NORMAL)
    
    while(True):

        # read one image
        o_image = cv2.imread("test.jpg", 1)
        width, height, channel = o_image.shape
    
        detect_image = cv2.resize(o_image, (720, 720))
        detect_out = detect_sess.run(detect_y, feed_dict={detect_x: [detect_image]})
        results = detect_util.detection_out(detect_out) 
        det_conf = results[0][:, 1]
        det_xmin = results[0][:, 2]
        det_ymin = results[0][:, 3]
        det_xmax = results[0][:, 4]
        det_ymax = results[0][:, 5]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.57]
        top_conf = det_conf[top_indices]
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
    
        for i in range(top_conf.shape[0]):
            
            # box
            xmin = int(round(top_xmin[i] * height))
            ymin = int(round(top_ymin[i] * width))
            xmax = int(round(top_xmax[i] * height))
            ymax = int(round(top_ymax[i] * width))
          
            # draw some lines, points, text
            cv2.rectangle(o_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        cv2.imshow("DeepFace", o_image)
            
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    
    # release all resources
    cv2.destroyAllWindows()
    detect_sess.close() 
