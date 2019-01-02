# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:05:12 2018

@author: shen1994
"""

import os
import keras
import pickle
import operator
import numpy as np

from ssd import SSD300
from ssd_utils import BBoxUtility
from ssd_loss import MultiboxLoss
from ssd_generator import Generator

def concatenate_ssd300_boxes():
    
    prior_boxes_list = []
    images_path = 'images'
    prior_boxes_path = []
    for file_name in os.listdir(images_path):
        if len(file_name) > 8 and operator.eq(file_name[0:8], 'priorbox'):
            prior_boxes_path.append(file_name)
      
    prior_boxes_number = []
    for elem in prior_boxes_path:
        flag = True
        k = ''
        for i in range(len(elem)):
            
            if elem[i] == '_' and flag:
                flag = False
                continue
            if elem[i] == '_' and not flag:
                break
            if not flag:
                k += elem[i]

        prior_boxes_number.append(int(k))
        
    prior_boxes_index = np.argsort(-np.array(prior_boxes_number))
    
    for index in range(len(prior_boxes_path)):
        true_index = prior_boxes_index[index]
        full_path = images_path + os.sep + prior_boxes_path[true_index]
        prior_boxes_list.append(pickle.load(open(full_path, 'rb')))
            
    prior_boxes = np.concatenate(prior_boxes_list, axis=0)
        
    return prior_boxes

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    model_path = 'model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    NUM_CLASSES = 2
    input_shape = [720, 720, 3]
    
    batch_size = 8
    epochs = 30
    base_lr = 1e-5
    decay = 0.9

    optimizer = keras.optimizers.Adam(lr=base_lr)
    model = SSD300(input_shape=input_shape, num_classes=NUM_CLASSES)
    model.load_weights('model/weights.30-1.81.hdf5', by_name=True)
    model.compile(optimizer=optimizer, loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)
    priors = concatenate_ssd300_boxes()
    bbox_util = BBoxUtility(NUM_CLASSES, priors)
    train_bbx_gt = pickle.load(open('images/train_bbx_gt.pkl', 'rb'))
    valid_bbx_gt = pickle.load(open('images/valid_bbx_gt.pkl', 'rb'))
    train_keys = list(train_bbx_gt.keys())
    valid_keys = list(valid_bbx_gt.keys())

    gen = Generator(bbox_util, batch_size, (input_shape[0], input_shape[1]),
                    train_bbx_gt, valid_bbx_gt, train_keys, valid_keys, do_crop=False)
    
    def schedule(epoch, decay=decay):
        return base_lr #  * decay ** (epoch)
    callbacks = [keras.callbacks.ModelCheckpoint('model/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 verbose=1,
                                                 save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule)]

    history = model.fit_generator(generator=gen.generate(True), 
                                  steps_per_epoch = int(gen.train_batches / gen.batch_size),
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=gen.generate(False),
                                  validation_steps=int(gen.val_batches / gen.batch_size),
                                  workers=1)
	