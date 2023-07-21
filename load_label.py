import tensorflow as tf
import numpy as np
import pandas as pd
import skimage.transform as st

def get_data_label(dataset='mimic', split='test', return_demo=False):
    
    filename = 'tfrecords/copd_{a}_{b}.tfrecords'.format(a=dataset, b=split)
      
    y = []
    demo = []

    #load the test files
    raw_dataset = tf.data.TFRecordDataset(filename)
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        label = example.features.feature['COPD'].int64_list.value[0]
        
        y.append(label)
        
        if (return_demo):
        
            gender = example.features.feature['gender'].int64_list.value[0]
            race = example.features.feature['race'].int64_list.value[0]
            age = example.features.feature['age'].int64_list.value[0]

            demo.append({'Age':age, 'Gender':gender, 'Race':race})

        
    if (return_demo):
        return np.array(y), np.array(demo)
    else:
        return np.array(y)