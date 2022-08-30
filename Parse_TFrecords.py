import tensorflow as tf
import collections
import pandas as pd

Labels_diseases = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Lung Opacity', 'Pleural Effusion', 'Support Devices']
features = { i : tf.io.FixedLenFeature([], tf.int64) for i in Labels_diseases }

def filter_1(img, label):
    return (tf.math.equal(label, 1))

def filter_0(img, label):
    return (tf.math.equal(label, 0))

def Convert(string):
    li = list(string.split(", "))
    new_li = []
    for i in li:
        new_li.append(int(i))
    return new_li

def parse_TFrecord_pretrain(example):
    img = tf.io.parse_single_example(example, {'jpg_bytes': tf.io.FixedLenFeature([], tf.string)})
    img = tf.io.decode_jpeg(img['jpg_bytes'], channels=3)
    # Normalize the pixel values to be between 0 and 1
    img = (1.0 / 255.0) * tf.cast(tf.image.resize(img, [256, 256]), tf.float32)
    
    label = tf.io.parse_single_example(example, features)
    Label = tf.stack([label[l] for l in Labels_diseases], axis=0)
        
    return img, Label

@tf.autograph.experimental.do_not_convert
def parse_TFrecord_train(example):
    img = tf.io.parse_single_example(example, {'jpg_bytes': tf.io.FixedLenFeature([], tf.string)})
    img = tf.io.decode_jpeg(img['jpg_bytes'], channels=3)
    # Normalize the pixel values to be between 0 and 1
    img = (1.0 / 255.0) * tf.cast(tf.image.resize(img, [256, 256]), tf.float32)
    
    copd = tf.io.parse_single_example(example, {'COPD' :tf.io.FixedLenFeature([], tf.int64)})
        
    return img, copd['COPD']

def parse_TFrecord_test(example):
    img = tf.io.parse_single_example(example, {'jpg_bytes': tf.io.FixedLenFeature([], tf.string)})
    img = tf.io.decode_jpeg(img['jpg_bytes'], channels=3)
    # Normalize the pixel values to be between 0 and 1
    img = (1.0 / 255.0) * tf.cast(tf.image.resize(img, [256, 256]), tf.float32)
        
    return img

@tf.autograph.experimental.do_not_convert
def parse_TFrecord_demo(example):
    img = tf.io.parse_single_example(example, {'jpg_bytes': tf.io.FixedLenFeature([], tf.string)}) # Parse images to keras dataset
    img = tf.io.decode_jpeg(img['jpg_bytes'], channels=3) #RBG image
    # Normalize the pixel values to be between 0 and 1
    img = (1.0 / 255.0) * tf.cast(tf.image.resize(img, [256, 256]), tf.float16) #Normalize pixel 
    
    label = tf.cast(tf.io.parse_single_example(example, {'COPD' :tf.io.FixedLenFeature([], tf.int64)})['COPD'], tf.int32) # parses the label copd
        
    race = tf.cast(tf.io.parse_single_example(example, {'race' :tf.io.FixedLenFeature([], tf.int64)})['race'], tf.int32)
    age = tf.cast(tf.io.parse_single_example(example, {'age' :tf.io.FixedLenFeature([], tf.int64)})['age'], tf.int32)
    gender = tf.cast(tf.io.parse_single_example(example, {'gender' :tf.io.FixedLenFeature([], tf.int64)})['gender'], tf.int32)
        
    age = tf.one_hot(age, 4)
    race = tf.one_hot(race, 5)
    
    demo = tf.experimental.numpy.append(tf.cast(race, tf.float32), tf.cast(age, tf.float32))
    demo = tf.experimental.numpy.append(tf.cast(demo, tf.float32), tf.cast(gender, tf.float32))

    feature_dict = collections.OrderedDict(
      input_cxr=img,
      input_demo=demo
  )

    return feature_dict, label