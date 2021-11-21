import tensorflow as tf
import numpy as np
import cv2 as cv
import skimage.transform as st

def get_data_label(split='test', category=None, types=0):
    
    if (split=='train'):
        filename = 'copd_train_new.tfrecords'
    elif (split=='val'):
        filename = 'copd_val_new.tfrecords'
    else:
        filename = 'copd_test_new.tfrecords'
      
    y = []

    #load the test files
    raw_dataset = tf.data.TFRecordDataset(filename)
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        label = example.features.feature['COPD'].int64_list.value[0]
        
        gender = example.features.feature['gender'].int64_list.value[0]
        race = example.features.feature['race'].int64_list.value[0]
        age = example.features.feature['age'].int64_list.value[0]
        
        if (race > 4): # combine indigenous, unknown to others
            race = 3
            
        if (age > 0): # combine 0-20 to 20-40
            age -= 1
            
        if (category == 'Age'):
            if (age != types):
                continue
        elif (category == 'Gender'):
            if (Gender != types):
                continue
        elif (category == 'Race'):
            if (race != types):
                continue
        else:
            pass

        y.append(label)
     
    return np.array(y)

def get_test_data_demo(category=None, types=0):
        
    filename = 'copd_test_new.tfrecords' 
        
    image = []
    demo = []
    label = []

    #load the test files
    raw_dataset = tf.data.TFRecordDataset(filename)
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        nparr = np.fromstring(example.features.feature['jpg_bytes'].bytes_list.value[0], np.uint8) #extract  img info --in bytes list format convert to np array
        img_np = cv.imdecode(nparr, cv.IMREAD_COLOR) #open cv convert to gray scale IMREAD_GRAYSCALE -- RBG uses pretrained weights need RGB

        gender = example.features.feature['gender'].int64_list.value[0]
        race = example.features.feature['race'].int64_list.value[0]
        age = example.features.feature['age'].int64_list.value[0]
        
        if (race > 4): # combine indigenous, unknown to others
            race = 3
            
        if (age > 0): # combine 0-20 to 20-40
            age -= 1
            
        if (category == 'Age'):
            if (age != types):
                continue
        elif (category == 'Gender'):
            if (gender != types):
                continue
        elif (category == 'Race'):
            if (race != types):
                continue
        else:
            pass
        
        race = np.eye(5)[race]
        age = np.eye(4)[age]
        
        temp = np.concatenate((race, age, gender), axis=None)
        demo.append(temp)
        
        image.append(np.float32(st.resize(img_np, (256, 256))))
        label.append(example.features.feature['COPD'].int64_list.value[0])
                
    return np.array(image), np.array(demo), np.array(label)