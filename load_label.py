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
     

def Convert(string):
    li = list(string.split(", "))
    new_li = []
    for i in li:
        new_li.append(int(i))
    return new_li

# def get_seg_data(filename):
#     df = pd.read_csv('seg_coordinate', index_col=0)
    
#     image = []
#     label = []
#     none = 0
#     #load the test files
#     raw_dataset = tf.data.TFRecordDataset(filename)
#     for raw_record in raw_dataset:
#         example = tf.train.Example()
#         example.ParseFromString(raw_record.numpy())

#         nparr = np.fromstring(example.features.feature['jpg_bytes'].bytes_list.value[0], np.uint8) #extract  img info --in bytes list format convert to np array
#         img_np = cv.imdecode(nparr, cv.IMREAD_COLOR) #open cv convert to gray scale IMREAD_GRAYSCALE -- RBG uses pretrained weights need RGB
#         subject_id = example.features.feature['subject_id'].int64_list.value[0]
#         study_id = example.features.feature['study_id'].int64_list.value[0]
#         d = df.loc[lambda df: df['subject_id'] == subject_id].loc[lambda df: df['study_id'] == study_id]
#         try:
#             coordinate = d['coordinate'].iloc[0]
#             coordinate = Convert(coordinate[1:-1])
#             if ( coordinate[0] != 0 or coordinate[1] != 0 or coordinate[2] != 0 or coordinate[3] != 0):
#                 img_np = img_np[coordinate[0]:coordinate[1], coordinate[2]:coordinate[3]]
#         except:
#             none += 1
#         image.append(np.float32(st.resize(img_np, (256, 256))))
#         label.append(example.features.feature['COPD'].int64_list.value[0])

#     print(none)
    
#     return np.array(image), np.array(label)

