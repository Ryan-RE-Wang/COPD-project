import tensorflow as tf

Labels_diseases = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Lung Opacity', 'Pleural Effusion', 'Support Devices']
features = { i : tf.io.FixedLenFeature([], tf.float32) for i in Labels_diseases }

def filter_1(img, label):
    return (tf.math.equal(label, 1))

def filter_0(img, label):
    return (tf.math.equal(label, 0))

def parse_TFrecord_pretrain(example):
    img = tf.io.parse_single_example(example, {'jpg_bytes': tf.io.FixedLenFeature([], tf.string)})
    img = tf.io.decode_jpeg(img['jpg_bytes'], channels=3)
    # Normalize the pixel values to be between 0 and 1
    img = (1.0 / 255.0) * tf.cast(tf.image.resize(img, [256, 256]), tf.float32)
    
    label = tf.io.parse_single_example(example, features)
    Label = tf.stack([label[l] for l in Labels_diseases], axis=0)
        
    return img, Label

def parse_TFrecord_train(example):
    img = tf.io.parse_single_example(example, {'jpg_bytes': tf.io.FixedLenFeature([], tf.string)})
    img = tf.io.decode_jpeg(img['jpg_bytes'], channels=3)
    # Normalize the pixel values to be between 0 and 1
    img = (1.0 / 255.0) * tf.cast(tf.image.resize(img, [256, 256]), tf.float32)
    
    copd = tf.io.parse_single_example(example, {'COPD' :tf.io.FixedLenFeature([], tf.int64)})
    if (tf.math.equal(copd['COPD'], 0)):
        label = tf.constant(0)
    elif(tf.math.equal(copd['COPD'], 1)):
        label = tf.constant(1) 
    else:
        label = tf.constant(-1)
        
    return img, label

def parse_TFrecord_test(example):
    img = tf.io.parse_single_example(example, {'jpg_bytes': tf.io.FixedLenFeature([], tf.string)})
    img = tf.io.decode_jpeg(img['jpg_bytes'], channels=3)
    # Normalize the pixel values to be between 0 and 1
    img = (1.0 / 255.0) * tf.cast(tf.image.resize(img, [256, 256]), tf.float32)
        
    return img

def parse_TFrecord_train_demo(example):
    img = tf.io.parse_single_example(example, {'jpg_bytes': tf.io.FixedLenFeature([], tf.string)}) # Parse images to keras dataset
    img = tf.io.decode_jpeg(img['jpg_bytes'], channels=3) #RBG image
    # Normalize the pixel values to be between 0 and 1
    img = (1.0 / 255.0) * tf.cast(tf.image.resize(img, [256, 256]), tf.float16) #Normalize pixel 
    
    copd = tf.io.parse_single_example(example, {'COPD' :tf.io.FixedLenFeature([], tf.int64)}) # parses the label copd

    if (tf.math.equal(copd['COPD'], 0)):
        label = tf.constant(0)
    elif(tf.math.equal(copd['COPD'], 1)):
        label = tf.constant(1) 
    else:
        label = tf.constant(-1)
        
    race = tf.cast(tf.io.parse_single_example(example, {'race' :tf.io.FixedLenFeature([], tf.int64)})['race'], tf.int32)
    age = tf.cast(tf.io.parse_single_example(example, {'age' :tf.io.FixedLenFeature([], tf.int64)})['age'], tf.int32)
    gender = tf.cast(tf.io.parse_single_example(example, {'gender' :tf.io.FixedLenFeature([], tf.int64)})['gender'], tf.int32)
    
    race = tf.cond(tf.less(4, race), true_fn=lambda: tf.constant(3), false_fn=lambda: race)
    age = tf.cond(tf.less(age, 1), true_fn=lambda: age, false_fn=lambda: age-1)
    
    race = tf.one_hot(race, 5)
    age = tf.one_hot(age, 4)
    
    demo = tf.experimental.numpy.append(tf.cast(race, tf.float32), tf.cast(age, tf.float32))
    demo = tf.experimental.numpy.append(demo, tf.cast(gender, tf.float32))

    feature_dict = collections.OrderedDict(
      input_1=img,
      input_2=demo
  )

    return feature_dict, label