import tensorflow as tf
from tensorflow.keras import backend as K

INPUT_SHAPE = (256, 256, 3)

def swish_activation(x):
    return (K.sigmoid(x) * x)

def scheduler(epoch, lr):
    if epoch % 2 == 0:
        return lr * tf.math.exp(-0.05)
    else:
        return lr
    
def define_model(archi='Dnet121'):
    if (archi=='Dnet121'):
        base_model = tf.keras.applications.densenet.DenseNet121(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='IV3'):
        base_model = tf.keras.applications.InceptionV3(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    
    pred_layer = tf.keras.layers.Dense(6, activation='sigmoid')(base_model.output)
 
    model = tf.keras.Model(inputs=base_model.input, outputs=pred_layer, name='model')  
  
    return model
    
def get_ensemble_mlp():
    inputs_a = tf.keras.Input(shape=(2048))
    a = tf.keras.layers.Dense(128, activation=swish_activation)(inputs_a)

    inputs_b = tf.keras.Input(shape=(1024))
    b = tf.keras.layers.Dense(128, activation=swish_activation)(inputs_b)

    concate = tf.keras.layers.Concatenate()([a, b])
    concate = tf.keras.layers.Dense(64, activation=swish_activation)(concate)
    concate = tf.keras.layers.Dense(32, activation=swish_activation)(concate)
    concate = tf.keras.layers.Dense(8, activation=swish_activation)(concate)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(concate)

    model = tf.keras.Model(inputs=[inputs_a, inputs_b], outputs=pred)
    
    return model

def define_model_demo():
    input_2 = tf.keras.Input(shape=(10,), name='input_2') # sex - 1, age - 4, ethnicity - 5

    base_model = tf.keras.models.load_model('saved_model/Chexpert_pretrained_256_6_labels')
    
    flag = 0
    for layer in base_model.layers:
    #     number of blocks to be freezed is also a hyperparameter
        if (layer.name == 'pool3_pool'):
            break
        else:
            layer.trainable = False
    
    y = tf.keras.layers.Dense(54, activation='swish')(base_model.get_layer('max_pool').output)
    y = tf.keras.Model(inputs=base_model.input, outputs=y)
    
    # combine the output of the two branches
    combined = tf.keras.layers.concatenate([input_2, y.output])
    x = tf.keras.layers.Dense(16, activation="swish")(combined)
    x = tf.keras.layers.Dense(4, activation="swish")(x)
    pred_COPD = tf.keras.layers.Dense(1, activation='sigmoid', name='pred_COPD')(x)

    model = tf.keras.Model(inputs=[base_model.input, input_2], outputs=pred_COPD)

    return model