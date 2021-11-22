import tensorflow as tf
from tensorflow.keras import backend as K
from utilities import *

INPUT_SHAPE = (256, 256, 3)

def swish_activation(x):
    return (K.sigmoid(x) * x)
    
def freeze_layer(model, layer_name):
    for layer in model.layers:
        if (layer.name == layer_name):
            break
        else:
            layer.trainable = False
            
    return model
    
def define_model_pretrain(archi='Dnet121'):
    if (archi=='Dnet121'):
        base_model = tf.keras.applications.densenet.DenseNet121(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='IV3'):
        base_model = tf.keras.applications.InceptionV3(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    
    pred_layer = tf.keras.layers.Dense(6, activation='sigmoid')(base_model.output)
 
    model = tf.keras.Model(inputs=base_model.input, outputs=pred_layer, name='model')  
  
    return model

def define_model(archi='Dnet121'):
    if (archi=='Dnet121'):
        base_model = tf.keras.applications.densenet.DenseNet121(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='IV3'):
        base_model = tf.keras.applications.InceptionV3(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    
    pred_layer = tf.keras.layers.Dense(1, activation='sigmoid')(base_model.output)
 
    model = tf.keras.Model(inputs=base_model.input, outputs=pred_layer, name='model')  
  
    return model

def load_model_from_pretrain(archi='Dnet121'):
    if (archi == 'Dnet121'):
        base_model = define_model_pretrain(archi='Dnet121')
        base_model.load_weights('checkpoints/AUC/checkpoint_pretrain_Dnet121')
    else:
        base_model = define_model_pretrain(archi='IV3')
        base_model.load_weights('checkpoints/AUC/checkpoint_pretrain_InceptionV3')

    pred_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='pred_layer')(base_model.get_layer(base_model.layers[-2].name).output)

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

def define_model_demo(archi='Dnet121'):
    Input = tf.keras.Input(shape=(10,), name='input_demo') # sex - 1, age - 4, ethnicity - 5

    base_model = define_model_pretrain(archi)
    
    if (archi == 'Dnet121'):
        base_model.load_weights('checkpoints/AUC/checkpoint_pretrain_Dnet121')
#         base_model = tf.keras.models.load_model('saved_model/Chexpert_pretrained_256_6_labels')
 
        base_model = freeze_layer(base_model, 'pool_pool3')
    else:
        base_model.load_weights('checkpoints/AUC/checkpoint_pretrain_InceptionV3')
    
    base_model.layers[0]._name = 'input_cxr'
        
    y = tf.keras.layers.Dense(54, activation='swish')(base_model.get_layer(base_model.layers[-2].name).output)
    y = tf.keras.Model(inputs=base_model.input, outputs=y)
    
    # combine the output of the two branches
    combined = tf.keras.layers.concatenate([y.output, Input])
    x = tf.keras.layers.Dense(16, activation="swish")(combined)
    x = tf.keras.layers.Dense(4, activation="swish")(x)
    pred_COPD = tf.keras.layers.Dense(1, activation='sigmoid', name='pred_COPD')(x)

    model = tf.keras.Model(inputs=[base_model.input, Input], outputs=pred_COPD)
    
    return model