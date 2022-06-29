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

def define_model(archi='DenseNet121', nodes=1):
    if (archi=='DenseNet121'):
        base_model = tf.keras.applications.densenet.DenseNet121(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='DenseNet201'):
        base_model = tf.keras.applications.densenet.DenseNet201(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='InceptionV3'):
        base_model = tf.keras.applications.InceptionV3(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='ResNet50V2'):
        base_model = tf.keras.applications.ResNet50V2(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='ResNet152V2'):
        base_model = tf.keras.applications.ResNet152V2(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='InceptionResNetV2'):
        base_model = tf.keras.applications.InceptionResNetV2(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='Xception'):
        base_model = tf.keras.applications.Xception(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')    
    elif (archi=='MobileNetV2'):
        base_model = tf.keras.applications.MobileNetV2(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='EfficientNetV2S'):
        base_model = tf.keras.applications.EfficientNetV2S(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='EfficientNetV2M'):
        base_model = tf.keras.applications.EfficientNetV2M(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='NASNetMobile'):
        base_model = tf.keras.applications.NASNetMobile(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    else:
        raise Exception('No matching archi!')
        
    if (nodes==1):
        pred_layer = tf.keras.layers.Dense(1, activation='sigmoid')(base_model.output)
    else:  
        pred_layer = tf.keras.layers.Dense(nodes, activation='sigmoid')(base_model.output)
 
    model = tf.keras.Model(inputs=base_model.input, outputs=pred_layer, name='model')  
  
    return model

def load_model_from_pretrain(archi='Dnet121'):
    
    base_model = define_model(archi, nodes=6)
    base_model.load_weights('checkpoints_new/checkpoint_pretrain_{i}'.format(i=archi))
    
#     x = tf.keras.layers.Dense(256, activation='sigmoid')(base_model.get_layer(base_model.layers[-2].name).output)
    
    pred_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='pred_layer')(base_model.get_layer(base_model.layers[-2].name).output)

    model = tf.keras.Model(inputs=base_model.input, outputs=pred_layer, name='model')
    
    depth = len(model.layers)
    for i, layer in enumerate(model.layers):
        if (i == int(depth*0.3)):
            break
        else:
            layer.trainable = False
        
    return model
    
def get_ensemble_mlp():
    Input = tf.keras.Input(shape=(2560))

    x = tf.keras.layers.Dense(512, activation=swish_activation)(Input)
    x = tf.keras.layers.Dense(128, activation=swish_activation)(x)
    x = tf.keras.layers.Dense(32, activation=swish_activation)(x)
    x = tf.keras.layers.Dense(8, activation=swish_activation)(x)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=Input, outputs=pred)
    
    return model

def define_model_demo(archi='Dnet121'):
    Input = tf.keras.Input(shape=(10,), name='input_demo') # sex - 1, age - 4, ethnicity - 5

    base_model = define_model(archi)
    
    base_model.load_weights('checkpoints/AUC/checkpoint_BCE_Dnet121')
    base_model = freeze_layer(base_model, 'max_pool')
    
    base_model.layers[0]._name = 'input_cxr'
        
    y = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer(base_model.layers[-2].name).output)
    
    # combine the output of the two branches
    combined = tf.keras.layers.concatenate([y.output, Input])
    pred_COPD = tf.keras.layers.Dense(1, activation='sigmoid', name='pred_COPD')(combined)

    model = tf.keras.Model(inputs=[base_model.input, Input], outputs=pred_COPD)
    
    return model