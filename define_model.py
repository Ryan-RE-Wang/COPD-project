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
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='EfficientNetV2M'):
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    elif (archi=='NASNetMobile'):
        base_model = tf.keras.applications.NASNetMobile(
                include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
    else:
        raise Exception('No matching archi!')

    pred_layer = tf.keras.layers.Dense(nodes, activation='sigmoid')(base_model.output)
 
    model = tf.keras.Model(inputs=base_model.input, outputs=pred_layer, name='model')  
  
    return model

def load_model_from_pretrain(archi='DenseNet121'):
    
    base_model = define_model(archi, nodes=6)
    base_model.load_weights('checkpoints/checkpoints_pretrain/checkpoint_pretrain_{i}'.format(i=archi))
    
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

def get_model_demo():
    Input = tf.keras.Input(shape=(10,), name='input_demo') # age - 4, race - 5, gender - 1

    base_model = load_model_from_pretrain('Xception')

    base_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer(base_model.layers[-2].name).output)

    cxr_model = tf.keras.Sequential()
    cxr_model.add(base_model)
    cxr_model.add(tf.keras.layers.Dense(64, name='cxr_embeddings'))

    cxr_model.layers[0]._name = 'base_model'

    combined = tf.keras.layers.concatenate([cxr_model.output, Input], name='fusion_layer')
    pred_COPD = tf.keras.layers.Dense(1, activation='sigmoid', name='pred_COPD')(combined)

    model = tf.keras.Model(inputs=[cxr_model.input, Input], outputs=pred_COPD)
    model.layers[0]._name = 'input_cxr'

    print(model.summary())
    
    return model