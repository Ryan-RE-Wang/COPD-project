import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from skimage.transform import resize

def make_gradcam_heatmap(img_array, last_conv_layer_model, classifier_model, target_class=None):

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        
        if (target_class is not None):
            top_pred_index = tf.constant(target_class)
        else:
            top_pred_index = tf.argmax(preds[0])
        
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap

def show_heatmap(img_array, last_conv_layer_model, classifier_model, target_class=None):

    heatmap = make_gradcam_heatmap(img_array, last_conv_layer_model, classifier_model, target_class)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
   

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((256, 256))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    return ((jet_heatmap/255)*0.4+img_array)


def grad_cam_plus(img_array, last_conv_layer_model, classifier_model, target_class=None):
    """Get a heatmap by Grad-CAM++.
    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list or None,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.
    Return:
        A heatmap ndarray(without color).
    """

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                last_conv_layer_output = last_conv_layer_model(img_array)
                # Compute class predictions
                preds = classifier_model(last_conv_layer_output)

                if (target_class is not None):
                    top_pred_index = tf.constant(target_class)
                else:
                    top_pred_index = tf.argmax(preds[0])

                top_class_channel = preds[:, top_pred_index]
                
            conv_first_grad = gtape3.gradient(top_class_channel, last_conv_layer_output)
            
        conv_second_grad = gtape2.gradient(conv_first_grad, last_conv_layer_output)
        
    conv_third_grad = gtape1.gradient(conv_second_grad, last_conv_layer_output)

    global_sum = np.sum(last_conv_layer_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num/alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0,1))
    alpha_normalization_constant = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, 1)
    alphas /= alpha_normalization_constant
    
    weights = np.maximum(conv_first_grad[0], 0.0)
    
    deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
    grad_cam_map = np.sum(deep_linearization_weights*last_conv_layer_output[0], axis=2)
    
    arr_min, arr_max  = np.min(grad_cam_map), np.max(grad_cam_map)
    heatmap = (grad_cam_map - arr_min) / (arr_max - arr_min + 1e-18)
    
    heatmap = np.uint8(255 * heatmap)
    
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((256, 256))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    return ((jet_heatmap/255)*0.4+img_array)