#import cv2
import matplotlib.cm as cm
#from IPython.display import Image, display
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image as siva
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("saved_model/bestmodel.h5")
model_vgg=tf.keras.models.load_model("saved_model/bestmodel_vgg.h5")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://sollio.coop/sites/default/files/article_images/COVID-19.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

# Resnet methods
def get_img_array(img_path):
  """
  Input : Takes in image path as input
  Output : Gives out Pre-Processed image
  """
  path = img_path
  img = siva.load_img(path, target_size=(224,224,3))
  img = siva.img_to_array(img)
  img = np.expand_dims(img , axis= 0 )
  return img

#Vgg methods
def get_img_array_vgg(img_path):
  """
  Input : Takes in image path as input
  Output : Gives out Pre-Processed image
  """
  path = img_path
  img = siva.load_img(path, target_size=(224,224,3))
  img = siva.img_to_array(img)/255
  img = np.expand_dims(img , axis= 0 )
  return img

#Grad-Cam Methods

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = siva.load_img(img_path)
    img = siva.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = siva.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = siva.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = siva.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    st.image(cam_path)

def image_prediction_and_visualization(path, last_conv_layer_name="conv5_block3_3_conv",model=model):
    img_array = get_img_array(path)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    st.markdown("<span style='color:white'>The original input image: </span>",unsafe_allow_html=True)
    st.image(Image.open(path))
    st.markdown("<span style='color:white'>Heatmap of the image :</span>",unsafe_allow_html=True)
    st.image(heatmap, use_column_width=True, channels="GRAY")
    st.markdown("<span style='color:white'>Image with heatmap representing the region of interest:</span>",unsafe_allow_html=True)
    save_and_display_gradcam(path, heatmap)
new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">HYBRID MODEL COVID-19 DETECTION</p>'
st.markdown(new_title, unsafe_allow_html=True)
### load file
uploaded_file = st.file_uploader("Choose a image file", type=["png","jpg","jpeg"])
# Covid +ve X-Ray is represented by 0 and Normal is represented by 1
class_type = {0:'Covid',  1 : 'Normal'}
if uploaded_file is not None:
    st.image(uploaded_file)
    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        img = get_img_array(uploaded_file)
        img1 = get_img_array_vgg(uploaded_file)
        res = class_type[np.argmax(model.predict(img))]
        res_vgg = class_type[np.argmax(model_vgg.predict(img1))]

        cr=model.predict(img)[0][0]*100
        nr=model.predict(img)[0][1]*100
        cv=model_vgg.predict(img1)[0][0]*100
        nv=model_vgg.predict(img1)[0][1]*100
        resc=(cr+cv)/2
        resn=(nr+nv)/2
        st.markdown("<span style='color:white'>The given X-Ray image is of type : {}</span>".format(res),unsafe_allow_html=True)
        st.markdown("<span style='color:white'>The chances of image being Covid is : {}</span>".format(resc),unsafe_allow_html=True)
        st.markdown("<span style='color:white'>The chances of image being Normal is : {}</span>".format(resn),unsafe_allow_html=True)
        image_prediction_and_visualization(uploaded_file)
        
