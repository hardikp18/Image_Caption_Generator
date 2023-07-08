import streamlit as st
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Input, Dense, LSTM ,add,Dropout, Embedding 
import numpy as np
import cv2
from PIL import Image, ImageOps


# Obtained earlier 
# MAX_LENGTH=35
    
def idx_to_word(integer,tokenizer):
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def read_list(file_path):
    lst = []
    with open(file_path, 'r') as file:
        for line in file:
            lst.append(line.strip())
    return lst

def preprocess_image(img):
    size = (224,224)  
    image = ImageOps.fit(img, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis,...]
    image = preprocess_input(img_reshape)
    return image

def predict_caption(model,feature,tokenizer):
    max_length =35
    # add start tag for generation process 
    in_text='startseq'
    #iterate over the max length of sequence
    for i in range(max_length):
        # Encode input sequence
        sequence=tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence= pad_sequences([sequence],max_length)
        # Predict the next word
        # image=image.reshape((1,4096))
        yhat = model.predict([feature,sequence],verbose=0) 
        # Get the index with the highest probability
        yhat=np.argmax(yhat)
        # Convert the index to word
        word = idx_to_word(yhat,tokenizer)
        if word is None:
            break
        # Append word as input
        in_text+= " "+word
        # Stop if we reach end tag
        if word=='endseq':
            break
    return in_text

def generate_clean_caption(in_text):
    # Split the string into a list of words
    words = in_text.split()
    # Join all words except the first and last
    result = ' '.join(words[1:-1])  
    return result
    
    

## Layout ##
rad = st.sidebar.radio("Navigation",["Home","Caption Generator"])


if rad=="Home":
    st.title("Image Caption Generator")
    st.subheader("By Hardik Pahwa")
    st.markdown("""Upload an Image in the Caption Generator page to generate Caption""")
    
    # st.image("sample.jpg")
    
elif rad=="Caption Generator":
    st.write("""
            # Caption Generation
    """)
   
    vocab_size =8485
    vgg = VGG16()
    # restructure the model
    vgg_model = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
  
    with st.spinner("Loading Model..."):
        model = load_model("model.h5") 
     
    file = st.file_uploader("Upload an Image")
    st.write(file)
    st.set_option('deprecation.showfileUploaderEncoding', False)
        
    if file != None:
        image = Image.open(file)
        st.image(image,caption="Uploaded Image",use_column_width=True)
        #  Prepocessing the Image
        image=preprocess_image(image)
        feature =vgg_model.predict(image,verbose=0)
        
        all_captions = read_list('all_captions.txt')
        #Tokenize the text
        tokenizer=Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        
        
       
        # feature = vgg_model.predict(image,verbose=0)
        
        in_text=predict_caption(model,feature,tokenizer) 
        
        # Removing Startseq and Endseq
        caption= generate_clean_caption(in_text)
        
        # # Display the Caption
        st.markdown('<b>Generated Caption:</b>',True)
        st.markdown(caption.upper())
    else:
        st.write("No file uploaded")
