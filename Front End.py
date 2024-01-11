# Import necessary libraries
import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import requests
from tensorflow.keras.preprocessing import image

# Load the pretrained model
model = load_model('C:/Users/MSI/Downloads/trained_model')

# Download labels for ImageNet
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n") # Applied to the obtained text, splitting it into a list of strings using the newline character (\n) as the delimiter.

# Define the prediction function for Gradio
# Load labels specific to your model
class_labels = ["Covid", "Normal", "Pneumonia"]

# Define the prediction function for Gradio
def classify_image(inp):
    print("Input shape:", inp.shape)
    
    # Convert the Gradio image to a numpy array
    img_array = inp.astype(np.uint8) # Ensures that the pixel values in the image are within the valid range of 0 to 255
    img_array = Image.fromarray(img_array) #The img_array obtained from the previous step is converted into a PIL Image
    img_array = img_array.resize((224, 224)) # The image is resized to a square shape with dimensions (224, 224)
    
    # Convert the image to a numpy array for prediction
    img_array = image.img_to_array(img_array) # Performed to prepare the image for input to a neural network.
    img_array = np.expand_dims(img_array, axis=0) # Neural network models often expect input data to have a batch dimension, so this line is adding a batch dimension to the image.
    img_array /= 255.0  # Rescale to match the preprocessing used during training , form of normalization, rescaling the pixel values from the original range of 0 to 255 to a new range of 0.0 to 1.0
    
    # Make predictions with the model
    predictions = model.predict(img_array)
    
    # Get the index of the predicted class
    predicted_class_index = np.argmax(predictions)
    
    # Return the result as the predicted class
    return class_labels[predicted_class_index]

# Create the Gradio interface
iface = gr.Interface( # Initializes a Gradio interface
    fn=classify_image, # Specifies the function that will be used to perform predictions on input data
    inputs=gr.Image(), # Specifies the type of input expected by the interface
    outputs=gr.Label(num_top_classes=3),# Specifies the type of output expected by the interface
    examples=["C:/Users/MSI/Downloads/COVID.png", "C:/Users/MSI/Downloads/Dataset/PNEUMONIA/PNEUMONIA_92.png", "C:/Users/MSI/Downloads/Dataset/NORMAL/NORMAL_96.png", "C:/Users/MSI/Downloads/Dataset/COVID/COVID_983.png", "C:/Users/MSI/Downloads/Dataset/NORMAL/NORMAL_97.png", "C:/Users/MSI/Downloads/Dataset/PNEUMONIA/PNEUMONIA_93.png", "C:/Users/MSI/Downloads/Dataset/COVID/COVID_984.png", "C:/Users/MSI/Downloads/Dataset/NORMAL/NORMAL_99.png"]
) # provides a list of example inputs for testing the interface

# Launch the Gradio interface
iface.launch()
