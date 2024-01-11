# Import necessary libraries
import splitfolders
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the dataset
path = "Dataset"

# Display the contents of the dataset directory
print(os.listdir(path))

# Split the dataset into training and validation sets
splitfolders.ratio(path, seed=1337, output="Dataset-Splitted", ratio=(0.8, 0.2), group_prefix=None)

# Define paths for the training and validation sets
train_data_dir = "Dataset-Splitted/train"
test_data_dir = "Dataset-Splitted/val"

# Set image dimensions for preprocessing
img_width, img_height = 224, 224

# Set training parameters
#number of times the entire training dataset is passed through the neural network during training
epochs = 10
#the number of samples that are processed together in each iteration of training
batch_size = 32
num_classes = 3  # Number of classes in the dataset

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255, # Rescale : to normalize pixel values in the images
    shear_range=0.2, # Shear_range : transformation that shifts one part of an image, keeping the rest fixed
    zoom_range=0.2, # Zoom_Range: defines the range for random zooming applied to the training images
    horizontal_flip=True # Horizontal_Flip : helps the model generalize better by exposing it to variations in orientation
)

# Rescaling for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Loading the training set using ImageDataGenerator
training_set = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),#specifies the dimensions to which all images found in the specified directory will be resized
    batch_size=batch_size, #determines the number of samples in each batch
    class_mode='categorical',  # Use 'categorical' for multi-class classification : specifies the type of label arrays returned by the generator
    shuffle=True # determines whether to shuffle the order of the images in each batch
)

# Loading the test set using ImageDataGenerator
test_set = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height), 
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    shuffle=False  # No shuffling for the test set
)

# Building the Convolutional Neural Network (CNN) model
model = Sequential() #A linear stack of layers in Keras that allows for the creation of a neural network layer-by-layer in a step-by-step fashion
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu')) # Adds a 2D convolutional layer to the model.
model.add(MaxPooling2D(pool_size=(2, 2))) # This adds a 2D max-pooling layer to the model
model.add(Flatten()) # Flattens the input, transforming it into a one-dimensional array
model.add(Dense(128, activation='relu')) # Fully connected (dense) layers in the neural network
model.add(Dense(num_classes, activation='softmax'))  # Softmax for multi-class classification: Softmax converts the output into probability scores for each class

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit( #a method in Keras used to train the neural network model.
    training_set, #The generator providing training data.
    steps_per_epoch=training_set.samples // batch_size, # Number of steps (batches) to consider one epoch finished during training.
    epochs=epochs, # Number of times to iterate over the entire training dataset
    validation_data=test_set, # The generator providing validation data
    validation_steps=test_set.samples // batch_size # Number of steps (batches) to consider one validation epoch finished
)

# Save the trained model
model.save('trained_model')

# Define class labels
class_labels = ['COVID', 'NORMAL', 'PNEUMONIA']

# Make predictions on the test set
predictions = model.predict(test_set)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Get true labels
true_labels = test_set.classes

# Compute accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Create confusion matrix
conf_mat = confusion_matrix(true_labels, predicted_labels)

# Display confusion matrix using seaborn and matplotlib
plt.figure(figsize=(8, 6)) # Line creates a figure (plot) with a specific size of 8 inches in width and 6 inches in height
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels) #  creates a heatmap of the confusion matrix for better visualization. Set the labels for the x and y-axis ticks, respectively
plt.xlabel('Predicted') #  set the labels for the x  of the plot
plt.ylabel('True')# set the labels for the y-axis  of the plot  
plt.title('Confusion Matrix') # Sets the title of the plot
plt.show() # Displays the plot.