import cv2
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model architecture from JSON file
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Create the model from the loaded architecture
loaded_model = model_from_json(loaded_model_json)

# Load the model weights from H5 file
loaded_model.load_weights("model/emotion_model.h5")

# Initialize image data generator with rescaling
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all validation images
validation_generator = validation_data_gen.flow_from_directory(
    'data/validation',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# Compile the model
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Evaluate the model on the validation dataset
loss, accuracy = loaded_model.evaluate(validation_generator)

# Print the accuracy
print("Validation Accuracy:", accuracy)
