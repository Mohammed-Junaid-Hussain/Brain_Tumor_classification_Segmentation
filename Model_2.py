import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define some constants
img_size = (299, 299)
batch_size = 32

# Use absolute paths for directories (Update these paths to the correct ones)
train_dir = os.path.abspath("Training")
test_dir = os.path.abspath("Testing")

# Define ImageDataGenerator
_gen = ImageDataGenerator(rescale=1/255.0, brightness_range=(0.8, 1.2))
ts_gen = ImageDataGenerator(rescale=1/255.0)

# Flow from Directory
train_data = _gen.flow_from_directory(
    directory=train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = ts_gen.flow_from_directory(
    directory=test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define Classes Names
class_names = train_data.class_indices
print("Class names:", class_names)

# Define Callback list
callback_list = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(factor=0.8, monitor="val_accuracy", patience=3)
]

# Define the input shape
img_shape = (299, 299, 3)

# Load the Xception model with pre-trained weights, excluding the top layer
base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet',
                                            input_shape=img_shape, pooling='max')

# Optional: Freeze the layers of the base model to prevent them from being trained
# for layer in base_model.layers:
#     layer.trainable = False

# Create the model
model_1 = Sequential([
    base_model,
    Flatten(),
    Dropout(rate=0.3),
    Dense(128, activation='relu'),
    Dropout(rate=0.25),
    Dense(len(class_names), activation='softmax')  # Ensure the output layer matches the number of classes
])

# Compile the model
model_1.compile(Adamax(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

# Fit the model
model_1.fit(
    train_data,
    epochs=10,
    validation_data=test_data,
    shuffle=False,
    callbacks=callback_list
)

# Save the model
model_1.save('model_1.h5')