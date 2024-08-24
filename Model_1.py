import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense
import os

# Define some constants
img_size = (224, 224)
batch_size = 32

# Use absolute paths for directories (Update these paths to the correct ones)
train_dir = os.path.abspath("Training")
test_dir = os.path.abspath("Testing")

# Getting train data
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Getting test data
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    label_mode='categorical',
    batch_size=batch_size
)

# Define Classes Names
class_names = train_data.class_names
print("Class names:", class_names)

# Define Callback list
callback_list = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(factor=0.8, monitor="val_accuracy", patience=3)
]

# Define Base_Model (EfficientNetB7)
base_model = tf.keras.applications.EfficientNetB7(include_top=False)
base_model.trainable = False

inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="Input_layer")
x = base_model(inputs)
x = Conv2D(32, 3, padding='same', activation="relu", name="Top_Conv_Layer")(x)
x = GlobalAveragePooling2D(name="Global_avg_Pooling_2D")(x)
outputs = Dense(4, activation="softmax", name="Output_layer")(x)

Model_2 = tf.keras.Model(inputs, outputs)

# Compile the model
Model_2.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Fitting the model for 5 epochs
Model_2_History = Model_2.fit(
    train_data,
    validation_data=test_data,
    epochs=5,
    verbose=1,
    callbacks=callback_list
)

# Un freaze the Base_model
base_model_2.trainable = True

#Freezing all the layers except last 10
for layer in base_model_2.layers[:-10]:
  layer.trainable = False


#ReCompile the model
Model_2.compile(loss = "categorical_crossentropy" ,
                optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001), #--< When FineTuning u want to lower the LR by 10x
                metrics = ["accuracy"]
               )


#FineTune for 10 epochs
initial_epoch = 5

Fine_Tune_epoch = initial_epoch + 10

#Refit the model
Model_2_Stage_2_history = Model_2.fit(train_data ,
                              epochs = Fine_Tune_epoch ,
                              validation_data = test_data ,
                              validation_steps = len(test_data) ,
                              initial_epoch = initial_epoch-1 )

Model_2.save('model_22.h5')