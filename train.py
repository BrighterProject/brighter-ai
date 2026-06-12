import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers

# initial configuration
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
img_size = (224, 224)
batch_size = 32
epochs = 10
learning_rate = 1e-3

# load datasets(80/20 split)
train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
    labels="inferred",
    label_mode="int"
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
    labels="inferred",
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes found: {class_names} (Total: {num_classes})")

# prefetching for more optimal performance
autotune = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=autotune)
val_ds = val_ds.prefetch(buffer_size=autotune)

# data augmentation to prevent overfitting
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="augmentation")

# normalization layer
normalize = layers.Rescaling(1./127.5, offset=-1)

# build transfer model architecture
# специализира модела на Google, MobileNetV2, за нашата цел
def build_transfer_model(num_classes):
    inputs = keras.Input(shape=(224, 224, 3))
    x = augmentation(inputs)
    x = normalize(x)

    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model

model = build_transfer_model(num_classes)
model.summary()

# model compile
model.compile(
    optimizer=optimizers.Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# model train and store best performance
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# model save
model.save(os.path.join(data_dir, "room_classifier.keras"))
best_val_acc = max(history.history["val_accuracy"])
print(f"\nTraining complete! Best validation accuracy achieved: {best_val_acc:.4f}")