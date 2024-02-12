
import tensorflow as tf
from tensorflow.keras import layers, models

class SimpleCNN(models.Model):
    """
    A simple Convolutional Neural Network (CNN) for image classification
    using TensorFlow/Keras. This model is designed for demonstration
    purposes and can be extended for more complex tasks.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # First convolutional block
        self.conv1 = layers.Conv2D(32, (3, 3), activation=\'relu’, input_shape=(32, 32, 3))
        self.pool1 = layers.MaxPooling2D((2, 2))
        # Second convolutional block
        self.conv2 = layers.Conv2D(64, (3, 3), activation=\'relu’)
        self.pool2 = layers.MaxPooling2D((2, 2))
        # Third convolutional block
        self.conv3 = layers.Conv2D(64, (3, 3), activation=\'relu’)
        # Flatten the output for the dense layers
        self.flatten = layers.Flatten()
        # Dense layers for classification
        self.dense1 = layers.Dense(64, activation=\'relu’)
        self.dropout = layers.Dropout(0.5) # Added dropout for regularization
        self.dense2 = layers.Dense(num_classes, activation=\'softmax’)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

def compile_and_train_model(model, train_dataset, val_dataset, epochs=10):
    """
    Compiles and trains the given TensorFlow Keras model.
    
    Args:
        model (tf.keras.Model): The Keras model to compile and train.
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
        epochs (int): Number of training epochs.
    """
    print("Compiling model...")
    model.compile(optimizer=\'adam’,
                  loss=\'sparse_categorical_crossentropy’,
                  metrics=[‘accuracy’])
    print("Model compiled successfully.")

    print(f"Training model for {epochs} epochs...")
    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
    print("Model training complete.")
    return history

def preprocess_image_dataset(image, label):
    """
    Preprocesses a single image for the CNN model.
    Resizes, normalizes, and converts to appropriate data types.
    """
    image = tf.image.resize(image, (32, 32))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def load_and_prepare_cifar10(batch_size=32):
    """
    Loads the CIFAR-10 dataset, preprocesses it, and prepares it
    for training and validation with TensorFlow.
    """
    print("Loading CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    # Preprocess and batch the datasets
    train_dataset = train_dataset.map(preprocess_image_dataset).shuffle(10000).batch(batch_size)
    test_dataset = test_dataset.map(preprocess_image_dataset).batch(batch_size)
    
    print("CIFAR-10 dataset loaded and prepared.")
    return train_dataset, test_dataset

if __name__ == "__main__":
    # Example usage:
    # Load and prepare data
    train_ds, test_ds = load_and_prepare_cifar10()
    
    # Initialize and train model
    cnn_model = SimpleCNN(num_classes=10)
    compile_and_train_model(cnn_model, train_ds, test_ds, epochs=1)
    
    # Evaluate the model
    print("Evaluating model...")
    loss, accuracy = cnn_model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("TensorFlow model example completed.")
