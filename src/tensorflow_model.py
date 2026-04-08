# src/tensorflow_model.py

import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape, num_classes):
    """Creates a simple Convolutional Neural Network (CNN) model using TensorFlow/Keras."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, train_images, train_labels, epochs=10, batch_size=64):
    """Compiles and trains the given Keras model."""
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2)
    return history

def evaluate_model(model, test_images, test_labels):
    """Evaluates the trained Keras model on test data."""
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'
Test accuracy: {test_acc}')
    return test_loss, test_acc

if __name__ == '__main__':
    # Example usage with dummy data (replace with actual data loading)
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Select a subset for faster execution in example
    train_images = train_images[:1000]
    train_labels = train_labels[:1000]
    test_images = test_images[:200]
    test_labels = test_labels[:200]

    input_shape = train_images.shape[1:]
    num_classes = 10

    model = create_cnn_model(input_shape, num_classes)
    model.summary()

    print("
Training TensorFlow model...")
    train_model(model, train_images, train_labels, epochs=1)

    print("
Evaluating TensorFlow model...")
    evaluate_model(model, test_images, test_labels)
