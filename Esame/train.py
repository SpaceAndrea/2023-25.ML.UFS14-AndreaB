import os
import pickle
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    # Percorsi gestiti tramite SageMaker
    MODEL_SAVE_PATH = os.environ['SM_MODEL_DIR']
    INPUT_TRAIN_PATH = os.path.join(os.environ['SM_INPUT_DIR'], 'data/training')

    # Parametri configurabili
    batch_size = 32
    image_size = (224, 224)
    epochs = 4

    # Preparazione dei dati
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        INPUT_TRAIN_PATH,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        INPUT_TRAIN_PATH,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Salva la mappatura delle classi
    class_indices = train_generator.class_indices
    with open(os.path.join(MODEL_SAVE_PATH, 'class_indices.pkl'), 'wb') as f:
        pickle.dump(class_indices, f)

    # Costruzione del modello
    input_shape = (image_size[0], image_size[1], 3)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    # Compilazione e training
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    # Salva il modello
    model.save(os.path.join(MODEL_SAVE_PATH, 'model.keras'))
    print("Modello salvato con successo in:", MODEL_SAVE_PATH)
