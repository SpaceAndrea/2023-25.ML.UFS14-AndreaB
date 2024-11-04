import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# Directory di input e output
input_dir = os.environ['SM_CHANNEL_TRAINING']
model_dir = os.environ['SM_MODEL_DIR']

def train_model(path):
    # Carica il dataset
    df = pd.read_csv(path)

    # Genera e salva il grafico della distribuzione dei tipi di pianeti
    if 'planet_type' in df.columns:
        planet_type_counts = df['planet_type'].value_counts()
        plt.figure(figsize=(10, 6))
        planet_type_counts.plot(kind='bar')
        plt.xlabel('Tipo di Pianeta')
        plt.ylabel('Conteggio')
        plt.title('Distribuzione dei Tipi di Pianeti')

        plot_path = os.path.join(model_dir, 'planet_type_distribution.png')
        plt.savefig(plot_path)
        plt.close()
    else:
        print("La colonna 'planet_type' non è presente nel dataset.")


    # Resto del codice di preprocessamento e addestramento
    # Preprocessamento dei dati (adatta questo codice al tuo dataset)
    df_numeric = df.select_dtypes(include=[np.number]).dropna()

    # Assicurati che ci siano dati numerici per l'addestramento
    if not df_numeric.empty:
        # Sostituisci 'sy_dist' con la tua colonna target se diversa
        X = df_numeric.drop('sy_dist', axis=1, errors='ignore').values
        y = df_numeric['sy_dist'].values if 'sy_dist' in df_numeric else None

        if y is not None:
            # Dividi i dati in training e test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Definisci e compila il modello
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)  # Per regressione
            ])
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

            # Addestra il modello
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

            # Salva il modello
            model.save(os.path.join(model_dir, '1'))

            # Salva un file di output per confermare il completamento
            with open(os.path.join(model_dir, 'output_model.txt'), 'w') as f:
                f.write('Il modello è stato addestrato con successo!')
        else:
            print("La colonna target 'sy_dist' non è presente nei dati.")
    else:
        print("Non ci sono dati numerici disponibili per l'addestramento.")

# Avvia l'addestramento
if __name__ == '__main__':
    try:
        # Il percorso al dataset è passato come argomento
        train_model(path=os.path.join(input_dir, 'nasa_exoplanets.csv'))
    except Exception as e:
        # Scrive eventuali errori nel file di log
        with open(os.path.join(model_dir, 'error_log.txt'), 'w') as f:
            f.write(str(e))
