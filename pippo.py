"""
Optuna example that optimizes a neural network classifier configuration for the
MNIST dataset using Keras.

In this example, we optimize the validation accuracy of MNIST classification using
Keras. We optimize the filter and kernel size, kernel stride and layer activation.

"""

import urllib
import matplotlib.pyplot as plt
import optuna
from optuna.pruners import HyperbandPruner
from optuna.pruners import MedianPruner
from optuna.pruners import PercentilePruner
from optuna.pruners import ThresholdPruner 
from optuna.pruners import NopPruner
from optuna.importance import get_param_importances
from optuna.samplers import TPESampler
from optuna.samplers import RandomSampler

from keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Sequential
from keras.optimizers import RMSprop

from plotly.io import show

import logging
import sys


# TODO(crcrpar): Remove the below three lines once everything is ok.
# Register a global custom opener to avoid HTTP Error 403: Forbidden when downloading MNIST.
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10

# Callback per contare i trial validi
def count_valid_trials(study, trial):
    # Aggiungi solo i trial che non sono pruned
    if trial.state != optuna.trial.TrialState.PRUNED:
        study.valid_trials += 1
    # Se abbiamo raggiunto il numero di trial validi desiderato, fermiamo l'ottimizzazione
    if study.valid_trials >= 100:
        print("Raggiunto il numero massimo di trial validi!")
        study.stop()

def objective(trial):
    # Clear clutter from previous Keras session graphs.
    

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    img_x, img_y = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(-1, img_x, img_y, 1)[:N_TRAIN_EXAMPLES].astype("float32") / 255
    x_valid = x_valid.reshape(-1, img_x, img_y, 1)[:N_VALID_EXAMPLES].astype("float32") / 255
    y_train = y_train[:N_TRAIN_EXAMPLES]
    y_valid = y_valid[:N_VALID_EXAMPLES]
    input_shape = (img_x, img_y, 1)

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(
        Conv2D(
            filters=trial.suggest_categorical("filters", [32, 64]),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5]),
            strides=trial.suggest_categorical("strides", [1, 2]),
            activation = trial.suggest_categorical("activation", [
                "relu",        # classica ReLU
                "linear",      # funzione identità (nessuna attivazione)
                "sigmoid",     # utile per output tra 0 e 1
                "tanh"        # compressa tra -1 e 1, zero-centered
            ])
        )
    )
    model.add(Flatten())
    model.add(Dense(CLASSES, activation="softmax"))

    # We compile our model with a sampled learning rate.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=RMSprop(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    '''model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        batch_size=BATCHSIZE,
        epochs=EPOCHS,
        verbose=False,
    )'''

    #best_accuracy = 0  # Per tenere traccia della miglior accuratezza durante l'allenamento

    for epoch in range(EPOCHS):
        # Allenamento per una epoca
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            shuffle=True,
            batch_size=BATCHSIZE,
            epochs=1,
            verbose=True,  # Non stampare dettagli durante il training con False o 0
        )
        
        # Recupera l'accuratezza di validazione per questa epoca
        # ! history.history['val_accuracy'] e' una lista che contiene per ciascuna epoca
        # ! eseguita il relativo valore della accuracy. se intendo eseguire nel ciclo
        # ! una epoca alla volta posso considerare l'unico valore con history.history['val_accuracy'][0],
        # ! ma se invece voglio guardare un numero maggiore di epoche e poi decidere da li se prunare,
        # ! allora mi conviece history.history['val_accuracy'][-1], prendendo the last one
        val_accuracy = history.history['val_accuracy'][-1]
        #print(len(history.history['val_accuracy']))

        print(f'\n validation accuracy is = ', val_accuracy, '\n')
        
        # Segnala il valore intermedio (accuratezza di validazione)
        trial.report(val_accuracy, epoch)
        
        # Aggiorna la miglior accuratezza
        #if val_accuracy > best_accuracy:
        #    best_accuracy = val_accuracy
        
        # Verifica se il trial deve essere interrotto
        if trial.should_prune():
            # ! check the epoch when the pruning happens
            print(f'\n we are at epoch:', epoch, "\n")
            raise optuna.exceptions.TrialPruned()

        # Valutazione finale sulla validazione
        score = model.evaluate(x_valid, y_valid, verbose=0)
        return score[1]  # Restituisce l'accuratezza


if __name__ == "__main__":

    # Abilita logging per Optuna
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Definisci il pruner (es. MedianPruner)
    #pruner = MedianPruner(n_startup_trials=5)
    pruner = PercentilePruner(percentile=30)
    #pruner = ThresholdPruner(0.8)
    #pruner = NopPruner()

    # inizia lo studio...
    # ! ho messo multivariate a True, che dovrebbe contemplare la possibilita' che
    # ! i parametri abbiano relazioni tra di loro
    study = optuna.create_study(direction="maximize", pruner=pruner,
                              sampler=TPESampler(multivariate=True, seed = 42), 
                              #sampler=RandomSampler(seed = 42), 
                              study_name="studio_MNIST", 
                              storage="sqlite:///example.db",
                              load_if_exists=True)

    # Aggiungi una variabile per tracciare i trial validi
    study.valid_trials = 0

    # Esegui l'ottimizzazione
    study.optimize(objective, n_trials=800, timeout=600,callbacks=[count_valid_trials])

    print("Numero di trial validi:", study.valid_trials)

    # Stampa i risultati
    '''print("Number of finished trials: {}".format(len(study.trials)))'''

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Ottieni lo spazio di ricerca usato (suggerito) nei trial eseguiti
    search_space = optuna.search_space.IntersectionSearchSpace()
    print('\n Print of intersection space of parameters checked in the valid studies:\n')
    print(search_space.calculate(study))
    print('\n')
    
    fig1 = optuna.visualization.plot_parallel_coordinate(study)
    show(fig1)

    # Ora possiamo analizzare le importanze
    importances = get_param_importances(study)

    print("\n🎯 Importanza degli iperparametri:")
    for param, importance in importances.items():
        print(f"  {param}: {importance:.4f}")

    # Ordina per importanza
    params = list(importances.keys())
    values = [importances[k] for k in params]

    # Plot a barre
    plt.figure(figsize=(8, 5))
    plt.barh(params, values, color="skyblue")
    plt.xlabel("Importanza")
    plt.title("Importanza degli iperparametri (Optuna API)")
    plt.gca().invert_yaxis()  # Parametri più importanti in alto
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()