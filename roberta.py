from sklearn.model_selection import KFold
from transformers import TFAutoModel, AutoTokenizer

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# get paths to TensorFlow XLM-RoBERTa base and large models
roberta_base = "jplu/tf-xlm-roberta-base"
roberta_large = 'jplu/tf-xlm-roberta-large'


def load_data(path_path: str):
    train = pd.read_csv(path_path)
    return train


TOKENIZER = AutoTokenizer.from_pretrained(roberta_large)
LR_RATE = 1e-5
EPOCHS = 10
FOLDS = 4
MAX_LEN = 85
TTA = 3
VERBOSE = 2
BATCH_SIZE, steps_per_epoch
DEVICE = 'CPU'

# helper function to create our model
def build_model(transformer_layer, max_len, learning_rate):
    # must use this to send to TPU cores
    with strategy.scope():
        # define input(s)
        input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32)
        print("input")

        # insert roberta layer
        roberta = TFAutoModel.from_pretrained(transformer_layer)
        roberta = roberta(input_ids)[0]
        print("roberta")
        # only need <s> token here, so we extract it now
        out = roberta[:, 0, :]

        # add our softmax layer
        out = tf.keras.layers.Dense(3, activation='softmax')(out)
        print("dense")
        # assemble model and compile
        model = tf.keras.Model(inputs=input_ids, outputs=out)
        print("model")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    return model

def setRuntime():
    if DEVICE == "TPU":
        print("connecting to TPU...")
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.master())
        except ValueError:
            print("Could not connect to TPU")
            tpu = None

        if tpu:
            try:
                print("initializing  TPU ...")
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                strategy = tf.distribute.experimental.TPUStrategy(tpu)
                print("TPU initialized")
            except _:
                print("failed to initialize TPU")
        else:
            DEVICE = "GPU"

    if DEVICE != "TPU":
        print("Using default strategy for CPU and single GPU")
        strategy = tf.distribute.get_strategy()

    if DEVICE == "GPU":
        print("Num GPUs Available: ", len(
            tf.config.experimental.list_physical_devices('GPU')))

    AUTO = tf.data.experimental.AUTOTUNE
    REPLICAS = strategy.num_replicas_in_sync
    print(f'REPLICAS: {REPLICAS}')

    global BATCH_SIZE, steps_per_epoch
    BATCH_SIZE = 16 * REPLICAS

    STEPS_PER_EPOCH = len(train) // BATCH_SIZE


#function to encode text and convert dataset to tensor dataset
def to_tf_dataset(dataset, max_len, repeat = False, shuffle = False, labeled = True, batch_size = BATCH_SIZE):
    """
    Convert Dataset to tensorflow format
    """
    dataset_text = dataset[['premise', 'hypothesis']].values.tolist()
    dataset_enc = TOKENIZER.batch_encode_plus(dataset_text, pad_to_max_length = True, max_length = max_len)
    
    if labeled:
        tf_dataset = tf.data.Dataset.from_tensor_slices((dataset_enc['input_ids'], dataset['label']))
    else:
        tf_dataset = tf.data.Dataset.from_tensor_slices((dataset_enc['input_ids']))
    
    if repeat: tf_dataset = tf_dataset.repeat()  
        
    if shuffle: 
        tf_dataset = tf_dataset.shuffle(2048)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        tf_dataset = tf_dataset.with_options(opt)
        
    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(AUTO)
    
    return tf_dataset


if __name__ == '__main__':
    setRuntime()

    test = load_data("./data/testing_data/test.csv")
    train = load_data("./data/training_data/train.csv")

    skf = KFold(n_splits=FOLDS, shuffle=True)

    train_index, val_index = skf.split(train)

    # clear TPU memory to save memory in cloud environment
    if DEVICE == 'TPU':
        if tpu:
            tf.tpu.experimental.initialize_tpu_system(tpu)

    # build model
    K.clear_session()
    model = build_model(roberta_large, max_len=MAX_LEN, learning_rate=LR_RATE)

    train_ds = to_tf_dataset(
        train.loc[train_index], labeled=True, shuffle=True, repeat=True, max_len=MAX_LEN)
    val_ds = to_tf_dataset(
        train.loc[val_index], labeled=True, shuffle=False, repeat=False, max_len=MAX_LEN)


    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                        verbose=VERBOSE)


    # convert test to tensor dataset
    test_tf = to_tf_dataset(
        test, shuffle=False, labeled=False, repeat=False, max_len=MAX_LEN)

    # predict with augmentated validation sets
    pred = model.predict(test_tf, verbose=VERBOSE)
    print("Prediction:", pred)