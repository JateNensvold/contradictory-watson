
import numpy as np
import pandas as pd

from transformers import BertTokenizer, TFBertModel

import matplotlib.pyplot as plt
import tensorflow as tf
import os

os.environ["WANDB_API_KEY"] = "0"  # to silence warning

# try:
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)
#     strategy = tf.distribute.experimental.TPUStrategy(tpu)
# except ValueError:
#     strategy = tf.distribute.get_strategy()  # for CPU and single GPU
#     print('Number of replicas:', strategy.num_replicas_in_sync)


def load_data(path_path: str):

    train = pd.read_csv(path_path)
    return train


def run_BERT(train: pd.DataFrame, test: pd.DataFrame):
    """
    Runs the 'bert-base-multilingual-cased' model against the input dataset
    """
    model_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name, progress=False)

    def encode_sentence(s):
        """
        Encode sentence from ascii/natural language to array of numbers
        """
        tokens = list(tokenizer.tokenize(s))
        tokens.append('[SEP]')
        return tokenizer.convert_tokens_to_ids(tokens)

    def encode_data(hypotheses, premises, tokenizer, max_length=80):
        """
        Prepare data to be passed to BERT model
        """
        num_examples = len(hypotheses)

        sentence1 = tf.ragged.constant([
            encode_sentence(s)
            for s in np.array(hypotheses)])
        sentence2 = tf.ragged.constant([
            encode_sentence(s)
            for s in np.array(premises)])
        # print(sentence1.shape, sentence2.shape)
        cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
        input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)
        input_mask = tf.ones_like(input_word_ids).to_tensor()

        type_cls = tf.zeros_like(cls)
        # print("cls", type_cls.shape)
        type_s1 = tf.zeros_like(sentence1)
        # print("s1", type_s1.shape)
        type_s2 = tf.ones_like(sentence2)
        # print("s2", type_s2.shape)

        input_type_ids = tf.concat(
            [type_cls, type_s1, type_s2], axis=-1).to_tensor()
        # print("ids", input_type_ids.shape)
        # print(input_type_ids)
        # for i in range(input_type_ids.shape[0]):
        #     print("s1", type_s1[i])
        #     print("id", input_type_ids[i])
        cleaned_data = {
            'input_word_ids': input_word_ids.to_tensor(),
            'input_mask': input_mask,
            'input_type_ids': input_type_ids}
        # x = [h + ' [SEP] ' + p for h,
        #      p in zip(np.array(hypotheses), np.array(premises))]
        # x = tokenizer(x, padding=True, truncation=True, max_length=max_length)

        # inputs = {
        #     'input_word_ids': tf.ragged.constant(x['input_ids']).to_tensor(),
        #     'input_mask': tf.ragged.constant(x['attention_mask']).to_tensor(),
        #     'input_type_ids': tf.ragged.constant(x['token_type_ids']).to_tensor()}

        # return inputs
        return cleaned_data

    def build_model(name):
        max_len = 259

        bert_encoder = TFBertModel.from_pretrained(name)
        # input_word_ids = tf.keras.Input(
        #     shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        # input_mask = tf.keras.Input(
        #     shape=(max_len,), dtype=tf.int32, name="input_mask")
        # input_type_ids = tf.keras.Input(
        #     shape=(max_len,), dtype=tf.int32, name="input_type_ids")

        input_word_ids = tf.keras.Input(
            shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(
            shape=(max_len,), dtype=tf.int32, name="input_mask")
        input_type_ids = tf.keras.Input(
            shape=(max_len,), dtype=tf.int32, name="input_type_ids")

        embedding = bert_encoder(
            [input_word_ids, input_mask, input_type_ids])[0]
        output = tf.keras.layers.Dense(
            3, activation='softmax')(embedding[:, 0, :])
        model = tf.keras.Model(
            inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)
        model.compile(tf.keras.optimizers.Adam(
            lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    train_input = encode_data(train.premise.values,
                              train.hypothesis.values,
                              tokenizer)

    # with strategy.scope():
    model = build_model(model_name)
    model.summary()
    print(type(train_input))




    print(train_input)
    print(train.label.shape)
    model.fit(train_input, train.label.values, epochs=2,
              verbose=1, batch_size=64, validation_split=0.2)

    test_input = bert_encode(
        test.premise.values, test.hypothesis.values, tokenizer)


if __name__ == '__main__':
    data = load_data("./data/training_data/train.csv")
    test = load_data("./data/testing_data/test.csv")
    #  Create Pie chart of languages
    labels, frequencies = np.unique(data.language.values, return_counts=True)

    plt.figure(figsize=(10, 10))
    plt.pie(frequencies, labels=labels, autopct='%1.1f%%')
    plt.show()

    run_BERT(data, test)



