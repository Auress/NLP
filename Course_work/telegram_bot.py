# -- coding: utf-8 -

import os
import logging
from telegram import Update
from telegram.ext  import Updater, CommandHandler, MessageHandler, CallbackQueryHandler, Filters, CallbackContext
import dialogflow
import torch
import string
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import annoy
from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import notebook
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from lightgbm import LGBMClassifier
import pickle
import numpy as np
from tqdm import tqdm_notebook
import pandas as pd
import re

# Обработка команд
def startCommand(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Добрый день')


def textMessage(bot, update):
    response = 'Ваше сообщение принял ' + update.message.text  # формируем текст ответа
    bot.send_message(chat_id=update.message.chat_id, text=response)


def start(update: Update, context: CallbackContext):
    update.message.reply_text('Hi!')


def echo(update: Update, context: CallbackContext):
    txt = update.message.text
    update.message.reply_text('Ваше сообщение! ' + update.message.text)


def preprocess_txt(line, mode='q'):  # mode='q'; mode='a' - for questions and answers
    txt = str(line)
    txt = re.split(r"[?!.]", txt)[0]  # Берем только по одномк предложению
    txt = re.sub("<.{,3}>", " ", txt)  # Убираем разметку
    txt = re.sub(r"[^а-яА-ЯёЁa-zA-Z,]+", " ", txt)  # Заменяем на пробел все кроме нужных символов
    txt = txt.lower()
    txt = re.sub(r"([?.!,])", r" \1 ", txt)  # Пробелы вокруг пунктуации
    txt = re.sub("\s{2,}", " ", txt)  # убираем лишние пробелы

    if mode == 'q':
        txt = re.sub("\sне\s", " не", txt)
        txt = [morpher.parse(word)[0].normal_form for word in txt.split() if word not in sw]  # if word not in sw
    elif mode == 'a':
        txt = [word for word in txt.split()]
    else:
        assert mode in ['a', 'q'], "Mode error"

    return txt


def embed_txt(txt, idfs, midf):
    n_ft = 0
    vector_ft = np.zeros(100)
    for word in txt:
        if word in modelFT:
            vector_ft += modelFT[word] * idfs.get(word, midf)
            n_ft += idfs.get(word, midf)
    return vector_ft / n_ft


def preprocess_txt_transformer(line, mode='q'):  # mode='q'; mode='a' - for questions and answers
    txt = str(line)
    txt = re.split(r"[?!.]", txt)[0]  # Берем только по одномк предложению
    txt = re.sub("<.{,3}>", " ", txt)  # Убираем разметку
    txt = re.sub(r"[^а-яА-ЯёЁa-zA-Z,]+", " ", txt)  # Заменяем на пробел все кроме нужных символов
    txt = txt.lower()
    txt = re.sub(r"([?.!,])", r" \1 ", txt)  # Пробелы вокруг пунктуации
    txt = re.sub("\s{2,}", " ", txt)  # убираем лишние пробелы

    if mode == 'q':
        txt = re.sub("\sне\s", " не", txt)
        txt = [morpher.parse(word)[0].normal_form for word in txt.split() if word not in sw]  # if word not in sw
    elif mode == 'a':
        txt = [word for word in txt.split()]
    else:
        assert mode in ['a', 'q'], "Mode error"

    return " ".join(txt)


def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': look_ahead_mask
    })
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
        'query': attention1,
        'key': enc_outputs,
        'value': enc_outputs,
        'mask': padding_mask
    })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
        }
        return config


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


def evaluate(sentence):
    sentence = preprocess_txt_transformer(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model_trs(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    return predicted_sentence

    # Все загружаем


def startCommand(update: Update, context: CallbackContext):
    update.message.reply_text('Добрый день!')


def modeCommand(update: Update, context: CallbackContext):
    global mode
    if mode == 'trs':
        mode = 'gpt'
        update.message.reply_text('mode: ruGPT-3 (скачанная модель)')
    else:
        mode = 'trs'
        update.message.reply_text('mode: transformer (обученная модель)')


def textMessage(update: Update, context: CallbackContext):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
    text_input = dialogflow.types.TextInput(text=update.message.text, language_code=DIALOGFLOW_LANGUAGE_CODE)
    query_input = dialogflow.types.QueryInput(text=text_input)

    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
    except InvalidArgument:
        raise

    if response.query_result.action == 'input.welcome':  # Приветствие из DialogFlow
        text = response.query_result.fulfillment_text
        if text:
            update.message.reply_text(response.query_result.fulfillment_text)
        else:
            update.message.reply_text('А вот это не совсем понятно.')

    else:
        input_txt = preprocess_txt(update.message.text)
        vect = vectorizer.transform([" ".join(input_txt)])
        prediction = lr.predict(vect)
        result = []

        if prediction[0] == 1:  # магазин
            vect_ft = embed_txt(input_txt, idfs, midf)
            ft_index_shop_val = ft_index_shop.get_nns_by_vector(vect_ft, 2000)

            for item in ft_index_shop_val:
                title, image, info = index_map_shop[item]
                if len(set(info).intersection(set(input_txt))) > 1:
                    result.insert(0, [title, image])
                elif len(set(info).intersection(set(input_txt))) > 0:
                    result.append([title, image])

            for title, image in result[:5]:
                text = "title: {} image: {}".format(title, image)
                update.message.reply_text(text)

        else:
            if mode == 'gpt':  # ruGRP-3
                start = tokenizer_gpt.encode(update.message.text, return_tensors="pt")
                result = model_gpt.generate(start.to(device),
                                            max_length=50,
                                            num_beams=10,
                                            early_stopping=True,
                                            no_repeat_ngram_size=3)
                text = (tokenizer_gpt.decode(result.cpu().flatten().numpy(), skip_special_tokens=True))
                try:
                    text = re.split('\n', text)[1]
                except IndexError:
                    pass

            else:  # transformer
                with tf.device("/cpu:0"):
                    text = predict(update.message.text)

            if len(text) > 0:
                update.message.reply_text(text)
            else:  # Если ничего, то ответ из DialogFlow
                update.message.reply_text(response.query_result.fulfillment_text)


if __name__ == '__main__':
    telegram_api_key = '' # Вставить telegram_api_key
    df_api_json = '' # Вставить dialogflow_api_json

    updater = Updater(token=telegram_api_key)  # Токен API к Telegram
    dispatcher = updater.dispatcher

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger()

        # Shop
    morpher = MorphAnalyzer()
    sw = set(get_stop_words("ru"))
    exclude = set(string.punctuation)

    modelFT = FastText.load("ft_model")

    ft_index = annoy.AnnoyIndex(100, 'angular')
    ft_index.load('speaker.ann')

    ft_index_shop = annoy.AnnoyIndex(100, 'angular')
    ft_index_shop.load('shop.ann')

    with open("index_speaker.pkl", "rb") as f:
        index_map = pickle.load(f)

    with open("index_shop.pkl", "rb") as f:
        index_map_shop = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("idfs.pkl", "rb") as f:
        idfs = pickle.load(f)

    with open("midf.pkl", "rb") as f:
        midf = pickle.load(f)

    with open("lr.pkl", "rb") as f:
        lr = pickle.load(f)

        # Transformer
    with open("questions_1000k.txt", "rb") as fp:
        questions = pickle.load(fp)
    with open("answers_1000k.txt", "rb") as fp:
        answers = pickle.load(fp)

    t_name = "1000k_tf_tokenizer_14"
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(t_name)

    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    VOCAB_SIZE = tokenizer.vocab_size + 2

    MAX_LENGTH = 40
    BATCH_SIZE = 64
    BUFFER_SIZE = 20000

    questions, answers = tokenize_and_filter(questions, answers)

    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': questions,
                'dec_inputs': answers[:, :-1]
            },
            {
                'outputs': answers[:, 1:]
            },
        ))

        dataset = dataset.cache()
        dataset = dataset.shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Hyper-parameters
        NUM_LAYERS = 3  # 2
        D_MODEL = 256  # 256
        NUM_HEADS = 8
        UNITS = 1024  # 512
        DROPOUT = 0.1

        model_trs = transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            units=UNITS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dropout=DROPOUT)

        learning_rate = CustomSchedule(D_MODEL)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        model_trs.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

        model_trs.load_weights('20_epoches_model_1000k')

        # ruGPT-3
    tokenizer_gpt = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
    model_gpt = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2", output_attentions=True)

    device = "cpu"
    model_gpt.to(device)

        # Bot
    updater = Updater(telegram_api_key, use_context=True)  # Токен API к Telegram
    dispatcher = updater.dispatcher
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = df_api_json  # скачнный JSON

    DIALOGFLOW_PROJECT_ID = 'df-nlp-cwork-bot-ratf'  # PROJECT ID из DialogFlow
    DIALOGFLOW_LANGUAGE_CODE = 'ru'  # язык
    SESSION_ID = 'ES_nlp_CWork_bot'  # ID бота из телеграма

    mode = 'trs'  # trs - transformer; gpt - downloaded ruGPT-3

    dispatcher.add_handler(CommandHandler("start", startCommand))
    dispatcher.add_handler(CommandHandler("m", modeCommand))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, textMessage))

    # Start the Bot
    updater.start_polling()
    updater.idle()
