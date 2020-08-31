"""Model architecture."""
import itertools as it
import json
import random

import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import GRU, LSTM, Dense, Embedding, Reshape, TimeDistributed  # noqa: F401
from keras.models import Sequential
from keras.utils import Sequence

RNNLayer = LSTM  # to easily change RNN implementation (GRU. LSTM, etc).


def build_model(
        num_unique_chars=None, batch_size=None, seq_length=None, optim=None,
        embed_dim=None, rnn_units=[], rnn_dropout=0.0, rnn_rec_dropout=0.0,
        rnn_init='truncated_normal', rnn_rec_init='orthogonal', rnn_reg=None,
        rnn_rec_reg=None, rnn_activ_reg=None, rnn_return_sequences=True,
        dense_init='truncated_normal', dense_reg=None):
    model = Sequential()

    # LSTM needs batch_input_shape == (batch_size, seq_length, num_features)
    # use Embedding or Reshape as first layer.
    model.add(Embedding(
        input_dim=num_unique_chars,
        output_dim=embed_dim,
        batch_input_shape=(batch_size, seq_length),
        # input_shape=(seq_length,),
    ))

    for units in rnn_units[:-1]:
        model.add(RNNLayer(
            units=units,
            return_sequences=True,  # True when stacking RNN layers so shapes match.
            stateful=True,
            dropout=rnn_dropout,
            recurrent_dropout=rnn_rec_dropout,
            kernel_initializer=rnn_init,
            recurrent_initializer=rnn_rec_init,
            kernel_regularizer=rnn_reg,
            recurrent_regularizer=rnn_rec_reg,
            activity_regularizer=rnn_activ_reg,
        ))
    # last RNN layer may have a different return_sequences value.
    model.add(RNNLayer(
        units=rnn_units[-1],
        return_sequences=rnn_return_sequences,  # True when training, False when generating.
        stateful=True,
        dropout=rnn_dropout,
        recurrent_dropout=rnn_rec_dropout,
        kernel_initializer=rnn_init,
        recurrent_initializer=rnn_rec_init,
        kernel_regularizer=rnn_reg,
        recurrent_regularizer=rnn_rec_reg,
        activity_regularizer=rnn_activ_reg,
    ))

    if rnn_return_sequences:
        model.add(TimeDistributed(Dense(
            num_unique_chars,
            activation='softmax',
            kernel_initializer=dense_init,
            kernel_regularizer=dense_reg,
        )))
    else:
        model.add(Dense(
            num_unique_chars,
            activation='softmax',
        ))

    model.compile(
        optimizer=optim,
        loss='categorical_crossentropy',  # 'kullback_leibler_divergence'
        metrics=['accuracy']
    )

    return model


def gen_sequences(x, y, seq_length):
    for offset in range(0, x.shape[0] - seq_length, seq_length):
        yield x[offset:offset+seq_length], y[offset:offset+seq_length, :]


class BatchGenerator(Sequence):

    def __init__(self, dataset, batch_size, seq_length, shuffle=True):
        if shuffle:
            random.shuffle(dataset)
        xs, ys = zip(*dataset)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        # x.shape == (len_of_all_abcs, )
        # y.shape == (len_of_all_abcs, num_unique_chars)
        x, y = zip(*gen_sequences(x, y, seq_length))
        # x.shape == (len_of_all_abcs // seq_length, seq_length)
        # y.shape == (len_of_all_abcs // seq_length, seq_length, num_unique_chars)
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.len = len(x) // batch_size
        self.mod = len(x) % batch_size

    def __len__(self):
        return self.len

    def __getitem__(self, batch_index):
        # generate batches compatible with the stateful=True parameter of RNN.
        i = batch_index % self.len
        return self.x[np.arange(i, self.x.shape[0] - self.mod, self.len), :],\
            self.y[np.arange(i, self.x.shape[0] - self.mod, self.len), :, :]

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.dataset)
            xs, ys = zip(*self.dataset)
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            x, y = zip(*gen_sequences(x, y, self.seq_length))
            self.x, self.y = np.asarray(x), np.asarray(y)


def train_model(
        dataset, model_dir, build_params, epochs, from_epoch=0, model=None,
        weights_to_load=None):
    with (model_dir/'build_params.json').open('w') as f:
        json.dump(build_params, f)

    if not model:
        model = build_model(**build_params)

    if weights_to_load:
        model.load_weights(model_dir/weights_to_load)

    model.summary()

    # split dataset.
    random.shuffle(dataset)
    num_samples = len(dataset)
    val_set, train_set = dataset[:num_samples//5], dataset[num_samples//5:]
    # batch generators for model.fit method.
    batch_size = build_params['batch_size']
    seq_length = build_params['seq_length']
    train_batches = BatchGenerator(train_set, batch_size, seq_length)
    val_batches = BatchGenerator(val_set, batch_size, seq_length, shuffle=False)

    # callbacks.
    cp_path = model_dir/'weights-{epoch:03}-{acc:.4f}-{val_acc:.4f}.h5'
    checkpoint = ModelCheckpoint(
        filepath=str(cp_path),
        monitor='val_acc',
        save_best_only=True,
        save_weights_only=True,
        period=1
    )
    logger = CSVLogger(str(
        model_dir/'log.csv'),
        append=(from_epoch != 0)  # only append if resuming training.
    )
    callbacks = [checkpoint, logger]

    # train the model.
    history = model.fit_generator(
        train_batches,
        shuffle=False,  # already handled by BatchGenerator.on_epoch_end method.
        epochs=epochs,
        initial_epoch=from_epoch,
        validation_data=val_batches,
        verbose=1,
        callbacks=callbacks,
        # use_multiprocessing=True
    )

    return history


def grid_search(dataset, model_dir, param_grid, epochs, from_epoch=0,
                weights_to_load=None):
    with (model_dir/'param_grid.json').open('w') as f:
        json.dump(param_grid, f)

    param_names = sorted(param_grid.keys())
    param_comb = it.product(*(param_grid[p] for p in param_names))

    for params in param_comb:
        build_params = dict(zip(param_names, params))
        prms = sorted(build_params.keys()).remove('num_unique_chars')
        prm_str = ','.join(['{}={}'.format(p, build_params[p]) for p in prms])
        grid_dir = model_dir/prm_str
        grid_dir.mkdir(parents=True, exist_ok=True)
        print('\n>>> trying params:', prm_str)
        try:
            train_model(dataset, grid_dir, build_params, epochs, from_epoch,
                        weights_to_load=weights_to_load)
        except Exception as e:
            print('error while trying params', prm_str)
            print(e)
            error_log = grid_dir/'ERROR.log'
            with error_log.open('w') as f:
                f.write('params: {}\n'.format(prm_str))
                f.write(str(e))


def generate_sequence(model_dir, weights_to_load, output_length,
                      initial_seq='X: 1\n', mode='prob'):
    with (model_dir/'token2id.json').open() as f:
        token2id = json.load(f)

    with (model_dir/'build_params.json').open() as f:
        build_params = json.load(f)

    build_params.update(
        batch_size=1,
        seq_length=1,
        rnn_return_sequences=False  # the last RNN layer returns a single value.
    )

    model = build_model(**build_params)
    model.load_weights(model_dir/weights_to_load)

    num_unique_chars = build_params['num_unique_chars']
    seq = [token2id[c] for c in initial_seq]

    for i in range(len(seq) - 1):
        batch = np.asarray([seq[i]])
        model.predict_on_batch(batch)

    for _ in range(output_length):
        batch = np.asarray(seq[-1:])
        preds = model.predict_on_batch(batch).ravel()

        if mode == 'prob':
            sample = np.random.choice(range(num_unique_chars), p=preds)
        elif mode == 'highest':
            sample = np.argmax(preds)

        seq.append(sample)

    id2token = dict(zip(token2id.values(), token2id.keys()))

    return ''.join([id2token[i] for i in seq])
