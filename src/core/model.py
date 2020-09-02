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
        batch_size,
        seq_length,
        num_unique_chars,
        embedding_dim,
        rnn_units,
        is_training=True,
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        **kwargs
        ):
    model = Sequential()

    # input layer. RNNs need batch input shape to be (batch_size, seq_length, num_features).
    model.add(Embedding(
        batch_input_shape=(batch_size, seq_length),
        input_dim=num_unique_chars,
        output_dim=embedding_dim,
    ))

    for units in rnn_units[:-1]:
        model.add(RNNLayer(
            units=units,
            return_sequences=True,  # True when stacking RNN layers so shapes match.
            stateful=True,
            **kwargs
        ))
    # last RNN layer may have a different return_sequences value.
    model.add(RNNLayer(
        units=rnn_units[-1],
        return_sequences=is_training,
        stateful=True,
        **kwargs
    ))

    if is_training:
        model.add(TimeDistributed(Dense(
            num_unique_chars,
            activation='softmax',
        )))
    else:
        model.add(Dense(
            num_unique_chars,
            activation='softmax',
        ))

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    return model


class BatchGenerator(Sequence):

    def __init__(self, dataset, batch_size, seq_length, shuffle=True):
        if shuffle:
            random.shuffle(dataset)

        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.shuffle = shuffle

        self.gen_sequences()
        # num of batches the generator has.
        self.len = len(self.x) // batch_size
        # this is used to ensure all batches have same length (no extra item in first batches). check `__getitem__`.
        self.mod = len(self.x) % batch_size

    def gen_sequences(self):
        xs, ys = zip(*self.dataset)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        # x.shape == (len_of_all_abcs, )
        # y.shape == (len_of_all_abcs, num_unique_chars)
        sequences = (
            (x[offset:offset + self.seq_length],
             y[offset:offset + self.seq_length, :])
            for offset in range(0, x.shape[0] - self.seq_length, self.seq_length)
        )
        self.x, self.y = map(np.asarray, zip(*sequences))
        # self.x.shape == (len_of_all_abcs // seq_length, seq_length)
        # self.y.shape == (len_of_all_abcs // seq_length, seq_length, num_unique_chars)

    # methods required by keras.utils.Sequence

    def __len__(self):
        return self.len

    def __getitem__(self, batch_index):
        # generate batches compatible with the `stateful=True` parameter of RNN.
        i = batch_index % self.len
        return \
            self.x[np.arange(i, self.x.shape[0] - self.mod, self.len), :],\
            self.y[np.arange(i, self.x.shape[0] - self.mod, self.len), :, :]

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.dataset)
            self.gen_sequences()


def train_model(
        dataset,
        model_dir,
        model_params,
        epochs,
        initial_epoch=0,
        weights_to_load=None,
        ):
    # build model.
    with (model_dir/'model_params.json').open('w') as f:
        json.dump(model_params, f)

    model = build_model(**model_params)
    if weights_to_load:
        model.load_weights(model_dir/weights_to_load)

    model.summary()

    # split dataset.
    bound = int(0.2 * len(dataset))
    random.shuffle(dataset)
    val_set, train_set = dataset[:bound], dataset[bound:]
    # batch generators for model training.
    batch_size = model_params['batch_size']
    seq_length = model_params['seq_length']
    train_batches = BatchGenerator(train_set, batch_size, seq_length)
    val_batches = BatchGenerator(val_set, batch_size, seq_length, shuffle=False)

    # callbacks.
    checkpoint = ModelCheckpoint(
        filepath=str(model_dir/'weights-{epoch:03}-{acc:.4f}-{val_acc:.4f}.h5'),
        monitor='val_acc',
        save_best_only=True,
        save_weights_only=True,
        period=1
    )
    logger = CSVLogger(
        filename=str(model_dir/'log.csv'),
        append=(initial_epoch != 0)  # only append if resuming training.
    )
    callbacks = [checkpoint, logger]

    # train the model.
    history = model.fit_generator(
        train_batches,
        shuffle=False,  # already handled by `BatchGenerator.on_epoch_end` method.
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=val_batches,
        callbacks=callbacks,
        verbose=1,
        # use_multiprocessing=True
    )

    return history


def grid_search(
        dataset,
        model_dir,
        param_grid,
        epochs,
        initial_epoch=0,
        weights_to_load=None,
        ):
    with (model_dir/'param_grid.json').open('w') as f:
        json.dump(param_grid, f)

    param_comb = it.product(*param_grid.values())
    for param_values in param_comb:
        build_params = dict(zip(param_grid.keys(), param_values))
        param_str = ','.join(f'{k}={v}' for k, v in build_params.items() if k != 'num_unique_chars')
        grid_dir = model_dir/param_str
        grid_dir.mkdir(parents=True, exist_ok=True)
        print('\n>>> trying params:', param_str)
        try:
            train_model(
                dataset,
                grid_dir,
                build_params,
                epochs,
                initial_epoch,
                weights_to_load,
            )
        except Exception as e:
            print('error while trying params', param_str)
            print(e)
            error_log = grid_dir/'ERROR.log'
            with error_log.open('w') as f:
                f.write(f'params: {param_str}\n')
                f.write(str(e))


def generate_sequence(
        model_dir,
        weights_to_load,
        output_length,
        initial_seq='X: 1\n',
        mode='prob',
        ):
    # build model.
    with (model_dir/'model_params.json').open() as f:
        model_params = json.load(f)

    model_params.update(
        batch_size=1,
        seq_length=1,
        is_training=False  # the last RNN layer returns a single value.
    )

    model = build_model(**model_params)
    model.load_weights(model_dir/weights_to_load)

    # warm up model processing `initial_seq`.
    with (model_dir/'token2id.json').open() as f:
        token2id = json.load(f)

    seq = [token2id[c] for c in initial_seq]
    for i in range(len(seq) - 1):
        batch = np.asarray([seq[i]])
        model.predict_on_batch(batch)

    # generate sequence.
    num_unique_chars = model_params['num_unique_chars']
    for _ in range(output_length):
        batch = np.asarray([seq[-1]])
        predictions = model.predict_on_batch(batch).ravel()

        if mode == 'prob':
            sample = np.random.choice(range(num_unique_chars), p=predictions)
        elif mode == 'highest':
            sample = np.argmax(predictions)

        seq.append(sample)

    id2token = dict(zip(token2id.values(), token2id.keys()))
    result_seq = ''.join([id2token[i] for i in seq])

    return result_seq
