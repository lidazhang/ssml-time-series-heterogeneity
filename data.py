import numpy as np
import os
import threading
import random

import utils
import reader
import preprocessing
from utils import get_bin_custom, get_bin_log


def load_all_data(data_path,batch_size,nbatches,mode='train'):
    data_mode = mode
    if mode =='val':
        data_mode = 'train'
    data_reader = reader.DecompensationReader(dataset_dir=os.path.join(data_path, data_mode),
                                      listfile=os.path.join(data_path, mode+'_listfile.csv'))
    # val_reader = reader.LengthOfStayReader(dataset_dir=os.path.join(data_path, 'train'),
    #                                 listfile=os.path.join(data_path, 'val_listfile.csv'))
    discretizer = preprocessing.Discretizer(timestep=1.0, #args.timestep
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')
    discretizer_header = discretizer.transform(data_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
    normalizer = preprocessing.Normalizer(fields=cont_channels)
    normalizer_state = 'decomp_ts{}.input_str:previous.n1e5.start_time:zero.normalizer'.format(1.0)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)
    loss_function = 'sparse_categorical_crossentropy'

    data_data_gen = BatchGen(reader=data_reader,
                                    discretizer=discretizer,
                                    normalizer=normalizer,
                                    batch_size=batch_size,
                                    steps=nbatches,
                                    shuffle=True)
    # if mode == 'train':
    #     data = data_data_gen._generator()
    # else:
    data = data_data_gen._generator()
    return data #np.expand_dims(ys, -1)

class BatchGen(object):

    def __init__(self, reader, discretizer, normalizer,
                 batch_size, steps, shuffle, return_names=False):
        self.reader = reader
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_names = return_names

        if steps is None:
            self.n_examples = reader.get_number_of_examples()
            self.steps = (self.n_examples + batch_size - 1) // batch_size
        else:
            self.n_examples = steps * batch_size
            self.steps = steps

        self.chunk_size = min(1024*10, self.steps) * batch_size
        self.lock = threading.Lock()
        #self.generator = self._generator()


    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                self.reader.random_shuffle()
            remaining = self.n_examples
            while remaining > 0:
                current_size = min(self.chunk_size, remaining)
                # remaining -= current_size

                current_size = min(self.chunk_size, remaining)

                ret = utils.read_chunk(self.reader, current_size)
                Xs = ret["X"]
                ts = ret["t"]
                ys = ret["y"]
                names = ret["name"]
                Xs = utils.preprocess_chunk(Xs, ts, self.discretizer, self.normalizer)
                (Xs, ys, ts, names) = utils.sort_and_shuffle([Xs, ys, ts, names], B)
                # Xs = utils.pad_zeros(Xs)
                # ys = np.array(ys)

                #return Xs, ys
                data_all = []
                data_aug = []
                zeros = np.zeros((24))
                for i in range(len(Xs)):
                    X = Xs[i]
                    X = list(np.reshape(X[-(len(X)//24)*24:], (-1, 24, 76)))
                    y = ys[i]
                    y_true = np.array(y)
                    batch_ts = ts[i]

                    for xx in X[:-1]:
                        idx = random.randint(0,75)
                        x_new = np.copy(xx)
                        x_new[:,idx] = zeros
                        data_aug.append(x_new)

                    y = np.array(y)
                    #print(np.shape(X),np.shape(y),np.shape(y_true))
                    data_all.append((X,y,y_true))

                return data_all, data_aug


    def _generator_val(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                self.reader.random_shuffle()
            remaining = self.n_examples
            while remaining > 0:
                current_size = min(self.chunk_size, remaining)
                # remaining -= current_size

                current_size = min(self.chunk_size, remaining)

                ret = utils.read_chunk(self.reader, current_size)
                Xs = ret["X"]
                ts = ret["t"]
                ys = ret["y"]
                names = ret["name"]
                Xs = utils.preprocess_chunk(Xs, ts, self.discretizer, self.normalizer)
                (Xs, ys, ts, names) = utils.sort_and_shuffle([Xs, ys, ts, names], B)
                # Xs = utils.pad_zeros(Xs)
                # ys = np.array(ys)

                #return Xs, ys
                data_all = []
                for i in range(len(Xs)):
                    X = Xs[i]
                    X = list(np.reshape(X[-(len(X)//24)*24:], (-1, 24, 76)))
                    y = ys[i]
                    y_true = np.array(y)
                    batch_ts = ts[i]

                    y = np.array(y)
                    #print(np.shape(X),np.shape(y),np.shape(y_true))
                    data_all.append((X,y,y_true))

                return data_all, []



    def __iter__(self):
        return self.generator

    def next(self, return_y_true=False):
        with self.lock:
            self.return_y_true = return_y_true
            return next(self.generator)

    def __next__(self):
        return self.next()
