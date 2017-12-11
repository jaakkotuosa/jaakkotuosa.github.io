
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.utils import multi_gpu_model
import numpy as np
import random
import sys
import json

text = open('ainamoinen.txt').read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
blank_marker = '\0'
start_marker = '\1'
chars.append(blank_marker)
chars.append(start_marker)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

with open('models/char_indices.json', 'w') as f:
    f.write(json.dumps(char_indices))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 6
sentences = []
next_chars = []
used_fragments = set()
for i in range(0, len(text) - maxlen, step):
    for j in range(1, maxlen - 2, 2):
        fragment = text[i: i + j]
        assert(len(fragment) == j)
        if fragment not in used_fragments:
            used_fragments.add(fragment)
            assert(maxlen - 1 - j > 0)
            sentence = start_marker * (maxlen - 1 - j) + fragment + blank_marker
            assert(len(sentence) == maxlen)
            sentences.append(sentence)
            next_chars.append(text[i + j])
            if len(sentences) < 100:
                print(fragment, '->', text[i + j])

print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


print('Build model...')
model = Sequential()
recurrent_init = 'zeros' # use zeros because keras-js didn't seem to have initializers for kernel state (at least in 0.3.0)
kernel_init = 'glorot_uniform' # can't use zeros here, did not converge, hopefully affect the training only
model.add(LSTM(256, return_sequences=True, input_shape=(maxlen, len(chars)), kernel_initializer=kernel_init, recurrent_initializer=recurrent_init))
model.add(LSTM(256, input_shape=(maxlen, len(chars)), kernel_initializer=kernel_init, recurrent_initializer=recurrent_init))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# Had some issue saving the model with this multi-gpu model, pity.
#model = multi_gpu_model(model, gpus=2)

optimizer = Adam() #RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 1000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x, y,
              batch_size=1024,
              epochs=1)

    model.save('models/model' + str(iteration)+'.h5')
    model.save_weights('models/model' + str(iteration)+'.hdf5')
    with open('models/model' + str(iteration)+'.json', 'w') as f:
        f.write(model.to_json())

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 1.0]:
        print()
        print('----- diversity:', diversity)

        sentence = text[start_index: start_index + 10]
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(sentence)


        for i in range(80):
            x_pred = np.zeros((1, maxlen, len(chars)))

            input = start_marker * (maxlen - len(sentence) - 1) + sentence + blank_marker
            for t, char in enumerate(input):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[-(maxlen-2):] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
