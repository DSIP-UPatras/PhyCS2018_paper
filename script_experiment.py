
import numpy as np
import tensorflow as tf
import random

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(1234)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
random.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see:
#    https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
#   https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

##############################################################################
import sys
import matplotlib.pyplot as plt
from keras import optimizers, initializers, regularizers, constraints
from keras.utils import plot_model
from utils import *
from generator import *
from models import *
import preprocessing
import json
import re
import datetime
from sklearn import metrics

# 1. Logging
TIMESTAMP = '{}'.format(
    re.sub('[^A-Za-a0-9]+', '', '{}'.format(datetime.datetime.now())))
CONFIG_FILE = str(sys.argv[1])
INTER_SUBJECT = bool(int(sys.argv[2]))
with open(CONFIG_FILE) as json_file:
    config_data = json.load(json_file)

LOGGING_ENABLE = config_data['logging']['enable']
if LOGGING_ENABLE:
    LOGGING_FILE = os.path.abspath(
        'logs') + os.sep + TIMESTAMP + '_' + config_data['logging']['log_file'] + '.log'
    LOGGING_TESNORBOARD_FILE = os.path.abspath(
        'logs/tblogs') + os.sep + TIMESTAMP + '_' + config_data['logging']['log_file']

MODEL_SAVE_ENABLE = config_data['model']['save']
if MODEL_SAVE_ENABLE:
    MODEL_SAVE_FILE = os.path.abspath(
        'models') + os.sep + TIMESTAMP + '_' + config_data['model']['save_file'] + '_{}.json'
    MODEL_WEIGHTS_SAVE_FILE = os.path.abspath(
        'models') + os.sep + TIMESTAMP + '_' + config_data['model']['save_file'] + '_{}.h5'

METRICS_SAVE_FILE = os.path.abspath(
    'metrics') + os.sep + TIMESTAMP + '_' + config_data['logging']['log_file'] + '.mat'

# 2. Config params
PARAMS_TRAINING = config_data['training']
PARAMS_MODEL = config_data['model']
PARAMS_DATASET = config_data['dataset']

PARAMS_TRAIN_GENERATOR = DEFAULT_GENERATOR_PARAMS.copy()
params_gen = PARAMS_DATASET.get('train_generator', {}).copy()
for key in params_gen.keys():
    PARAMS_TRAIN_GENERATOR[key] = params_gen[key]

PARAMS_VALID_GENERATOR = DEFAULT_GENERATOR_PARAMS.copy()
params_gen = PARAMS_DATASET.get('valid_generator', {}).copy()
for key in params_gen.keys():
    PARAMS_VALID_GENERATOR[key] = params_gen[key]

PARAMS_TEST_GENERATOR = DEFAULT_GENERATOR_PARAMS.copy()
params_gen = PARAMS_DATASET.get('test_generator', {}).copy()
for key in params_gen.keys():
    PARAMS_TEST_GENERATOR[key] = params_gen[key]

SUBJECTS = config_data.get('subjects', [i for i in range(1, 28)])

# 3. Initialization
if PARAMS_DATASET['name'] == 'DB1':
    input_directory = '/home/etro/ptsinganos/emgdl/Datasets/Ninapro-DB1-Final'
    PARAMS_TRAIN_GENERATOR['preprocess_function_1'] = [preprocessing.lpf]
    PARAMS_TRAIN_GENERATOR['preprocess_function_1_extra'] = [{'fs':100}]
    PARAMS_TRAIN_GENERATOR['data_type'] = 'rms'
    PARAMS_TRAIN_GENERATOR['classes'] = [i for i in range(53)]

    PARAMS_VALID_GENERATOR['preprocess_function_1'] = [preprocessing.lpf]
    PARAMS_VALID_GENERATOR['preprocess_function_1_extra'] = [{'fs':100}]
    PARAMS_VALID_GENERATOR['data_type'] = 'rms'
    PARAMS_VALID_GENERATOR['classes'] = [i for i in range(53)]

    PARAMS_TEST_GENERATOR['preprocess_function_1'] = [preprocessing.lpf]
    PARAMS_TEST_GENERATOR['preprocess_function_1_extra'] = [{'fs':100}]
    PARAMS_TEST_GENERATOR['data_type'] = 'rms'
    PARAMS_TEST_GENERATOR['classes'] = [i for i in range(53)]

elif PARAMS_DATASET['name'] == 'DB2':
    input_directory = '/home/etro/ptsinganos/emgdl/Datasets/Ninapro-DB2-Final-1_1'
    PARAMS_TRAIN_GENERATOR['preprocess_function_1'] = [preprocessing.rms, preprocessing.lpf, preprocessing.subsample]
    PARAMS_TRAIN_GENERATOR['preprocess_function_1_extra'] = [{'fs':2000}, {'fs':2000}, {'factor':20}]
    PARAMS_TRAIN_GENERATOR['data_type'] = 'raw'
    PARAMS_TRAIN_GENERATOR['classes'] = [i for i in range(41)]

    PARAMS_VALID_GENERATOR['preprocess_function_1'] = [preprocessing.rms, preprocessing.lpf, preprocessing.subsample]
    PARAMS_VALID_GENERATOR['preprocess_function_1_extra'] = [{'fs':2000}, {'fs':2000}, {'factor':20}]
    PARAMS_VALID_GENERATOR['data_type'] = 'raw'
    PARAMS_VALID_GENERATOR['classes'] = [i for i in range(41)]
    
    PARAMS_TEST_GENERATOR['preprocess_function_1'] = [preprocessing.rms, preprocessing.lpf, preprocessing.subsample]
    PARAMS_TEST_GENERATOR['preprocess_function_1_extra'] = [{'fs':2000}, {'fs':2000}, {'factor':20}]
    PARAMS_TEST_GENERATOR['data_type'] = 'raw'
    PARAMS_TEST_GENERATOR['classes'] = [i for i in range(41)]

PARAMS_TRAIN_GENERATOR.pop('input_directory', '')
PARAMS_VALID_GENERATOR.pop('input_directory', '')
PARAMS_TEST_GENERATOR.pop('input_directory', '')

MODEL = AtzoriNet

PARAMS_TRAIN_GENERATOR["label_proc"] = smooth_labels_with_dist
#PARAMS_TRAIN_GENERATOR["label_proc_extra"] = {"smooth_factor": 0.1}
PARAMS_TRAIN_GENERATOR["label_proc_extra"] = {"smooth_factor": 0.3, "smooth_dist": group_dist}

mean_train, mean_test, mean_test_vote, mean_test_3, mean_test_5 = [], [], [], [], []
mean_cm, mean_cm_vote = [], []
mean_train_loss, mean_test_loss = [], []

if LOGGING_ENABLE:
    with open(LOGGING_FILE, 'w') as f:
        f.write(
            'TIMESTAMP: {}\n'
            'KERAS: {}\n'
            'TENSORFLOW: {}\n'
            'DATASET: {}\n'
            'TRAIN_GENERATOR: {}\n'
            'VALID_GENERATOR: {}\n'
            'TEST_GENERATOR: {}\n'
            'MODEL: {}\n'
            'MODEL_PARAMS: {}\n'
            'TRAIN_PARAMS: {}\n'.format(
                TIMESTAMP, keras.__version__, tf.__version__, PARAMS_DATASET['name'], PARAMS_TRAIN_GENERATOR,
                PARAMS_VALID_GENERATOR, PARAMS_TEST_GENERATOR,
                PARAMS_MODEL['name'], PARAMS_MODEL['extra'],
                PARAMS_TRAINING)
        )
        f.write(
            'SUBJECT,TRAIN_SHAPE,TEST_SHAPE,TRAIN_LOSS,TRAIN_ACC,TEST_LOSS,TEST_ACC,TEST_ACC_VOTE,TEST_TOP_3_ACC,TEST_TOP_5_ACC\n')

for subject in SUBJECTS:
    print('Subject: {}'.format(subject))
    input_dir = '{}/subject-{:02d}'.format(input_directory, subject)


    if INTER_SUBJECT:
        train_generator = DataGenerator(
            input_directory=['{}/subject-{:02d}'.format(input_directory, s) for s in SUBJECTS if s != subject],
            **PARAMS_TRAIN_GENERATOR)
    else:
        train_generator = DataGenerator(input_directory=input_dir, **PARAMS_TRAIN_GENERATOR)
    valid_generator = DataGenerator(input_directory=input_dir, **PARAMS_VALID_GENERATOR)
    X_test, Y_test, test_reps = valid_generator.get_data()

    y_test = np.argmax(Y_test, axis=1)

    model = MODEL(
        input_shape=train_generator.dim,
        classes=train_generator.n_classes,
        **PARAMS_MODEL['extra'])
    model.summary()

    if PARAMS_TRAINING['optimizer'] == 'adam':
        optimizer = optimizers.Adam(
            lr=PARAMS_TRAINING['l_rate'], epsilon=0.001)
    elif PARAMS_TRAINING['optimizer'] == 'sgd':
        optimizer = optimizers.SGD(lr=PARAMS_TRAINING['l_rate'], momentum=0.9)
        
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[
                  'accuracy', top_3_accuracy, top_5_accuracy])

    train_callbacks = []
    if LOGGING_ENABLE:
        tensorboardCallback = MyTensorboard(log_dir=LOGGING_TESNORBOARD_FILE + "/{}".format(subject),
                                            batch_size=100,
                                            histogram_freq=5)
        train_callbacks.append(tensorboardCallback)
    lrScheduler = MyLRScheduler(**PARAMS_TRAINING['l_rate_schedule'])
    train_callbacks.append(lrScheduler)

    history = model.fit_generator(train_generator, epochs=PARAMS_TRAINING['epochs'],
                                  validation_data=(X_test, Y_test), callbacks=train_callbacks)
    Y_pred = model.predict(X_test)

    y_pred = np.argmax(Y_pred, axis=1)

    if MODEL_SAVE_ENABLE:
        # serialize model to JSON
        model_json = model.to_json()
        with open(MODEL_SAVE_FILE.format(subject), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(MODEL_WEIGHTS_SAVE_FILE.format(subject))
        print("Saved model to disk")

    # CM
    # C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j.
    cnf_matrix_frame = metrics.confusion_matrix(y_test, y_pred)
    if np.array(mean_cm).shape != cnf_matrix_frame.shape:
        mean_cm = cnf_matrix_frame
    else:
        mean_cm += cnf_matrix_frame
    # Vote
    accuracy_vote, cnf_matrix_vote = evaluate_vote(y_test, y_pred, test_reps)
    mean_test_vote.append(accuracy_vote)
    if np.array(mean_cm_vote).shape != cnf_matrix_vote.shape:
        mean_cm_vote = cnf_matrix_vote
    else:
        mean_cm_vote += cnf_matrix_vote

    mean_train.append(np.mean(history.history['acc'][-5:]))
    mean_test.append(np.mean(history.history['val_acc'][-5:]))
    mean_train_loss.append(np.mean(history.history['loss'][-5:]))
    mean_test_loss.append(np.mean(history.history['val_loss'][-5:]))
    mean_test_3.append(np.mean(history.history['val_top_3_accuracy'][-5:]))
    mean_test_5.append(np.mean(history.history['val_top_5_accuracy'][-5:]))
    K.clear_session()

    if LOGGING_ENABLE:
        with open(LOGGING_FILE, 'a') as f:
            f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(subject, train_generator.__len__() * PARAMS_TRAIN_GENERATOR['batch_size'], valid_generator.__len__(),
                                                             mean_train_loss[-1], mean_train[-1], mean_test_loss[-1], mean_test[-1], mean_test_vote[-1], mean_test_3[-1], mean_test_5[-1]))

if LOGGING_ENABLE:
    with open(LOGGING_FILE, 'a') as f:
        f.write('Train loss: {} +- {}\n'.format(np.mean(mean_train_loss),
                                                np.std(mean_train_loss)))
        f.write(
            'Train accuracy: {} +- {}\n'.format(np.mean(mean_train), np.std(mean_train)))
        f.write('Test loss: {} +- {}\n'.format(np.mean(mean_test_loss),
                                               np.std(mean_test_loss)))
        f.write(
            'Test accuracy: {} +- {}\n'.format(np.mean(mean_test), np.std(mean_test)))
        f.write('Vote accuracy: {} +- {}\n'.format(np.mean(mean_test_vote),
                                                   np.std(mean_test_vote)))
        f.write(
            'Top-3 accuracy: {} +- {}\n'.format(np.mean(mean_test_3), np.std(mean_test_3)))
        f.write(
            'Top-5 accuracy: {} +- {}\n'.format(np.mean(mean_test_5), np.std(mean_test_5)))
print('Train accuracy: {} +- {}\n'.format(np.mean(mean_train), np.std(mean_train)))
print('Test accuracy: {} +- {}\n'.format(np.mean(mean_test), np.std(mean_test)))
print('Vote accuracy: {} +- {}\n'.format(np.mean(mean_test_vote), np.std(mean_test_vote)))
print('Top-3 accuracy: {} +- {}\n'.format(np.mean(mean_test_3), np.std(mean_test_3)))
print('Top-5 accuracy: {} +- {}\n'.format(np.mean(mean_test_5), np.std(mean_test_5)))

metrics_dict = {
    'mean_cm': mean_cm,
    'mean_cm_vote': mean_cm_vote,
    'mean_test': mean_test,
    'mean_test_3': mean_test_3,
    'mean_test_5': mean_test_5,
    'mean_test_vote': mean_test_vote,
    'mean_train': mean_train,
    'mean_train_loss': mean_train_loss,
    'mean_test_loss': mean_test_loss
}
scipy.io.savemat(METRICS_SAVE_FILE, metrics_dict)
