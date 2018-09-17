import numpy as np
from sklearn import metrics
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import TensorBoard, Callback
import keras.backend as K
import tensorflow as tf


def evaluate_vote(y_true, y_pred, reps):
    """ Evaluates model based on majority voting of a complete repetition segment.

    Arguments:
        y_true -- array-like, true class labels of EMG images
        y_pred -- array-like, predicted class labels of EMG images
        reps -- array-like, repetition of each sample
        group -- array-like, similar to y_true, but rest labels '0' are assigned to the movement that follows
        vote_window -- integer, size in frames of majority voting window. The increment of voting windows is 1 frame.

    Returns:
        accuracy_vote -- float, accuracy metric for majority voting
        cnf_matrix_vote -- array-like, confusion matrix
    """
    # Vote
    y_true_vote = []
    y_pred_vote = []
    max_label = np.max(y_true)
    assert (y_true.size == reps.size), 'Error'
    for m in np.unique(y_true):
        im = np.isin(y_true, m)
        for r in np.unique(reps):
            ir = np.isin(reps, r)

            # For movement
            irm = np.logical_and(im, ir)
            y_test_rm = y_true[irm].astype(int)
            y_pred_rm = y_pred[irm].astype(int)

            if y_pred_rm.size > 0:
                bins = np.bincount(y_pred_rm, minlength=max_label + 1)
                k = np.argmax(bins)
                y_true_vote.append(m)
                y_pred_vote.append(k)

    # Vote accuracy
    cnf_matrix_vote = metrics.confusion_matrix(
        y_true_vote, y_pred_vote, labels=np.unique(y_true))
    accuracy_vote = metrics.accuracy_score(y_true_vote, y_pred_vote)
    return accuracy_vote, cnf_matrix_vote


def top_1_accuracy(y_true, y_pred):
    """ Calculates top-1 accuracy of the predictions. To be used as evaluation metric in model.compile().

    Arguments:
        y_true -- array-like, true labels
        y_pred -- array-like, predicted labels

    Returns:
        top-1 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=1)


def top_3_accuracy(y_true, y_pred):
    """ Calculates top-3 accuracy of the predictions. To be used as evaluation metric in model.compile().

    Arguments:
        y_true -- array-like, true labels
        y_pred -- array-like, predicted labels

    Returns:
        top-3 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_5_accuracy(y_true, y_pred):
    """ Calculates top-5 accuracy of the predictions. To be used as evaluation metric in model.compile().

    Arguments:
        y_true -- array-like, true labels
        y_pred -- array-like, predicted labels

    Returns:
        top-5 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


class MyTensorboard(TensorBoard):
    """ Tensorboard callback to store the learning rate at the end of each epoch.
    """

    def on_epoch_end(self, epoch, logs=None):
        lr = K.eval(self.model.optimizer.lr)
        lr_summary = tf.Summary(
            value=[tf.Summary.Value(tag='lr', simple_value=lr)])
        self.writer.add_summary(lr_summary, epoch)
        self.writer.flush()
        super(MyTensorboard, self).on_epoch_end(epoch, logs)


class MyLRScheduler(Callback):
    def __init__(self, schedule_type = 'constant', decay = 0, step = 0, lr_start = 0, lr_end = 0, verbose=0):
        super(MyLRScheduler, self).__init__()
        self.schedule_type = schedule_type
        self.decay = float(decay)
        self.step = step
        self.lr_start = float(lr_start)
        self.lr_end = float(lr_end)
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, lr))

    def schedule(self, epoch):
        """ Defines the learning rate schedule. This is called at the begin of each epoch through the LearningRateScheduler callback.
            Arguments:
                epoch -- integer, current epoch, [0, #epochs-1]

            Returns:
                rate -- calculated learning rate
        """
        if self.schedule_type == 'constant':
            rate = self.lr_start
        elif self.schedule_type == 'step':
            rate = self.lr_start * (self.decay ** np.floor(epoch / self.step))
        elif self.schedule_type == 'anneal':
            rate = self.lr_start / (1 + self.decay * epoch)
        elif self.schedule_type == 'clr_triangular':
            e = epoch + self.step
            c = np.floor(1 + e / (2 * self.step))
            x = np.abs(e / self.step - 2 * c + 1)
            rate = self.lr_end + (self.lr_start - self.lr_end) * \
                np.maximum(0, (1 - x)) * float(self.decay**(c - 1))
        elif self.schedule_type == 'clr_restarts':
            c = np.floor(epoch / self.step)
            x = 1 + np.cos((epoch % self.step) / self.step * np.pi)
            rate = self.lr_end + 0.5 * (self.lr_start - self.lr_end) * x * self.decay**c
        return float(rate)


DEFAULT_GENERATOR_PARAMS = {
    "repetitions": [],
    "input_directory": '',
    "batch_size": 128,
    "sample_weight": False,
    "dim": [None,],
    "classes": 5,
    "shuffle": False,
    "noise_snr_db": 0,
    "window_size": 0,
    "window_step": 0,
    "data_type": 'rms',
    "preprocess_function_1": None,
    "size_factor": 0,
    "min_max_norm": True,
    "update_after_epoch": True
}


