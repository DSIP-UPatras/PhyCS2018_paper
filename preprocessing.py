import scipy.signal
import numpy as np

FS = 2000
SUBSAMPLE_FACTOR = 20

def _rms(x, win):
	output = np.zeros((x.shape))
	npad = np.floor(win / 2).astype(int)
	win = int(win)
	x_ = np.pad(x,((npad, npad),(0,0)),'symmetric')
	for i in range(output.shape[0]):
		output[i,:] = np.sqrt(np.sum(x_[i:i+win, :]**2, axis=0) / win)
	return output

def _arv(x, win):
	output = np.zeros((x.shape))
	npad = np.floor(win / 2).astype(int)
	win = int(win)
	x_ = np.pad(x,((npad, npad),(0,0)),'symmetric')
	for i in range(output.shape[0]):
		output[i,:] = np.mean(np.abs(x_[i:i+win, :]), axis=0)
	return output

def rms(x, fs=FS):
	win = 0.2*fs
	return _rms(x, win)

def arv(x, fs=FS):
	win = 0.2*fs
	return _arv(x, win)

def lpf(x, f=1., fs=FS):
	f = f/(fs/2)
	x = np.abs(x)
	b,a = scipy.signal.butter(1,f,'low')
	output = scipy.signal.filtfilt(b,a,x,axis=0,padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
	return output

def subsample(x, axis=0, factor=SUBSAMPLE_FACTOR):
	inds = np.arange(0,x.shape[axis],factor)
	return np.take(x, inds, axis)
