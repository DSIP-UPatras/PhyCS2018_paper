import numpy as np

def jitter(x, snr_db=25):
	if isinstance(snr_db, list):
		snr_db_low = snr_db[0]
		snr_db_up = snr_db[1]
	else:
		snr_db_low = snr_db
		snr_db_up = 45
	snr_db = np.random.randint(snr_db_low, snr_db_up, (1,))[0]
	snr = 10 ** (snr_db/10)
	Xp = np.sum(x**2, axis=0, keepdims=True) / x.shape[0]
	Np = Xp / snr
	n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)
	xn = x + n
	return xn

