import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

metrics = np.load('Metrics/metrics_4.npz', allow_pickle=True)['history'].item()
train_loss = metrics['train_loss']
val_loss = metrics['val_loss']

#plt.plot(train_loss)
#plt.plot(val_loss)
#plt.show()
