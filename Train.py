from Model import make_model
import numpy as np
dataset = np.load('dataset/final_dataset001.npz')
print(dataset.files)
X_train = dataset['X_train']
y_train = dataset['y_train']
X_test = dataset['X_test']
y_test = dataset['y_test']
X_validate = dataset['X_validate']
y_validate = dataset['y_validate']
print("Training input shape: ",X_train.shape)
print("Test input shape: ",X_test.shape)
print("Validation input shape: ",X_validate.shape)
print("Training target shape: ",y_train.shape)
print("Test target shape: ",y_test.shape)
print("Validation targer shape: ",y_validate.shape)
import matplotlib.pyplot as plt
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_train))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(X_train[random_index, :])
        ax.set_title(y_train[random_index])
plt.show()
from keras.utils import to_categorical
import keras
from keras import backend as K
y_train2 = to_categorical(y_train)
y_test2 = to_categorical(y_test)
y_validate2 = to_categorical(y_validate)
INIT_LR = 1e-2
BATCH_SIZE = 32
EPOCHS = 10
model = make_model()
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.adamax(lr=INIT_LR),
    metrics=['accuracy']
)
def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch
class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model.optimizer.lr))

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from time import time
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
save_model = ModelCheckpoint('models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                             monitor='val_loss', verbose=0,
                             save_best_only=False, save_weights_only=False,
                             mode='auto', period=1)
model.fit(
    X_train, y_train2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler), 
               LrHistory(),tensorboard,save_model],
    validation_data=(X_validate, y_validate2),
    shuffle=True
)
