import numpy as np
dataset = np.load('dataset/final_dataset001.npz')
X_test = dataset['X_validate']
y_test = dataset['y_validate']
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
model = load_model('models/weights.05-0.03.hdf5')
print(model.summary())
y_pred_test = model.predict(X_test)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
print("Test accuracy:", accuracy_score(y_test, y_pred_test_classes))
