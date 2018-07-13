# Traffic-Sign-Classification
Traffic Sign Recognition is one of the most important part in creation of a complete self driving system. Traffic Sign Classification is a multi class classification problem. It is a challange to create a realtime traffic sign classification system.<br>
In this project we are creating a classifier using **Convolutional Neural Network** to classify **43 different traffic signs**.
## Dataset
A dataset of near about 40,000 images of 43 different classes is obtained from *[German Traffic Sign Benchmark*](http://benchmark.ini.rub.de/) website.<br>
[Download Dataset from Here](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip) 263MB<br>
Dataset consists of images in ppm format in 43 different folders of different class.<br>

****[DataPreparation.py](https://github.com/pranjulsingh/Traffic-Sign-Classification/blob/master/DataPreparation.py):****
   * Reads images from complete dataset.
   * Converts images to jpg format.
   * Divides data in three different folders Teain, Validate and Test.<br>

****[DatasetCreation.py](https://github.com/pranjulsingh/Traffic-Sign-Classification/blob/master/DatasetCreation.py):****
   * Reads data from Train, Validate and Test folders.
   * Splits in X_train, y_train, X_validate, y_validate, X_test, y_test
   * Saves the numpy array in a file
   [Download processed numpy file](www.google.com)<br>

****[Model.py](https://github.com/pranjulsingh/Traffic-Sign-Classification/blob/master/Model.py):****
   * A Convolutional Neural Network Model
   * 4 Convolution Layer
   * 2 Max Pooling Layer
   * Activaion Function: LeakyRelu
   * 3 Dense Layers with 512, 256 and 43 neurons respectively
   * Softmax as final activation function.

![alt text](https://github.com/pranjulsingh/Traffic-Sign-Classification/blob/master/graph_large.png).

****[Train.py](https://github.com/pranjulsingh/Traffic-Sign-Classification/blob/master/Train.py)****
   * Batch Size 32
   * Epoch 10
   * Decreasing learning rate
   * Saves Logs in *logs/* directory
   * Saves model in *models/* directory
   * [Download model with the weight of 6<sup>th</sup> epoch](http://www.google.com)

****[Predict.py](https://github.com/pranjulsingh/Traffic-Sign-Classification/blob/master/Predict.py):****
   * Validation accuracy 99.15
   * Test accuracy 99.05
