# A semi-supervised variational recurrent neural network
Tensorflow code that implements a semi-supervised variational recurrent neural network.

The model is trained to simultaneously infer the label of the current time step and to predict the future continuous feature observations. This code tests the model on the Human Activity Recognition Using Smartphones Data Set [0].


# Training

Run HAR_ssvrnn_main.py to train the model. See the comments for the different flags.  



# Data
Download the Human Activity Recognition Using Smartphones Data Set: 

https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

Unzip it and place the test and train folder in the UCIHAR folder.




# References
[0] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
