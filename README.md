# GAN-based-Unsupervised-Outlier-Detection
To achieve unsupervised outlier detection, our model consists of two main networks: the GAN network and the autoencoder.
The GAN network fits the data to be detected as real data. After the GAN network is trained, the generated fake data are used as fake "normal objects" to train the autoencoder. When the second stage of the autoencoder training is completed, the data set to be detected is fed into the autoencoder for a forward propagation.
Finally, the reconstruction error of the object is computed at the output layer of the autoencoder; the larger the reconstruction error, the more likely it is an outlier.
When running our project, the path where the fake data is saved in the GAN network needs to be adjusted; also, the appropriate parameters need to be adjusted for different datasets.
In this project, we give the example data, other datasets can be found in ODDS official website.
#ODDS：http://odds.cs.stonybrook.edu/
