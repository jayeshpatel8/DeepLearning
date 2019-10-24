# DeepLearning : RF Signal Classification (modulation)
                  Classify the modulated RF Signal, DNN trained using I/Q data
                  
## Objective 

### To get first hands-on experience of :
	How Deep Neural Network/Deep Learning works in practice
	Working with Environment/setup: Anaconda, Google collaborator, Docker
	Working with Keras, Tensorflow, Theano, numpy, matplot
	Migrating from Python 2.x to Python 3.x
	Generating and using Dataset
	How RF I/Q data based Deep Learning DNN/CNN Model trains 
	Compare Model Accuracy Vs Hyper parameter Vs dataset size Vs epoch 
	CPU vs GPU learning rate difference

## Software Dependencies
    ENVIRONMENT: 
      	Anaconda, Google Collaborator, Docker
    DL:
	      Keras, Tensorflow, Theano 
    SDR: 
	      GNU Radio 
	      out-of-tree (OOT)  gr-Module

## Dataset
    Synthetic
      Tiny [20,000 I/Q sample] Modulation[2] SNR [-20 to 18 dB] 
	      Generated locally by limiting the modulation class to only two i.e. ( CPFSK & GFSK ). 10,000 samples/modulation
      Big [110,000 I/Q sample] Modulation[11] SNR [-20 to 18 dB] 
	      Downloaded from deepsig-datasets-RML2016.10a.tar.bz2
    Real
  	  Downloaded from Featurized_RF_Signal_Classification Dataset *
## Modification made in Template
    To make it compatible with Python 3.x
      Map and cPickle adaptation
    Run using Tensorflow as backend
      To Use GPU acceleration
    Run using Theano as backend
    Run on google collaboration platform
    Data set correct loading  
      encoding=latin1
![](https://github.com/jayeshpatel8/DeepLearning/blob/master/RF/Image/Slide6.JPG)
![](https://github.com/jayeshpatel8/DeepLearning/blob/master/RF/Image/Slide7.JPG)
![](https://github.com/jayeshpatel8/DeepLearning/blob/master/RF/Image/Slide8.JPG)
## References
    Deepsig - Over the Air Deep Learning Based Radio Signal Classification
    RWTCH Aachen - iNETS_RFSig_v1_documentation.pdf
    git hub - radioML
    Gnuradio
    gtc-deep-learning-applications-for-radio-frequency-rf-data
    Training - Oreilly Deep Learning with Keras,Tensorflow 
