### This is a classification example with normal dataset of 110,000 sample of synthetic I/Q using Threano as backend of Keras
```python

# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
%matplotlib inline
import os,random
os.environ["KERAS_BACKEND"] = "theano"
os.environ["KERAS_IMAGE_DATA_FORMAT"] = "channels_first"
os.environ["KERAS_IMAGE_DIM_ORDERING"] = "th"
#os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(1)
import numpy as np
#import theano as th
#import theano.tensor as T
#from keras.utils import np_utils
import keras.models as models
from keras import Sequential
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import _pickle as cPickle
import random, sys, keras, bz2, tarfile
```

    Using Theano backend.
    


```python
if ("channels_first" != keras.backend.image_data_format()):
    print("ERROR: [%s] set it for Keras in  ~/.keras/keras.json file % keras.backend.image_data_format()")
```


```python
# Load the dataset ...
#  You will need to seperately download or generate this file

#Xd = cPickle.load(open("RML2014.04c_dict_cpfsk_gfsk.dat","rb") )
Xd = cPickle.load(open("..\Dataset\RML2016.10a_dict_unix.pkl","rb"),encoding="latin1" )
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
print("Modulations:  %s, " % mods)
for mod in mods:
#    print("\t\t %s" %  mod)
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
```

    Modulations: 
    		 8PSK
    		 AM-DSB
    		 AM-SSB
    		 BPSK
    		 CPFSK
    		 GFSK
    		 PAM4
    		 QAM16
    		 QAM64
    		 QPSK
    		 WBFM
    


```python

# Partition the data
#  into training and test sets of the form we can train/test on 
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
```


```python
in_shp = list(X_train.shape[1:])
print (X_train.shape, in_shp)
classes = mods
```

    (110000, 2, 128) [2, 128]
    


```python
# Build VT-CNN2 Neural Net model using Keras primitives -- 
#  - Reshape [N,2,128] to [N,1,2,128] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization

dr = 0.5 # dropout rate (%)
model = models.Sequential()
#model = Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))
model.add(ZeroPadding2D((0, 2)))
#model.add(Convolution2D(256, 1, 3, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform'))
model.add(Convolution2D(256, (1, 3), activation="relu", name="conv1", kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
#model.add(Convolution2D(80, 2, 3, border_mode="valid", activation="relu", name="conv2", init='glorot_uniform'))
model.add(Convolution2D(80, (2, 3), activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', name="dense1", kernel_initializer='he_normal'))
model.add(Dropout(dr))
model.add(Dense( len(classes) , name="dense2", kernel_initializer='he_normal' ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))

#new_model = Sequential()
#new_model.add(model)
#new_model.add(Activation('softmax'))
#new_model.add(Reshape([len(classes)]))

#new_model.compile(loss='categorical_crossentropy', optimizer='adam')
#new_model.summary()

#model.compile(loss='categorical_crossentropy', optimizer='adam')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    reshape_1 (Reshape)          (None, 1, 2, 128)         0         
    _________________________________________________________________
    zero_padding2d_1 (ZeroPaddin (None, 1, 2, 132)         0         
    _________________________________________________________________
    conv1 (Conv2D)               (None, 256, 2, 130)       1024      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 256, 2, 130)       0         
    _________________________________________________________________
    zero_padding2d_2 (ZeroPaddin (None, 256, 2, 134)       0         
    _________________________________________________________________
    conv2 (Conv2D)               (None, 80, 1, 132)        122960    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 80, 1, 132)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 10560)             0         
    _________________________________________________________________
    dense1 (Dense)               (None, 256)               2703616   
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense2 (Dense)               (None, 11)                2827      
    _________________________________________________________________
    activation_1 (Activation)    (None, 11)                0         
    _________________________________________________________________
    reshape_2 (Reshape)          (None, 11)                0         
    =================================================================
    Total params: 2,830,427
    Trainable params: 2,830,427
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Set up some params 
nb_epoch = 50    # number of epochs to train on
batch_size = 1024  # training batch size
```


```python
# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
#    show_accuracy=False,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)
```

    ipykernel_launcher.py:13: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      del sys.path[0]
    

    Train on 110000 samples, validate on 110000 samples
    Epoch 1/50
     - 1642s - loss: 2.2543 - acc: 0.1463 - val_loss: 2.1239 - val_acc: 0.1980
    Epoch 2/50
     - 1392s - loss: 2.0338 - acc: 0.2518 - val_loss: 1.9203 - val_acc: 0.2836
    Epoch 3/50
     - 1419s - loss: 1.9002 - acc: 0.2896 - val_loss: 1.7992 - val_acc: 0.3251
    Epoch 4/50
     - 1382s - loss: 1.7970 - acc: 0.3280 - val_loss: 1.6711 - val_acc: 0.3959
    Epoch 5/50
     - 1533s - loss: 1.6855 - acc: 0.3711 - val_loss: 1.5733 - val_acc: 0.4189
    Epoch 6/50
     - 1506s - loss: 1.6183 - acc: 0.3925 - val_loss: 1.5076 - val_acc: 0.4399
    Epoch 7/50
     - 1536s - loss: 1.5720 - acc: 0.4080 - val_loss: 1.4658 - val_acc: 0.4460
    Epoch 8/50
     - 1567s - loss: 1.5346 - acc: 0.4189 - val_loss: 1.4285 - val_acc: 0.4640
    Epoch 9/50
     - 1545s - loss: 1.4999 - acc: 0.4324 - val_loss: 1.4066 - val_acc: 0.4700
    Epoch 10/50
     - 1523s - loss: 1.4733 - acc: 0.4386 - val_loss: 1.3854 - val_acc: 0.4729
    Epoch 11/50
     - 1539s - loss: 1.4551 - acc: 0.4445 - val_loss: 1.3522 - val_acc: 0.4847
    Epoch 12/50
     - 1462s - loss: 1.4347 - acc: 0.4507 - val_loss: 1.3372 - val_acc: 0.4867
    Epoch 13/50
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-8-99b3eb9b8abc> in <module>
         11     callbacks = [
         12         keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
    ---> 13         keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
         14     ])
         15 # we re-load the best weights once training is finished
    

    ~\AppData\Local\Continuum\anaconda3\envs\gr_p3.6\lib\site-packages\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
       1176                                         steps_per_epoch=steps_per_epoch,
       1177                                         validation_steps=validation_steps,
    -> 1178                                         validation_freq=validation_freq)
       1179 
       1180     def evaluate(self,
    

    ~\AppData\Local\Continuum\anaconda3\envs\gr_p3.6\lib\site-packages\keras\engine\training_arrays.py in fit_loop(model, fit_function, fit_inputs, out_labels, batch_size, epochs, verbose, callbacks, val_function, val_inputs, shuffle, callback_metrics, initial_epoch, steps_per_epoch, validation_steps, validation_freq)
        202                     ins_batch[i] = ins_batch[i].toarray()
        203 
    --> 204                 outs = fit_function(ins_batch)
        205                 outs = to_list(outs)
        206                 for l, o in zip(out_labels, outs):
    

    ~\AppData\Local\Continuum\anaconda3\envs\gr_p3.6\lib\site-packages\keras\backend\theano_backend.py in __call__(self, inputs)
       1401     def __call__(self, inputs):
       1402         assert isinstance(inputs, (list, tuple))
    -> 1403         return self.function(*inputs)
       1404 
       1405 
    

    ~\AppData\Local\Continuum\anaconda3\envs\gr_p3.6\lib\site-packages\theano\compile\function_module.py in __call__(self, *args, **kwargs)
        901         try:
        902             outputs =\
    --> 903                 self.fn() if output_subset is None else\
        904                 self.fn(output_subset=output_subset)
        905         except Exception:
    

    ~\AppData\Local\Continuum\anaconda3\envs\gr_p3.6\lib\site-packages\theano\ifelse.py in thunk()
        243         outputs = node.outputs
        244 
    --> 245         def thunk():
        246             if not compute_map[cond][0]:
        247                 return [0]
    

    KeyboardInterrupt: 



```python
# Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print (score)
```


```python
# Show loss curves 
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-acb44adba319> in <module>
          1 # Show loss curves
    ----> 2 plt.figure()
          3 plt.title('Training performance')
          4 plt.plot(history.epoch, history.history['loss'], label='train loss+error')
          5 plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    

    NameError: name 'plt' is not defined



```python
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```


```python
# Plot confusion matrix
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)
```


```python
# Plot confusion matrix
acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
```


```python
# Save results to a pickle file for plotting later
print (acc)
fd = open('results_cnn2_d0.5.dat','wb')
cPickle.dump( ("CNN2", 0.5, acc) , fd )
fd.close()
```


```python
# Plot accuracy curve
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
```


```python
print("Done")
```
