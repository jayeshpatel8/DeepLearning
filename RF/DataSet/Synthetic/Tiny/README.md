### Tiny Dataset:
      Generated locally by limiting the modulation class to only two i.e. ( CPFSK & GFSK ). 10000 samples/modulation
      [20,000 I/Q sample] , Modulation[2] , SNR [-20 to 18 dB]

### Generated dataset files:
      RML2014.04c_dict.pkl
      RML2014.04c.pkl


### ipynb

```python

```


```python

from transmitters import transmitters
from source_alphabet import source_alphabet
import timeseries_slicer
import analyze_stats
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft, gzip
import _pickle as cPickle
print(transmitters)
```

    Mapper warning: Could not load the module “fastcluster”.
    The module “scipy.cluster.hierarchy“ is used instead, but it will be slower.
    The 'cmappertools' module could not be imported.
    The 'cmappertools' module could not be imported.
    Intrinsic metric is not available.
    The 'cmappertools' module could not be imported.
    

    {'discrete': [<class 'transmitters.transmitter_gfsk'>, <class 'transmitters.transmitter_cpfsk'>]}
    


```python

```


```python
'''
Generate dataset with dynamic channel model across range of SNRs
'''

apply_channel = True
output = {}
min_length = 9e9
snr_vals = range(-20,20,2)
for snr in snr_vals:
    for alphabet_type in transmitters.keys():
        print (alphabet_type)
        for i,mod_type in enumerate(transmitters[alphabet_type]):
            print ("running test", i,mod_type)

            tx_len = int(10e3)
            if mod_type.modname == "QAM64":
                tx_len = int(30e3)
            if mod_type.modname == "QAM16":
                tx_len = int(20e3)
            src = source_alphabet(alphabet_type, tx_len, True)
            mod = mod_type()
            fD = 1
            delays = [0.0, 0.9, 1.7]
            mags = [1, 0.8, 0.3]
            ntaps = 8
            noise_amp = 10**(-snr/10.0)
            print (noise_amp)
            #noise_amp = 0.1
            chan = channels.dynamic_channel_model( 200e3, 0.01, 1e2, 0.01, 1e3, 8, fD, True, 4, delays, mags, ntaps, noise_amp, 0x1337 )

            snk = blocks.vector_sink_c()

            tb = gr.top_block()

            # connect blocks
            if apply_channel:
                tb.connect(src, mod, chan, snk)
            else:
                tb.connect(src, mod, snk)
            tb.run()

            modulated_vector = np.array(snk.data(), dtype=np.complex64)
            if len(snk.data()) < min_length:
                min_length = len(snk.data())
                min_length_mod = mod_type
            output[(mod_type.modname, snr)] = modulated_vector

print ("min length mod is %s with %i samples" % (min_length_mod, min_length))
# trim the beginning and ends, and make all mods have equal number of samples
start_indx = 100
fin_indx = min_length-100
for mod, snr in output:
 output[(mod,snr)] = output[(mod,snr)][start_indx:fin_indx]
X = timeseries_slicer.slice_timeseries_dict(output, 128, 64, 1000)
file = open("RML2014.04c_dict.pkl", "wb" )
cPickle.dump( X, file )
file.close()
X = np.vstack(X.values())
file = open("RML2016.04c.pkl", "wb" )
cPickle.dump( X, file )
file.close()
```

    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    100.0
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    100.0
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    63.09573444801933
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    63.09573444801933
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    39.810717055349734
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    39.810717055349734
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    25.118864315095795
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    25.118864315095795
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    15.848931924611133
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    15.848931924611133
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    10.0
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    10.0
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    6.309573444801933
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    6.309573444801933
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    3.9810717055349722
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    3.9810717055349722
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    2.51188643150958
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    2.51188643150958
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    1.5848931924611136
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    1.5848931924611136
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    1.0
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    1.0
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    0.6309573444801932
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    0.6309573444801932
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    0.3981071705534972
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    0.3981071705534972
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    0.251188643150958
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    0.251188643150958
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    0.15848931924611134
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    0.15848931924611134
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    0.1
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    0.1
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    0.06309573444801933
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    0.06309573444801933
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    0.039810717055349734
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    0.039810717055349734
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    0.025118864315095794
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    0.025118864315095794
    discrete
    running test 0 <class 'transmitters.transmitter_gfsk'>
    0.015848931924611134
    running test 1 <class 'transmitters.transmitter_cpfsk'>
    0.015848931924611134
    min length mod is <class 'transmitters.transmitter_gfsk'> with 79992 samples
    

    ipykernel_launcher.py:58: FutureWarning: arrays to stack must be passed as a "sequence" type such as list or tuple. Support for non-sequence iterables such as generators is deprecated as of NumPy 1.16 and will raise an error in the future.
    


```python

```

    


```python

```


```python

```
