
### Log from TF1.5

## Step1. prepare align.json

 python -m datasets.generate_data ./datasets/LJSpeech_1_0/alignment.json


## Step2. convert 



# problem : 
'''
root@379ff2ce4937:/purescratch/hryu/github/tacotron-ms# python -m datasets.datasets.generate_data ./datasets/LJSpeech_1_0/alignment.json 
.gitignore        README.md         assets/           download.py       logs/             requirements.txt  synthesizer.py    utils/            
DISCLAIMER        __pycache__/      audio/            eval.py           models/           run.sh            text/             web/              
LICENSE           app.py            datasets/         hparams.py        recognition/      scripts/          train.py          
root@379ff2ce4937:/purescratch/hryu/github/tacotron-ms# python -m datasets.datasets.generate_data ./datasets/LJSpeech_1_0/alignment.json 
/usr/bin/python: Error while finding spec for 'datasets.datasets.generate_data' (ImportError: No module named 'datasets.datasets')
root@379ff2ce4937:/purescratch/hryu/github/tacotron-ms# vi datasets/generate_data.py 
root@379ff2ce4937:/purescratch/hryu/github/tacotron-ms# python -m datasets.generate_data ./datasets/LJSpeech_1_0/alignment.json 
Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/pywrap_tensorflow.py", line 58, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 28, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
  File "/usr/lib/python3.5/imp.py", line 242, in load_module
    return load_dynamic(name, filename, file)
  File "/usr/lib/python3.5/imp.py", line 342, in load_dynamic
    return _load(spec)
ImportError: libcuda.so.1: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.5/runpy.py", line 184, in _run_module_as_main
    "__main__", mod_spec)
  File "/usr/lib/python3.5/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/purescratch/hryu/github/tacotron-ms/datasets/generate_data.py", line 19, in <module>
    from hparams import hparams
  File "/purescratch/hryu/github/tacotron-ms/hparams.py", line 1, in <module>
    import tensorflow as tf
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/__init__.py", line 24, in <module>
    from tensorflow.python import *
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/__init__.py", line 49, in <module>
    from tensorflow.python import pywrap_tensorflow
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/pywrap_tensorflow.py", line 74, in <module>
    raise ImportError(msg)
ImportError: Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/pywrap_tensorflow.py", line 58, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 28, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
  File "/usr/lib/python3.5/imp.py", line 242, in load_module
    return load_dynamic(name, filename, file)
  File "/usr/lib/python3.5/imp.py", line 342, in load_dynamic
    return _load(spec)
ImportError: libcuda.so.1: cannot open shared object file: No such file or directory


Failed to load the native TensorFlow runtime.

See https://www.tensorflow.org/install/install_sources#common_installation_problems

for some common reasons and solutions.  Include the entire stack trace
above this error message when asking for help.
root@379ff2ce4937:/purescratch/hryu/github/tacotron-ms# ldconfig
root@379ff2ce4937:/purescratch/hryu/github/tacotron-ms# find /usr -name libcoda.so.1
root@379ff2ce4937:/purescratch/hryu/github/tacotron-ms# find /usr -name libcuda.so  
/usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/libcuda.so
root@379ff2ce4937:/purescratch/hryu/github/tacotron-ms# find /usr -name libcuda.so.1
/usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/libcuda.so.1
'''
# solution 

'''
export LD_LIBRARY_PATH=:$LD_LIBRARY_PATH:/usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs
'''

## Result

'''
root@379ff2ce4937:/purescratch/hryu/github/tacotron-ms# python -m datasets.generate_data ./datasets/LJSpeech_1_0/alignment.json 
========================================
 [!] Sampling rate: 22050
========================================

 [*] Make directories : ./datasets/LJSpeech_1_0/data
 [!] Skip recognition level: 0 (use all)
 [!] Converting to english mode
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13100/13100 [02:46<00:00, 78.71it/s]
 [*] Loaded metadata for 13100 examples (24.00 hours)
 [*] Max length: 810
 [*] Min length: 90
 [*] After filtered: 12889 examples (23.91 hours)
 [*] Max length: 810
 [*] Min length: 150

'''


# check the fiels 


'''
197.npz  LJ040-0039.npz  LJ045-0232.npz  LJ050-0271.npz
LJ005-0185.npz  LJ010-0023.npz  LJ014-0160.npz  LJ018-0097.npz  LJ023-0090.npz  LJ029-0077.npz  LJ034-0198.npz  LJ040-0040.npz  LJ045-0233.npz  LJ050-0272.npz
LJ005-0186.npz  LJ010-0024.npz  LJ014-0161.npz  LJ018-0098.npz  LJ023-0091.npz  LJ029-0078.npz  LJ034-0199.npz  LJ040-0041.npz  LJ045-0234.npz  LJ050-0273.npz
LJ005-0187.npz  LJ010-0025.npz  LJ014-0162.npz  LJ018-0099.npz  LJ023-0092.npz  LJ029-0079.npz  LJ034-0200.npz  LJ040-0042.npz  LJ045-0235.npz  LJ050-0274.npz
LJ005-0188.npz  LJ010-0026.npz  LJ014-0163.npz  LJ018-0100.npz  LJ023-0093.npz  LJ029-0080.npz  LJ034-0201.npz  LJ040-0043.npz  LJ045-0236.npz  LJ050-0275.npz
LJ005-0189.npz  LJ010-0027.npz  LJ014-0164.npz  LJ018-0101.npz  LJ023-0094.npz  LJ029-0081.npz  LJ034-0202.npz  LJ040-0044.npz  LJ045-0237.npz  LJ050-0276.npz
LJ005-0190.npz  LJ010-0028.npz  LJ014-0165.npz  LJ018-0102.npz  LJ023-0095.npz  LJ029-0082.npz  LJ034-0203.npz  LJ040-0045.npz  LJ045-0238.npz  LJ050-0277.npz
LJ005-0191.npz  LJ010-0029.npz  LJ014-0166.npz  LJ018-0103.npz  LJ023-0096.npz  LJ029-0083.npz  LJ034-0204.npz  LJ040-0046.npz  LJ045-0239.npz  LJ050-0278.npz

'''
