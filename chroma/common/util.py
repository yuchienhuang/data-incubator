# util.py

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import wave
import sys
import os
import os.path

import IPython.display as ipd
from ipywidgets import IntSlider, FloatSlider

        
PI = np.pi


######### Wave Files ##########

def load_wav(filepath, t_start = 0, t_end = sys.maxsize, only_22k = True) :
    """Load a wave file, which must be 22050Hz and 16bit and must be either
    mono or stereo. 
    Inputs:
        filepath: audio file
        t_start, t_end:  (optional) subrange of file to load (in seconds)
        only_22k: if True (default), assert if sample rate is different from 22050.
    Returns:
        a numpy floating-point array with a range of [-1, 1]
    """
    
    wf = wave.open(filepath)
    num_channels, sampwidth, fs, end, comptype, compname = wf.getparams()
    
    # for now, we will only accept 16 bit files at 22k
    assert(sampwidth == 2)
    # assert(fs == 22050)

    # start frame, end frame, and duration in frames
    f_start = int(t_start * fs)
    f_end = min(int(t_end * fs), end)
    frames = f_end - f_start

    wf.setpos(f_start)
    raw_bytes = wf.readframes(frames)

    # convert raw data to numpy array, assuming int16 arrangement
    samples = np.fromstring(raw_bytes, dtype = np.int16)

    # convert from integer type to floating point, and scale to [-1, 1]
    samples = samples.astype(np.float)
    samples *= (1 / 32768.0)

    if num_channels == 1:
        return samples

    elif num_channels == 2:
        return 0.5 * (samples[0::2] + samples[1::2])

    else:
        raise('Can only handle mono or stereo wave files')

def save_wav(channels, fs, filepath) :
    """Interleave channels and write out wave file as 16bit audio.
    Inputs:
        channels: a tuple or list of np.arrays. Or can be a single np.array in which case this will be a mono file.
                  format of np.array is floating [-1, 1]
        fs: sampling rate
        filepath: output filepath
    """

    if type(channels) == tuple or type(channels) == list:
        num_channels = len(channels)
    else:
        num_channels = 1
        channels = [channels]

    length = min ([len(c) for c in channels])
    data = np.empty(length*num_channels, np.float)

    # interleave channels:
    for n in range(num_channels):
        data[n::num_channels] = channels[n][:length]

    data *= 32768.0
    data = data.astype(np.int16)
    data = data.tostring()

    wf = wave.open(filepath, 'w')
    wf.setnchannels(num_channels)
    wf.setsampwidth(2)
    wf.setframerate(fs)
    wf.writeframes(data)


######### ipywidgets helper function for sliders ##########

def slider(min, max, value=None):
    """Create a FloatSlider or IntSlider, depending on type of input args.
    Turn off continuous_update
    """
    if type(min) is float or type(max) is float:
        return FloatSlider(min=float(min), max=float(max), continuous_update=False, value=value)
    else:
        return IntSlider(min=int(min), max=int(max), continuous_update=False, value=value)


######### IPython Display ##########

class Audio(ipd.Audio):
    """Audio object, providing browser-based audio controls in notebook
    
    Almost identical to IPython.display.Audio (see that module for docs).
    Provides additional parameter: norm, for controlling normalization of 
    audio data whereas IPython.display.Audio will always normalize.
    
    Parameters
    ----------
    data : numpy array
    url : unicode string
    filename : unicode string
    rate : integer
        The sampling rate of the raw data.
        Only required when data parameter is being used as an array
    norm : bool
        Normalize the data array so max value == 1
        Default is `False`
    """
    
    def __init__(self, data=None, filename=None, url=None, rate=None, norm=True):
        self.norm = norm # normalize audio data or not
        super(Audio, self).__init__(data=data, filename=filename, url=url, rate=rate)        

    # override this behavior to add normalization option
    def _make_wav(self, data, rate):
        """ Transform a numpy array to a PCM bytestring """
        import struct
        from io import BytesIO
        import wave

        data = np.array(data, dtype=float)
        if len(data.shape) == 1:
            nchan = 1
        elif len(data.shape) == 2:
            nchan = data.shape[0]
            data = data.T.ravel()
        else:
            raise ValueError('Array audio input must be a 1D or 2D array')
        
        if self.norm:
            max_val = np.max(np.abs(data))
            data = data/max_val
        else: # otherwise, clip data to [-1, 1]
            data = np.clip(data, -1, 1)
            
        scaled = np.int16(data*32767).tolist()

        fp = BytesIO()
        waveobj = wave.open(fp,mode='wb')
        waveobj.setnchannels(nchan)
        waveobj.setframerate(rate)
        waveobj.setsampwidth(2)
        waveobj.setcomptype('NONE','NONE')
        waveobj.writeframes(b''.join([struct.pack('<h',x) for x in scaled]))
        val = fp.getvalue()
        waveobj.close()

        return val
        
        


######### Annotations ##########

def load_annotations(filepath) :
    '''Load annotations from a tab-separated text file where each line is one annotation and all annotations are numbers
    Input:
        filepath: annotation file
    Return:
        np.array (MxN) with annotation data
    '''
    lines = open(filepath).readlines()
    return np.array([float(l.split('\t')[0]) for l in lines])

def write_annotations(data, filepath) :
    f = open(filepath, 'w')
    for d in data:
        f.write('%f\n' % d)


######### Plotting ##########

def plot_and_listen(filepath, len_t = 0) :
    """Plot the audio waveform and create an audio listening widget.
    Inputs:
        filepath: audio file
        len_t: (optional) only load the first len_t seconds of audio.
    Returns:
        IPython.display.Audio object for listening
    """
    if len_t != 0:
        x = load_wav(filepath, 0, len_t)
    else:
        x = load_wav(filepath)
    fs = 22050
    t = np.arange(len(x)) / float(fs)
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("time (secs)")
    plt.show()
    return Audio(x, rate=fs)

def plot_fft_and_listen(filepath, raw_axis = False) :
    """Plot the audio waveform and create an audio listening widget.
    Inputs:
        filepath: audio file
        raw_axis: (optional)
    Returns:
        IPython.display.Audio object for listening
    """
    fs = 22050
    x = load_wav(filepath)
    x_ft = np.abs(np.fft.fft(x))

    time = np.arange(len(x),dtype=np.float) / fs
    freq = np.arange(len(x_ft), dtype=np.float) / len(x_ft) * fs

    if raw_axis:
        print('sample rate:', fs)
        print('N: ', len(x))

    plt.figure()
    plt.subplot(2,1,1)
    if raw_axis:
        plt.plot(x)
        plt.xlabel('n')
        plt.ylabel('$x(n)$')
    else:
        plt.plot(time, x)
        plt.xlabel('time')

    plt.subplot(2,1,2)
    if raw_axis:
        plt.plot(x_ft)
        plt.xlabel('k')
        plt.ylabel('$|X(k)|$')
        plt.xlim(0, 3000*len(x) / fs)
    else:
        plt.plot(freq, x_ft)
        plt.xlim(0, 3000)
        plt.xlabel('Frequency (Hz)')

    return Audio(x, rate=fs)


def plot_spectrogram(spec, cmap=None, colorbar=True, fs = None) :
    '''plot a spectrogram using a log scale for amplitudes (ie, color brightness)
    Inputs:
        spec: the spectrogram, |STFT|^2
        cmap: (optional), provide a cmap
        colorbar: (optional, default True), plot the colobar
        fs: the original sampling frequency - will show frequency labels for y-axis
    '''

    extent = None 
    if fs:
        extent = (0, spec.shape[1], 0, fs / 2)

    maxval = np.max(spec)
    minval = .1
    plt.imshow(spec, origin='lower', interpolation='nearest', aspect='auto', 
        norm=LogNorm(vmin=minval, vmax=maxval), cmap=cmap, extent=extent)
    if colorbar:
        plt.colorbar()


def plot_two_chromas(c1, c2, cmap = 'Greys'):
    '''plot two chromagrams with subplots(2,1,1) and (2,1,2). Ensure that vmin and vmax are the same
    for both chromagrams'''

    plt.subplot(2,1,1)
    _min = 0.5 * ( np.min(c1) + np.min(c2) )
    _max = 0.5 * ( np.max(c1) + np.max(c2) )
    plt.imshow(c1, origin='lower', interpolation='nearest', aspect='auto', cmap=cmap, vmin=_min, vmax=_max)
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.imshow(c2, origin='lower', interpolation='nearest', aspect='auto', cmap=cmap, vmin=_min, vmax=_max)
    plt.colorbar()



######### File ##########

def get_directory_files(dirpath, file_ext = None):
    '''Return all files in a directory
    Inputs:
        dirpath: directory name
        file_ext: (optional) only return files ending with that extension.
    '''
    files = sorted(os.listdir(dirpath))
    return [os.path.join(dirpath, f) for f in files if file_ext == None or f.endswith(file_ext)]

