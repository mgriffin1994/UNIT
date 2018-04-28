import os 
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.io import wavfile
import numpy as np
from scipy.signal import stft, istft
from audio2spectrum import shape_patch
import sys

def reconstruct(spec_mag, out_filename, sample_rate, nperseg,  nfft=None, iters=100,plot_result=False):
    length = istft(spec_mag, sample_rate, nperseg=nperseg, nfft=nfft)[1].shape[0]
    xt = np.random.normal(size=length)
    for i in range(iters):
        # Code based on the answer here: https://dsp.stackexchange.com/a/3410
        Xf = stft(xt, sample_rate, nperseg=nperseg, nfft=nfft)[2]
        
        Z = spec_mag * np.exp(np.angle(Xf) * 1j)
        t, x_rec = istft(Z, sample_rate, nperseg=nperseg)
        x_rec = x_rec/np.amax(abs(x_rec))
    if plot_result:
        plt.figure()
        plt.plot(t,x_rec)
        plt.title("reconstructed samples")
        plt.show()
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    wavfile.write(out_filename, sample_rate,x_rec)
    return t, x_rec

def rec_from_complex_spectrum(spec, out_filename, sample_rate, nperseg, nfft, plot_result=False): 
    t, x_rec = istft(spec, sample_rate, nperseg=nperseg, nfft=nfft)
    x_rec = x_rec/np.amax(abs(x_rec));
    
    if plot_result:
        plt.figure()
        plt.plot(t,x_rec)
        plt.title("reconstructed samples")
        plt.show()
    
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    wavfile.write(out_filename, sample_rate,x_rec)
    return t, x_rec

def magphase2complex(spec_mag_phase):
    if spec_mag_phase.shape[2] == 2:
        mag = spec_mag_phase[:,:,0]
        phase = spec_mag_phase[:,:,1]/255*360 - 180
        
        return np.multiply(mag, np.exp(phase/180*np.pi*1j))
        
    elif spec_mag_phase.shape[2] == 4:
        mag0 = spec_mag_phase[:,:,0]
        phase0 = spec_mag_phase[:,:,1]/255*360 - 180
        complex0 = np.multiply(mag0, np.exp(phase0/180*np.pi*1j))
        
        mag1 = spec_mag_phase[:,:,2]
        phase1 = spec_mag_phase[:,:,3]/255*360 - 180
        complex1 = np.multiply(mag1, np.exp(phase1/180*np.pi*1j))
        
        return np.stack([mag1, phase1], asix=2)

def rec_from_magphase_spectrum(spec_mag_phase, out_filename, sample_rate, nperseg, nfft, plot_result=False):
    spec_complex = magphase2complex(spec_mag_phase)
    t, x_rec = rec_from_complex_spectrum(spec_complex, out_filename, sample_rate, nperseg, nfft, plot_result=False)
    return t, x_rec

if __name__ == "__main__":  
    #format of calling: python3 spectrum2audio.py "/Users/tom/Downloads/vcc2016_training/SF1" 3
    #argv[1]: directory of audio file
    #argv[2]: which "piture" you want to convert into audio
    prfx = sys.argv[1]+'/'     # prfx = "/Users/tom/Downloads/vcc2016_training/SF1/"
    
    spectrum_filename = prfx + "spec_data/image_with_params/pic_" + str(sys.argv[2]) + ".npz"
    out_filename = prfx + str(sys.argv[2]) + "_reconstructed.wav"
     
    data = np.load(spectrum_filename)
    # true_shape = data['true_spec_shape']
    # print("*********************** true shape:",true_shape)
    
    temp = data['spec4train']
    # print("*********************** spectrum shape:", temp.shape) 
    # spec_mag = shape_patch(temp[:,:,0], true_shape)
    spec_mag = temp[:,:,0]
    # spec_phase = shape_patch(temp[:,:,1], true_shape)
    spec_phase = temp[:,:,1]
    spec_complex = magphase2complex(np.stack([spec_mag, spec_phase], axis=2))
    # print("*********************** pathched spectrum shape:",spec_complex.shape)
    sample_rate = data['sample_rate']
    nperseg = data['nperseg']
    nfft = data["nfft"]
    print("nfft, nperseg", nfft, nperseg)
    # nfft =
    # nfft = 1023
    # nperseg = 70
    spec_mag = np.abs(spec_complex)
    
    plt.figure()
    plt.plot(np.reshape(spec_mag,-1,1))
    plt.show()
    
    # t, x_rec = reconstruct(spec_mag, out_filename, sample_rate, nperseg,  nfft=nfft, iters=100, plot_result=False)
    try:
        t, x_rec =rec_from_complex_spectrum(spec_complex, out_filename, \
                sample_rate, nperseg, nfft=nfft, plot_result=False)
    except:
        t, x_rec =rec_from_complex_spectrum(spec_complex, out_filename, \
                sample_rate, nperseg+3, nfft=nfft, plot_result=False)
    
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    wavfile.write(out_filename, sample_rate,x_rec)
    
    plt.figure()
    plt.plot(t,x_rec)
    plt.show()









