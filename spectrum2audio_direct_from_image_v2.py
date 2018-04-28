import os 
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.io import wavfile
import numpy as np
from scipy.signal import stft, istft
from scipy import ndimage as img
from audio2spectrum import shape_patch
import sys

def GriffinLim(spec_mag, out_filename, sample_rate, nperseg, noverlap,  nfft, iters=200, plot_result=False):
    length = istft(spec_mag, fs=sample_rate, window='boxcar', nperseg=nperseg, noverlap=noverlap, nfft=nfft)[1].shape[0]
    xt_old = np.random.normal(size=length)
    
    print("********* spec_mag.shape:", spec_mag.shape)
    # nfft = spec_mag.shape[0]*2-1
    for i in range(iters):
        spectrum = stft(xt_old, fs=sample_rate, window='boxcar', nperseg=nperseg, noverlap=noverlap, nfft=nfft)[2]
        spectrum = spectrum[0:spec_mag.shape[0],:]
        
        spectrum = spec_mag * np.exp(np.angle(spectrum) * 1j)
        t, xt_new = istft(spectrum, fs=sample_rate, window='boxcar', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        err = np.linalg.norm(xt_new-xt_old)/np.linalg.norm(xt_old)
        xt_old = xt_new
        
        if not(i%20):
            print(str(i)+'/'+str(iters)+' iteration: error = '+str(err))
    temp = np.angle(spectrum, deg=True)
    phase = (temp+180)/360*255
     
    return t, xt_new, phase
    

def reconstruct_from_mag(spec_mag, num_ch, out_filename, sample_rate, nperseg, noverlap, nfft, plot_result=False):
    spec_mag = np.reshape(spec_mag, (spec_mag.shape[0], spec_mag.shape[1], -1))
    if num_ch == 1:
        t, x_rec, phase= GriffinLim(spec_mag[:,:,0], out_filename, sample_rate, nperseg, noverlap,  nfft, iters=200, plot_result=False)
    elif num_ch == 2:
        t, x_rec0, phase = GriffinLim(spec_mag[:,:,0], out_filename, sample_rate, nperseg, noverlap,  nfft, iters=200, plot_result=False)
        t, x_rec1, phase = GriffinLim(spec_mag[:,:,1], out_filename, sample_rate, nperseg, noverlap,  nfft, iters=200, plot_result=False)
        x_rec0 = np.reshape(x_rec0,-1,1)
        x_rec1 = np.reshape(x_rec1,-1,1)
        x_rec = np.stack((x_rec0, x_rec1), axis=1)
        # print(out_filename)
        x_rec = x_rec/np.amax(abs(x_rec));
    wavfile.write(out_filename, sample_rate, x_rec)
    return t, x_rec

def rec_from_complex_spectrum(spec_complex, num_ch, out_filename, sample_rate, nperseg, noverlap, nfft, plot_result=False): 
    # works for mono/stereo
    shape0 = spec_complex.shape[0]
    shape1 = spec_complex.shape[1]
    
    if num_ch == 1:
        spec = np.reshape(spec_complex, (shape0,shape1,-1))
        try:
            t, x_rec = istft(spec[:,:,0], sample_rate, window='boxcar', noverlap=noverlap, nperseg=nperseg, nfft=nfft)
        except ValueError:
            print("****** parameters adjusted to satisfy COLA")
            t, x_rec = istft(spec[:,:,0], sample_rate, window='boxcar', noverlap=noverlap, nperseg=nperseg, nfft=nfft)
        
    elif num_ch == 2:
        try:
            t, x0 = istft(spec_complex[:,:,0], fs=sample_rate, window='boxcar', noverlap=noverlap,  nperseg=nperseg, nfft=nfft)
        except ValueError:
            print("****** parameters adjusted to satisfy COLA")
            t, x0 = istft(spec_complex[:,:,0], sample_rate, window='boxcar', noverlap=noverlap, nperseg=nperseg, nfft=nfft)
        try:    
            t, x1 = istft(spec_complex[:,:,1], fs=sample_rate, window='boxcar', noverlap=noverlap, nperseg=nperseg, nfft=nfft)
        except ValueError:
            print("****** parameters adjusted to satisfy COLA")
            t, x1 = istft(spec_complex[:,:,1], sample_rate, window='boxcar', noverlap=noverlap, nperseg=nperseg, nfft=nfft)
        
        x0 = np.reshape(x0,-1,1)
        x1 = np.reshape(x1,-1,1)
        x_rec = np.stack([x0,x1],axis=1)
    else:
        raise(ValueError("Wrong Number of Channels"))
    x_rec = x_rec/np.amax(abs(x_rec));
    if plot_result:
        plt.figure()
        if num_ch == 1:
            plt.plot(t,x_rec)
        elif num_ch ==2:
            plt.plot(t,x_rec[:,1])
        plt.title("reconstructed samples")
        plt.show()   
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    # print("x_rec.shape, type(x_rec[0][0]), sample_rate = ", x_rec.shape, type(x_rec[0][0]), sample_rate)
    # print(out_filename)
    wavfile.write(out_filename, sample_rate, x_rec)
    return t, x_rec

def magphase2complex(spec_mag_phase, num_ch):
    # works for mono/stereo
    if  num_ch == 1:  # mono channel audio
        mag = spec_mag_phase[:,:,0]
        phase = spec_mag_phase[:,:,1]/255*360 - 180
        
        return np.multiply(mag, np.exp(phase/180*np.pi*1j))
        
    elif num_ch == 2: # stereo audio
        mag0 = spec_mag_phase[:,:,0]
        phase0 = spec_mag_phase[:,:,1]/255*360 - 180
        complex0 = np.multiply(mag0, np.exp(phase0/180*np.pi*1j))
        
        mag1 = spec_mag_phase[:,:,2]
        phase1 = spec_mag_phase[:,:,3]/255*360 - 180
        complex1 = np.multiply(mag1, np.exp(phase1/180*np.pi*1j))
        return np.stack([complex0, complex1], axis=2)
    else:
        raise(ValueError("Wrong Number of Channels"))
   
def rec_from_magphase_spectrum(spec_mag_phase, num_ch, out_filename, sample_rate, nperseg, noverlap, nfft, plot_result=False):
    spec_complex = magphase2complex(spec_mag_phase, num_ch)
    t, x_rec = rec_from_complex_spectrum(spec_complex, num_ch, out_filename, sample_rate, nperseg, noverlap, nfft, plot_result=plot_result)
    return t, x_rec

#******************* the following two are out-most-layer functions to call #*********************
def matrix2audio(M, filename_from_dataset, out_filename, mode_rec='mag_only', mode_saved='mag_only', plot_result=False):
    data = np.load(filename_from_dataset)
    try: 
        sample_rate = data['sample_rate']
    except KeyError:
        raise(KeyError("parameter sample_rate should be specified along with the image matrix!"))        
    try:
        num_ch = data['num_channel']
    except KeyError:
        print("Number of audio channels not given, default to 2")
        num_ch = 2             
    try:     
        nperseg = data['nperseg']
    except KeyError:
        raise(KeyError("parameter nperseg should be specified along with the image matrix!"))      
    try:     
        noverlap = data['noverlap']
    except KeyError:
        raise(KeyError("parameter noverlap should be specified along with the image matrix!"))   
    try:
        nfft = data["nfft"]
    except KeyError:
        raise(KeyError("parameter nfft should be specified along with the image matrix!"))
    # try:
    #     mode_saved = data["mode"]
    # except KeyError:
    #     raise(KeyError("parameter mode should be specified along with the image matrix!"))
        
    print("sample_rate =", sample_rate)
    print("nperseg =", nperseg)
    print("noverlap =", noverlap)
    print("nfft =", nfft)

    if mode_rec == 'mag_only':
        if num_ch == 2 and mode_saved == "mag+phase":
            M0 = M[:,:,0]
            M1 = M[:,:,2]
            M = np.stack([M0,M1],axis=2)
        t, x_rec = reconstruct_from_mag(M, num_ch, out_filename, sample_rate, nperseg, noverlap, nfft, plot_result=False)
    elif mode_rec == 'mag+phase':
        t, x_rec = rec_from_magphase_spectrum(M, num_ch, out_filename, sample_rate, nperseg, noverlap, nfft, plot_result=plot_result)
    else:
        raise(ValueError("Wrong string for mode!"))
    return t, x_rec
    
def jpg2audio(jpgname, filename_from_dataset, out_filename, mode_rec='mag_only', mode_saved='mag_only', plot_result=False):
    M = img.read(jpgname)
    matrix2audio(M, filename_from_dataset, out_filename, mode_rec, mode_saved, plot_result=plot_result)
    return t, x_rec
    

if __name__ == "__main__":
    # ***************************** test: stereo, mag+phase, matrix *****************************
    # filename_from_dataset = "/Volumes/Mac_Ext/dataset/IRMAS-TrainingData/flu/spec_data/image_with_params/pic_1.npz"
    filename_from_dataset = "/Volumes/Mac_Ext/dataset/vcc2016_training/SF1/spec_data/image_with_params/pic_1.npz"
    
    # out_filename = "/Volumes/Mac_Ext/dataset/IRMAS-TrainingData/flu/1_rec.wav"
    out_filename = "/Volumes/Mac_Ext/dataset/vcc2016_training/SF1/1_rec.wav"
    data = np.load(filename_from_dataset)
    M = data['spec4train']
    # print(M.shape)
    matrix2audio(M, filename_from_dataset, out_filename, mode_rec='mag_only', mode_saved='mag+phase', plot_result = 1)
    
    
    
    
    
    
    
    
    
    
    