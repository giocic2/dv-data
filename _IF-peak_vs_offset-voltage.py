import os
import numpy as np
import matplotlib.pyplot as plt

filenames = []
voltageOffsets = 0 # initialization, evalued automatically
offsetStep = 0.1 # V
startStep = - 10 # steps
stopStep = 10 # steps (the last one is not included by np.arange)
stopStep += 1 # the last step is not included by np.arange
currentCycle = 0

print('List of *.csv files in the current directory: ')
with os.scandir(path='.') as directoryScan:
    for entry in directoryScan:
            if entry.name.endswith('.csv') and entry.is_file():
                print(entry.name)
                
with os.scandir(path='.') as directoryScan:
    for entry in directoryScan:
            if entry.name.endswith('.csv') and entry.is_file():
                filenames.append(entry.name)
                print(entry.name)
            if 'ChA' in entry.name:
                voltageOffsets += 1
                
offsetAxis = np.arange(start = startStep, stop = stopStep) * offsetStep

TXf_FREQ = 24e9 # Hz
wavelength = 3e8 / 24e9
print('Wavelength: ' + str(wavelength * 1e3) + ' mm')
            
print('Number of different DC voltage offsets: ' + str(voltageOffsets))

SAMPLING_FREQUENCY = 10e3 # Hz
print('Sampling frequency: ' + str(SAMPLING_FREQUENCY) + ' Hz')
ACQUISITION_TIME = 2 # s
print('Acquisition time: ' + str(ACQUISITION_TIME) + ' s')
samplingInterval = 1/SAMPLING_FREQUENCY
totalSamples = round(ACQUISITION_TIME/samplingInterval)
FFT_FREQ_BINS = 2**19
print('FFT resolution: ' + str(SAMPLING_FREQUENCY/FFT_FREQ_BINS) + ' Hz')

FFTpeaks = np.ndarray((2, voltageOffsets)) # First row for ChA (IFI), second row for ChB (IFQ)

currentCycle = 0

CONE_FREQUENCY = 35 # Hz
ARGMAX_RANGE = 100 # bins

for entry in filenames:
    if 'ChA' in entry:
        rawSamples = np.genfromtxt(entry, delimiter = ',')
        voltageAxis_mV = rawSamples[:,0]
        timeAxis_s = rawSamples[:,1]
        # FFT computation
        FFT = np.fft.rfft(voltageAxis_mV, n = FFT_FREQ_BINS) # FFT of real signal
        FFT_mV = np.abs(2/(totalSamples)*FFT) # FFT magnitude
        FFT_dBV = 20*np.log10(FFT_mV/1000)
        freqAxis = np.fft.rfftfreq(FFT_FREQ_BINS) # freqBins/2+1
        freqAxis_Hz = freqAxis * SAMPLING_FREQUENCY
        argmax_startBin = int(round((FFT_FREQ_BINS / (SAMPLING_FREQUENCY) * CONE_FREQUENCY) - ARGMAX_RANGE / 2))
        argmax_endBin = int(round((FFT_FREQ_BINS / (SAMPLING_FREQUENCY) * CONE_FREQUENCY) + ARGMAX_RANGE / 2))
        FFTpeaks[0, currentCycle] = np.amax(FFT_mV[argmax_startBin:argmax_endBin])
        print("{0:.0f}".format(FFTpeaks[0, currentCycle]), end = ' ')
    if 'ChB' in entry:
        rawSamples = np.genfromtxt(entry, delimiter = ',')
        voltageAxis_mV = rawSamples[:,0]
        timeAxis_s = rawSamples[:,1]
        # FFT computation
        FFT = np.fft.rfft(voltageAxis_mV, n = FFT_FREQ_BINS) # FFT of real signal
        FFT_mV = np.abs(2/(totalSamples)*FFT) # FFT magnitude
        FFT_dBV = 20*np.log10(FFT_mV/1000)
        freqAxis = np.fft.rfftfreq(FFT_FREQ_BINS) # freqBins/2+1
        freqAxis_Hz = freqAxis * SAMPLING_FREQUENCY
        argmax_startBin = int(round((FFT_FREQ_BINS / (SAMPLING_FREQUENCY) * CONE_FREQUENCY) - ARGMAX_RANGE / 2))
        argmax_endBin = int(round((FFT_FREQ_BINS / (SAMPLING_FREQUENCY) * CONE_FREQUENCY) + ARGMAX_RANGE / 2))
        FFTpeaks[1, currentCycle] = np.amax(FFT_mV[argmax_startBin:argmax_endBin])
        print("{0:.0f}".format(FFTpeaks[1, currentCycle]), end = ' ')
        currentCycle += 1

# IFI and IFQ plots
plt.plot(offsetAxis, FFTpeaks[0, :], label = 'IFI magnitude @35Hz [mV]')
plt.plot(offsetAxis, FFTpeaks[1, :], label = 'IFQ magnitude @35Hz [mV]')
plt.xlabel('Offset voltage [V]')
plt.legend()
plt.grid(True)
plt.show()
