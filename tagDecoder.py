from os import listdir
from multiprocessing import Process as Proc
from multiprocessing import Queue
import numpy as np
import csv
from numpy.random import normal

class tagDecoder:
    def __init__(self, sampleStream, sampleRate, dataRate):
        self.samples = sampleStream
        self.sampleRate = sampleRate
        self.dataRate = dataRate
        self.samplePerBit = int(sampleRate/float(dataRate))
        self.preambleMask = [1,1,-1,1,-1,-1,1,-1,-1,-1,1,1]
        self.oneMask = [-1,1,1,-1]
        self.zeroMask = [-1,1,-1,1]
        self.minT1 = int(250e-6 * sampleRate)
        # self.preambleRange = int((self.minT1*0.04 + 2e-6*self.sampleRate)*2)
        self.preambleRange = self.samplePerBit
        self.shiftingLog = []
        self.tagAmp = complex(0.0, 0.0)

        self.minT1_ratio = 0.9 # with 210805 use 0.5. Otherwise use 0.9

    def getMaskedValue(self, samples, cur_index, mask, prev_value = None):
        halfBitSamples = int(self.samplePerBit/2)
        cur_value = complex(0.0, 0.0)

        if prev_value != None:
            cur_value = prev_value

        endIdx = cur_index
        for idx_m in range(len(mask)):

            startIdx = endIdx
            endIdx = int(startIdx + halfBitSamples)

            if prev_value == None:
                for sample in samples[startIdx: endIdx]:
                    cur_value += (mask[idx_m] * sample)
            else:
                cur_value += (mask[idx_m]*samples[endIdx - 1] - mask[idx_m]*samples[startIdx - 1])
        return cur_value


    def getNoiseStd(self) -> complex:
        T1_samples = np.array(self.samples[0:int((self.minT1*self.minT1_ratio)*0.9)])
        noise_std = complex(np.std(T1_samples.real), np.std(T1_samples.imag))

        return noise_std

    def getAmpAvg(self) -> complex:
        T1_samples = np.array(self.samples[0:int(self.minT1*self.minT1_ratio/2)])
        amp_avg = np.mean(T1_samples)

        return amp_avg

    def getTagCorr(self) -> complex:
        start_idx, preambleAmp = self.findPreamble()
        
        # tag_signal_len = int(((len(self.preambleMask)/2 + 16)*self.samplePerBit)*1.04)
        # end_idx = start_idx+tag_signal_len
        # amp_avg = self.getAmpAvg()

        # self.tagAmp = np.mean(np.array(self.samples[start_idx:end_idx])-amp_avg)

        self.tagAmp = preambleAmp
        return self.tagAmp

    def findPreamble(self):
        maskValue = None
        bestIdx = 0
        bestValue = 0.0

        # T1_duration_with_minimum_tolerance = int((self.minT1*0.9) - (self.minT1*0.04) - (2e-6*self.sampleRate))
        T1_duration = int((self.minT1*self.minT1_ratio))

        tag_signal_len = len(self.preambleMask)*(self.samplePerBit/2)

        for i in range(-self.preambleRange, self.preambleRange+1):
            maskValue = self.getMaskedValue(self.samples, i + T1_duration, self.preambleMask, maskValue)

            if abs(bestValue) < abs(maskValue):
                bestIdx = i
                bestValue = maskValue

        return bestIdx + T1_duration, bestValue/tag_signal_len

    def decode(self, expectedBit):
        shift_size = int(self.samplePerBit*0.2)
        start_idx, self.tagAmp = self.findPreamble()

        preamble_threshold = abs(self.getNoiseStd())*0.407
        if abs(self.tagAmp) < preamble_threshold:
            return -1

        cur_center = int(start_idx + (len(self.preambleMask)-1)*self.samplePerBit/2)
        
        bits = []

        for bit_idx in range(expectedBit):
            bestShift = -shift_size
            bestValue = 0.0
            cur_value_0 = None
            cur_value_1 = None
            cur_bit = 0
            bit_updown = 1
            for shift in range(-shift_size, shift_size+1):
                cur_value_0 = self.getMaskedValue(self.samples, cur_center + shift, self.zeroMask, cur_value_0)
                cur_value_1 = self.getMaskedValue(self.samples, cur_center + shift, self.oneMask, cur_value_1)

                if abs(cur_value_0) > bestValue:
                    bestShift = shift
                    bestValue = abs(cur_value_0)
                    cur_bit = 0

                if abs(cur_value_1) > bestValue:
                    bestShift = shift
                    bestValue = abs(cur_value_1)
                    cur_bit = 1

            bits.append(cur_bit)
            cur_center += (bestShift + self.samplePerBit)
            self.shiftingLog.append(bestShift)
        return bits
        

def subSampling(samples, subSampleRate, shift):
    subSamples = []
    for i in range(shift, len(samples), subSampleRate):
        subSamples.append(samples[i])

    return subSamples


def gaussianNoiseAdder(samples, noiseStd):
    if np.iscomplex(noiseStd):
        noiseVal_real = normal(0, noiseStd.real, len(samples))
        noiseVal_imag = normal(0, noiseStd.imag, len(samples))
    else:
        noiseVal_real = normal(0, noiseStd, len(samples))
        noiseVal_imag = normal(0, noiseStd, len(samples))
    
    noiseVal = (noiseVal_real + noiseVal_imag*1j).view(np.complex128).reshape(len(samples),)
    noiseAddedSample = np.array(samples) + noiseVal

    return noiseAddedSample.tolist()

from random import random

def testProc(filepath_list : list, q, dataPath : str, sub_sample_ratio=1, sampling_rate=400e3, noise_coef=None):
    sub_result = []
    for filepath in filepath_list:
        gateFilePath = dataPath + "/" + filepath
        reader = IQreader(gateFilePath)
        if reader.eof() or reader.getTotalSize()<=0.0:
            continue
        else:
            stream = reader.read()

            for shift in range(sub_sample_ratio):
                subStream = subSampling(stream, sub_sample_ratio, shift)

                if noise_coef != None:
                    noise = complex(noise_coef*random(), noise_coef*random())
                    subStream = gaussianNoiseAdder(subStream, noise)

                decoder = tagDecoder(subStream, sampling_rate/sub_sample_ratio, 40e3)

                tag_corr = decoder.getTagCorr()

                noise_signal = decoder.getNoiseStd()

                rn16_list = decoder.decode(16)
                rn16 = 0
                if rn16_list == -1:
                    rn16 = -1
                else:
                    for bit in rn16_list:
                        rn16 *= 2
                        rn16 += bit

                sub_result.append((filepath, shift, tag_corr, noise_signal, rn16))#, start_idx])
    q.put(sub_result)

import glob
from typing import Dict, List, Tuple

def process_gate_data(gatePath : str, sub_sample_rate=1, sampling_rate=400e3, noise_coef=None) -> Dict[Tuple[int, int], Tuple[complex, complex, int]]:
    file_list = listdir(gatePath)

    rn16_file_list = [file for file in file_list if not (file.endswith("_EPC") or file.startswith("fail"))]

    result = []
    
    q = Queue()

    num_worker = 32
            
    divided_list = list(np.array_split(rn16_file_list, num_worker))
    decodeProc = []

    for i in range(num_worker):
        p = Proc(target=testProc, args=(divided_list[i], q, gatePath, sub_sample_rate, sampling_rate, noise_coef))
        decodeProc.append(p)

    for p in decodeProc:
        p.start()

    for i in range(num_worker):
        result += q.get()

    for p in decodeProc:
        p.join()

    result_dict = {}

    for result_data in result:
        result_dict[(int(result_data[0]), result_data[1])] = result_data[2:]

    return result_dict


from complex_reader.IqDataReader import IqDataReader as IQreader
import sys

def main():
    dataPath = sys.argv[1]
    dir_list = listdir(dataPath)

    # extract only gate dir
    gate_dir_list = [dir for dir in dir_list if dir.endswith("gate")]

    for gate_dir_name in gate_dir_list:
        print(gate_dir_name)

        file_list = listdir(dataPath +"/" + gate_dir_name + "/")

        rn16_file_list = [file for file in file_list if not (file.endswith("_EPC") or file.startswith("fail"))]
        #print(rn16_file_list)

        result = []
        
        q = Queue()

        num_worker = 1
               
        divided_list = list(np.array_split(rn16_file_list, num_worker))
        decodeProc = []

        for i in range(num_worker):
            p = Proc(target=testProc, args=(divided_list[i], q, dataPath+'/'+gate_dir_name))
            decodeProc.append(p)

        for p in decodeProc:
            p.start()

        for i in range(num_worker):
            result += q.get()

        for p in decodeProc:
            p.join()

        with open("result/"+gate_dir_name+".csv", "w") as result_file:
            csv_writer = csv.writer(result_file)
            csv_writer.writerows(result)

    return          


if __name__ == "__main__":
    main()

