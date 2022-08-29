from IqDataReader import IqDataReader as IQreader
from os import listdir
from multiprocessing import Process as Proc
from multiprocessing import Queue
import numpy as np
from numpy.random import normal

class tagDecoder:
    def __init__(self, sampleStream, sampleRate, dataRate):
        self.samples = sampleStream
        self.sampleRate = sampleRate
        self.dataRate = dataRate
        self.samplePerBit = sampleRate/float(dataRate)
        self.preambleMask = [1,1,-1,1,-1,-1,1,-1,-1,-1,1,1]
        self.oneMask = [-1,1,1,-1]
        self.zeroMask = [-1,1,-1,1]
        self.minT1 = int(0.0001 * sampleRate)
        self.preambleRange = int(self.minT1/2)
        self.shiftingLog = []

    def getMaskedValue(self, samples, cur_index, mask, prev_value = complex(0.0,0.0)):
        halfBitSamples = self.samplePerBit/2
        cur_value = prev_value
        for idx_m in range(len(mask)):
            startIdx = int(cur_index + idx_m * halfBitSamples)
            endIdx = int(startIdx + halfBitSamples)

            if prev_value == complex(0.0,0.0):
                for sample in samples[startIdx: endIdx]:
                    cur_value += mask[idx_m] * sample
            else:
                cur_value += (mask[idx_m]*samples[endIdx - 1] - mask[idx_m]*samples[startIdx - 1])

        return cur_value


    def findPreamble(self):
        maskValue = complex(0.0,0.0)
        bestIdx = 0
        bestValue = 0.0

        for i in range(self.preambleRange):
            maskValue = self.getMaskedValue(self.samples, i + self.minT1, self.preambleMask, maskValue)

            if bestValue < abs(maskValue):
                bestIdx = i
                bestValue = abs(maskValue)

        return bestIdx + self.minT1


    def decode(self, expectedBit):
        shift_size = int(self.samplePerBit/4)
        cur_center = self.findPreamble() + (len(self.preambleMask)-1)*self.samplePerBit/2
        bits = []

        for bit_idx in range(expectedBit):
            bestShift = -shift_size
            bestValue = 0.0
            cur_value_0 = complex(0.0, 0.0)
            cur_value_1 = complex(0.0, 0.0)
            cur_bit = 0

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
    noiseVal = normal(0, noiseStd, (len(samples),2)).view(np.complex128).reshape(len(samples),)
    noiseAddedSample = np.array(samples) + noiseVal

    return noiseAddedSample.tolist()


def testProc(samples, subSampleRate, q, expectedData=21845):
    returnVal = 0.0
    for i in range(subSampleRate):
        subStream = subSampling(samples, subSampleRate, i)
        decoder = tagDecoder(subStream, 2e6/subSampleRate, 40e3)

        bits = decoder.decode(16)
        data = 0

        for bit in bits:
            data *= 2
            data += bit

        if expectedData == data:
            returnVal += 1
    returnVal /= subSampleRate

    q.put([subSampleRate, returnVal])



def main():
    dataPath = "./data/"
    dir_list = listdir(dataPath)


    resultDict = {}
    iteration_step = 10
    print("dir name, fileNumber, Added Noise, 1, 2, 3, 4, 5, 6, 8, 12, 25")

    for dir_name in dir_list:
        file_list = listdir(dataPath + dir_name + "/")
        for gateFile in file_list:
            gateFilePath = dataPath + dir_name + "/" + gateFile
            q = Queue()
            subSampleStep = [1, 2, 3, 4, 5, 6, 8, 12, 25]


            reader = IQreader(gateFilePath)
            if reader.eof():
                continue
            else:
                stream = reader.read(1750)

            
            for noiseStep in range(10, 40, 5):
                decodeSuccessDict = {}

                for sub in subSampleStep:
                    decodeSuccessDict[sub] = 0.0

                for i in range(iteration_step):
                    noise = noiseStep * 0.00001
                    noisyStream = gaussianNoiseAdder(stream, noise)

                    decodeProc = []


                    for subSampleRate in subSampleStep:
                        p = Proc(target=testProc, args=(noisyStream, subSampleRate, q))
                        decodeProc.append(p)

                    for p in decodeProc:
                        p.start()

                    for p in decodeProc:
                        p.join()
                        result = q.get()
                        decodeSuccessDict[result[0]] += result[1]

                for sub in subSampleStep:
                    decodeSuccessDict[sub] /= iteration_step

                print(dir_name, ", ", gateFile,", ", noiseStep,", ", decodeSuccessDict[1],", ", decodeSuccessDict[2],", ", decodeSuccessDict[3],", ", decodeSuccessDict[4],", ", decodeSuccessDict[5],", ", decodeSuccessDict[6],", ", decodeSuccessDict[8],", ", decodeSuccessDict[12],", ", decodeSuccessDict[25])


        


if __name__ == "__main__":
    main()

