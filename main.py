from csv import reader
from complex_reader.IqDataReader import IqDataReader
from readerDecoder import readerDecoder
from glob import glob
import numpy as np
import sys

def __main__():
    for filename in sys.argv[1:]:
        iq_reader = IqDataReader(filename)
        reader_decoder = readerDecoder(2e6)

        idx = 0

        if iq_reader.eof():
            continue
        else:
            stream = np.array(iq_reader.read(5000))
            idx += 5000
            avg_dc = np.mean(stream)

            stream = np.array(iq_reader.read(20000) - avg_dc)

            for i in range(20000):
                if abs(stream[i]) > 0.1:
                    stream = stream[i:]
                    idx += i
                    break
            
            while not iq_reader.eof():
                if len(stream) < 25000:
                    stream = np.append(stream, np.array(iq_reader.read(50000))- avg_dc)
                #print(idx)

                result, cur_idx = reader_decoder.find_query(stream)

                stream = stream[cur_idx:]
                idx += cur_idx

                if result == -1:
                    pass
                else:
                    print(result>>5 & 0x3fff)


if __name__=="__main__":
    __main__()