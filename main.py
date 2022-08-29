from complex_reader.IqDataReader import IqDataReader
from glob import glob
import sys

def __main__():
    for filename in sys.argv[1:]:
        iq_reader = IqDataReader(filename)
        


if __name__=="__main__":
    __main__()