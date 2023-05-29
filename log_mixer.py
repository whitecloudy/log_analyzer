import numpy as np
from math import pi
import sys
import cmath
import csv
import tagDecoder
from typing import List, Dict

class parsed_data:
    def __init__(self):
        self.phase = list()
        self.tag_corr = complex(0.0, 0.0)
        self.noise = complex(0.0, 0.0)
        self.rn16 = 0
        self.round = 0
        self.set_number = 0
        self.filename = []
        self.sub_sampling_shift = 0

    def to_list(self):
        result_list = []
        result_list += self.phase
        for i in range(len(self.filename)):
            result_list.append(self.filename[i])
        
        for i in range(4 - len(self.filename)):
            result_list.append(0)

        result_list.append(self.round)
        result_list += [self.noise.real, self.noise.imag]
        result_list += [self.tag_corr.real, self.tag_corr.imag]

        return result_list


def row_parser(row) -> parsed_data:
    data = parsed_data()

    phase = row[0:6]
    for i, p in enumerate(phase):
        phase[i] = int(p)

    data.phase = phase

    data.tag_corr = complex(float(row[9]), float(row[10]))
    data.noise = complex(float(row[13]), float(row[14]))
    data.rn16 = row[15]
    if data.rn16 == ' -':
        data.rn16 = -1
    else:
        data.rn16 = int(data.rn16)
    data.round = int(row[16])
    data.set_number = int(row[18])

    return data


def log_file_reader(filename) -> List[parsed_data]:
    datas = []
    with open(filename, "r") as log_file:
        csv_reader = csv.reader(log_file)
        first = True
        for row in csv_reader:
            if first:
                first = False
                continue
            datas.append(row_parser(row))

    return datas


def prefix_parser(file_addr: str) -> str:
    filename = file_addr.split('/')[-4:]
    prefix = filename[-1].replace('_log', '')
    prefix = prefix.replace('_gate', '')
    prefix = prefix.replace('.csv', '')
    filename[-1] = prefix
    return filename

def gate2log_path(file_path: str):
    return file_path.replace('gate', 'log.csv')

def file_path_parser(file_path : str):
    filename = file_path.split('/')[-1]
    split_filename = filename.split('_')[0:-1]

    return split_filename


from os import listdir
from os.path import isdir
from glob import glob

def find_gate_dir(dataPath : list):
    path_list = glob(dataPath+"/*")
    result = []
    for path in path_list:
        if isdir(path):
            if path.endswith("gate"):
                result.append(path)
            else:
                result += find_gate_dir(path)
    
    return result


def __main__():
    result_filename = "result/"+sys.argv[1]
    dataPath_l = sys.argv[2:]

    result_list = []
    accurate_list = []

    for dataPath in dataPath_l:
        gate_dir_list = find_gate_dir(dataPath)

        for gate_dir in gate_dir_list:
            print(prefix_parser(gate_dir))
            log_path = gate2log_path(gate_dir)
            
            gate_data = tagDecoder.process_gate_data(gate_dir, 5, 2e6)

            log_data_list = log_file_reader(log_path)

            log_data_dict : Dict[int, parsed_data] = {data.round : data for data in log_data_list}

            for round, shift in gate_data:
                corr, noise, rn16 = gate_data[(round, shift)]
                if rn16 != -1 and round in log_data_dict:
                    log_data = log_data_dict[round]
                    log_data.tag_corr = corr
                    log_data.noise = noise
                    log_data.sub_sampling_shift = shift
                    log_data.filename = prefix_parser(gate_dir)
                    log_data.rn16 = rn16

                    result_list.append(log_data.to_list())

            gate_data = tagDecoder.process_gate_data(gate_dir, 1, 2e6)

            for round, shift in gate_data:
                corr, noise, rn16 = gate_data[(round, shift)]
                if rn16 == 21845 and round in log_data_dict:
                    log_data = log_data_dict[round]
                    log_data.tag_corr = corr
                    log_data.noise = noise
                    log_data.sub_sampling_shift = shift
                    log_data.filename = prefix_parser(gate_dir)
                    log_data.rn16 = rn16

                    accurate_list.append(log_data.to_list())


    first_row = ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "key1", "key2", "key3", "key4", "round", "noise real", "noise imag", "corr real", "corr imag"]

    with open(result_filename+"_nosub.csv", 'w') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(first_row)
        log_writer.writerows(accurate_list)

    with open(result_filename+"_result.csv", 'w') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(first_row)
        log_writer.writerows(result_list)


            

            


if __name__=="__main__":
    __main__()