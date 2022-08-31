from re import search
import numpy as np
from crccheck.crc import Crc5EpcC1G2

pw_len = 24e-6
delim_len = 12e-6
rtcal_len = 144e-6
trcal_len = 200e-6

avg_amp_refresh_rate = 0.95
down_threshold_ratio = 0.3
up_threshold_ratio = 0.7

class readerDecoder:
    def __init__(self, sampleRate):
        self.sampleRate = sampleRate
        self.sampleStream = None
        self.cur_idx = 0
        self.avg_amp = 0.0
        
        self.pw_d = pw_len * self.sampleRate
        self.delim_d = delim_len * self.sampleRate
        self.rtcal_d = rtcal_len * self.sampleRate
        self.trcal_d = trcal_len * self.sampleRate

        self.query_bit_size = 21 - 5
        self.crc_bit_size = 5


    def find_query(self, sampleStream):
        self.sampleStream = sampleStream
        avg_amp = sampleStream[0]

        for i in range(len(sampleStream)-10-5000):
            sample = sampleStream[i+10]

            if abs(sample) < (abs(avg_amp) * down_threshold_ratio):
                vote = 1
                for forward in range(1, 5):
                    if abs(sampleStream[i+10+forward]) < (abs(avg_amp) * down_threshold_ratio):
                        vote += 1
                
                if vote >= 4:
                    #we found down pulse by voting
                    self.avg_amp = avg_amp
                    return self.decode_preamble(i, avg_amp, 21)
                    
            sample = sampleStream[i]
            avg_amp = avg_amp * (1 - avg_amp_refresh_rate) + sample * avg_amp_refresh_rate

        return -1, len(sampleStream)-10-3600

    def detect_down(self, cur_idx, search_len):
        search_len = int(search_len)
        for i in range(cur_idx, cur_idx+search_len):
            if abs(self.sampleStream[i]) < (self.avg_amp * down_threshold_ratio):
                return i
        return -1
    
    def detect_up(self, cur_idx, search_len):
        search_len = int(search_len)
        for i in range(cur_idx, cur_idx+search_len):
            if abs(self.sampleStream[i]) > (self.avg_amp * up_threshold_ratio):
                return i
        return -1

    def check_length(self, length, expected_len, tolerance):
        if length >= expected_len * (1 - tolerance) and length <= expected_len * (1 + tolerance):
            return True
        else:
            return False

    def check_pulse(self, up_pulse_len, down_pulse_len, up_tolerance, down_tolerance=-1):
        cur_idx = self.cur_idx
        if down_tolerance == -1:
            down_tolerance = up_tolerance
        down_idx = self.detect_down(cur_idx, up_pulse_len * (1+up_tolerance*2))  #down edge
        #check up pulse
        if down_idx == -1:
            return -1, cur_idx
        else:
            length = down_idx - cur_idx
            if self.check_length(length, up_pulse_len, up_tolerance):
                pass
            else:
                return -1, down_idx
        
        cur_idx = down_idx 

        up_idx = self.detect_up(cur_idx, down_pulse_len * (1+down_tolerance*2))  #up edge
        #check down pulse
        if up_idx == -1:
            return -1, cur_idx
        else:
            length = up_idx - cur_idx
            if self.check_length(length, down_pulse_len, down_tolerance):
                pass
            else:
                return -1, up_idx
        
        cur_idx = up_idx 
        return 0, cur_idx
    
    def decode_preamble(self, start_idx, avg_amp, expected_bit):
        avg_amp = abs(avg_amp)
        self.cur_idx = start_idx

        ######################################################
        #find delimiter
        self.cur_idx = self.detect_down(self.cur_idx, 100)  #down edge
        up_idx = self.detect_up(self.cur_idx, self.delim_d * 1.05) #up edge

        #check down pulse
        if up_idx == -1:
            return -1, self.cur_idx
        else:
            length = up_idx - self.cur_idx
            if self.check_length(length, self.delim_d, 0.05):
                pass
            else:
                #print("12.5us fail")
                return -1, self.cur_idx

        self.cur_idx = up_idx        

        #find data_0
        result, idx = self.check_pulse(self.pw_d, self.pw_d, 0.05)
        self.cur_idx = idx
        if result != 0:
            #print("data_0 fail")
            return -1, self.cur_idx

        #find RT_cal
        result, idx = self.check_pulse(self.rtcal_d - self.pw_d, self.pw_d, 0.1)
        self.cur_idx = idx
        if result != 0:
            #print("rtcal fail")
            return -1, self.cur_idx

        #find TR_cal
        result, idx = self.check_pulse(self.trcal_d - self.pw_d, self.pw_d, 0.1)
        self.cur_idx = idx
        if result != 0:
            #print("trcal fail")
            return -1, self.cur_idx

        #######################################################
        #decoding phase
        data = 0
        for i in range(expected_bit):
            data = data << 1
            data_0, idx_0 = self.check_pulse(self.pw_d, self.pw_d, 0.1)
            data_1, idx_1 = self.check_pulse(self.pw_d*3, self.pw_d, 0.1)

            if data_0 == -1 and data_1 == -1:
                #print("data part fail : ", i)
                self.cur_idx = idx_1
                return -1, self.cur_idx
            else:
                if data_1 == 0:
                    self.cur_idx = idx_1
                    data += 1
                elif data_0 == 0:
                    self.cur_idx = idx_0

        return data, self.cur_idx

