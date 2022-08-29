import numpy as np
from crccheck.crc import Crc5EpcC1G2

pw_len = 24e-6
delim_len = 12e-6
rtcal_len = 144e-6
trcal_len = 200e-6

avg_amp_refresh_rate = 0.9
down_threshold_ratio = 0.3
up_threshold_ratio = 0.7

class readerDecoder:
    def __init__(self, sampleRate):
        self.sampleRate = sampleRate
        
        self.pw_d = pw_len * self.sampleRate
        self.delim_d = delim_len * self.sampleRate
        self.rtcal_d = rtcal_len * self.sampleRate
        self.trcal_d = trcal_len * self.sampleRate

        self.query_bit_size = 21 - 5
        self.crc_bit_size = 5


    def find_query(self, sampleStream):
        avg_amp = sampleStream[0]

        for i in range(0, len(sampleStream-10-3600)):
            sample = sampleStream[i+10]

            if abs(sample) < abs(avg_amp):
                vote = 1
                for forward in range(1, 5):
                    if abs(sampleStream[i+10+forward]) < (abs(avg_amp) * down_threshold_ratio):
                        vote += 1
                
                if vote >= 4:
                    #we found down pulse by voting
                    self.query_decode(sampleStream, i)
                    
            sample = sampleStream[i]
            avg_amp = avg_amp * (1 - avg_amp_refresh_rate) + sample * avg_amp_refresh_rate

    
    def query_decode(self, sampleStream, start_idx, avg_amp):
        avg_amp = abs(avg_amp)
        cur_idx = start_idx

        def detect_down(search_len):
            for i in range(cur_idx, cur_idx+search_len):
                if abs(sampleStream[i]) < (avg_amp * down_threshold_ratio):
                    return i
            return -1
        
        def detect_up(search_len):
            for i in range(cur_idx, cur_idx+search_len):
                if abs(sampleStream[i]) > (avg_amp * up_threshold_ratio):
                    return i
            return -1

        def check_length(length, expected_len, tolerance):
            if length >= expected_len * (1 - tolerance) and length <= expected_len * (1 + tolerance):
                return True
            else:
                return False

        def check_pulse(up_pulse_len, down_pulse_len, up_tolerance, down_tolerance=-1):
            if down_tolerance == -1:
                down_tolerance = up_tolerance
            down_idx = detect_down(up_pulse_len * (1+up_tolerance*2))  #down edge
            #check up pulse
            if down_idx == -1:
                return -1
            else:
                length = down_idx - cur_idx
                if check_length(length, up_pulse_len, up_tolerance):
                    pass
                else:
                    return -1
            
            cur_idx = down_idx 

            up_idx = detect_down(down_pulse_len * (1+down_tolerance*2))  #up edge
            #check down pulse
            if up_idx == -1:
                return -1
            else:
                length = up_idx - cur_idx
                if check_length(length, down_pulse_len, down_tolerance):
                    pass
                else:
                    return -1
            
            cur_idx = up_idx 

            return 0

        ######################################################
        #find delimiter
        cur_idx = detect_down(100)  #down edge
        up_idx = detect_up(self.delim_d * 1.05) #up edge

        #check down pulse
        if up_idx == -1:
            return -1
        else:
            length = up_idx - cur_idx
            if check_length(length, self.delim_d, 0.05):
                pass
            else:
                return -1

        cur_idx = up_idx        

        #find data_0
        if check_pulse(self.pw_d, self.pw_d, 0.05) != 0:
            return -1

        #find RT_cal
        if check_pulse(self.rtcal_d - self.pw_d, self.pw_d, 0.1) != 0:
            return -1

        #find TR_cal
        if check_pulse(self.trcal_d - self.pw_d, self.pw_d, 0.1) != 0:
            return -1

        #######################################################
        #decoding phase
        data = 0
        for i in range(self.query_bit_size):
            data = data << 1
            data_0 = check_pulse(self.pw_d, self.pw_d, 0.1)
            data_1 = check_pulse(self.pw_d*3, self.pw_d, 0.1)

            if data_0 == -1 and data_1 == -1:
                return -1
            else:
                if data_1 == 0 and data_0 == -1:
                    data += 1

        #crc phase
        crc = 0
        for i in range(self.crc_bit_size):
            crc = data << 1
            data_0 = check_pulse(self.pw_d, self.pw_d, 0.1)
            data_1 = check_pulse(self.pw_d*3, self.pw_d, 0.1)

            if data_0 == -1 and data_1 == -1:
                return -1
            else:
                if data_1 == 0 and data_0 == -1:
                    crc += 1

        crc_cal = Crc5EpcC1G2(data)
        if crc_cal != crc:
            return 0
        else:
            data = data << 5
            data += crc

        return data

