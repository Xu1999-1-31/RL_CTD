import re
import numpy as np

def ReadRouteCongestion(inrpt, scale):
    H_congestion = np.zeros((scale, scale))
    V_congestion = np.zeros((scale, scale))
    with open(inrpt, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            if line.startswith('##'):
                row = int(re.findall(r'\d+', line.split()[5])[0])
                column = int(re.findall(r'\d+', line.split()[6])[0])
            if line.startswith('H routing'):
                H_congestion[row][column] = int(line.split()[3])
            if line.startswith('V routing'):
                V_congestion[row][column] = int(line.split()[3])
    
    return H_congestion[::-1, :], V_congestion[::-1, :]