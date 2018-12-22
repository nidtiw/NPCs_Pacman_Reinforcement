import re
import matplotlib.pyplot as plt
from tqdm import tqdm  
import argparse
import numpy as np

def process_line(line):
    data = line.split("|")
    js =  {
        'index': data[0][1::].strip(),
    }
    for each in data[1::]:
        x, y = each.split(':')
        js[x.strip()] = y.strip()
    return js

def process_file(fname):
    data = open(fname, 'r').read()
    log_data = []
    for each in tqdm(data.split("\n")):
        log_data.append(process_line(each))
    return log_data

if __name__ == "__main__":
    # file_name = "ghost-2Thu_13_Dec_2018_16_09_28-l-8-m-7-x-400.log"
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',  type=str, help='log file')
    args = parser.parse_args()
    file_name = args.file
    processed = process_file(file_name)
    Q = []
    Q_win = []
    Q_loss = []
    itr = []
    for i in range(len(processed)):
        if processed[i]['Q'] == 'nan':
            qtemp = 0.0
        else:
            qtemp = float(processed[i]['Q'])
        
        if (processed[i]['won']) == 'True':
            Q_win.append(qtemp)
        else:
            Q_loss.append(qtemp)

        Q.append(qtemp)
        itr.append(processed[i]['index'])
    
    print("Average Q: ")
    print(np.mean(np.asarray(Q)))
    print("Average Q (Win): ")
    print(np.mean(np.asarray(Q_win)))
    print("Average Q (Loss): ")
    print(np.mean(np.asarray(Q_loss)))
    print("Win % : ")
    print(len(Q_win)/len(Q))
    print("Loss % : ")
    print(len(Q_loss)/len(Q))
    plt.plot(np.asarray(Q), 'b+')
    plt.xlabel('Iteration')
    plt.ylabel('Q Value ')
    plt.savefig((file_name.replace('.log','.jpg')))

    # print(itr)
        

