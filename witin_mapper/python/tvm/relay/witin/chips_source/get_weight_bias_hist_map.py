#coding=utf8

import os
import argparse
import shutil
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

def witin_hist_map(data, title_name, hist_name):

    if(len(data.shape) > 1):
        data = data.reshape(-1)
    elif(len(data.shape) == 0):
        return

    plt.hist(data, 30)
    plt.xlabel('data')
    plt.ylabel('frequency')
    plt.title(title_name)

    plt.savefig(hist_name)
    plt.close()

#original_path = './output/params'
#target_path = './output/map/weight_hist'

def witin_get_weight_bias_hist_map(original_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    else:
        shutil.rmtree(target_path)
        os.makedirs(target_path)

    if not os.path.exists(original_path):
        #raise("Error!!! Please check the output folder of build folder carefully!!!")
        return

    else:
        sub_original_files = os.listdir(original_path)
        for sub_file in sub_original_files:
            sub_abs_original_file = os.path.join(original_path, sub_file)
            sub_abs_target_path = os.path.join(target_path, sub_file)

            if not os.path.exists(sub_abs_target_path):
                os.makedirs(sub_abs_target_path)

            for root, dirs, filenames in os.walk(sub_abs_original_file):

                for name in filenames:

                    file_name = os.path.join(root, name)
                    file_path = os.path.split(file_name)[0]
                    file_sub_path= os.path.split(file_path)[1]
                    target_file_path = os.path.join(sub_abs_target_path, file_sub_path)

                    if(('weight' in file_name) | ('bias' in file_name)):
                        if not os.path.exists(target_file_path):
                            os.makedirs(target_file_path)

                        if('weight' in file_name):
                            weight_data = np.loadtxt(file_name, dtype=int)
                            weight_title_name = file_sub_path + "_weight"
                            weight_hist_name = os.path.join(target_file_path, os.path.splitext(name)[0]+'.jpg')
                            witin_hist_map(weight_data,weight_title_name,weight_hist_name)

                        elif('bias' in file_name):
                            bias_data_orig = np.loadtxt(file_name, dtype=int)
                            bias_data = np.sum(bias_data_orig,0) * 128
                            bias_title_name = file_sub_path + "_bias"
                            bias_hist_name = os.path.join(target_file_path, os.path.splitext(name)[0]+'.jpg')
                            witin_hist_map(bias_data, bias_title_name, bias_hist_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_bias_file_path',
                        type=str,
                        default='./output/params',
                        help='Output directory for weight files and bias files.')
    parser.add_argument('--weight_bias_hist_path',
                        type=str,
                        default='./output/map/weight_hist',
                        help='Output directory for weight histograms and bias histograms.')
    args, unparsed = parser.parse_known_args()
    witin_get_weight_bias_hist_map(args.weight_bias_file_path, args.weight_bias_hist_path)
    print('((Created),', '(The weight histograms and bias histograms is located in \''+args.weight_bias_hist_path+'\'!))')

#'''







