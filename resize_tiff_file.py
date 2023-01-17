# This class is a modified version of the code here
# https://www.kaggle.com/code/realneuralnetwork/coatnet-strip-ai-inference

import numpy as np

import tifffile
import cv2
import gc


def resize_image(in_file, out_file):
    new_size = 300
    try:
        sz = os.path.getsize(in_file)
    except:
        sz = 1e9
    if(sz > 8e8):
        print("truncating file", in_file)
        img = np.zeros((new_size,new_size,3), np.uint8)
    else:
        try:
            img = cv2.resize(tifffile.imread(in_file), (new_size, new_size))
        except Exception as e:
            print("can't convert file", in_file, e)
            img = np.zeros((new_size,new_size,3), np.uint8)
    cv2.imwrite(out_file, img)
    del img
    gc.collect()