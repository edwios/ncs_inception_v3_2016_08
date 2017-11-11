#! /usr/bin/env python3

# Copyright 2017 Intel Corporation. 
# The source code, information and material ("Material") contained herein is  
# owned by Intel Corporation or its suppliers or licensors, and title to such  
# Material remains with Intel Corporation or its suppliers or licensors.  
# The Material contains proprietary information of Intel or its suppliers and  
# licensors. The Material is protected by worldwide copyright laws and treaty  
# provisions.  
# No part of the Material may be used, copied, reproduced, modified, published,  
# uploaded, posted, transmitted, distributed or disclosed in any way without  
# Intel's prior express written permission. No license under any patent,  
# copyright or other intellectual property rights in the Material is granted to  
# or conferred upon you, either expressly, by implication, inducement, estoppel  
# or otherwise.  
# Any license under such intellectual property rights must be express and  
# approved by Intel in writing.

from mvnc import mvncapi as mvnc
import sys
import time
import numpy
import cv2
import csv

path_to_networks = './'
path_to_images = '../../data/images/'
graph_filename = 'graph'
image_filename = path_to_images + 'cat.jpg'
lookuptable = {}

#mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

#Load graph
with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()

#Load preprocessing data
mean = 128
std = 1/128

#Load categories
categories = []
with open(path_to_networks + 'classes.txt', 'r') as f:
    for line in f:
        cat = line.split('\n')[0]
        if cat != 'classes':
            categories.append(cat)
    f.close()
    print('Number of categories:', len(categories))

with open(path_to_networks + 'dict.csv', 'r') as f:
	for line in f:
		words = [word.strip(' "\n') for word in line.split(',', 1)]
		lookuptable[words[0]] = words[1]
	f.close()
	print('Number of classes:', len(lookuptable), lookuptable.get('/m/0100nhbf'))

#Load image size
with open(path_to_networks + 'inputsize.txt', 'r') as f:
    reqsize = int(f.readline().split('\n')[0])

graph = device.AllocateGraph(graphfile)

img = cv2.imread(image_filename).astype(numpy.float32)
dx,dy,dz= img.shape
delta=float(abs(dy-dx))
if dx > dy: #crop the x dimension
    img=img[int(0.5*delta):dx-int(0.5*delta),0:dy]
else:
    img=img[0:dx,int(0.5*delta):dy-int(0.5*delta)]
img = cv2.resize(img, (reqsize, reqsize))

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

for i in range(3):
    img[:,:,i] = (img[:,:,i] - mean) * std

print('Start download to NCS...')
graph.LoadTensor(img.astype(numpy.float16), 'user object')
starttime=time.time()
output, userobj = graph.GetResult()
stoptime=time.time()
graph.DeallocateGraph()
device.CloseDevice()
print('{0:0.3f} ms used'.format((stoptime-starttime)*1000))

top_inds = output.argsort()[::-1][:9]

print(''.join(['*' for i in range(79)]))
print('inception-v3 on NCS')
print(''.join(['*' for i in range(79)]))
print('Number of classes:', len(lookuptable), lookuptable.get('/m/0100nhbf'))
for i in range(9):
	v = categories[top_inds[i]]
	dispname = lookuptable.get(v, 'unknown')
	print(top_inds[i], v, dispname, output[top_inds[i]])

print(''.join(['*' for i in range(79)]))

print('Finished')
