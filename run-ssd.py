#!/usr/bin/python3

import os
import tvm
import numpy as np
import time

from tvm.contrib.download import download
from tvm.contrib import graph_runtime

current_milli_time = lambda: int(round(time.time() * 1000))

test_image = "dog.jpg"
dshape = (1, 3, 512, 512)
#dshape = (1, 3, 608, 608)
dtype = "float32"
test_image_npy = "{}.{}.npy".format(test_image, dshape[2])

image_url = "https://cloud.githubusercontent.com/assets/3307514/20012567/" \
                    "cbb60336-a27d-11e6-93ff-cbc3f09f5c9e.jpg"

download(image_url, test_image)

# Preprocess image

if os.path.isfile(test_image_npy):
    print("File {} exists, skip image preprocessing.".format(test_image_npy))
    img_data = np.load(test_image_npy)
else:
    import cv2
    test_image_path = test_image
    image = cv2.imread(test_image_path)
    img_data = cv2.resize(image, (dshape[2], dshape[3]))
    img_data = img_data[:, :, (2, 1, 0)].astype(np.float32)
    img_data -= np.array([123, 117, 104])
    img_data = np.transpose(np.array(img_data), (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    np.save(test_image_npy, img_data.astype(dtype))

ctx = tvm.cl()
target="opencl"

#base = "deploy_ssd_resnet50_512/{}/".format(target)
#base = "deploy_ssd_inceptionv3_512/{}/".format(target)
#base = "deploy_ssd_mobilenet_512/{}/".format(target)
#base = "deploy_ssd_mobilenet_608/{}/".format(target)
#base = "cpu-model/"
base = "./"
path_lib = base + "model.so"
path_graph = base + "model.json"
path_param = base + "model.params"

graph = open(path_graph).read()
params = bytearray(open(path_param, "rb").read())
lib = tvm.module.load(path_lib)

class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

######################################################################
# Create TVM runtime and do inference

# Build TVM runtime
m = graph_runtime.create(graph, lib, ctx)
m.load_params(params)
input_data = tvm.nd.array(img_data.astype(dtype))
# dryrun
m.run(data = input_data)
# execute
t1 = current_milli_time()
m.run(data = input_data)
# get outputs
tvm_output = m.get_output(0)
t2 = current_milli_time()
print(base)
print("time: {} ms".format(t2 - t1))
out = tvm_output.asnumpy()[0]
i = 0
for det in out:
    cid = int(det[0])
    if cid < 0:
        continue
    score = det[1]
    if score < 0.5:
         continue
    i += 1

    print(i, class_names[cid], det)

######################################################################
# Display result

def display(img, out, thresh=0.5):
    import random
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['figure.figsize'] = (10, 10)
    pens = dict()
    plt.clf()
    plt.imshow(img)
    for det in out:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        if cid not in pens:
            pens[cid] = (random.random(), random.random(), random.random())
        scales = [img.shape[1], img.shape[0]] * 2
        xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                             edgecolor=pens[cid], linewidth=3)
        plt.gca().add_patch(rect)
        text = class_names[cid]
        plt.gca().text(xmin, ymin-2, '{:s} {:.3f}'.format(text, score),
                       bbox=dict(facecolor=pens[cid], alpha=0.5),
                       fontsize=12, color='white')
    plt.show()

#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#display(image, tvm_output.asnumpy()[0], thresh=0.45)
