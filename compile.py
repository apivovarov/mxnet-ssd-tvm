#!/usr/bin/python3
import os
import zipfile
import tvm
import nnvm
import mxnet as mx
import numpy as np

from nnvm import compiler
from nnvm.frontend import from_mxnet
from tvm import relay
from tvm.contrib.download import download
from mxnet.model import load_checkpoint


dshape = (1, 3, 512, 512)
#dshape = (1, 3, 300, 300)
#dshape = (1, 3, 608, 608)
dtype = "float32"

# Target settings
# Use these commented settings to build for cuda.
#target = 'cuda'
# Use these commented settings to build for opencl.
#target = 'opencl'
#target = "llvm"
#target = tvm.target.arm_cpu('rasp3b')
#target = tvm.target.intel_graphics()
#target = tvm.target.mali('rk3399')
target = tvm.target.mali()

#target_host = 'llvm -target=armv7l-linux-gnueabihf'
target_host = 'llvm -target=aarch64-linux-gnu'
#target_host = 'llvm'
#target_host = None

######################################################################
# Convert and compile model with NNVM or Relay for CPU.

#inf_json = "deploy_ssd_resnet50_512/deploy_ssd_resnet50_512-symbol.json"
#inf_json = "deploy_ssd_inceptionv3_512/deploy_ssd_inceptionv3_512-symbol.json"
#inf_json = "deploy_ssd_vgg16_reduced_512/deploy_ssd_vgg16_reduced_512-symbol.json"
#inf_json = "deploy_ssd_vgg16_reduced_300/deploy_ssd_vgg16_reduced_300-symbol.json"
inf_json = "deploy_ssd_mobilenet_512/deploy_ssd_mobilenet_512-symbol.json"
#inf_json = "deploy_ssd_mobilenet_608/deploy_ssd_mobilenet_608-symbol.json"
print("mx.sym.load: " + inf_json)
sym = mx.sym.load(inf_json)

#checkp = "deploy_ssd_resnet50_512/deploy_ssd_resnet50_512"
#checkp = "deploy_ssd_inceptionv3_512/deploy_ssd_inceptionv3_512"
#checkp = "deploy_ssd_vgg16_reduced_512/deploy_ssd_vgg16_reduced_512"
#checkp = "deploy_ssd_vgg16_reduced_300/deploy_ssd_vgg16_reduced_300"
checkp = "deploy_ssd_mobilenet_512/deploy_ssd_mobilenet_512"
#checkp = "deploy_ssd_mobilenet_608/deploy_ssd_mobilenet_608"
print("load_checkpoint: " + checkp)
_, arg_params, aux_params = load_checkpoint(checkp, 0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--frontend",
    help="Frontend for compilation, nnvm or relay",
    type=str,
    default="nnvm")
args = parser.parse_args()
if args.frontend == "relay":
    net, params = relay.frontend.from_mxnet(sym, {"data": dshape}, arg_params=arg_params, \
                                            aux_params=aux_params)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(net, target, params=params, target_host=target_host)
elif args.frontend == "nnvm":
    net, params = from_mxnet(sym, arg_params, aux_params)
    with compiler.build_config(opt_level=3):
        graph, lib, params = compiler.build(
            net, target, {"data": dshape}, params=params, target_host=target_host)
else:
    parser.print_help()
    parser.exit()

print("Saving files")
# save the graph, lib and params into separate files
path_lib = "model.so"
#lib.export_library(path_lib)
#lib.export_library(path_lib, cc="arm-linux-gnueabihf-g++")
lib.export_library(path_lib, cc="aarch64-linux-gnu-g++")
with open("model.json", "w") as fo:
    fo.write(graph.json())
with open("model.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
print("Files saved")
