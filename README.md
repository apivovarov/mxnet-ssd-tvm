# mxnet-ssd-tvm

To Compile mxnet-ssd ssd_mobilenet_512 model
```
python3 compile.py
```

To Run on ARMv8 Mali gpu (e.g. RK3399)
```
python3 run-ssd.py
```

Error
```
python3 run-ssd.py 
File dog.jpg exists, skip.
File dog.jpg.512.npy exists, skip image preprocessing.
Traceback (most recent call last):
  File "run-ssd.py", line 69, in <module>
    m.run(data = input_data)
  File "/usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/contrib/graph_runtime.py", line 151, in run
    self._run()
  File "/usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/_ffi/_ctypes/function.py", line 185, in __call__
    ctypes.byref(ret_val), ctypes.byref(ret_tcode)))
  File "/usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/_ffi/base.py", line 71, in check_call
    raise TVMError(py_str(_LIB.TVMGetLastError()))
tvm._ffi.base.TVMError: [04:53:18] /home/firefly/tvm/src/runtime/module_util.cc:53: Check failed: ret == 0 (-1 vs. 0) [04:53:18] /home/firefly/tvm/src/runtime/opencl/opencl_module.cc:63: Check failed: e == CL_SUCCESS OpenCL Error, code=-5: CL_OUT_OF_RESOURCES

Stack trace returned 10 entries:
[bt] (0) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x105d4) [0x7f9a3425d4]
[bt] (1) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x10f60) [0x7f9a342f60]
[bt] (2) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x8ef64) [0x7f9a3c0f64]
[bt] (3) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x8f170) [0x7f9a3c1170]
[bt] (4) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(TVMFuncCall+0x74) [0x7f9a345f9c]
[bt] (5) ./model.so(+0x22aa4) [0x7f98044aa4]
[bt] (6) ./model.so(fuse_nms+0x298) [0x7f98044704]
[bt] (7) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x349ac) [0x7f9a3669ac]
[bt] (8) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x6f3c8) [0x7f9a3a13c8]
[bt] (9) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x6d5c8) [0x7f9a39f5c8]



Stack trace returned 8 entries:
[bt] (0) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x105d4) [0x7f9a3425d4]
[bt] (1) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x34dc8) [0x7f9a366dc8]
[bt] (2) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x6f3c8) [0x7f9a3a13c8]
[bt] (3) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x6d5c8) [0x7f9a39f5c8]
[bt] (4) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(TVMFuncCall+0x74) [0x7f9a345f9c]
[bt] (5) /usr/lib/python3.5/lib-dynload/_ctypes.cpython-35m-aarch64-linux-gnu.so(ffi_call_SYSV+0x64) [0x7fa3088980]
[bt] (6) /usr/lib/python3.5/lib-dynload/_ctypes.cpython-35m-aarch64-linux-gnu.so(ffi_call+0xc0) [0x7fa30881e0]
[bt] (7) /usr/lib/python3.5/lib-dynload/_ctypes.cpython-35m-aarch64-linux-gnu.so(_ctypes_callproc+0x448) [0x7fa3080f68]


terminate called after throwing an instance of 'dmlc::Error'
  what():  [04:53:19] /home/firefly/tvm/src/runtime/workspace_pool.cc:96: Check failed: allocated_.size() == 1 (3 vs. 1) 

Stack trace returned 5 entries:
[bt] (0) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x105d4) [0x7f9a3425d4]
[bt] (1) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(tvm::runtime::WorkspacePool::Pool::Release(DLContext, tvm::runtime::DeviceAPI*)+0x470) [0x7f9a37b9f8]
[bt] (2) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(tvm::runtime::WorkspacePool::~WorkspacePool()+0x48) [0x7f9a37a5e0]
[bt] (3) /usr/local/lib/python3.5/dist-packages/tvm-0.6.dev0-py3.5-linux-aarch64.egg/tvm/libtvm_runtime.so(+0x84908) [0x7f9a3b6908]
[bt] (4) /lib/aarch64-linux-gnu/libc.so.6(__call_tls_dtors+0x44) [0x7fa345e474]

```
