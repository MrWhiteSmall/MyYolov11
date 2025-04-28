# (yolo11) ❰root|yolo11❱ find / -name "libnvinfer.so*" 2>/dev/null

# /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvinfer.so
# /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvinfer.so.8
# /usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs/libnvinfer.so
# /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvinfer.so.8.5.1
# /root/TensorRT-8.5.1.7/targets/x86_64-linux-gnu/lib/libnvinfer.so
# /root/TensorRT-8.5.1.7/targets/x86_64-linux-gnu/lib/libnvinfer.so.8
# /root/TensorRT-8.5.1.7/targets/x86_64-linux-gnu/lib/stubs/libnvinfer.so
# /root/TensorRT-8.5.1.7/targets/x86_64-linux-gnu/lib/libnvinfer.so.8.5.1

try:
    import tensorrt as trt
    print("TensorRT is installed.")
except ImportError:
    print("TensorRT is not installed.")
