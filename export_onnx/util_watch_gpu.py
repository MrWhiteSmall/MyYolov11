import torch
import pynvml
 
pynvml.nvmlInit()#初始化
#设备情况
deviceCount = pynvml.nvmlDeviceGetCount()
print('显卡数量：',deviceCount)
for i in range(deviceCount):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    print('GPU %d is :%s'%(i,gpu_name))
 
    #显存信息
    memo_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("GPU %d Memory Total: %.4f G"%(i,memo_info.total/1024/1024/1000) )
    print("GPU %d Memory Free: %.4f G"%(i,memo_info.free/1024/1024/1000))
    print("GPU %d Memory Used: %.4f G"%(i,memo_info.used/1024/1024/1000))
 
    #温度
    Temperature = pynvml.nvmlDeviceGetTemperature(handle, 0)
    print("Temperature is %.1f C" %(Temperature))
 
    #风扇转速
    speed = pynvml.nvmlDeviceGetFanSpeed(handle)
    print("Fan speed is ",speed)
 
    #电源状态
    power_ststus = pynvml.nvmlDeviceGetPowerState(handle)
    print("Power ststus", power_ststus)
#关闭
pynvml.nvmlShutdown()