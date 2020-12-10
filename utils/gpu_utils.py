import nvidia_smi
# pip install nvidia-ml-py3
class GPU_Utils:
    def __init__(self,gpu_index=0):
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)

    def gpu_mem_usage(self):
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        return mem_res.used / (1024**2)

    def gpu_utilization(self):
        gpu_util = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
        return gpu_util.gpu
