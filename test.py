import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_allocated(0))
print(torch.cuda.memory_reserved(0))
print(torch.cuda.memory_summary(0, abbreviated=False))