"""import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("PyTorch Version:", torch.__version__)
print("GPU Name:", torch.cuda.get_device_name(0))"""

'''import torch
a = torch.randn(3, 3).cuda()
b = torch.randn(3, 3).cuda()
c = a + b
print(c)
'''
import torch
print(torch.__version__)  # Check PyTorch version
print(torch.cuda.is_available())  # Should return True if CUDA is available
