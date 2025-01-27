# Check GPU availability
import sys
#import tensorflow.keras
#import tensorflow as tf
import platform
import torch

print(f"Python Platform: {platform.platform()}")
print("A GOOD result is macOS-12.4.-arm64-64bit")
print("A BAD  result is macOS-11.8-x86_64-i386-64bit")
#print(f"Tensor Flow Version: {tf.__version__}")
#print(f"Keras Version: {tensorflow.keras.__version__}")
print(f"Python {sys.version}")
# print(f"Pandas {pd.__version__}")
# print(f"Scikit-Learn {sk.__version__}")
# print(f"SciPy {sp.__version__}")
#gpu = len(tf.config.list_physical_devices('GPU'))>0
#print("GPU is", "available" if gpu else "NOT AVAILABLE")
print(f"PyTorch {torch.__version__}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
print("if device='mps' then the Mac M1 GPU is being used")
if device=='mps':print("PASSED: This Mac is using the Metal Performance Shaders (MPS)")
x = torch.rand(5, 3)
print(x.to(device)) # This pushes the tensor the GPU if available.
# ref: https://saturncloud.io/blog/how-to-check-if-pytorch-is-using-the-gpu/
