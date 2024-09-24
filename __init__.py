import platform
import sys
import os
import subprocess

# Python version and platform checks
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
platform_system = platform.system()
platform_machine = platform.machine()

print(f"Detected Python version: {python_version}")
print(f"Detected platform system: {platform_system}")
print(f"Detected platform machine: {platform_machine}")

# List of dependencies
requirements = [
    "transformers>=4.0.0",
    "torch>=1.7.1",
    "accelerate"
]

# llama-cpp-python (CPU only, AVX2)
cpu_wheels = {
    "3.11": {
        "Linux": {
            "x86_64": "https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/cpu/llama_cpp_python-0.2.89+cpuavx2-cp311-cp311-linux_x86_64.whl"
        },
        "Windows": {
            "amd64": "https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/cpu/llama_cpp_python-0.2.89+cpuavx2-cp311-cp311-win_amd64.whl"
        }
    },
    "3.10": {
        "Linux": {
            "x86_64": "https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/cpu/llama_cpp_python-0.2.89+cpuavx2-cp310-cp310-linux_x86_64.whl"
        },
        "Windows": {
            "amd64": "https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/cpu/llama_cpp_python-0.2.89+cpuavx2-cp310-cp310-win_amd64.whl"
        }
    }
}

# llama-cpp-python (CUDA, no tensor cores)
cuda_wheels = {
    "3.11": {
        "Linux": {
            "x86_64": "https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.89+cu121-cp311-cp311-linux_x86_64.whl"
        },
        "Windows": {
            "amd64": "https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.89+cu121-cp311-cp311-win_amd64.whl"
        }
    },
    "3.10": {
        "Linux": {
            "x86_64": "https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.89+cu121-cp310-cp310-linux_x86_64.whl"
        },
        "Windows": {
            "amd64": "https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.89+cu121-cp310-cp310-win_amd64.whl"
        }
    }
}

# Function to install a package
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Installing base requirements
for req in requirements:
    install_package(req)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Installing llama-cpp-python based on platform and CUDA availability
if cuda_available:
    print("CUDA is available. Installing the CUDA version of llama-cpp-python.")
    wheel_url = cuda_wheels.get(python_version, {}).get(platform_system, {}).get(platform_machine)
else:
    print("CUDA is not available. Installing the CPU-only version of llama-cpp-python.")
    wheel_url = cpu_wheels.get(python_version, {}).get(platform_system, {}).get(platform_machine)

if wheel_url:
    install_package(wheel_url)
else:
    print(f"No compatible llama-cpp-python wheel found for Python {python_version}, platform {platform_system}, machine {platform_machine}.")



from .Searge_LLM_Node import *
