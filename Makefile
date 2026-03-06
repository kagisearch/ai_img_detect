ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
VENV_DIR := $(ROOT_DIR)/.venv



setup:
	(\
		cd $(ROOT_DIR)\
		&& export VIRTUAL_ENV=$(VENV_DIR)\
		&& uv venv\
		&& . $(VENV_DIR)/bin/activate\
		&& $(MAKE) install_torch\
		&& uv sync --inexact --all-extras\
	)

install_torch:
# MacOS has no CPU wheel for pytorch; separate install path
ifeq ($(UNAME), Darwin)
	uv pip install torch torchvision -qq
# If there's CI flags, we're on a prod linux CPU only build
# Specified CUDA Build
else ifeq ($(strip $(GPU_BACKEND)), cuda)
	echo "\nCUDA Specified";
	uv pip install torch torchvision --torch-backend=cu128
# Using Intel GPU; needs to force the backend
# Not supporting JAX at the moment, to install it see 
else ifeq ($(strip $(GPU_BACKEND)), xpu)
	echo "\nIntel XPU specified";
	uv pip install torch torchvision --torch-backend=xpu
# Flexible fallback for other backend (eg. cu125)
else ifneq ($(strip $(GPU_BACKEND)),)
	echo "\nTorch Backend $(GPU_BACKEND)";
	uv pip install torch torchvision --torch-backend=$(GPU_BACKEND)
else
	echo "\nNo GPU specified, using auto backend";
	uv pip install torch torchvision --torch-backend=auto
endif
	# Now Print Torch Device Capabilities
	uv run python $(ROOT_DIR)/model_serving/src/torch_compatibility_checking.py


install_llama_cpp:
# Specified CUDA Build
ifeq ($(strip $(GPU_BACKEND)), cuda)
	echo "\nCUDA Specified";
	FORCE_CMAKE=1 CC=gcc CXX=g++ CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --force-reinstall --no-cache-dir 'llama-cpp-python[server]'
# Using Intel GPU; use the portable installer for llamacpp convenience
#   github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/llamacpp_portable_zip_gpu_quickstart.md
else ifeq ($(strip $(GPU_BACKEND)), xpu)
	echo "\nIntel XPU specified; Using IPEX-LLM";
	wget https://github.com/ipex-llm/ipex-llm/releases/download/v2.3.0-nightly/llama-cpp-ipex-llm-2.3.0b20250724-ubuntu-core.tgz
# Vulkan backend, use prebuilt llamacpp
else ifeq ($(strip $(GPU_BACKEND)), vulkan)
	wget https://github.com/ggml-org/llama.cpp/releases/download/b6759/llama-b6759-bin-ubuntu-vulkan-x64.zip
else
	echo "\nNo GPU specified, using auto backend";
	uv pip install llama-cpp-python[server]
endif


