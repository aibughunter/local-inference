# AIBugHunter Local Inference Script

This script is used in the [AIBugHunter](https://github.com/aibughunter/aibughunter) VSCode Extension

## Install

```bash
pip install -r requirements.txt
# or manually
pip3 install numpy onnxruntime torch transformers
```

or 

```bash
pip3 install -r requirements.txt
# or manually
pip3 install numpy onnxruntime torch transformers
```

Refer to the [ONNX Runtime Website](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime) to determine whether to install `onnxruntime-gpu` or `onnxruntime` on your machine

The extension does not support running script on virtual environments (Python venv, Anaconda etc...) yet, please install dependencies globally with virtual environments deactivated