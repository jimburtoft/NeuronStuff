#!/bin/bash
# OPTIONAL: only needed if you cannot use the Neuron torch-neuronx 2.14.0 DLAMI (see README).
# Sets up TorchNeuron (Neuron SDK 2.33, PyTorch 2.14) + Hugging Face TRL on a trn2 instance
# launched from the stock Ubuntu Server 26.04 AMI (AWS manual-install path).
# Note: the torch-neuronx 2.13/2.14 wheels on pip.repos.neuron.amazonaws.com need glibc 2.43, which means
# Ubuntu 26.04. On Ubuntu 24.04 (glibc 2.39) `import torch_neuronx` fails; use the DLAMI there.
set -euo pipefail

# --- Neuron driver, runtime and tools ---
. /etc/os-release
sudo install -m 0755 -d /etc/apt/keyrings
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/neuron.gpg
echo "deb [signed-by=/etc/apt/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" | sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y linux-headers-$(uname -r) git
sudo apt-get install -y aws-neuronx-dkms=2.* aws-neuronx-collectives=2.* aws-neuronx-runtime-lib=2.* aws-neuronx-tools=2.*
# The Neuron compiler backend links libarchive. It is not on the stock Ubuntu 26.04 image and not yet in the AWS docs.
sudo apt-get install -y libarchive13t64 python3-venv g++

# --- Python environment ---
python3 -m venv ~/aws_neuron_venv_pytorch_2_14
source ~/aws_neuron_venv_pytorch_2_14/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch-neuronx==2.14.*" "neuronx-cc==2.*" nki \
  --extra-index-url https://pip.repos.neuron.amazonaws.com --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install "trl==1.16.0" "peft==0.21.2" "transformers>=5.20" datasets

# --- Swap: the compiler can use a lot of host RAM on multi-process runs ---
if ! swapon --show | grep -q /swapfile; then
  sudo dd if=/dev/zero of=/swapfile bs=1M count=65536 status=none
  sudo chmod 600 /swapfile && sudo mkswap --force /swapfile > /dev/null && sudo swapon /swapfile
fi

# --- Verify ---
export PATH=/opt/aws/neuron/bin:$PATH
python -c "import torch, torch_neuronx; print(f'PyTorch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}')"
python -c "import torch, torch_neuronx; x = torch.randn(4, 4, device='neuron'); print('neuron device OK', (x @ x).cpu().shape)"
neuron-ls
