# Navi48 VLLM Demo Setup for Large Language Model Inference

## Introduction 

As a starting point, it is assumed that the user has an Ubuntu 24.04 installation ready to go.  For ease of setup, it is preferred to SSH into the Ubuntu platform with an SSH client such as Putty or any SSH client of your choice.

## Before you start
- Ensure that you have sufficient disk storage in your platform to accommodate the docker images and LLMs. A minimum 1 TB of storage space is recommended

- It is preferred that SecureBoot is disabled on the platform. Check this as follows:

```
mokutil --sb-state
```
This should report back "SecureBoot disabled"
<br><br>

- Ahead of accessing the platform via SSH, install OpenSSH-Server

```
sudo apt install openssh-server -y
```
<br>

- Confirm the Ubuntu IP address

```
ip add | grep inet
```
<br>

- Connect to the system via your SSH client

```
ssh username@<system-IP-address>
```
<br><br>

##  Ubuntu Linux Setup 
Update Ubuntu to the latest and greatest for the installed release
```
sudo apt update && sudo apt upgrade -y
```
<br>

Install some prerequisites:
```
sudo apt install python3-pip git curl git-lfs -y
```
<br>

Blacklist the amdgpu driver (Upstream amdgpu may interfere with the ROCm installation)
```
sudo sh -c 'echo "blacklist amdgpu" >>  /etc/modprobe.d/blacklist.conf'
sudo update-initramfs -uk all
```

Reboot the system

After the system is back up, SSH into the system with your SSH client again.
<br><br>

## Install ROCm
```
wget https://repo.radeon.com/amdgpu-install/6.4/ubuntu/jammy/amdgpu-install_6.4.60400-1_all.deb
sudo apt install ./amdgpu-install_6.4.60400-1_all.deb -y
sudo apt update &&
sudo apt install python3-setuptools python3-wheel -y &&
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" &&
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install amdgpu-dkms rocm -y
```
<br>

When done, verify that amdgpu dkms is properly installed.  

```
dkms status
```
<br>
This should report back similar to the following:
<br>

amdgpu/6.12.6-2107834.22.04, 6.11.0-24-generic, x86_64: installed
<br>


Reboot the system before continuing.


<br>
After the reboot, load the amdgpu driver:

```
sudo modprobe amdgpu
```
<br>
If ROCm is properly installed, confirm that ROCm identifies all of the Navi48 GPUs installed:

```
/opt/rocm-6.4.0/bin/rocm-smi
```
<br>
In the example output below, four Navi48 GPUs are installed and identified:
<br>
<img src="https://github.com/dasilvajm/vllm-tools/blob/main/rocm-smi.jpg" alt="Screenshot" width="800"/>


<br><br>
 
## Install Docker
Use the steps below to install Docker:

```
sudo apt-get install ca-certificates curl -y
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt 
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
```

<br>

## Download the LLM
Create a models directly on your HOME path of the Linux installation and cd into that directory:

```
mkdir models
cd models
```
<br>

Download the LLM from Huggingface.  In the example below, we are cloning the DeepSeek-R1-Distill-Llama-70B-FP8-dynamic model
```
git clone https://huggingface.co/RedHatAI/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
```

## Set Up and Run the VLLM Server

Pull the image:

```
sudo docker pull hyoon11/vllm-dev:20250417_fp8_navi_main_a1c35e7_dev
```
<br>

Run the Docker image:

```
sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/models:/models hyoon11/vllm-dev:20250417_fp8_navi_main_a1c35e7_dev
```

Note that this example mounts the $HOME/models folder to the container.  Inside the docker, when a path to the model folder is required, use /models/<model-folder-name>, for example, '/models/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic'
<br>

Inside the container, launch the VLLM server:
```
python -m vllm.entrypoints.openai.api_server --model='/models/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic/' --dtype=bloat16 --max_model_len=4096 --gpu-memory-utilization=0.98
```
<br>
The VLLM server should start up without any errors

<br><br>
 
## Set up Open WebUI

Pull the Open WebUI docker image:

```
sudo docker run -d -p 3000:8080 -e OPENAI_API_KEY=1234 -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
```
<br>
The command above pulls the image and sets the API key to '1234'. This key will be used to connect to the VLLM server from an HTML client below.




