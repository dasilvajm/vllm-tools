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
If ROCm is properly installed, confirm that ROCm identifies all of the Navi48 GPUs installed]

```
/opt/rocm-6.4.0/bin/rocm-smi
```
<br>
In the example output below, four Navi48 GPUs are installed and identified:
<br>
<img src="https://github.com/dasilvajm/vllm-tools/blob/main/rocm-smi.jpg" alt="Screenshot" width="500"/>


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

## Pull and run the recommended Navi48 VLLM Docker image

Pull the image:

```
sudo docker pull xxxxxxxxxxx
```

<br>

Run the Docker image:

```
sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx rocm/vllm-dev:v710inference_rocm6.3-release_ubuntu22.04_py3.10_pytorch_release-2.6
```

Note that this example mounts the $HOME/dockerx folder to the container.  In the benchmarking examples below, the language models are assumed to be already downloaded to the /dockerx folder.

<br><br>
 
## Benchmarking Inference Performance with vLLM

