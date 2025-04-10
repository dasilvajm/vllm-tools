# vLLM Setup for Benchmarking Large Language Models on an NVads V710 v5-series Instance

## Introduction 

vLLM is a framework designed to streamline the deployment, testing, and benchmarking of large language models (LLMs) in a virtualized environment.  This setup is particularly valuable for benchmarking, as it provides a consistent and reproducible platform to measure key inference performance metrics.


## Prerequisites
- Access to an NVads V710 v5 instance, preferably NV24ads (full GPU instance) for a fast interactive experience.
- Sufficient disk storage in your instance to accommodate the docker images and LLMs under test.
- Ubuntu 22.04 LTS image



## Install ROCm
The example below outlines the steps for installing the latest available public AMD ROCm release, in this case, ROCm 6.3.2.

```
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME 
wget https://repo.radeon.com/amdgpu-install/6.3.2/ubuntu/jammy/amdgpu-install_6.3.60302-1_all.deb
sudo apt install ./amdgpu-install_6.3.60302-1_all.deb -y
sudo apt update
sudo apt install amdgpu-dkms rocm -y
```

When done, verify that amdgpu dkms is properly installed.  

```
dkms status
```

If successful, it should report back as “installed” as shown below:
amdgpu/6.10.5-2109964.22.04, 6.8.0-1021-azure, x86_64: installed

Reboot the instance before continuing.  After the reboot, load the amdgpu driver:

```
sudo modprobe amdgpu
```


 
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



## Pull and run the recommended AMD V710 VLLM Docker image

A V710 vLLM docker image has been made available on AMD’s rocm/vllm-dev dockerhub repository.  

Pull the image:

```
sudo docker pull rocm/vllm-dev:v710inference_rocm6.3-release_ubuntu22.04_py3.10_pytorch_release-2.6
```



Run the Docker image:

```
sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx rocm/vllm-dev:v710inference_rocm6.3-release_ubuntu22.04_py3.10_pytorch_release-2.6
```

Note that this example mounts the $HOME/dockerx folder to the container.  In the benchmarking examples below, the language models are assumed to be already downloaded to the /dockerx folder.


 
## Benchmarking Inference Performance with vLLM

vLLM includes three key benchmarking scripts to evaluate different aspects of inference performance: benchmark_latency.py, benchmark_throughput.py, and benchmark_serving.py.   This document will focus on the latency benchmark.

The benchmark_latency.py script is designed to measure the latency of processing a single batch of requests in an offline inference scenario. This test evaluates the end-to-end latency for a single batch of requests, from input processing to output generation, excluding network or serving overhead and will provide the total latency (in seconds) for processing the batch.



## Example Command 
The command below runs the Llama-3.1-8B-Instruct model with batch size 1, input length 1024 and output length 1024 and expects that the model is present in the $HOME/dockerx folder:

```
python benchmark_latency.py --input-len=1024 --output-len=1024 --batch-size=1 --num-iters=5 --model='/dockerx/Llama-3.1-8B-Instruct/' --num-iters-warmup 2 --dtype=float16 --max_model_len=4096
```


