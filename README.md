# vLLM Setup for Benchmarking Large Language Models on an NVads V710 v5-series Instance

## Introduction 

vLLM is a framework designed to streamline the deployment, testing, and benchmarking of large language models (LLMs) in a virtualized environment.  This setup is particularly valuable for benchmarking, as it provides a consistent and reproducible platform to measure key inference performance metrics.


## Prerequisites


o	Access to an NVads V710 v5 instance, preferably NV24ads (full GPU instance) for a fast interactive experience.
o	Sufficient disk storage in your instance to accommodate the docker images and LLMs under test.

•	Software:
o	Ubuntu 22.04.4 LTS image 

•	Accounts and Access:
o	A Hugging Face account with a read-only API token (for downloading models).
o	Alternatively, models copied locally to the instance under test


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

run_models.py is the main MAD CLI(Command Line Interface) for running models locally. While the tool has many options, running a singular model is very easy. To run any model simply look for its name or tag in the models.json and the command is of the form:

For each model in models.json, the script

* builds docker images associated with each model. The images are named 'ci-$(model_name)', and are not removed after the script completes.
* starts the docker container, with name, 'container_$(model_name)'. The container should automatically be stopped and removed whenever the script exits.
* clones the git 'url', and runs the 'script'
* compiles the final perf.csv and perf.html

### Tag functionality

With the tag functionality, the user can select a subset of the models, that have the corresponding tags matching user specified tags, to be run. User specified tags can be specified in 'tags.json' or with the --tags argument. If multiple tags are specified, all models that match any tag is selected. Each model name in models.json is automatically a tag that can be used to run that model. Tags are also supported in comma-separated form

"python3 tools/run_models.py --tags TAG" so for example to run the pyt_huggingface_bert model use "python3 tools/run_models.py --tags pyt_huggingface_bert" or to run all pytorch models "python3 tools/run_models.py --tags pyt".

### Custom timeouts

The default timeout for model run is 2 hrs. This can be overridden if the model in models.json contains a 'timeout' : TIMEOUT entry. Both the default timeout and/or timeout specified in models.json can be overridden using --timeout TIMEOUT command line argument. Having TIMEOUT set to 0 means that the model run will never timeout.

### Debugging

Some of the more useful flags to be aware of are "--liveOutput" and "–keepAlive". "--liveOutput" will show all the logs as MAD is running, otherwise they are saved to log files on the current directory. "–keepAlive" will prevent MAD from stopping and removing the container when it is done, which can be very useful for manual debugging or experimentation. Note that when running with the "–keepAlive" flag the user is responsible for stopping and deleting that container. The same MAD model cannot run again until that container is cleared. 

For a more details on the tool please look at the "–help" flag 

## To add a model to the MAD repo

0. create workload name. The names of the modules should follow a specfic format. First it should be the framework(tf_, tf2_, pyt_, ort_, ...) , the name of the project and finally the workload. For example

    ```
    tf2_huggingface_gpt2
    ```
    Use this name in the models.json, as the dockerfile name and the scripts folder name.

1. add the necessary info to the models.json file. Here is a sample model info entry for bert
    ```json
        {
            "name": "tf2_bert_large",
            "url": "https://github.com/ROCmSoftwarePlatform/bert",
            "dockerfile": "docker/tf2_bert_large",
            "scripts": "scripts/tf2_bert_large",
            "n_gpus": "4",
            "owner": "john.doe@amd.com",
            "training_precision": "fp32",
            "tags": [
                "per_commit",
                "tf2",
                "bert",
                "fp32"
            ],
            "args": ""
        }
    ```
   | Field               | Description                                                                |
   |---------------------| ---------------------------------------------------------------------------|
   | name                | a unique model name                                                        |
   | url                 | model url to clone                                                         |
   | dockerfile          | initial search path dockerfile collection                                  |
   | scripts             | model script to execute in dockerfile under cloned model directory         |
   | data                | Optional field denoting data for script                                    |
   | n_gpus              | number of gpus exposed inside docker container. '-1' => all available gpus |
   | timeout             | model specific timeout, default of 2 hrs                                   |
   | owner               | email address for model owner                                              |
   | training\_precision | precision, currently used only for reporting                               |
   | tags                | list of tags for selecting model. The model name is a default tag.         |
   | multiple\_results   | optional parameter for multiple results, pointing to csv that holds results|
   | args                | extra arguments passed to model scripts                                    | 

2. create a dockerfile, or reuse an existing dockerfile in the docker directory. Here is an example below that should serve as a template. 
    ```docker
    # CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
    FROM rocm/tensorflow

    # Install dependencies
    RUN apt update && apt install -y \
        unzip 
    RUN pip3 install pandas

    # Download data
    RUN URL=https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip && \
        wget --directory-prefix=/data -c $URL && \
        ZIP_NAME=$(basename $URL) && \
        unzip /data/$ZIP_NAME -d /data
    ```
3. create a directory in the scripts directory that contains everything necessary to do a run and report performance. The contents of this directory will be copied to the model root directory. Make sure the directory has a run script. If the script name is not explicitly specified, MAD assumes that the script name is 'run.sh'. Here is a sample run.sh script for bert.
    ```bash
        # setup model
        MODEL_CONFIG_DIR=/data/uncased_L-24_H-1024_A-16
        BATCH=2
        SEQ=512
        TRAIN_DIR=bert_large_ba${BATCH}_seq${SEQ}
        TRAIN_STEPS=100
        TRAIN_WARM_STEPS=10
        LEARNING_RATE=1e-4
        DATA_SOURCE_FILE_PATH=sample_text.txt
        DATA_TFRECORD=sample_text_seq${SEQ}.tfrecord
        MASKED_LM_PROB=0.15
        calc_max_pred() {
            echo $(python3 -c "import math; print(math.ceil($SEQ*$MASKED_LM_PROB))")
        }
        MAX_PREDICTION_PER_SEQ=$(calc_max_pred)

        python3 create_pretraining_data.py \
            --input_file=$DATA_SOURCE_FILE_PATH \
            --output_file=$DATA_TFRECORD \
            --vocab_file=$MODEL_CONFIG_DIR/vocab.txt \
            --do_lower_case=True \
            --max_seq_length=$SEQ \
            --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
            --masked_lm_prob=$MASKED_LM_PROB \
            --random_seed=12345 \
            --dupe_factor=5

        # train model
        python3 run_pretraining.py \
            --input_file=$DATA_TFRECORD \
            --output_dir=$TRAIN_DIR \
            --do_train=True \
            --do_eval=True \
            --bert_config_file=$MODEL_CONFIG_DIR/bert_config.json \
            --train_batch_size=$BATCH \
            --max_seq_length=$SEQ \
            --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
            --num_train_steps=$TRAIN_STEPS \
            --num_warmup_steps=$TRAIN_WARM_STEPS \
            --learning_rate=$LEARNING_RATE \
            2>&1 | tee log.txt

        # report performance metric
        python3 get_bert_model_metrics.py $TRAIN_DIR
    ```
    Note that there is a python script for reporting performance that was also included in the model script directory. For single result reporting scripts, make sure that you print performance in the following format, `performance: PERFORMANCE_NUMBER PERFORMANCE_METRIC`. For example, the performance reporting script for bert, get_bert_model_metrics.py, prints `performance: 3.0637370347976685 examples/sec`.

    For scripts that report multiple results, signal MAD to expect multiple results with 'multiple\_results' field in models.json. This points to a csv generated by the script. The csv should have 3 columns, `models,performance,metric`, with different rows for different results. 

4. For a particular model, multiple tags such as the precision, the framework, workload may be given.  For example, the "tf2_mlperf_resnet50v1.nchw" could have the "tf2" and "resnet50" tag.  If this workload also specified the precision then this would be a valid tag as well (e.g. "fp16" or "fp32").  Also, MAD considers each model name to be a default tag, that need not be explicitly specified.


## Special environment variables 

MAD uses special environment variables to provide additional functionality within MAD. These environment variables always have a MAD_ prefix. These variables are accessible within the model scripts. 

  | Variable                    | Description                          |
  |-----------------------------|--------------------------------------|
  | MAD_SYSTEM_GPU_ARCHITECTURE | GPU Architecture for the host system |
  | MAD_RUNTIME_NGPUS           | Number of GPU available to the model |

### Model environment variables

MAD also exposes model-environment variables to allow for model tuning at runtime. These environment variables always have a MAD_MODEL_ prefix. These variables are accessible within the model scripts and are set to default values if not specified at runtime. 

   | Field                       | Description                                                                       |
   |-----------------------------| ----------------------------------------------------------------------------------|
   | MAD_MODEL_NAME              | model's name in `models.json`                                                     |
   | MAD_MODEL_NUM_EPOCHS        | number of epochs                                                                  |
   | MAD_MODEL_BATCH_SIZE        | batch-size                                                                        |

