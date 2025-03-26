"""Benchmark offline inference throughput."""
import os
import argparse
import json
import random
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

def build_chat_prompt(prompt, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


@dataclass
class Metrics:
    num_requests: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    latency: float

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    dataset=[]
    with open(dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data["question"])

    # Tokenize the prompts and completions.
    prompts = [build_chat_prompt(prompt,tokenizer) for prompt in dataset]
    #print('---------------------------')
    prompt_token_ids = tokenizer(prompts).input_ids
    # completions = [completion for _, completion in dataset]
    # completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        # output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    sampled_requests = tokenized_dataset[:num_requests]
    return sampled_requests

def calculate_metrics(
    outputs,
    elapsed_time: float,
):
    input_tokens = 0
    output_tokens = 0
    num_requests = len(outputs)
    for output in outputs:
        input_tokens += len(output.prompt_token_ids)
        output_tokens += len(output.outputs[0].token_ids)
    request_throughput = num_requests / elapsed_time
    output_throughput = output_tokens / elapsed_time

    metrics = Metrics(
        num_requests=num_requests,
        total_input=input_tokens,
        total_output=output_tokens,
        request_throughput=request_throughput,
        output_throughput=output_throughput,
        latency=elapsed_time,
    )

    return metrics

def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    enable_prefix_caching: bool,
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,
) -> float:

    from vllm import LLM, SamplingParams
    llm = LLM(model=model,
              tokenizer=tokenizer,
              quantization=quantization,
              tensor_parallel_size=tensor_parallel_size,
              seed=seed,
              trust_remote_code=trust_remote_code,
              dtype=dtype,
              max_model_len=2048,
             # max_model_len=4096,
             # max_model_len=32768,
              gpu_memory_utilization=gpu_memory_utilization,
              enforce_eager=False,
             # enforce_eager=True,
              kv_cache_dtype=kv_cache_dtype,
              device=device,
              enable_prefix_caching=enable_prefix_caching,
           #   enable_prefix_caching=True,
              download_dir=download_dir,
             
              max_num_batched_tokens = 1024*16, # org
              #max_num_batched_tokens =1024*2, # Qwen2-72B-Instruct-GPTQ-Int4
              #max_num_batched_tokens = 1024*2,
             
              max_num_seqs = 256, # org
             # max_num_seqs = 512, #388 + 16*12, # Qwen2-72B-Instruct-GPTQ-Int4
             
              block_size=16,
         #     block_size=16*2,
           #   enable_chunked_prefill=False, # org
              enable_chunked_prefill=True,
              #num_scheduler_steps=2,
             use_v2_block_manager=False,
            #  use_v2_block_manager=True,
              # enforce_eager = True
              disable_log_stats = False
             )
    # llm.llm_engine.log_stats = True
    # Add the requests to the engine.

    prompts = []
    sampling_params = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
            #    use_beam_search=use_beam_search,
                # ignore_eos=True,
                max_tokens=output_len,
            ))

    start = time.perf_counter()
    if 0:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
            with record_function("generate"):
                outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        prof.export_chrome_trace("glm4-base-ck-line.json")
    else:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    end = time.perf_counter()
    metrics = calculate_metrics(outputs, end - start)

    # llm.llm_engine.do_log_stats()
    return end - start, metrics


def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    use_beam_search: bool,
    max_batch_size: int,
    trust_remote_code: bool,
) -> float:
    assert not use_beam_search
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (max(max_prompt_len, next_prompt_len) +
                    max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt",
                              padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_mii(
    requests: List[Tuple[str, int, int]],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
) -> float:
    from mii import client, serve
    llm = serve(model, tensor_parallel=tensor_parallel_size)
    prompts = [prompt for prompt, _, _ in requests]

    start = time.perf_counter()
    llm.generate(prompts, max_new_tokens=output_len)
    end = time.perf_counter()
    client = client(model)
    client.terminate_server()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code,use_fast=True)
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_prompts)]
    else:
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                                   args.output_len)

    if args.backend == "vllm":
        elapsed_time, metrics = run_vllm(requests, args.model, args.tokenizer,
                                args.quantization, args.tensor_parallel_size,
                                args.seed, args.n, args.use_beam_search,
                                args.trust_remote_code, args.dtype,
                                args.max_model_len, args.enforce_eager,
                                args.kv_cache_dtype, args.device,
                                args.enable_prefix_caching,
                                args.gpu_memory_utilization, args.download_dir)
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size,
                              args.trust_remote_code)
    elif args.backend == "mii":
        elapsed_time = run_mii(requests, args.model, args.tensor_parallel_size,
                               args.output_len)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    print("{s:{c}^{n}}".format(s=' Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.num_requests))
    print("{:<40} {:<10}".format("Input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Output tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Latency (s):", metrics.latency))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("=" * 50)
    
    if args.save_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = args.backend
        result_json["model_id"] = args.model
        result_json["dataset"] = args.dataset
        result_json["tokenizer_id"] = args.tokenizer
        result_json["gpu_memory_utilization"] = args.gpu_memory_utilization
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts
        result_json["input_len"] = args.input_len
        result_json["output_len"] = args.output_len
        

        # Merge with benchmark result
        result_json["num_requests"] = metrics.num_requests
        result_json["total_input"] = metrics.total_input
        result_json["total_output"] = metrics.total_output
        result_json["latency"] = metrics.latency
        result_json["request_throughput"] = metrics.request_throughput
        result_json["output_throughput"] = metrics.output_throughput

        # Save to file
        base_model_id = args.model.split("/")[-1]
        file_name = f"{args.backend}-{args.output_len}output-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8_e5m2"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type.')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help='device type for vLLM execution, supporting CUDA only currently.')
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="enable automatic prefix caching for vLLM backend.")
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
    main(args)
