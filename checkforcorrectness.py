import sys
import sysconfig

from vllm import LLM, SamplingParams
import time
import numpy as np
from scipy import stats
from transformers import AutoTokenizer

model = '/dockerx/Meta-Llama-3.1-8B'

number_of_generated_seq = 1
enforce_eager = False

def checkReasonability():
    llm = LLM(model=model,
        tokenizer=None,
#        quantization="fp8",
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype='float16',
        enforce_eager=enforce_eager,
#        kv_cache_dtype='fp8',
        device="cuda",
        ray_workers_use_nsight=False,
        enable_chunked_prefill=False,
        download_dir=None,
        block_size=16,
        disable_log_stats=True,
        use_v2_block_manager=True,
        disable_custom_all_reduce=False,
        enable_prefix_caching=False,
        # max_num_batched_tokens=4,
        # max_num_seqs=4,
        # max_model_len=2048,
        gpu_memory_utilization=0.9,
        quantization_param_path="",
#       distributed_executor_backend="mp",
        load_format="auto",
        # num_scheduler_steps=num_scheduler_steps,
        disable_async_output_proc=False,
        worker_use_ray=False,
        # engine_use_ray=False,
        # disable_log_requests=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model)
    # inputs = ["what is the capital of Great Britain?",
    #             "Hello what is your name?",
    #             "What do you like?",
    #             "How to cook chicken?",
    #             "what is quantum entanglement?"]
    inputs = ["How to cook chicken?",]

    sampling_params = SamplingParams(
        n=number_of_generated_seq,
        temperature=0.0,
        top_p=1.0,
        use_beam_search=False,
        ignore_eos=True,
        max_tokens=256)

    for prompt in inputs:
        prompt_enc = tokenizer(prompt)
        outs = llm.generate(prompt_token_ids=[prompt_enc['input_ids']],
                            sampling_params=sampling_params,
                            use_tqdm=False)
        print(f"{prompt=}")
        print(f"{outs[0].outputs[0].text=}")

if __name__ == "__main__":
    checkReasonability()