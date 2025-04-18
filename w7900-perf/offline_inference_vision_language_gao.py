"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import time

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser
from PIL import Image
from dataclasses import dataclass
import torch

# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.


@dataclass
class Metrics:
    num_requests: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    latency: float


def calculate_metrics(
    outputs,
    elapsed_time: float,
):
    input_tokens = 0
    output_tokens = 0
    num_requests = len(outputs)
    for output in outputs:
        print(f"input_tokens:{len(output.prompt_token_ids)}")
        input_tokens += len(output.prompt_token_ids)
        print(f"output_tokens:{len(output.outputs[0].token_ids)}")
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

# LLaVA-1.5
def run_llava(question: str, modality: str):
    assert modality == "image"

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    llm = LLM(model="llava-hf/llava-1.5-7b-hf", max_model_len=4096)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(question: str, modality: str):
    assert modality == "image"

    prompt = f"[INST] <image>\n{question} [/INST]"
    llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf", max_model_len=8192)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LlaVA-NeXT-Video
# Currently only support for video input
def run_llava_next_video(question: str, modality: str):
    assert modality == "video"

    prompt = f"USER: <video>\n{question} ASSISTANT:"
    llm = LLM(model="llava-hf/LLaVA-NeXT-Video-7B-hf", max_model_len=8192)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLaVA-OneVision
def run_llava_onevision(question: str, modality: str):

    if modality == "video":
        prompt = f"<|im_start|>user <video>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    elif modality == "image":
        prompt = f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    llm = LLM(model="llava-hf/llava-onevision-qwen2-7b-ov-hf",
              max_model_len=16384)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Fuyu
def run_fuyu(question: str, modality: str):
    assert modality == "image"

    prompt = f"{question}\n"
    llm = LLM(model="adept/fuyu-8b", max_model_len=2048, max_num_seqs=2)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Phi-3-Vision
def run_phi3v(question: str, modality: str):
    assert modality == "image"

    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"  # noqa: E501
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (128k) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.

    # num_crops is an override kwarg to the multimodal image processor;
    # For some models, e.g., Phi-3.5-vision-instruct, it is recommended
    # to use 16 for single frame scenarios, and 4 for multi-frame.
    #
    # Generally speaking, a larger value for num_crops results in more
    # tokens per image instance, because it may scale the image more in
    # the image preprocessing. Some references in the model docs and the
    # formula for image tokens after the preprocessing
    # transform can be found below.
    #
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct#loading-the-model-locally
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/main/processing_phi3_v.py#L194
    llm = LLM(
        model="microsoft/Phi-3-vision-128k-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={"num_crops": 16},
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# PaliGemma
def run_paligemma(question: str, modality: str):
    assert modality == "image"

    # PaliGemma has special prompt format for VQA
    prompt = "caption en"
    llm = LLM(model="google/paligemma-3b-mix-224")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Chameleon
def run_chameleon(question: str, modality: str):
    assert modality == "image"

    prompt = f"{question}<image>"
    llm = LLM(model="facebook/chameleon-7b", max_model_len=4096)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# MiniCPM-V
def run_minicpmv(question: str, modality: str):
    assert modality == "image"

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # For more details, please see: See: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630 # noqa
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    # model_name = "openbmb/MiniCPM-Llama3-V-2_5"

    #2.6
    model_name = "openbmb/MiniCPM-V-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
    )
    # NOTE The stop_token_ids are different for various versions of MiniCPM-V
    # 2.0
    # stop_token_ids = [tokenizer.eos_id]

    # 2.5
    # stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

    # 2.6
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    messages = [{
        'role': 'user',
        'content': f'(<image>./</image>)\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return llm, prompt, stop_token_ids


# H2OVL-Mississippi
def run_h2ovl(question: str, modality: str):
    assert modality == "image"

    model_name = "h2oai/h2ovl-mississippi-2b"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for H2OVL-Mississippi
    # https://huggingface.co/h2oai/h2ovl-mississippi-2b
    stop_token_ids = [tokenizer.eos_token_id]
    return llm, prompt, stop_token_ids


# InternVL
def run_internvl(question: str, modality: str):
    assert modality == "image"

    model_name = "/app/data-new/model/InternVL2-26B"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B#service
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, prompt, stop_token_ids


# NVLM-D
def run_nvlm_d(question: str, modality: str):
    assert modality == "image"

    model_name = "nvidia/NVLM-D-72B"

    # Adjust this as necessary to fit in GPU
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        tensor_parallel_size=4,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# BLIP-2
def run_blip2(question: str, modality: str):
    assert modality == "image"

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = f"Question: {question} Answer:"
    llm = LLM(model="Salesforce/blip2-opt-2.7b")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Qwen
def run_qwen_vl(question: str, modality: str):
    assert modality == "image"

    llm = LLM(
        model="Qwen/Qwen-VL",
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=2,
    )

    prompt = f"{question}Picture 1: <img></img>\n"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Qwen2-VL
def run_qwen2_vl(question: str, modality: str):
    assert modality == "image"

    model_name = "/app/data-new/model/Qwen2-VL-7B-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        dtype=torch.float16,
        
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
    )

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Pixtral HF-format
def run_pixtral_hf(question: str, modality: str):
    assert modality == "image"

    model_name = "mistral-community/pixtral-12b"

    llm = LLM(
        model=model_name,
        max_model_len=8192,
    )

    prompt = f"<s>[INST]{question}\n[IMG][/INST]"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLama 3.2
def run_mllama(question: str, modality: str):
    assert modality == "image"

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        enforce_eager=True,
    )

    prompt = f"<|image|><|begin_of_text|>{question}"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Molmo
def run_molmo(question, modality):
    assert modality == "image"

    model_name = "allenai/Molmo-7B-D-0924"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    prompt = question
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# GLM-4v
def run_glm4v(question: str, modality: str):
    assert modality == "image"
    model_name = "/mnt/data/glm-4v-9b"

    llm = LLM(model=model_name,
              max_model_len=4096,
              max_num_seqs=256,
              tensor_parallel_size=1,
              max_num_batched_tokens=1024 * 8,
              trust_remote_code=True,
              enforce_eager=False,
             # enforce_eager=True,

              disable_log_stats = False,
              gpu_memory_utilization=0.97,
              dtype=torch.float16,
              )
    prompt = question
    stop_token_ids = [151329, 151336, 151338]
    return llm, prompt, stop_token_ids


# Idefics3-8B-Llama3
def run_idefics3(question: str, modality: str):
    assert modality == "image"
    model_name = "HuggingFaceM4/Idefics3-8B-Llama3"

    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        enforce_eager=True,
        # if you are running out of memory, you can reduce the "longest_edge".
        # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
        mm_processor_kwargs={
            "size": {
                "longest_edge": 3 * 364
            },
        },
    )
    prompt = (
        f"<|begin_of_text|>User:<image>{question}<end_of_utterance>\nAssistant:"
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


model_example_map = {
    "llava": run_llava,
    "llava-next": run_llava_next,
    "llava-next-video": run_llava_next_video,
    "llava-onevision": run_llava_onevision,
    "fuyu": run_fuyu,
    "phi3_v": run_phi3v,
    "paligemma": run_paligemma,
    "chameleon": run_chameleon,
    "minicpmv": run_minicpmv,
    "blip-2": run_blip2,
    "h2ovl_chat": run_h2ovl,
    "internvl_chat": run_internvl,
    "NVLM_D": run_nvlm_d,
    "qwen_vl": run_qwen_vl,
    "qwen2_vl": run_qwen2_vl,
    "pixtral_hf": run_pixtral_hf,
    "mllama": run_mllama,
    "molmo": run_molmo,
    "glm4v": run_glm4v,
    "idefics3": run_idefics3,
}


def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        # Input image and question
        #image = ImageAsset("cherry_blossom") \
        #    .pil_image.convert("RGB")

        image_path = "./glm4v-test.jpg"
        image=Image.open(image_path).convert("RGB")
        img_question = "图片是一张快递面单照片。输出位于条形码下方的`快递面单号`(以JD、KY或KK开头的字母数字组合，不包括-后部分)字段，结果以JSON格式输出。"

        return {
            "data": image,
            "question": img_question,
        }

    if args.modality == "video":
        # Input video and question
        video = VideoAsset(name="sample_demo_1.mp4",
                           num_frames=args.num_frames).np_ndarrays
        vid_question = "Why is this video funny?"

        return {
            "data": video,
            "question": vid_question,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    question = mm_input["question"]

    llm, prompt, stop_token_ids = model_example_map[model](question, modality)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=1.0,
                                     top_p=1.0,
                                     top_k=50,
                                     max_tokens=2048,
                                     stop_token_ids=stop_token_ids)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                modality: data
            },
        }

    else:
        # Batch inference
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                modality: data
            },
        } for _ in range(args.num_prompts)]
        
    start = time.perf_counter()
    if 0:
        from torch.profiler import profile, record_function, ProfilerActivity
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        with profile(activities=[ProfilerActivity.CUDA], with_stack=False) as prof:
           # with record_function("generate"):
                outputs = llm.generate(inputs, sampling_params=sampling_params)
        prof.export_chrome_trace("qwen2-vl-100.json")
    else:
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        
    end = time.perf_counter()
    metrics = calculate_metrics(outputs, end - start)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

    print("{s:{c}^{n}}".format(s=' Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.num_requests))
    print("{:<40} {:<10}".format("Input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Output tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Latency (s):", metrics.latency))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("=" * 50)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="llava",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=4,
                        help='Number of prompts to run.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=['image', 'video'],
                        help='Modality of the input.')
    parser.add_argument('--num-frames',
                        type=int,
                        default=16,
                        help='Number of frames to extract from the video.')
    args = parser.parse_args()
    main(args)
