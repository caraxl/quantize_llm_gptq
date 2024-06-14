import torch
import time,gc,os
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
from typing import List,Dict
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def bytes_to_giga_bytes(bytes)->int:
    '''
    计算模型显存占用大小为多少GB
    :param bytes:
    :return:
    '''
    gb = bytes / 1024 / 1024 / 1024
    return gb

def flush():
    # 清理内存空间
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def load_model(quantized_model_dir:str):
    model = AutoModelForCausalLM.from_pretrained(
        quantized_model_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)
    return model, tokenizer

def build_conversation():
    prompt = "简单介绍一下什么是人工智能技术"
    messages = [
        {"role": "system", "content": "你是一个计算机领域的专家，擅长回答用户提出的计算机方面的问题。"},
        {"role": "user", "content": prompt}
    ]
    return messages

def predict(messages:List[dict],tokenizer:AutoTokenizer,model:AutoModelForCausalLM):
    device = "cuda:1"  # 指定推理设备
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generation_config = GenerationConfig(
            do_sample=True,
            top_k=3,
            temperature=0.1,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    t1 = time.time()
    generated_ids = model.generate(
        **model_inputs,
        generation_config=generation_config,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    gb = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())
    print(f"显存占用：{str(gb)}GB") # fp16:15Gb int4:5Gb int8:
    t2 = time.time()
    print(f"模型生成答案用时：{t2-t1}s，吞吐量：{len(response)/int(t2-t1)}Tokens/S") # fp16:9s int4:65s int8:
    print(f"模型生成答案:{response}")

def main():
    # quantized_model_dir = "/media/cara/文档/PythonProjects/quantize_llm/results/Llama-3-8B-Instruct-gptq-int4"
    quantized_model_dir = "/media/cara/文档/PythonProjects/quantize_llm/results/Llama-3-8B-Instruct-gptq-int8"
    # raw_model_dir = "/media/cara/文档/PythonProjects/models/llm/llama/Llama-3-8B-Instruct" # 未量化版本
    model, tokenizer = load_model(quantized_model_dir=quantized_model_dir)
    messages = build_conversation()
    predict(model=model,tokenizer=tokenizer,messages=messages)
    flush()

if __name__ == '__main__':
    main()


