from optimum.gptq import GPTQQuantizer
import torch,gc
from transformers import AutoTokenizer,AutoModelForCausalLM
from typing import Dict,List
import shutil,os
from utils import load_json
# 参考：https://medium.com/@sharmamridul1612/optimizing-llama-3-on-gptq-787830f97ea4

def flush():
    # 清理内存空间
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
def build_prompt(sample:Dict):
    label = "负面" if int(sample["label"])==0 else "正面"
    prompt = f"""
    <|im_start|>system
    你是一个外卖评论情绪识别专家，擅长根据用户评论的内容判断用户的情绪类型。总共有两种情绪类型：正面，负面。<|im_end|>
    <|im_start|>user
    {sample["review"]}<|im_end|>
    <|im_start|>assistant
    {label}<|im_end|>
    """
    return prompt


def quantize_model(pretrained_model_path:str,data:List[Dict]):

    quantizer = GPTQQuantizer(bits=8, group_size=128, dataset=data, model_seqlen=2048, disable_exllama=True)
    quantizer.quant_method = "gptq"
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path,
                                                 config=quantizer,
                                                 torch_dtype=torch.float16,
                                                 # max_memory = {0: "15GIB", 1: "15GIB"}
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=False)
    quantized_model = quantizer.quantize_model(model, tokenizer)
    return quantized_model

def save_quantized_model(quantized_model,quantized_model_save_path:str,pretrained_model_path:str):
    '''

    :param quantized_model:
    :param quantized_model_save_path:
    :param pretrained_model_path:
    :return:
    '''
    # 路径检查
    if os.path.exists(quantized_model_save_path):
        shutil.rmtree(quantized_model_save_path)
    os.makedirs(quantized_model_save_path)
    quantized_model.save_pretrained(quantized_model_save_path, safe_serialization=True)
    AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=True).save_pretrained(quantized_model_save_path)

def main():
    dataPath = "../data/waiMaiComment/test.json" # gptq量化需要提供一个校准数据集
    pretrained_model_path = "/media/cara/文档/PythonProjects/models/llm/llama/Llama-3-8B-Instruct" # 需要量化的模型
    quantized_model_save_path = "/media/cara/文档/PythonProjects/quantize_llm/results/Llama-3-8B-Instruct-gptq-int8" # 量化后的模型保存路径
    data = load_json(dataPath=dataPath)
    data = [build_prompt(sample=itm) for itm in data]  # list of str
    quantized_model = quantize_model(pretrained_model_path=pretrained_model_path,data=data)
    save_quantized_model(quantized_model=quantized_model,quantized_model_save_path=quantized_model_save_path,pretrained_model_path=pretrained_model_path)
    flush()


if __name__ == '__main__':
    main()




