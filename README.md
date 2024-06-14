# quantize_llm_gptq
## 脚本功能说明
- quant_gptq.py 对llama3-8b-instruct模型进行gptq量化，支持int4和int8。
- infer.py 对量化后的模型进行推理，观测推理速度和输出质量。
  
量化后的模型也可以直接使用vllm推理框架进行推理部署。
