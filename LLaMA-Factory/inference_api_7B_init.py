# from openai import OpenAI
# import json
# import numpy as np
# import pandas as pd
# import os
# client = OpenAI(api_key="0",base_url="http://0.0.0.0:8000/v1")

# # 读取json文件
# file = open('/home/njs4nu/Desktop/nlp_proj/LLaMA-Factory/data/climatex_test.json', "r", encoding="utf-8")
# # 读取prompts
# data = json.load(file)
# num_data = len(data)

# for i in range(0,num_data):
#     system = data[i]['system']
#     instruction = data[i]['instruction']
#     input = data[i]['input']
#     ans = data[i]['output']

#     prompt = system + '\n' + instruction + '\n' + input
#     messages = [{"role": "user", "content": prompt}]
#     result = client.chat.completions.create(messages=messages, model="Qwen/Qwen2.5-7B-Instruct")
#     print(result.choices[0].message)
#     print(ans == result.choices[0].message)

from openai import OpenAI
import json
import pandas as pd
import os

# 初始化 OpenAI 客户端
client = OpenAI(api_key="0", base_url="http://0.0.0.0:8001/v1")

# 读取 JSON 文件
file_path = '/home/njs4nu/Desktop/nlp_proj/LLaMA-Factory/data/climatex_test.json'
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

num_data = len(data)

# 结果保存路径
output_csv_path = '/home/njs4nu/Desktop/nlp_proj/inference_results_qwen_7B_init.csv'
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# 如果文件不存在，先写入 CSV 文件的表头
if not os.path.exists(output_csv_path):
    header_df = pd.DataFrame(columns=["Number","System", "Instruction", "Input", "Expected Output", "Generated Output", "Match"])
    header_df.to_csv(output_csv_path, index=False, encoding="utf-8")

# 遍历每个数据样本进行推理
for i in range(num_data):
    system = data[i]['system']
    instruction = data[i]['instruction']
    input_text = data[i]['input']
    expected_output = data[i]['output']

    # 构建提示信息
    prompt = f"{system}\n{instruction}\n{input_text}"
    messages = [{"role": "user", "content": prompt}]

    try:
        # 调用 API 进行推理
        result = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages
        )

        # 获取模型生成的内容
        generated_output = result.choices[0].message.content

    except Exception as e:
        generated_output = f"Error: {str(e)}"

    # 将结果保存到 CSV 文件中
    result_data = {
        "Number": i + 1,
        "System": system,
        "Instruction": instruction,
        "Input": input_text,
        "Expected Output": expected_output,
        "Generated Output": generated_output,
        "Match": expected_output == generated_output
    }

    # 将当前结果转换为 DataFrame 并追加保存到 CSV 文件
    result_df = pd.DataFrame([result_data])
    result_df.to_csv(output_csv_path, mode='a', index=False, header=False, encoding="utf-8")

    # 打印当前处理进度
    print(f"Processed {i + 1}/{num_data}: Saved to CSV")

print(f"All results saved to {output_csv_path}")
