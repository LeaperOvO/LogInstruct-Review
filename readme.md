# LogInstruct: Knowledge-Driven Instruction Synthesis for Enhancing LLM-Based Log Analysis
This repository contains implementation for LogInstruct: Knowledge-Driven Instruction Synthesis for Enhancing LLM-Based Log Analysis.

Logs are vital for monitoring system health and troubleshooting. 
Recent advancements in large language models (LLMs) have demonstrated potential for automated log analysis. 
However, existing LLMs face challenges in this domain, including a lack of domain-specific knowledge and limitations in following task-specific instructions. Supervised fine-tuning can address these issues, but it relies on high-quality instruction datasets, which are costly and difficult to create for log analysis due to task diversity and manual annotation challenges.
To address these limitations, we propose LogInstruct, a novel knowledge-driven instruction synthesis framework designed to enhance the log analysis capabilities of LLMs.  LogInstruct introduces a structured knowledge tree for logs, capturing contextual information from documentation, and modeling instruction questions using Bloom’s Taxonomy to improve diversity. LogInstruct synthesizes 88K high-quality instruction datasets without manual intervention. Experiments demonstrate that models fine-tuned on our dataset achieve state-of-the-art performance across five distinct log analysis tasks, outperforming proprietary LLMs such as GPT-4o.

## Usage

This is the repo for the LogInstruct project, which aims to build and share an instruction-following log-specific open-source LLM. The repo contains:

- The [88K instruction data](#data-release) used for fine-tuning the model.
- The code for [generating the data](#data-generation-process).
- If you want to fine-tune the LLM (e.g., Qwen) using this data,  we suggest you refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### Data Release

[`final_instructions.json`](final_instructions/final_instructions.json) contains 88K instruction-following data we used for fine-tuning the log-specific model.
This JSON file is a list of dictionaries, each dictionary contains the following fields:

- `instruction`: `str`, describes the log analysis task the model should perform. 
- `input`: `str`, input raw log
- `output`: `str`, the answer to the instruction.

- for examples:
```
{
  "instruction": "After reading the log entry, determine what actions a network operator should take if they encounter this message during operations.",
  "input": "OSPF/6/STUB_RT_ONBT_TM_CRT: OSPF [Process-Id] creates TIMER of [Timer-Value] seconds to measure stub router state. In LS mode, this log is supported only by the service LSs.",
  "output": "1) Recognize that this log is informational and indicates that a timer is being set for a stub router. 2) Ensure that the stub router states are functioning correctly and that the OSPF process is operating as expected. 3) No immediate action is required due to its informational nature, but monitoring the stub router state could be beneficial."
 }
```


We used the following prompts for fine-tuning the model:

- for examples:
```
 
 ### Instruction:
 {instruction}
 
 ### Input:
 {input}
 
 ### Response:
```

## Data Generation Process

Our code is provided in [code_for_synthesize](code_for_synthesize) folder, if you want to synthesize the instruction with your own data, please following the following process.

<summary> <strong> Running the code </strong> </summary>

1. Set environment variables `OPENAI_API_KEY` to your OpenAI API key.
2. Install the dependencies with `pip install -r requirements.txt`.
3. Run `python code_for_synthesize/synthesize_instruction_with_gpt.py` to synthesize the instruction with your own data.

### Requirements

- transformers
- numpy
- torch
- huggingface-hub
- tokenizers
- scikit-learn
- tqdm
- requests 
- vllm
```
pip install -r requirements.txt
```





## License
MIT License

## Acknowledgements

Our downstream tasks datasets are collected public supports [Huawei](https://support.huawei.com/enterprise/en/index.html) and [H3C]().
