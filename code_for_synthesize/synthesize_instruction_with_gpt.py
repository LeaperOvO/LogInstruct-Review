
from tqdm import tqdm
import os
import time
import json
import random



def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content


def save_json(data, file):
    dict_json = json.dumps(data, indent=1)
    with open(file, 'w+', newline='\n') as file:
        file.write(dict_json)


def get_content(api_key,url,messages):
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }
    a = requests.post(url, json={
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.8,
        "stream": False,
    }, headers=headers)
    content = a.json()['choices'][0]['message']['content']
    return content


import requests


def generate(data,output_file, revise_num=10, wait_time=1):
    prompt_knowledge_tree = '''
You are an expert skilled in structured knowledge extraction. Please construct a Knowledge Tree about the given log. You need to build the tree from the provided document information according to the following rules.

1. Root Node
Objective:
    Identify the core issue about the log.
Rules:
    1. Extract the core issue from the log description using concise phrases or short sentences.
    2. Avoid redundant modifiers and ensure it directly reflects the essence of the problem.
    3. Use ROOT as the identifier for the root node.

2. Branch Nodes
Objective:
    Hierarchically classify the knowledge in the provided document information.
Rules:
    1. Each branch node represents a different type of knowledge.
    2. Summarize each type of knowledge using keywords.
    3. Refine branches as necessary, ensuring logical clarity when deriving sub-branches.
    4. Allow branch nodes to contain sub-branch nodes, judging whether the current node’s knowledge can be further refined to support multi-level knowledge granularity.
    5. Avoid content duplication by integrating related information into a single branch.
    6. Use NODE as the identifier for branch nodes.

3. Leaf Nodes
Objective:
    Refine specific knowledge into the smallest unit, representing the bottom layer of the Knowledge Tree.
Rules:
    1. Each leaf node describes only a single knowledge point, using concise phrases or short sentences, avoiding lengthy information.
    2. When extracting key information from the original data, summarize and organize it to ensure completeness and clarity as a standalone knowledge point.
    3. Ensure that the path from the root node to the leaf node forms a complete knowledge chain, with no overlap between branches and all branches covering the entire information.
    4. Use LEAF as the identifier for leaf nodes, with numerical numbering.

4. Output Requirements
Output the constructed Knowledge Tree in JSON format. Reference JSON format:
{{
    "ROOT": "xxx",
    "Children": [
        {{
            "NODE": "xxx",
            "Children": [
                {{"LEAF_1}}": "xxx"}},
                {{
                    "NODE": "xxx",
                    "Children": [{{"LEAF_2": "xxx"}}]
                }}
            ]
        }}
    ]
}}
Based on the above requirements, construct the Knowledge Tree using the following information.
Original Log: {}
Document information: {}
'''

    instruct = '''
As a log analysis expert, please construct instruction fine-tuning data covering multiple cognitive levels based on the provided logs and their corresponding knowledge tree. Follow these requirements:
1. Instruction Difficulty Modeling:
	Align with Bloom's Taxonomy of cognitive levels to build instruction data across three levels:
	L1 (Simple): {"Capability Requirement": "Remember & Understand", "Instruction Characteristics": "Require factual knowledge without reasoning, focus on the main issue of the log"}
	L2 (Medium): {"Capability Requirement": "Apply & Analyze", "Instruction Characteristics": "Involve associations across ≥2 knowledge branches, requiring logical deduction and knowledge integration"}
	L3 (Hard): {"Capability Requirement": "Synthesize & Predict", "Instruction Characteristics": "Include hypothetical scenarios and constraints, answers requiring systematic analysis, knowledge reorganization, and predictive reasoning."}
2. Data Construction Guidelines:
	Quantity: 3 instruction samples for L1 level, 2-3 instruction samples for L2 and L3 Level. The constructed instruction questions stand in the perspective of the operators to understand and analyze the logs. 
	Instruction Diversity: Ensure diverse instruction question and sentence structure, avoid repetitive interrogative words (e.g., what/how/why).
	Coverage: Use a breadth-first traversal algorithm approach to ensure the final instruction set covers more than 90% of the leaf nodes.  
	Answer Requirements: 
     - Provide answers for all constructed instructions.  
     - Simple instruction are answered directly, medium and difficult instructions need to be answered by providing step-by-step reasoning processes and avoiding direct replication of the content of the knowledge tree.

3. Data Output Specifications:  
   - include a generation plan specifying how to build instructions.
   - Refer to the following example and use the same JSON template, where the `instruction` field represents the generated question, the `input` field represents the raw log content (must follow the raw log), and the `output` field represents the reasoning result:
     {
       "Plan": "Using breadth-first traversal of the knowledge tree, instructions are designed to cover 100% of leaf nodes (7/7). L1 focuses on direct knowledge recall, L2 combines cause-procedure branches, and L3 introduces failure scenarios requiring synthesis of configuration limits, route attributes, and support procedures.",
       "Data": [
         {
            "Easy": {
             "instruction": "As an operations engineer, examine the following log entry and provide a clear and concise interpretation.",
             "input": "BGP/4/ASPATH_EXCEED_MAXNUM:The number of AS-PATHs exceeded the limit([limit-value]). (Operation=[STRING])",
             "output": "The number of AS_Paths has exceeded the maximum allowed value."
           }
         },
         {
            "Medium": {
                "instruction": "A network engineer sees this alert during maintenance. Outline their next diagnostic actions considering both configuration and route factors.",
                "input": "BGP/4/ASPATH_EXCEED_MAXNUM:The number of AS-PATHs exceeded the limit([limit-value]). (Operation=[STRING])",
                "output": "1) First assess service impact to determine urgency. 2) Compare current AS-Path limit (using 'display bgp' commands) with typical route attributes. 3) If limit is below observed AS-Path lengths, execute 'as-path-limit' adjustment in BGP view. 4) If limit appears adequate, analyze recent route updates for abnormal AS-Path extensions."
            }
         },
         {
            "Hard": {
                "instruction": "Predict the cascading network impacts if this alert is ignored in a multi-homed BGP environment with route reflectors.",
                "input": "BGP/4/ASPATH_EXCEED_MAXNUM:The number of AS-PATHs exceeded the limit([limit-value]). (Operation=[STRING])",
                "output": "1) Route instability: Exceeding AS-Path limits causes route flapping as paths are rejected/readvertised. 2) Routing table incompleteness: Critical routes might be missing on some nodes. 3) Traffic blackholes: Asymmetric path acceptance between routers. 4) CPU spikes: Constant route recalculations strain network devices. 5) Eventually, BGP sessions might reset due to update processing failures, causing service outages."
            }
         }
       ]
     }
    '''
    result = []
    cnt = 1

    cc = 0
    for i in range(cc, len(data)):
        first_question = prompt_knowledge_tree.format(data[i][0], data[i][1])
        msg = [{"role": "user", "content": first_question}]
        tree = get_content(msg)
        msg.append({"role": "assistant", "content": tree})
        msg.append({"role": "user", "content": instruct})
        full_content = get_content(msg)
        print(i,full_content)
        result.append([i,data[i][0], tree, full_content])

        if i == len(data) - 1:
            save_json(result, output_file)

    return result

