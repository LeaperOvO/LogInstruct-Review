from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json


def print_outputs(outputs):
    oo = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        oo.append(generated_text)
        print(f"Generated text: {generated_text!r}")

    print("-" * 80)
    print('\n')
    return oo


def save_json(data, file):
    dict_json = json.dumps(data, indent=1)
    with open(file, 'w+', newline='\n') as file:
        file.write(dict_json)


def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content

def generate(model_path,data,save_path):
    model_path = model_path



    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, top_k=50, max_tokens=10000)

    # sampling_params = SamplingParams(temperature=0.8, top_p=0.6,top_k=100, max_tokens=2048)
    # llm = LLM(model=model_path, enable_lora=True, tensor_parallel_size=1, trust_remote_code=True, max_model_len=10000)
    llm = LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True,max_model_len=2048)


    prompt = '''
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

    conversations = []
    labels = []
    tts = []
    for i in range(0, len(data)):
        conversation = [
            {
                "role": "user",
                "content": prompt.format(data[i][0], data[i][1])
            }
        ]

        conversations.append(conversation)
        labels.append(data[i][0])
    print(len(labels))
    result = []
    for i in range(0, len(conversations), 8):
        conversation = conversations[i: i + 8]
        label = labels[i: i + 8]
        outputs = llm.chat(conversation, sampling_params, use_tqdm=True)
        tree = print_outputs(outputs)
        print(i)
        for j in range(len(tree)):
            conversation[j].append({"role": "assistant", "content": tree[j]})
            conversation[j].append({"role": "user", "content": instruct})
        outputs = llm.chat(conversation, sampling_params, use_tqdm=True)
        oo = print_outputs(outputs)
        for j in range(len(oo)):
            result.append([label[j], oo[j]])
        print('\n')
    # base qwen1.5
    save_json(result,
              save_path)



