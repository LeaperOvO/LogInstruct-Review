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

def generate(data, save_path, revise_num=10, wait_time=1):
    prompt = '''
Perform semantically equivalent paraphrasing on given instruction data to enhance diversity of instructions.
Requirements:
1. Only rewrite the 'instruction' field to generate new instructions with identical semantics but varied expressions (e.g., rephrased questioning, sentence restructuring, voice conversion, etc.).
2. Technical terminology needs to be consistent. 
3. Keep the 'input' field unchanged and ensure responses to rewritten instructions remain aligned with the original 'output' content.
4. Output three modified data entry in the list while preserving the original JSON structure.

Raw data: {}    
'''
    result = []
    cnt = 1
    cc = 0
    for i in range(cc, len(data)):
        first_question = prompt.format(data[i][1])
        msg = [{"role": "user", "content": first_question}]
        full_content = get_content(msg)
        print(i,full_content)
        result.append([i,data[i][0],  full_content])
        if i == len(data) - 1:
            save_json(result, save_path)

    return result
