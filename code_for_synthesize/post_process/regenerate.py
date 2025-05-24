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
You are an advanced AI assistant designed to generate comprehensive instructions from structured knowledge tree. Follow these instructions rigorously:

Task Definition:
    1. Synthesize instructions that fully cover all leaf nodes of the knowledge tree derived from the input log.
    2. Ensure no critical information in the log is omitted.
    3. If coverage < 90%, generate additional simple instructions targeting missing leaf nodes until coverage ≥ 90%.
    
Raw Log: {}    
Knowledge Tree: {}
Raw Generation: {} 
'''
    result = []
    cnt = 1
    cc = 0
    for i in range(cc, len(data)):
        first_question = prompt.format(data[i][0],data[i][1],data[i][2])
        msg = [{"role": "user", "content": first_question}]
        full_content = get_content(msg)
        print(i,full_content)
        result.append([i,data[i][0],  full_content])
        if i == len(data) - 1:
            save_json(result, save_path)

    return result
