#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
@author: Leaper
"""
from tqdm import tqdm
import os
import time
import json
import random

random.seed(2)


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
    Given a log analytics data in json format, where INSTRUCTION denotes the question, INPUT is the log and OUTPUT is the answer. 
    Now provide the knowledge tree of the log and ask you to determine whether the knowledge in the output align with the Knowledge Tree. 
    If correct output True, else output False and regenerate that instruction data in the list and provide the correct outputs. Only analyze the medium and hard level data.
    Output format: 
    [
        {{
          "Correctness": True
        }},
        {{
          "Correctness": True
        }},
        {{
          "Correctness": False,
          "RevisedData": {{
              "instruction": raw instruction,
              "input": raw input,
              "output": modified output
          }}
        }},
        {{
          "Correctness": true
        }}
    ] 

    Knowledge Tree: {}
    Raw Data: {}
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
