import json

def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content


def save_json(data, file):
    dict_json = json.dumps(data, indent=1)
    with open(file, 'w+', newline='\n') as file:
        file.write(dict_json)

def cal_rouge(path):
    all_query_list = []
    data = read_json(path)
    reference = []
    candidate = []
    for item in data:
        reference.append(item[0])
        candidate.append(item[1])

    from rouge import Rouge

    # init
    rouge = Rouge()
    c = 0
    count1 = 0.0
    count2 = 0.0
    count3 = 0.0

    for i in range(len(reference)):
        print(i)
        scores = rouge.get_scores(hyps=candidate[i], refs=reference[i], avg=True)
        rouge_1_score = scores['rouge-1']['f']
        rouge_2_score = scores['rouge-2']['f']
        rouge_l_score = scores['rouge-l']['f']
        count1 += rouge_1_score
        count2 += rouge_2_score
        count3 += rouge_l_score
        print(scores)
        print('\n')

    # print(c)
    print('Rouge1->>>',count1/len(candidate))
    print('Rouge2->>>',count2/len(candidate))
    print('Rouge3->>>',count3/len(candidate))
    return count1/len(candidate), count2/len(candidate), count3/len(candidate)