import json
import random

random.seed(12)

with open('chats.json', 'r') as file:
    chats = json.load(file)

with open('paraphrase_one_chat.json', 'r') as file:
    chats_para = json.load(file)

with open('paraphrase_many__chats.json', 'r') as file:
    chats_para_many = json.load(file)
    
interactions = []
for i in chats:
    answer = i['chat'].split('\n\nassistant: ')
    question = answer[0].split('user: ')
    interaction = {'length': i['length'], 'question': question[1], 'answer': answer[1]}
    interactions.append(interaction)

# for i in chats_para:
#     if i['length'] <= 1024:
#         answer = i['chat'].split('\n\n')
#         question = answer[0].split('user: ')
#         interaction = {'length': i['length'], 'question': question[1], 'answer': answer[1]}
#         interactions.append(interaction)

for i in chats_para_many:
    if i['length'] <= 1024:
        answer = i['chat'].split('\n\n')
        question = answer[0].split('user: ')
        interaction = {'length': i['length'], 'question': question[1], 'answer': answer[1]}
        interactions.append(interaction)

train_len = int(0.9 * len(interactions))
random.shuffle(interactions)

train = interactions[:train_len]
test = interactions[train_len:]

with open ('data_train_one.json', 'w') as file:
    json.dump(train, file, indent=4)

with open ('data_test_one.json', 'w') as file:
    json.dump(test, file, indent=4)