import torch

# ==========================================
# Target Token Processing
# ==========================================

MODEL_CONFIGS = {
    'Qwen/Qwen3-4B-Instruct-2507': {
        "insert_prefix_space": True,
        "use_system_msg": True
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "insert_prefix_space": False,
        "use_system_msg": True
    },
    'meta-llama/Llama-3.1-8B-Instruct': {
        "insert_prefix_space": True,
        "use_system_msg": True
    },
    'mistralai/Mistral-7B-Instruct-v0.3': {
       "insert_prefix_space": True,
       "use_system_msg": False
    },
    'Qwen/Qwen2.5-32B-Instruct': {
       "insert_prefix_space": True,
       "use_system_msg": True
    },
}

# ==========================================
# Data Collation
# ==========================================

def add_special_token(prompt, tokenizer):
    return tokenizer.decode(
        tokenizer.encode(prompt, add_special_tokens=True)
    )

def collate_known(batch, tokenizer, use_system_prompt = True):
    ids, prompts, contexts, answers = [], [], [], []
    for sample in batch:
        ids.append(sample["known_id"])
        contexts.append(sample['prompt'])
        answers.append(sample["attribute"])
        prompts.append(add_special_token(sample['prompt'], tokenizer))
    return ids, prompts, contexts, answers

def collate_ioi(batch, tokenizer, use_system_prompt = True):
    ids, prompts, contexts, answers = [], [], [], []
    for sample in batch:
        ids.append(sample["case_id"])
        contexts.append(sample['prompt'])
        answers.append(sample["answer"])
        prompts.append(add_special_token(sample['prompt'], tokenizer))
    return ids, prompts, contexts, answers

def collate_squad(batch, tokenizer, use_system_prompt = True):
    ids, prompts, contexts, answers = [], [], [], []
    for sample in batch:
        ids.append(sample["id"])
        contexts.append(sample["context"])
        answers.append(sample["answer"])
        
        if use_system_prompt:
            chat = [
                {'role': 'system', 'content': "You are a highly precise reading comprehension AI. Your task is to extract the exact answer to the user's question from the provided context. Answer with ONLY the specific keyword or short phrase found in the text."},
                {'role': 'user', 'content': f"Context: {contexts[-1]}\nQuestion: {sample['question']}"},
                {'role': 'assistant', 'content': "Answer:"}
                ]
        else:
            chat = [
                {'role': 'user', 'content': f"You are a highly precise reading comprehension AI. Your task is to extract the exact answer to the user's question from the provided context. Answer with ONLY the specific keyword or short phrase found in the text.\nContext: {contexts[-1]}\nQuestion: {sample['question']}"},
                {'role': 'assistant', 'content': "Answer:"}
                ]

        prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True))
    return ids, prompts, contexts, answers

def collate_imdb(batch, tokenizer, use_system_prompt = True):
    ids, prompts, contexts, answers = [], [], [], []
    for sample in batch:
        ids.append(sample["id"])
        contexts.append(sample["text"])
        answers.append(sample["answer"])
        
        if use_system_prompt:
            chat = [
                {'role': 'system', 'content': "Classify the sentiment of the following review. Answer either 'Positive' or 'Negative'."},
                {'role': 'user', 'content': f"Review: {contexts[-1]}"},
                {'role': 'assistant', 'content': "Answer:"}
                ]
        else:
           chat = [
                {'role': 'user', 'content': f"Classify the sentiment of the following review. Answer either 'Positive' or 'Negative'.\n\nReview: {contexts[-1]}"},
                {'role': 'assistant', 'content': "Answer:"}
                ] 
        prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True))
    return ids, prompts, contexts, answers

DATASET_CONFIGS = {
    "known_1000": {
        "process_fn": collate_known
    },
    "ioi": {
        "process_fn": collate_ioi
    },
    "squad_v2.0": {
        "process_fn": collate_squad
    },
    "imdb": {
        "process_fn": collate_imdb
    }
}
