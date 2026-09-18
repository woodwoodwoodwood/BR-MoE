import torch
import pandas as pd
import json
from tqdm import tqdm
import time
import os

def eval_perplexity(model, tokenizer, result_save_path="."):
    if isinstance(result_save_path, str):
        save_file_path = os.path.join(result_save_path, "eval_result.json")
    else:
        # If no path provided, don't save results
        save_file_path = None
    begin = time.time()
    


    device = torch.device("cuda:0")
    # Try to find the wikitext2 test file in several common locations
    possible_paths = [
        "wikitext2_test-00000-of-00001.parquet",  # Current directory
        os.path.join(os.path.dirname(__file__), "wikitext2_test-00000-of-00001.parquet"),  # Evaluation directory
        "./evaluation/wikitext2_test-00000-of-00001.parquet",  # Evaluation directory
        "/data/wikitext2_test-00000-of-00001.parquet"  # Common data directory
    ]
    
    fname = None
    for path in possible_paths:
        if os.path.exists(path):
            fname = path
            break
    
    if fname is None:
        raise FileNotFoundError("Could not find wikitext2 test file. Please provide the correct path.")
    
    print(f"Using wikitext2 test file: {fname}")
    df = pd.read_parquet(fname)
    texts = df['text'].tolist()
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    max_length = 2048
    stride = 512
    seq_len = encodings.input_ids.size(1)
    

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
    # for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)

        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100


        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
            neg_log_likelihood = outputs.loss
       
        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()

    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as file:
            try:
                all_metrics = json.load(file)
            except json.JSONDecodeError:
                all_metrics = {} 
    else:
        all_metrics = {}
    all_metrics['wikitext2_ppl'] = ppl
    with open(save_file_path, 'w') as file:
        json.dump(all_metrics, file, indent=4)
    end = time.time()
    print(">>>>> Results <<<<<")
    print(f"Metrics: {all_metrics}")
    print(f"Evaluation time: {end - begin:.2f}s")

    return all_metrics


