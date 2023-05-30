import torch
import json
from tqdm import tqdm
#from query_expansion import expandQueryPipeline
from query_expansion import expandQueryPipeline

input_file_path = '/ssd0/byeongguk/DPR/results/NQ_test/NQ_test_top100_all.json'
#input_file_path = '/ssd0/byeongguk/DPR/dataset/downloads/data/retriever/nq-adv-hn-train.json'
output_file_path = '/ssd0/byeongguk/DPR/results/NQ_test/NQ_test_top100_all_QE_2.json'
#output_file_path = '/ssd0/byeongguk/DPR/dataset/downloads/data/retriever/nq-adv-hn-train_qe.json'


query_expansion_model = expandQueryPipeline(
    "mosaicml/mpt-7b-instruct",
    torch_dtype=torch.bfloat16,
    is_trust_remote_code=True,
)

with open(input_file_path, 'r', encoding="utf8") as input_file :
    data = json.load(input_file)
    for i, instance in tqdm(enumerate(data)) :
        if i%100 == 0:
            print(i)
        ## add expanded query
        original_query = instance['question']
        expanded_query = query_expansion_model.generate_one(original_query)
        data[i]['question'] = expanded_query
        
    
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile)