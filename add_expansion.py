## sample code for using our query expansion pipeline

import json
from query_expansion import expandQueryPipeline
import torch
from operator import itemgetter

file_path = '/' ## bench mark
output_file_path = '/' ## output file

query_expansion_model = expandQueryPipeline(
    "mosaicml/mpt-7b-instruct",
    torch_dtype=torch.bfloat16,
    is_trust_remote_code=True,
)


with open(file_path, 'r', encoding="utf8") as file :
    data = json.load(file)
    for i, instance in enumerate(data) :
        data[i]['expanded_query_4'] = query_expansion_model.generate_one(instance['question'])

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile)
