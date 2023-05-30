import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, Tuple, List
from tqdm import tqdm
import time
from collections import deque

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

Prompt_Template = [
"""
expand follwing query with various term and find implicit meaning of question. /
let me show a example 

Query: 'Who is the best soccer player?'
Query Expansion: best soccer player mean the player who score many goal in game. or best soccer player mena the player who won prize many time. So the original question rewrite
Who scored the most goals in the soccer game or receive most prize?

Query: '{original_query}'
Query Expansion:
""",
"""
Write a long paragraph about '{original_query}'. The paragraph has to be more than 300 words.
"""
]

BEST_PROMPT_TAMPLATE = """
expand follwing query with various term and find implicit meaning of question. 
let me show some example 

### Query: Who is the best soccer player?
### Expansion point : best soccer player  mean the player who score many goal in game. or best soccer player mena the player who won prize many time. 
### Query Expansion : Who scored the most goals in the soccer game or who receive most prize in the soccer?

### Query: What are the major cities in France?
### Expansion point : major city mean capital, the most populous cities, most wellness cities
### Query Expansion : What are the major cities like capital, most populous or most wellness cities in France?

### Query: {original_query}
"""

WITHOUT_EP_PROMPT_TAMPLATE = """
expand follwing query with various term and find implicit meaning of question. 
let me show some example 

### Query: Who is the best soccer player?
### Query Expansion : Who scored the most goals in the soccer game or who receive most prize in the soccer?

### Query: What are the major cities in France?
### Query Expansion : What are the major cities like capital, most populous or most wellness cities in France?

### Query: {original_query}
"""

OPTIONS = [(0.1,256), (0.4,256), (0.7,256)]

## query expansion pipeline based MPT-7B
class expandQueryPipeline :
    def __init__ (
        self,
        model_name = 'mosaicml/mpt-7b-instruct',
        torch_dtype = torch.bfloat16,
        is_trust_remote_code = True,
        is_use_auth_token = None,
    ) :
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=is_trust_remote_code,
            use_auth_token=is_use_auth_token,
        ) 
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=is_trust_remote_code,
            use_auth_token=is_use_auth_token
        )
        
        if tokenizer.pad_token_id is None :
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left' ## input encoding
        self.tokenizer = tokenizer
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device=device, dtype=torch_dtype)
        
        self.advanced_options = {
            "temperature": 0.1,
            "top_p": 0.92,
            "top_k": 0,
            "max_new_tokens": 512,
            "use_cache": False,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.1,  # 1.0 means no penalty, > 1.0 means penalty, 1.2 from CTRL paper
        }
        
        
    def generate(
        self, original_query, **advanced_option: Dict[str, Any]
    ) -> List[Tuple[str, str, float]]:
        prompts = [PROMPT_FOR_GENERATION_FORMAT.format(instruction = tamplate.format(original_query = original_query)) for tamplate in Prompt_Template]
        output_text = []
        for prompt in prompts :
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.model.device)
            options = {**self.advanced_options, **advanced_option} 
            with torch.no_grad():
                ouput_ids = self.model.generate(input_ids, **options)
            new_tokens =ouput_ids[0, len(input_ids[0]) :]
            output_text.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return output_text
    
    def multipleGenerate(
        self, original_query, **advanced_option: Dict[str, Any]
    ) -> List[Tuple[str, str, float]]:
        prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction = WITHOUT_EP_PROMPT_TAMPLATE.format(original_query = original_query))
        output_text = []
        for option in OPTIONS :
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.model.device)
            advanced_option['temperature'] = option[0]
            options = {**self.advanced_options, **advanced_option} 
            with torch.no_grad():
                ouput_ids = self.model.generate(input_ids, **options)
            new_tokens =ouput_ids[0, len(input_ids[0]) :]
            output_text.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return output_text
    
    def remove_prefix(self, text, prefix) :
        if text.startswith(prefix) :
            return text[len(prefix):]
        return text
        
    def generate_one(
        self, original_query, **advanced_option: Dict[str, Any]
    ) -> Tuple[str, str, float]:
        prompt = WITHOUT_EP_PROMPT_TAMPLATE.format(original_query = original_query)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)
        options = {**self.advanced_options, **advanced_option} 
        with torch.no_grad():
            ouput_ids = self.model.generate(input_ids, **options)
        new_tokens =ouput_ids[0, len(input_ids[0]) :]
        expansion_point = self.remove_prefix(self.tokenizer.decode(new_tokens, skip_special_tokens=True), '### expansion points : ')
        expansion_point = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return expansion_point + '\n' + original_query

                
def test():
    print("---start test---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_expansion_model = expandQueryPipeline(
        "mosaicml/mpt-7b-instruct",
        torch_dtype=torch.bfloat16,
        is_trust_remote_code=True,
    )
    
    original_query = 'the south west wind blows across nigeria between'
    
    print("-------------test1---------------\n")
    output = query_expansion_model.generate_one(original_query)
    print(output)
    
    print("-------------test2---------------\n")
    output = query_expansion_model.multipleGenerate(original_query)
    print(output)

if __name__ == "__main__":
    print("test")
    test()
    
