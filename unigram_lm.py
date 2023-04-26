from collections import Counter
import random
from transformers import AutoTokenizer
import math

class UnigramLM:
    def __init__(self, corpus):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        special_tokens = {"bos_token": "<START>", "eos_token": "<STOP>"}
        self.tokenizer.add_special_tokens(special_tokens)
        tokens = self.tokenizer.encode_plus(corpus.lower(), add_special_tokens=False)
        
        self.START_TOKEN = self.tokenizer.encode("<START>", add_special_tokens=False)[0]
        self.STOP_TOKEN = self.tokenizer.encode("<STOP>", add_special_tokens=False)[0]
        self.token_counts = Counter(tokens.input_ids)
    
    def compute_probability(self, token):
        return max([self.token_counts[token] / sum(self.token_counts.values()), 1e-10])

    def perplexity(self, tokens):
        log_probability = sum([math.log(self.compute_probability(token)) for token in tokens])
        perplexity = math.exp(-log_probability / len(tokens))
        return perplexity

    def evaluate(self, eval_df):
        """
        eval_df: pd.DataFrame of input and corresponding expected output

        returns:
        list of perplexities for each test string, and mean perplexity score for all texts
        """

        # ignore inputs here
        texts = eval_df["Output"]
        ps = []
        for text in texts:
            tokens = self.tokenizer.encode_plus(text.lower())
            ps.append(self.perplexity(tokens.input_ids))
        return ps, sum(ps) / len(ps)

    def generate(self, text = None, max_len=50):
        token = self.START_TOKEN # start token
        generated_tokens = [token]
        while token != self.STOP_TOKEN and len(generated_tokens) < max_len: # until stop
            token = random.choices(list(self.token_counts), weights=list(self.token_counts.values()))[0]
            if token == self.START_TOKEN:
                continue
            generated_tokens.append(token)
        mask = self.tokenizer.get_special_tokens_mask(generated_tokens, already_has_special_tokens=True)
        generated_tokens = [token for token, m in zip(generated_tokens, mask) if m != 1] 
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        