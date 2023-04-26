from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
import torch

class ZeroShotCommenter:
    def __init__(self, templates=[]):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.templates = [
            """
            If you were a witty person, how would you reply to: {}
            """,
            """
            You are witty, sarcastic, funny, and hilarious. Now respond to: {}
            """,
            """
            Summarize: {}
            """,
            """
            Laugh at: {}
            """,                    

        ]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(self.device)

    
    def generate(self, text=""):
        template = np.random.choice(self.templates)
        input = template.format(text)
        print(input)
        input = self.tokenizer(input, return_tensors="pt")
        output = self.model.generate(**input, temperature=0.9, max_new_tokens=500)
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return output[0]

    def evaluate(self, eval_df):
        perplexities = []
        for _, example in eval_df.iterrows():
            template = np.random.choice(self.templates)
            input = template.format(example["Input"])
            input_ids = self.tokenizer.encode(input, return_tensors='pt').to(self.device)
            output_ids = self.tokenizer.encode(example['Output'], return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                loss = self.model(input_ids=input_ids, labels=output_ids).loss
            perplexity = torch.exp(loss)
            perplexities.append(perplexity.item())
        return (perplexities, np.mean(perplexities))