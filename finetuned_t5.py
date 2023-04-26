from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.optim import AdamW

class T5Commenter:
    def __init__(self):
        self.model_name = "finetuned-t5-small"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

    def finetune(self, data_filename):
        dataloader = self.create_dataloader(data_filename)
        NUM_EPOCHS = 15
        progress_bar = tqdm(range(NUM_EPOCHS * len(dataloader)))
        for epoch in range(NUM_EPOCHS):
            for step, data in enumerate(dataloader):
                input_ids, labels, attention_mask = data

                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels[labels == self.tokenizer.pad_token_id] = -100
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
        self.model.save_pretrained(self.model_name)

    def create_dataloader(self, data_filename):
        dataset = load_dataset("csv", data_files="reddit.csv")
        def tokenize_function(example):
            example["text"] = "question: what is a witty response to this?\n context: " + example["text"]
            text_tokenized = tokenizer(example["text"], padding="max_length", truncation=True)
            label_tokenized = tokenizer(example["comment_text"], padding="max_length", truncation=True)
            example["text_input_ids"] = text_tokenized["input_ids"]
            example["label_input_ids"] = label_tokenized["input_ids"]
            example["text_attention"] = text_tokenized["attention_mask"]
            example["label_attention"] = label_tokenized["attention_mask"]
            return example
        dataset = (
            dataset.map(tokenize_function)
            .remove_columns(
                ["submission_text", "Unnamed: 0", "submission_ids", "permalinks", "comment_scores", "comment_ids", "titles", "comment_text"]
            )
        )

        training_data = []
        # [
        #     (
        #         [], # input ids
        #         [], # output ids
        #         [] # attention mask
        #     ),
        #     (
        #         [],
        #         [],
        #         []
        #     ),
        # ]
        for row in dataset["train"]:
            training_data.append((np.array(row["text_input_ids"]), np.array(row["label_input_ids"]), np.array(row["text_attention"])))
        
        return DataLoader(training_data, batch_size=16, shuffle=True)

    def generate(self, text=None):
        inputs = self.tokenizer(
            """question: what is a witty response to this?
            context: {}
            """.format(text), 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        outputs = self.model.generate(**inputs, temperature=0.9, max_new_tokens=1000)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    def evaluate(self, eval_df):
        perplexities = []
        for _, example in eval_df.iterrows():
            input_ids = self.tokenizer.encode(example['Input'], return_tensors='pt').to(self.device)
            output_ids = self.tokenizer.encode(example['Output'], return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                loss = self.model(input_ids=input_ids, labels=output_ids).loss
            perplexity = torch.exp(loss)
            perplexities.append(perplexity.item())
        return (perplexities, np.mean(perplexities))
