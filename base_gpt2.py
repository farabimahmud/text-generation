from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os 
import pandas as pd

DATASET_FILENAME = os.path.join(os.getcwd(),"dataset.csv")


def main():
  dir = DATASET_FILENAME
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  
  input_ids = []
  df = pd.read_csv(dir)
  for i,row in df.iterrows():
    input_ids.append(tokenizer.encode(row['abstract'], return_tensors='pt'))

  # for i in input_ids:
  #   for iid in i:
  #     print(tokenizer.decode(iid.numpy()))

  # print(input_ids)
  model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
  model.train()
  
if __name__ == "__main__":
  main()