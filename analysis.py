import os
import argparse
import re
import pandas as pd

from transformers import AutoTokenizer

def write_text(sentences):
  tmp = ' '.join(sentences)
  tmp = re.sub("<\/?S>", "", tmp)
  return tmp

    
def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_root", type=str, help="")
  parser.add_argument("--file", type=str, help="")
  parser.add_argument("--type", type=str, help="")
  parser.add_argument("--text_col", type=str)
  parser.add_argument("--sum_col", type=str)
  
  args, unknown = parser.parse_known_args()
  return args, unknown

def main():
  args, unknown = read_args()

  result_dir = os.path.join(args.data_root, "analysis")
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
  result_file = os.path.join(result_dir, 'analysis_'+args.type+'_'+args.file)

  df = pd.read_json(os.path.join(args.data_root, args.file), lines=True)

  if 'LEDtokens' in df.columns:
    df = df.drop('LEDtokens', axis=1)
    
  if 'PXtokens' in df.columns:
    df = df.drop('PXtokens', axis=1)

  if isinstance(df[args.sum_col][0], list):
    df['tmp_abstract'] = df[args.sum_col].apply(write_text)
    df = df.drop(args.sum_col, axis=1)
    df = df.rename(columns={'tmp_abstract': args.sum_col})

  if isinstance(df[args.text_col][0], list):
    df['tmp_text'] = df[args.text_col].apply(write_text)
    df = df.drop(args.text_col, axis=1)
    df = df.rename(columns={'tmp_text': args.text_col})
      
  LEDtokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384-arxiv")
  PXtokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-base")

  def getLEDTokenCount(text):
    return LEDtokenizer.encode(text, return_tensors='pt').size()[1]

  def getPXTokenCount(text):
    return PXtokenizer.encode(text, return_tensors='pt').size()[1]

  df['LEDsum'] = df[args.sum_col].apply(getLEDTokenCount)
  df['LEDtext'] = df[args.text_col].apply(getLEDTokenCount)
  df['PXsum'] = df[args.sum_col].apply(getPXTokenCount)
  df['PXtext'] = df[args.text_col].apply(getPXTokenCount)
    
 
  with open(result_file, "w+") as writer:
    writer.write("Total entries:"+str(df['article_id'].count())+"\n")
    writer.write("LED: Avg Summary Tokens:"+str(df['LEDsum'].mean())+"\n")
    writer.write("LED: Median Summary Tokens:"+str(df['LEDsum'].median())+"\n")
    writer.write("LED: Min Summary Tokens:"+str(df['LEDsum'].min())+"\n")
    writer.write("LED: Min Summary Tokens:"+str(df['LEDsum'].max())+"\n")
    writer.write("LED: Avg Text Tokens:"+str(df['LEDtext'].mean())+"\n")
    writer.write("LED: Median Text Tokens:"+str(df['LEDtext'].median())+"\n")
    writer.write("LED: Min Text Tokens:"+str(df['LEDtext'].min())+"\n")
    writer.write("LED: Min Text Tokens:"+str(df['LEDtext'].max())+"\n")  

    writer.write("Pegasus-X Avg Summary Tokens:"+str(df['PXsum'].mean())+"\n")
    writer.write("Pegasus-X Median Summary Tokens:"+str(df['PXsum'].median())+"\n")
    writer.write("Pegasus-X Min Summary Tokens:"+str(df['PXsum'].min())+"\n")
    writer.write("Pegasus-X Min Summary Tokens:"+str(df['PXsum'].max())+"\n")
    writer.write("Pegasus-X Avg Text Tokens:"+str(df['PXtext'].mean())+"\n")
    writer.write("Pegasus-X Median Text Tokens:"+str(df['PXtext'].median())+"\n")
    writer.write("Pegasus-X Min Text Tokens:"+str(df['PXtext'].min())+"\n")
    writer.write("Pegasus-X Min Text Tokens:"+str(df['PXtext'].max())+"\n")

if __name__ == "__main__":
  main()
