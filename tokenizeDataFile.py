import os
import argparse
import re

from transformers import AutoTokenizer

import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as spark_types

KEYWORDS = {
    'introduction': 'i',
    'case': 'i',
    'purpose': 'i',
    'objective': 'i',
    'objectives': 'i',
    'aim': 'i',
    'summary': 'i',
    'findings': 'l',
    'background': 'i',
    'background/aims': 'i',
    'literature': 'l',
    'studies': 'l',
    'related': 'l',
    'methods': 'm',
    'method': 'm',
    'techniques': 'm',
    'methodology': 'm',
    'results': 'r',
    'result': 'r',
    'experiment': 'r',
    'experiments': 'r',
    'experimental': 'r',
    'discussion': 'c',
    'limitations': 'd',
    'conclusion': 'c',
    'conclusions': 'c',
    'concluding': 'c'}

def clean_section_names(sections):
  cleaned = [None] * len(sections)
  for i in range(len(sections)):
    tmp = re.sub("\[sec\d*\]", "", sections[i])
    tmp = re.sub("\[sec:level\d*\]", "", tmp)
    tmp = re.sub("\*", "", tmp)
    cleaned[i] = tmp
  return cleaned


def section_match(keywords):
    def section_match_(sections):
        match = False
        for section in sections:
            section = section.lower().split()
            for wrd in section:
                try:
                    match = KEYWORDS[wrd]
                except KeyError:
                    continue
        return 1 if match else 0
    return F.udf(section_match_, spark_types.ByteType())

def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_root", type=str, help="")
  parser.add_argument("--partitions", type=int, default=500, help="")

  args, unknown = parser.parse_known_args()
  return args, unknown


def main():
  args, unknown = read_args()
  
  conf = pyspark.SparkConf()
  sc = pyspark.SparkContext(conf=conf)
  spark = pyspark.sql.SparkSession(sc)

  output_dir = os.path.join(args.data_root, "counting")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  b_keywords = sc.broadcast(KEYWORDS)

  LEDtokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384-arxiv")
  PXtokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-base")

  def count_LEDtokens(text):
    return LEDtokenizer.encode(text, return_tensors='pt').size()[1]
  def count_PXtokens(text):
    return PXtokenizer.encode(text, return_tensors='pt').size()[1]

  clean_section_names_udf = F.udf(clean_section_names, spark_types.ArrayType(spark_types.StringType()))
  count_LEDtokens_udf = F.udf(count_LEDtokens, spark_types.IntegerType())
  count_PXtokens_udf = F.udf(count_PXtokens, spark_types.IntegerType())
  orig_test = os.path.join(args.data_root, "test.txt")
  orig_val = os.path.join(args.data_root, "val.txt")
  test_df = spark.read.json(orig_test)
  df = test_df.union(spark.read.json(orig_val)).repartition(args.partitions, "article_id")

  df = df.withColumn("section_names", clean_section_names_udf("section_names")) \
      .withColumn("cleaned_abstract", F.concat_ws(" ", F.col("abstract_text"))) \
      .withColumn("cleaned_abstract", F.regexp_replace("cleaned_abstract", "<\/?S>", ""))
  df = df.withColumn("LEDtextT", count_LEDtokens_udf(F.concat_ws(" ", F.col("article_text")))).withColumn("PXtextT", count_PXtokens_udf(F.concat_ws(" ", F.col("article_text")))) \
      .withColumn("LEDabsT", count_LEDtokens_udf("cleaned_abstract")).withColumn("PXabsT", count_PXtokens_udf("cleaned_abstract"))

  df = df.drop('cleaned_abstract')

  df = df.where(F.col("LEDtextT") <= 16384)
  df = df.where(F.col("PXtextT") <= 16384)
  df = df.withColumn("match", section_match(b_keywords)("section_names")).where(F.col("match") == True)
  df = df.orderBy(F.col("LEDtokens"), F.col("PXtokens"), ascending=False).limit(5000).drop("LEDtokens", "PXtokens").orderBy(F.rand())
  
  df.write.json(path=output_dir, mode="overwrite")

  os.system("cat " + output_dir + "/part-* >" + args.data_root + "/countedTokens.txt")
  os.system("rm -r " + output_dir)

if __name__ == "__main__":
  main()
