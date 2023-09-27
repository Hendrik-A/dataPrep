import os
import argparse

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

  clean_section_names_udf = F.udf(clean_section_names, spark_types.ArrayType(spark_types.StringType()))

  b_keywords = sc.broadcast(KEYWORDS)
  
  data_path = os.path.join(args.data_root, 'countedTokens.txt')
  df = spark.read.json(data_path).repartition(500, "article_id")

  df = df.where(F.col('LEDtokens') <= 16384)
  df = df.where(F.col('PXtokens') <= 16384)
  df = df.withColumn('section_names', clean_section_names_udf('section_names'))
  df = df.withColumn('match', section_match(b_keywords)('section_names'))
  df = df.filter(df.match == True)
  df = df.orderBy(F.col('LEDtokens'), F.col('PXtokens'), ascending=False).limit(5000).drop("LEDtokens", "PXtokens").orderBy(F.rand())
  
  rows = df.count()

  train_df = df.limit(round(rows*0.8))
  valid_test_df = df.filter(~df["article_id"].isin(list(train_df.select(train_df.article_id).toPandas()['article_id'])))
  valid_df = valid_test_df.limit(round(valid_test_df.count()*0.5))
  test_df = valid_test_df.filter(~valid_test_df["article_id"].isin(list(valid_df.select(train_df.article_id).toPandas()['article_id'])))

  train_df.write.json(path=os.path.join(args.data_root, "train"), mode="overwrite")
  test_df.write.json(path=os.path.join(args.data_root, "test"), mode="overwrite")
  valid_df.write.json(path=os.path.join(args.data_root, "val"), mode="overwrite")

  os.system('cat ' + args.data_root + '/train/part-* >' + args.data_root + '/train.txt')
  os.system('cat ' + args.data_root + '/val/part-* >' + args.data_root + '/val.txt')
  os.system('cat ' + args.data_root + '/test/part-* >' + args.data_root + '/test.txt')

  os.system('rm -r ' + args.data_root + '/train')
  os.system('rm -r ' + args.data_root + '/val')
  os.system('rm -r ' + args.data_root + '/test')

if __name__ == "__main__":
  main()
