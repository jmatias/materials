import pandas as pd

splits = {'train': 'data/train-00000-of-00001-fa296d8e8d8c9c07.parquet', 'validation': 'data/validation-00000-of-00001-ca5407451a5f9454.parquet', 'test': 'data/test-00000-of-00001-6e352a5dc26e9a8a.parquet'}


df_train = pd.read_parquet("hf://datasets/zpn/clintox/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/zpn/clintox/" + splits["test"])
df_validation = pd.read_parquet("hf://datasets/zpn/clintox/" + splits["validation"])

df_train.drop("selfies", axis=1, inplace=True)
df_test.drop("selfies", axis=1, inplace=True)
df_validation.drop("selfies", axis=1, inplace=True)

df_train.to_csv("data/clintox_train.csv", index=False)
df_test.to_csv("data/clintox_test.csv", index=False)
df_validation.to_csv("data/clintox_validation.csv", index=False)
df = pd.concat([df_train, df_test, df_validation])
df.to_csv("data/clintox.csv", index=False)
