# pip install mlxtend pandas
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["milk", "bread", "butter"],
    ["bread", "butter"],
    ["milk", "bread"],
    ["milk", "eggs"],
    ["bread", "butter", "eggs"],
    ["milk", "bread", "butter", "eggs"],
    ["bread", "eggs"],
    ["milk", "butter"],
]

te = TransactionEncoder()
data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(data, columns=te.columns_)

frequent = apriori(df, min_support=0.3, use_colnames=True)
print("Frequent Itemsets:\n", frequent)

rules = association_rules(frequent, metric="confidence", min_threshold=0.6)
print(rules[["antecedents", "consequents", "support",
             "confidence", "lift"]])
