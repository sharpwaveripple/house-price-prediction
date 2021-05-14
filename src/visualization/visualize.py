import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../../data/raw/train.csv")
df['SalePrice_log'] = np.log(df["SalePrice"])

# df["MSSubClass"].replace(
#     {20: }
# )
# enc = OrdinalEncoder()
# df["MSSubClass"] = enc.fit_transform(df[["MSSubClass"]])

sns.catplot(data=df, x="MSSubClass", y="SalePrice")
plt.show()

sns.catplot(data=df, x="MSZoning", y="SalePrice")
plt.show()

sns.jointplot(x="LotFrontage", y="SalePrice", data=df, kind='reg')
plt.show()

sns.jointplot(x="LotArea", y="SalePrice", data=df, kind='reg')
plt.show()

sns.catplot(data=df, x="Neighborhood", y="SalePrice")
plt.show()
