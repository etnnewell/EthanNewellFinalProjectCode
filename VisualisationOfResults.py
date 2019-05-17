# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update(plt.rcParamsDefault)
plt.style.use("Solarize_Light2")
plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["font.weight"] = "550"
plt.rcParams["axes.labelweight"] = "550"

pd.set_option("display.max_columns",10)


df = pd.read_csv("ReplicationCSV.csv",index_col=None,header=0)

originals = df[df["Original"]=="Yes"]
replications = df[df["Original"]=="No"]

plt.bar(originals["Model"],originals["Acc"],color="darkblue", label="Originals")
plt.bar(replications["Model"],replications["Acc"],color="cornflowerblue",alpha=1, label="Replications")
plt.ylim(60,100)
plt.xlabel("Models")
plt.ylabel("Accuracy %")
plt.title("Accuracy of Replications vs Originals")
plt.xticks(rotation=-20)
plt.legend()
plt.show()

plt.plot(originals["Model"],originals["Acc"],color="darkblue", label="Originals",linestyle="--",marker="o")
plt.plot(replications["Model"],replications["Acc"],color="cornflowerblue",alpha=1, label="Replications",linestyle=":",marker="o")
plt.ylim(80,100)
plt.xlabel("Models")
plt.ylabel("Accuracy %")
plt.title("Accuracy of Replications vs Originals")
plt.xticks(rotation=-20)
plt.xticks(np.arange(5),("P4-48","P4M-24","P4M-64","DenseNet40\nCIFAR","DenseNet-BC100\nCIFAR"))
plt.legend()
plt.tight_layout()
#plt.savefig("ReplicationsComparison.png",dpi=300,transparent=True)
plt.show()

df2 = pd.read_csv("OptimiserCSV.csv",index_col=None,header=0)

RMS = df2[df2["Optimiser"]=="RMS"]
Adam = df2[df2["Optimiser"]=="Adam"]

plt.plot(RMS["Model"],RMS["Acc"],color="darkblue", label="RMSProp",linestyle="--",marker="o")
plt.plot(Adam["Model"],Adam["Acc"],color="cornflowerblue",alpha=1, label="Adam",linestyle=":",marker="o")
plt.plot(RMS["Model"],RMS["Aug_acc"],color="darkolivegreen",linestyle="--",marker="o",label="Aug_RMS")
plt.plot(Adam["Model"],Adam["Aug_acc"],color="yellowgreen",alpha=1, linestyle=":",marker="o",label="Aug_Adam")
plt.xlabel("Models")
plt.ylabel("Accuracy %")
plt.title("Optimiser Accuracy Comparison")
plt.legend()
plt.tight_layout()
#plt.savefig("OptimiserComparison.png",dpi=300,transparent=True)
plt.show()

df3 = pd.read_csv("HyperparameterCSV.csv",index_col=None,header=0)

aTypes = df3[df3["ModelType"]=="A"]
bTypes = df3[df3["ModelType"]=="B"]
cTypes = df3[df3["ModelType"]=="C"]

plt.plot(aTypes["Growth-Rate"],aTypes["Accuracy"], color="crimson", label="Reactive Train, 0.2 Dropout",linestyle="--",marker="o")
plt.plot(bTypes["Growth-Rate"],bTypes["Accuracy"], color="darkolivegreen", label="Set Train, 0.2 Dropout",linestyle="--",marker="o")
plt.plot(cTypes["Growth-Rate"],cTypes["Accuracy"], color="magenta", label="Set Train, 0.5 Dropout",linestyle="--",marker="o")

plt.xlabel("Filter Growth Rate")
plt.ylabel("Accuracy %")
plt.title("Accuracy of Models by Growth Rate and Type")
plt.xticks([6,12,16])
plt.legend()
plt.tight_layout()
#plt.savefig("HyperparameterComparison.png",dpi=300,transparent=True)

plt.show()

plt.clf()


plt.plot(df3.groupby(["Growth-Rate"])["Accuracy"].mean(),color="royalblue", label="Average Acc",linestyle="--",marker="o")
plt.plot(df3.groupby(["Growth-Rate"])["AUC"].mean(),color="darkorange", label="Average AUC",linestyle="--",marker="o")
#plt.ylim(83.6,85)
plt.xticks([6,12,16])
plt.xlabel("Filter Growth Rate")
plt.ylabel("Accuracy and AUC %")
plt.title("Average Accuracy and AUC of Models Grouped by Growth Rate")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("GrowthRateComparison.png",dpi=300,transparent=True)
plt.show()


df4 = pd.read_csv("DenseBlocksComparisonCSV.csv",index_col=None,header=0)
df4.set_index("Blocks",inplace=True,append=False)
plt.plot(df4["Acc"], color="royalblue", label="Accuracy",linestyle="--",marker="o")
plt.plot(df4["AUC"], color="darkorange", label="AUC",linestyle="--",marker="o")
plt.xlabel("Number of Dense Blocks")
plt.ylabel("Accuracy and AUC %")
plt.title("Accuracy and AUC of Narrow Models by Number of Dense Blocks")
plt.xticks(np.arange(3,8),[3,4,"5*",6,7])

plt.legend()
plt.tight_layout()
#plt.savefig("DenseBlocksComparison.png",dpi=300,transparent=True)
plt.show()













