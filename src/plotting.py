import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import oapackage
import seaborn as sns
from datetime import datetime

sns.set()

from matplotlib import rcParams

rcParams["font.size"] = "40"
rcParams['text.usetex'] = False
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = (16.0, 9.0)
rcParams['figure.frameon'] = True
rcParams['figure.edgecolor'] = 'k'
rcParams['grid.color'] = 'white'
rcParams['grid.linestyle'] = 'solid'
rcParams['grid.linewidth'] = 2
rcParams['axes.linewidth'] = 1
rcParams['axes.edgecolor'] = 'k'
rcParams['axes.grid.which'] = 'minor'
rcParams['legend.frameon'] = 'True'
rcParams['legend.framealpha'] = 1
rcParams['legend.fontsize'] = 15

rcParams['ytick.major.size'] = 15
rcParams['ytick.major.width'] = 2
rcParams['ytick.minor.size'] = 6
rcParams['ytick.minor.width'] = 1
rcParams['xtick.major.size'] = 15
rcParams['xtick.major.width'] = 2
rcParams['xtick.minor.size'] = 6
rcParams['xtick.minor.width'] = 1
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15

results = {"n_params": [], "precision": [], "top3": []}
with open('final_results/dehb_eta_3_precison_0.6_modelsize_1000000.json') as f:
    data = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")
precision_constraint = 0.6
n_params_constraint = 1000000
inc_data = {"id": [], "precision": [], "top3": [], "n_params": []}
for line in range(len(data)):
    if data[line]["budget"] == 50:
            if data[line]["precision"] >= precision_constraint and data[line]["n_params"] <= n_params_constraint:
                inc_data["id"].append(line)
                inc_data["top3"].append(data[line]["top3"])
                inc_data["precision"].append(data[line]["precision"])
                inc_data["n_params"].append(data[line]["n_params"])
            else:
                results["n_params"].append(data[line]["n_params"])
                results["precision"].append(data[line]["precision"])
                results["top3"].append(data[line]["top3"])

incumbent_id = inc_data["id"][np.argmax(inc_data["top3"])]
incumbent = data[incumbent_id]["configuration"]
print(incumbent)
n_params = np.array(results["n_params"])
top3 = np.array(results["top3"])
precision = np.array(results["precision"])
best_precision = np.argmax(precision)
best_top3_idx = inc_data["id"]  # np.argmax(top3)
# top3 = np.delete(top3, inc_data["id"])
datapoints = np.random.rand(2, 50)
print(datapoints.shape)
best_top3_n_param = inc_data["n_params"]
# Accuracy vs N_params
plt.title('MO-DEHB', fontsize=20)
plt.scatter(n_params, top3, alpha=0.6, label="Not Satisfied")
plt.scatter(best_top3_n_param, inc_data["top3"], color="green", alpha=0.6, label="Satisfied")
plt.xlabel("Number of parameters", fontsize=20)
plt.ylabel("Top3-Accuracy", fontsize=20)
plt.xscale('log')
plt.ylim((0, 1))
plt.axvline(x=n_params_constraint, color='red')
plt.axhline(y=0.8, color="red")
plt.legend()
plt.grid()
plt.show()
#
best_precision = np.amax(precision)
# best_precision_idx = np.argmax(precision)
# precision = np.delete(precision, inc_data["id"])
#
# best_precision_n_param = n_params[inc_data["id"]]
# n_params_precision = np.delete(n_params, inc_data["id"])
# Precision vs N_params
plt.title('MO-DEHB', fontsize=20)
plt.scatter(n_params, precision, label="Not Satisfied", alpha=0.6)
plt.scatter(inc_data["n_params"], inc_data["precision"], color="green", alpha=0.6,
            label="Satisfied")
plt.xlabel("Number of parameters", fontsize=20)
plt.ylabel("Precision", fontsize=20)
plt.xscale('log')
plt.axvline(x=n_params_constraint, color='red')
plt.axhline(y=precision_constraint, color='red')
plt.legend()
plt.grid()
plt.show()
datapoints = np.concatenate((precision, top3))
datapoints = datapoints.reshape(2, -1)
print(datapoints.shape)
pareto = oapackage.ParetoDoubleLong()

# Pareto front 1
for i in range(0, datapoints.shape[1]):
    w = oapackage.doubleVector((datapoints[0, i], datapoints[1, i]))
    pareto.addvalue(w, i)
pareto.show(verbose=1)
lst = pareto.allindices()  # the indices of the Pareto optimal designs
optimal_datapoints = datapoints[:, lst]

# # Pareto front 2
# precision_2 = np.delete(precision, lst[0])
# top3_2 = np.delete(top3, lst[0])
# datapoints_2 = np.concatenate((precision_2, top3_2))
# datapoints_2 = datapoints_2.reshape(2, -1)
# for i in range(0, datapoints_2.shape[1]):
#     w = oapackage.doubleVector((datapoints_2[0, i], datapoints_2[1, i]))
#     pareto.addvalue(w, i)
# pareto.show(verbose=1)
# lst_2 = pareto.allindices()  # the indices of the Pareto optimal designs
# optimal_datapoints_2 = datapoints_2[:, lst_2]
# #
# # Pareto front 3
# datapoints_3_1 = np.delete(datapoints_2[0], lst_2)
# datapoints_3_2 = np.delete(datapoints_2[1], lst_2)
# datapoints_3 = np.concatenate((datapoints_3_1, datapoints_3_2))
# datapoints_3 = datapoints_3.reshape(2, -1)
# for i in range(0, datapoints_3.shape[1]):
#     w = oapackage.doubleVector((datapoints_3[0, i], datapoints_3[1, i]))
#     pareto.addvalue(w, i)
# pareto.show(verbose=1)
# lst_3 = pareto.allindices()  # the indices of the Pareto optimal designs
# optimal_datapoints_3 = datapoints_3[:, lst_3]
plt.title('MO-DEHB', fontsize=20)
plt.scatter(precision, top3, label="Not Satisfied", alpha=0.6)
plt.scatter(inc_data["precision"], inc_data["top3"], color="green", label="Satisfied", alpha=0.6)
plt.axhline(y=0.8, color="red")
plt.axvline(x=precision_constraint, color="red")
# plt.scatter(optimal_datapoints_2[0, :], optimal_datapoints_2[1, :], color="green", label="Pareto Optimal")
# plt.scatter(optimal_datapoints_3[0, :], optimal_datapoints_3[1, :], color="purple", label="Pareto Optimal")
plt.xlabel('Precision', fontsize=20)
plt.ylabel('Top3-Accuracy', fontsize=20)
plt.grid()
plt.legend()
plt.show()

# #Accuracy vs Precision
# plt.scatter(precision, top3, alpha=0.6)
# plt.xlabel("precision")
# plt.ylabel("accuracy")
# plt.grid()
# plt.show()
