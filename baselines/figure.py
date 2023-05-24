import os
import numpy as np
from matplotlib import pyplot as plt

# The line chart of experiments w.r.t. different contexts
# x = [0, 1, 3, 5, 7, 9]
# # x = ['no context', 'context+1', 'context+3', 'context+5', 'context+7', 'context+9']
# ego4d_res = [[56.2, 56.9, 57.5, 58.9, 56.3, 55.5],
#              [57.5, 57.7, 57.8, 58.9, 55.8, 57.9],
#              [58.6, 59.3, 58.4, 61.0, 58.4, 58.3]]
#
# youtube_res = [[68.9, 69.8, 69.9, 70.4, 70.4, 69.4],
#                [70.4, 70.3, 70.4, 70.5, 70.4, 70.2],
#                [68.4, 69.8, 69.1, 69.7, 69.3, 69.1]]
#
# blue = '#4FA6E1'
# green = '#ABECC6'
# yellow = '#F7BB5C'
#
# plt.cla()
# plt.plot(x, ego4d_res[0], color=blue, marker='o', linewidth=2, label='BERT')
# plt.plot(x, ego4d_res[1], color=green, marker='o', linewidth=2, label='RoBERTa')
# plt.plot(x, ego4d_res[2], color=yellow, marker='o', linewidth=2, label='MT-BERT')
# plt.ylim(54.0, 62.0)
# plt.xticks(x, ['0', '1', '3', '5', '7', '9'], size=14)
# plt.yticks(size=14)
# plt.legend(prop={'size': 14}, loc='lower center', ncol=3)
# plt.show()
#
# plt.cla()
# plt.plot(x, youtube_res[0], color=blue, marker='o', linewidth=2, label='BERT')
# plt.plot(x, youtube_res[1], color=green, marker='o', linewidth=2, label='RoBERTa')
# plt.plot(x, youtube_res[2], color=yellow, marker='o', linewidth=2, label='MT-BERT')
# plt.ylim(67.5, 71.0)
# plt.xticks(x, ['0', '1', '3', '5', '7', '9'], size=14)
# plt.yticks(size=14)
# plt.legend(prop={'size': 14}, loc='lower center', ncol=3)
# plt.show()


# The bar chart of data domain generalization
labels_1 = ['BERT', 'RoBERTa', 'MT-BERT']
labels_2 = ['BERT+C', 'RoBERTa+C', 'MT-BERT+C']
original_1 = [56.2, 57.5, 58.6]
original_2 = [58.9, 58.9, 61.0]
without_finetune_1 = [60.1, 61.5, 58.3]
without_finetune_2 = [61.1, 62.2, 58.7]
with_finetune_1 = [60.7, 61.7, 60.8]
with_finetune_2 = [61.6, 63.4, 61.4]

yellow = '#F7BB5C'
blue = '#4FA6E1'
green = '#ABECC6'


x = np.arange(len(labels_1 + labels_2)) * 2  # the label locations
width = 0.5  # the width of the bars
interval = 0.09

fig, ax = plt.subplots(figsize=(12, 4))
rects1_1 = ax.bar(x[::2] - width - interval, original_1, width, facecolor=yellow, edgecolor='darkorange', label='Ego4D Only')
rects1_2 = ax.bar(x[1::2] - width - interval, original_2, width, facecolor=yellow, edgecolor='darkorange', hatch='//')
rects2_1 = ax.bar(x[::2], without_finetune_1, width, facecolor=blue, edgecolor='blue', label='w.o. Fine-tuning')
rects2_2 = ax.bar(x[1::2], without_finetune_2, width, facecolor=blue, edgecolor='blue', hatch='//')
rects3_1 = ax.bar(x[::2] + width + interval, with_finetune_1, width, facecolor=green, edgecolor='green', label='w. Fine-tuning')
rects3_2 = ax.bar(x[1::2] + width + interval, with_finetune_2, width, facecolor=green, edgecolor='green', hatch='//')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_title('Scores by group and gender')
# ax.set_ylabel('Scores')
ax.set_xticks(x, ['BERT', 'BERT+C', 'RoBERTa', 'RoBERTa+C', 'MT-BERT', 'MT-BERT+C'], size=14)
ax.set_yticks(list(range(50, 71, 5)), None, size=14)  # setting the size for y-axis doesn't work
ax.set_ylim(bottom=50, top=70)
ax.legend(prop={'size': 14}, loc='upper center', ncol=3)

ax.bar_label(rects1_1, fmt='%.1f', padding=3, size=12)
ax.bar_label(rects1_2, fmt='%.1f', padding=3, size=12)
ax.bar_label(rects2_1, fmt='%.1f', padding=3, size=12)
ax.bar_label(rects2_2, fmt='%.1f', padding=3, size=12)
ax.bar_label(rects3_1, fmt='%.1f', padding=3, size=12)
ax.bar_label(rects3_2, fmt='%.1f', padding=3, size=12)

plt.show()


# The bar chart of game domain generalization
# labels = ['BERT', 'RoBERTa', 'MT-BERT']
# vanilla = [41.3, 46.0, 42.6]
# plus_C = [42.1, 40.8, 43.7]
#
# yellow = '#F7BB5C'
# blue = '#4FA6E1'
#
# x = np.arange(len(labels))  # the label locations
# width = 0.25  # the width of the bars
# interval = 0.05
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2 - interval/2, vanilla, width, facecolor=yellow, edgecolor='darkorange', label='Base Model')
# rects2 = ax.bar(x + width/2 + interval/2, plus_C, width, facecolor=blue, edgecolor='blue', label='Base Model + Context')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_title('Scores by group and gender')
# # ax.set_ylabel('Scores')
# ax.set_xticks(x, labels, size=14)
# ax.set_yticks(list(range(35, 56, 5)), None, size=14)  # setting the size for y-axis doesn't work
# ax.set_ylim(bottom=35, top=55)
# ax.legend(prop={'size': 14}, loc='upper center', ncol=2)
#
# ax.bar_label(rects1, fmt='%.1f', padding=3, size=12)
# ax.bar_label(rects2, fmt='%.1f', padding=3, size=12)
#
# plt.show()

