import argparse
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

training_dir = "/home/duju/training"

def save_graph(record_dir, reward_scale=1000):

    target_dir = record_dir

    assert os.path.isdir(target_dir)

    txt_path = os.path.join(target_dir,"rewards.txt")
    image_path = os.path.join(target_dir,"rewards.png")

    with open(txt_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter='*')
        rewards = []
        for row in csvreader:
            rewards.append(float(row[3]))

    with open(txt_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter='*')
        max_q_values = []
        for row in csvreader:
            max_q_values.append(float(row[6]))

    with open(txt_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter='*')
        eval_rewards = []
        for row in csvreader:
            eval_rewards.append(float(row[9]))

    draw(rewards, max_q_values, eval_rewards, reward_scale)

    plt.savefig(image_path,dpi=300)
    plt.close()

def draw(rewards, max_q_values, eval_rewards, reward_scale):
    t = np.arange(1,len(rewards)+1)

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('episode')
    ax1.set_ylabel('reward',color=color)
    ax1.plot(t, rewards, color=color)
    ax1.tick_params(axis='y',labelcolor=color)
    ax1.set_ylim([0,reward_scale])

    color = 'tab:orange'
    ax1.plot(t,eval_rewards, color=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('max_q',color=color)
    ax2.plot(t,max_q_values,color=color)
    ax2.tick_params(axis='y',labelcolor=color)
    ax2.set_ylim([0,np.max([np.max(max_q_values)*1.05,1.0])])

    fig.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='draw reward graph from current reward.txt file')

    parser.add_argument('--experiment-code',help='which experiment do you want to draw?')
    parser.add_argument('--reward-scale',help="reward scale 100 or 1000",default=1000)

    args = parser.parse_args()

    exp_code = args.experiment_code

    reward_scale = int(args.reward_scale)

    save_graph(exp_code,reward_scale)

    print("Successfully saved reward curve")