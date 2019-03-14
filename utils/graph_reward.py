import argparse
import os
import csv
import matplotlib.pyplot as plt

training_dir = "/home/duju/training"

def draw(exp_code, reward_scale):
    assert exp_code is not None

    target_dir = os.path.join(training_dir,exp_code)

    assert os.path.isdir(target_dir)

    txt_path = os.path.join(target_dir,"reward.txt")
    image_path = os.path.join(target_dir,"rewards.png")

    with open(txt_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter='-')
        rewards = []
        for row in csvreader:
            rewards.append(float(row[3]))

    axes = plt.gca()
    axes.set_ylim([0,reward_scale])
    plt.plot(rewards)
    plt.savefig(image_path,dpi=300)

    print("Successfully saved reward curve at",image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='draw reward graph from current reward.txt file')

    parser.add_argument('--experiment-code',help='which experiment do you want to draw?')
    parser.add_argument('--reward-scale',help="reward scale 100 or 1000",default=1000)

    args = parser.parse_args()

    exp_code = args.experiment_code

    reward_scale = int(args.reward_scale)

    draw(exp_code,reward_scale)