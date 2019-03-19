import cv2
import numpy as np
import datetime
import os

class VideoSaver:
    def __init__(self, save_path, source_fps, target_fps = 30, width = None, height = None):

        self.save_path = save_path
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.counter = 0

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.save_path, self.fourcc, self.target_fps, (self.width,self.height))

        self.save_frame_index = uniform_fps_downsizer(self.source_fps, self.target_fps)

    def write(self,frame):
        if (self.counter)%self.source_fps in self.save_frame_index:
            self.out.write(frame)
        else:
            pass
        self.counter += 1
    def release(self):
        self.out.release()

def RGB2BGR(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def uniform_fps_downsizer(source_fps, target_fps):
    assert type(source_fps) == int
    interval = source_fps / float(target_fps)
    index_list = []

    for i in range(target_fps):
        index_list.append(int(np.round(i*interval)))


    return index_list

def state_1d_dim_calc(env):
    ob_spec = env.observation_spec()

    result = np.zeros((1,))

    for i,j in ob_spec.items():
        result = result + np.array(j.shape)

    return np.array(result,dtype=np.int)

def state_1d_flat(ob_dict):

    result = []

    for i, k in ob_dict.items():
        result.extend(list(k))

    return np.array(result,dtype=np.float32)

def directory_setting(base_dir,domain_name, task_name, step_size):
    dt = datetime.datetime.now()
    dirpath = str(dt.year) + str(dt.month).zfill(2) + str(dt.day).zfill(2) +\
    "_" + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)+str(dt.second).zfill(2)+"_"+domain_name+"_"+task_name+"_"+str(step_size)
    dirpath = os.path.join(base_dir,dirpath)

    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    else:
        raise Exception("Existing Directory")

    print(dirpath)

    return dirpath

def line_writer(dirpath, _str):
    with open(os.path.join(dirpath,"reward.txt"),"a") as f:
        f.write(_str)

def append_file_writer(dirpath, file_name, _str):
    with open(os.path.join(dirpath,file_name),"a") as f:
        f.write(_str)


def experiment_detail_saver(domain_name, task_name, step_size, actor_lr, critic_lr, tau, gamma, sigma, batch_size,
                            critic_reg_weight):
    _str = "domain_name : " + str(domain_name) + "\n" + \
           "task_name: " + str(task_name) + "\n" + \
           "step_size: " + str(step_size) + "\n" + \
           "actor_lr: " + str(actor_lr) + "\n" + \
           "critic_lr: " + str(critic_lr) + "\n" + \
           "tau: " + str(tau) + "\n" + \
           "gamma: " + str(gamma) + "\n" + \
           "sigma: " + str(sigma) + "\n" + \
           "batct_size: " + str(batch_size) + "\n" + \
           "critic_reg_weight: " + str(critic_reg_weight) + "\n"

    return _str