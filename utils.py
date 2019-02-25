import cv2

def VideoSaver(save_path, fps, width,height):
    tfourcc = cv2.VideoWriter_fourcc(*'XVID')
    tout = cv2.VideoWriter(save_path, tfourcc, fps,(width,height))

    return tout

def RGB2BGR(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
