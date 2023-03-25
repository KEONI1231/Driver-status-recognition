import cv2
import numpy as np

class monitor:
    def __init__(self, width = 900, height = 300):
        self.GRAPH_SIZE = width
        self.GRAPH_HEIGHT = height

        self.q_eye_l_seta = [0 for _ in range(self.GRAPH_SIZE)]
        self.q_eye_r_seta = [0 for _ in range(self.GRAPH_SIZE)]

        self.WHITE = [255, 255, 255]

    def pushEyeSeta(self, seta_l, seta_r):
        self.q_eye_l_seta.pop(0)
        self.q_eye_r_seta.pop(0)
        self.q_eye_l_seta.append(seta_l)
        self.q_eye_r_seta.append(seta_r)

    def DrawMonitorSeta(self):
        graph_l = np.zeros((self.GRAPH_HEIGHT, self.GRAPH_SIZE), np.uint8)
        graph_r = np.zeros((self.GRAPH_HEIGHT, self.GRAPH_SIZE), np.uint8)
        mid_h = int(self.GRAPH_HEIGHT / 2)
        for i in range(1, self.GRAPH_SIZE):
            cv2.line(graph_l, (i - 1, mid_h - int(self.q_eye_l_seta[i - 1])), (i, mid_h - int(self.q_eye_l_seta[i])), self.WHITE, 1)
            cv2.line(graph_r, (i - 1, mid_h - int(self.q_eye_r_seta[i - 1])), (i, mid_h - int(self.q_eye_r_seta[i])), self.WHITE, 1)
        cv2.imshow("Graph-Seta-L", graph_l)
        cv2.imshow("Graph-Seta-R", graph_r)