import cv2
import numpy as np
import time
import datetime
import pygame

class monitor:
    def __init__(self, width = 900, height = 300):
        self.GRAPH_SIZE = width
        self.GRAPH_HEIGHT = height

        self.q_eye_l_seta = [0 for _ in range(self.GRAPH_SIZE)]
        self.q_eye_r_seta = [0 for _ in range(self.GRAPH_SIZE)]
        self.q_eye_closed = [0 for _ in range(self.GRAPH_SIZE)]

        self.q_mouth_l_seta = [0 for _ in range(self.GRAPH_SIZE)]
        self.q_mouth_r_seta = [0 for _ in range(self.GRAPH_SIZE)]

        self.WHITE = [255, 255, 255]

        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

        self.prev_eye_status = False
        self.eye_last_open = None
        self.eye_last_close = None

        self.shouldBeep = False
        pygame.mixer.init()


    def pushEyeSeta(self, seta_l, seta_r):
        self.q_eye_l_seta.pop(0)
        self.q_eye_r_seta.pop(0)
        self.q_eye_l_seta.append(seta_l)
        self.q_eye_r_seta.append(seta_r)

    def pushMouthSeta(self, seta_l, seta_r):
        self.q_mouth_l_seta.pop(0)
        self.q_mouth_r_seta.pop(0)
        self.q_mouth_l_seta.append(seta_l)
        self.q_mouth_r_seta.append(seta_r)

    def DrawMonitorSeta(self, q_l_seta, q_r_seta):
        graph_l = np.zeros((self.GRAPH_HEIGHT, self.GRAPH_SIZE), np.uint8)
        graph_r = np.zeros((self.GRAPH_HEIGHT, self.GRAPH_SIZE), np.uint8)
        mid_h = int(self.GRAPH_HEIGHT / 2)
        for i in range(1, self.GRAPH_SIZE):
            cv2.line(graph_l, (i - 1, mid_h - int(q_l_seta[i - 1])), (i, mid_h - int(q_l_seta[i])),
                     self.WHITE, 1)
            cv2.line(graph_r, (i - 1, mid_h - int(q_r_seta[i - 1])), (i, mid_h - int(q_r_seta[i])),
                     self.WHITE, 1)
        return graph_l, graph_r

    def DrawMonitorEyeSeta(self):
        graph_l, graph_r = self.DrawMonitorSeta(self.q_eye_l_seta, self.q_eye_r_seta)
        cv2.imshow("Graph-Eye-L-Seta", graph_l)
        cv2.imshow("Graph-Eye-R-Seta", graph_r)

    def DrawMonitorMouthSeta(self):
        graph_l, graph_r = self.DrawMonitorSeta(self.q_mouth_l_seta, self.q_mouth_r_seta)
        cv2.imshow("Graph-Mouth-L-Seta", graph_l)
        cv2.imshow("Graph-Mouth-R-Seta", graph_r)

    def DrawMediapipe(self, img):
        cv2.imshow("Mediapipe Result", img)

    def putText(self, img, text, x, y):
        cv2.putText(img, text, (x, y), self.FONT, 0.5, self.WHITE, 1)
        return img

    def DrawStatus(self, eye_l_seta, eye_r_seta, mouth_l_seta, mouth_r_seta, fps):
        is_eye_closed = False
        if eye_l_seta < 30 and eye_r_seta < 30:
            is_eye_closed = True
        if self.prev_eye_status and not is_eye_closed:
            self.eye_last_open = datetime.datetime.now()
        if not self.prev_eye_status and is_eye_closed:
            self.eye_last_close = datetime.datetime.now()

        if self.eye_last_close is not None and self.eye_last_open is not None:
            if self.eye_last_close > self.eye_last_open:
                time_delta_ms = (self.eye_last_close - self.eye_last_open).microseconds / 1000
                print(time_delta_ms)
                if time_delta_ms > 400:
                    self.shouldBeep = True

        if self.shouldBeep:
            self.beepsound()
            print('Beeeeeeeeeeeep')
            self.shouldBeep = False

        self.prev_eye_status = is_eye_closed
        background = np.zeros((200, 400), np.uint8)
        background = self.putText(background, 'EYE_L: ' + str(eye_l_seta), 10, 15)
        background = self.putText(background, 'EYE_R: ' + str(eye_r_seta), 10, 30)
        background = self.putText(background, 'MOUTH_L: ' + str(mouth_l_seta), 10, 45)
        background = self.putText(background, 'MOUTH_R: ' + str(mouth_r_seta), 10, 60)
        background = self.putText(background, 'EYE_CLOSED: ' + str(is_eye_closed), 10, 75)
        background = self.putText(background, 'LAST_OPENED: ' + str(self.eye_last_open), 10, 90)
        background = self.putText(background, 'LAST_CLOSED: ' + str(self.eye_last_close), 10, 105)
        background = self.putText(background, 'FPS: ' + str(fps), 10, 120)
        cv2.imshow("Status", background)

    def beepsound(self):
        pygame.mixer.music.load('./music/beep.mp3')  # 배경 음악
        pygame.mixer.music.play(0)