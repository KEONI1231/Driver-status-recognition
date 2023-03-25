import mediapipe as mp
import math
import numpy as np

class Detector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 눈동자의 중앙 특징점 인덱스(좌)
        self.IRIS_L_IDX = 468
        # 눈동자의 중앙 특징점 인덱스(우)
        self.IRIS_R_IDX = 473

        self.EYE_L_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 473]
        self.EYE_R_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 468]

        self.EYE_L_SETA_IDX = [133, 157, 155]
        self.EYE_R_SETA_IDX = [362, 384, 381]

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        self.img_height = 0
        self.img_width = 0

        self.face_landmarks = None

    def processImg(self, img):
        self.img_height, self.img_width, _ = img.shape
        results = self.face_mesh.process(img)
        self.face_landmarks = results.multi_face_landmarks

    def getCoordinateByIDX(self, idx, face_idx):
        if self.face_landmarks is None:
            return None
        else:
            coordinate = self.face_landmarks[face_idx].landmark[idx]
            x, y, z = coordinate.x, coordinate.y, coordinate.z
            x *= self.img_width
            y *= self.img_height
            return x, y

    def getIrisCoordinates(self, face_idx):
        eye_l_coordinate = self.getCoordinateByIDX(self.IRIS_L_IDX, face_idx)
        eye_r_coordinate = self.getCoordinateByIDX(self.IRIS_R_IDX, face_idx)
        return eye_l_coordinate, eye_r_coordinate

    def getEyeCenter(self, face_idx):
        var_l_x = 0
        var_l_y = 0
        for l_idx in self.EYE_L_IDX:
            x, y = self.getCoordinateByIDX(l_idx, face_idx)
            var_l_x += x
            var_l_y += y
        var_l_x /= len(self.EYE_L_IDX)
        var_l_y /= len(self.EYE_L_IDX)

        var_r_x = 0
        var_r_y = 0
        for r_idx in self.EYE_R_IDX:
            x, y = self.getCoordinateByIDX(r_idx, face_idx)
            var_r_x += x
            var_r_y += y
        var_r_x /= len(self.EYE_R_IDX)
        var_r_y /= len(self.EYE_R_IDX)

        return (var_l_x, var_l_y), (var_r_x, var_r_y)

    def getDist(self, x1, y1, x2, y2):
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    def getEyeSeta(self, EYE_SETA_IDX):
        x_s, y_s = self.getCoordinateByIDX(EYE_SETA_IDX[0], 0)
        x1, y1 = self.getCoordinateByIDX(EYE_SETA_IDX[1], 0)
        x2, y2 = self.getCoordinateByIDX(EYE_SETA_IDX[2], 0)
        hypot1 = self.getDist(x_s, y_s, x1, y1)
        base1 = self.getDist(x_s, 0, x1, 0)
        seta1 = math.acos(base1 / hypot1)
        hypot2 = self.getDist(x_s, y_s, x2, y2)
        base2 = self.getDist(x_s, 0, x2, 0)
        seta2 = math.acos(base2 / hypot2)
        return np.rad2deg(seta1 + seta2)

    def getEyeSetas(self):
        eye_l_seta = self.getEyeSeta(self.EYE_L_SETA_IDX)
        eye_r_seta = self.getEyeSeta(self.EYE_R_SETA_IDX)
        return eye_l_seta, eye_r_seta

    def getDiff(self, xy1, xy2):
        x1, y1 = xy1
        x2, y2 = xy2
        return (x2 - x1), (y2 - y1)

    def getGazeTarget(self):
        eye_l, eye_r = self.getEyeCenter(0)
        iris_l, iris_r = self.getIrisCoordinates(0)
        diff_l = self.getDiff(eye_l, iris_l)
        diff_r = self.getDiff(eye_r, iris_r)
        return (eye_l + diff_l * 3), (eye_r + diff_r * 3)
