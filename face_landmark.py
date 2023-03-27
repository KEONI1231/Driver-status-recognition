import mediapipe as mp
import math
import numpy as np
import cv2

class Detector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 눈동자의 중앙 특징점 인덱스(좌)
        self.IRIS_L_IDX = 468
        # 눈동자의 중앙 특징점 인덱스(우)
        self.IRIS_R_IDX = 473

        # 좌측 눈 주변 특징점 리스트
        self.EYE_L_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 473]
        # 우측 눈 주변 특징점 리스트
        self.EYE_R_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 468]

        # 좌측 눈 각도 계산을 위한 눈 특징점 리스트(눈물샘, 위측, 아래측)
        self.EYE_L_SETA_IDX = [133, 158, 153]
        # 우측 눈 각도 계산을 위한 눈 특징점 리스트(눈물샘, 위측, 아래측)
        self.EYE_R_SETA_IDX = [362, 385, 380]

        # 좌측 입 벌어짐 각도(왼쪽 끝, 위측, 아래측)
        self.MOUTH_L_SETA_IDX = [78, 13, 14]
        # 우측 입 벌어짐 각도(오른쪽 끝, 위측, 아래측)
        self.MOUTH_R_SETA_IDX = [308, 13, 14]

        # Mediapipe 모델 가져오기(FaceMesh)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            #
            static_image_mode=True,
            #
            refine_landmarks=True,
            # 최대 얼굴 인식 개수
            max_num_faces=1,
            #
            min_detection_confidence=0.5
        )

        # 이미지 높이
        self.img_height = 0
        # 이미지 너비
        self.img_width = 0
        # 인식된 얼굴 정보 리스트
        self.face_landmarks = None

        self.GREEN = [0, 255, 0]
        self.RED = [0, 0, 255]

    def processImg(self, img):
        self.img_height, self.img_width, _ = img.shape
        results = self.face_mesh.process(img)
        self.face_landmarks = results.multi_face_landmarks

    def isFaceDetected(self):
        if self.face_landmarks is None:
            return False
        else:
            return True

    def getCoordinateByIDX(self, idx, face_idx):
        if self.face_landmarks is None:
            return None
        else:
            coordinate = self.face_landmarks[face_idx].landmark[idx]
            x, y, z = coordinate.x, coordinate.y, coordinate.z
            x *= self.img_width
            y *= self.img_height
            return int(x), int(y)

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

    def getSeta(self, x_s, y_s, x1, y1, x2, y2):
        hypot1 = self.getDist(x_s, y_s, x1, y1)
        base1 = self.getDist(x_s, 0, x1, 0)
        seta1 = math.acos(base1 / hypot1)
        hypot2 = self.getDist(x_s, y_s, x2, y2)
        base2 = self.getDist(x_s, 0, x2, 0)
        seta2 = math.acos(base2 / hypot2)
        return np.rad2deg(seta1 + seta2)

    def getEyeSeta(self, eye_seta_idx):
        x_s, y_s = self.getCoordinateByIDX(eye_seta_idx[0], 0)
        x1, y1 = self.getCoordinateByIDX(eye_seta_idx[1], 0)
        x2, y2 = self.getCoordinateByIDX(eye_seta_idx[2], 0)
        return self.getSeta(x_s, y_s, x1, y1, x2, y2)

    def getEyeSetas(self):
        eye_l_seta = self.getEyeSeta(self.EYE_L_SETA_IDX)
        eye_r_seta = self.getEyeSeta(self.EYE_R_SETA_IDX)
        return eye_l_seta, eye_r_seta

    def getMouthSeta(self, mouth_seta_idx):
        x_s, y_s = self.getCoordinateByIDX(mouth_seta_idx[0], 0)
        x1, y1 = self.getCoordinateByIDX(mouth_seta_idx[1], 0)
        x2, y2 = self.getCoordinateByIDX(mouth_seta_idx[2], 0)
        return self.getSeta(x_s, y_s, x1, y1, x2, y2)

    def getMouthSetas(self):
        mouth_l_seta = self.getMouthSeta(self.MOUTH_L_SETA_IDX)
        mouth_r_seta = self.getMouthSeta(self.MOUTH_R_SETA_IDX)
        return mouth_l_seta, mouth_r_seta

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

    def getFaceMaskedIMG(self, img, face_idx):
        self.mp_drawing.draw_landmarks(
            image=img,
            landmark_list=self.face_landmarks[face_idx],
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        return img

    def getEyeMaskedIMG(self, img, face_idx):
        self.mp_drawing.draw_landmarks(
            image=img,
            landmark_list=self.face_landmarks[face_idx],
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_contours_style())
        return img

    def getIrisMaskedIMG(self, img, face_idx):
        self.mp_drawing.draw_landmarks(
            # 그릴 배경 이미지
            image=img,
            # 그릴 얼굴 데이터
            landmark_list=self.face_landmarks[face_idx],
            # 그릴 부위
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            #
            landmark_drawing_spec=None,
            # 그릴 마스크 스타일
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
        return img

    def getMaskedIMG(self, img):
        if not self.isFaceDetected():
            return img
        img = self.getFaceMaskedIMG(img, 0)
        img = self.getEyeMaskedIMG(img, 0)
        img = self.getIrisMaskedIMG(img, 0)
        return img

    def getEyeResultIMG(self, img):
        if not self.isFaceDetected():
            return img

        eye_l_xy = []
        for idx in self.EYE_L_SETA_IDX:
            x, y = self.getCoordinateByIDX(idx, 0)
            eye_l_xy.append((x, y))
            cv2.circle(img, (x, y), 4, self.GREEN, -1)
        cv2.line(img, eye_l_xy[0], eye_l_xy[1], self.GREEN, 2)
        cv2.line(img, eye_l_xy[0], eye_l_xy[2], self.GREEN, 2)

        eye_r_xy = []
        for idx in self.EYE_R_SETA_IDX:
            x, y = self.getCoordinateByIDX(idx, 0)
            eye_r_xy.append((x, y))
            cv2.circle(img, (x, y), 4, self.GREEN, -1)

        cv2.line(img, eye_r_xy[0], eye_r_xy[1], self.GREEN, 2)
        cv2.line(img, eye_r_xy[0], eye_r_xy[2], self.GREEN, 2)

        return img

    def getMouthResultIMG(self, img):
        if not self.isFaceDetected():
            return img

        mouth_l_seta = []
        for idx in self.MOUTH_L_SETA_IDX:
            x, y = self.getCoordinateByIDX(idx, 0)
            mouth_l_seta.append((x, y))
            cv2.circle(img, (x, y), 4, self.RED, -1)
        cv2.line(img, mouth_l_seta[0], mouth_l_seta[1], self.RED, 2)
        cv2.line(img, mouth_l_seta[0], mouth_l_seta[2], self.RED, 2)

        mouth_r_seta = []
        for idx in self.MOUTH_R_SETA_IDX:
            x, y = self.getCoordinateByIDX(idx, 0)
            mouth_r_seta.append((x, y))
            cv2.circle(img, (x, y), 4, self.RED, -1)

        cv2.line(img, mouth_r_seta[0], mouth_r_seta[1], self.RED, 2)
        cv2.line(img, mouth_r_seta[0], mouth_r_seta[2], self.RED, 2)

        return img