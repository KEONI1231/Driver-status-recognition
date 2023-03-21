import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

LEFT_IRIS_IDX = 468
RIGHT_IRIS_IDX = 473

eye_r_landmark_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 473]
eye_l_landmark_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 468]

cam = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
    while True:
        check, image = cam.read()

        image_height, image_width, _ = image.shape
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        if not results.multi_face_landmarks:
            continue

        annotated_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            var_r_x = 0
            var_r_y = 0
            for idx in eye_r_landmark_idx:
                coordinates = face_landmarks.landmark[idx]
                x = coordinates.x * image_width
                y = coordinates.y * image_height
                z = coordinates.z
                var_r_x += x
                var_r_y += y

                # x, y 좌표 화면에 그리기
                cv2.circle(annotated_image, (int(x), int(y)), 1, (255, 0, 0), -1)

            var_r_x /= len(eye_r_landmark_idx)
            var_r_y /= len(eye_r_landmark_idx)

            cv2.circle(annotated_image, (int(var_r_x), int(var_r_y)), 1, (0, 255, 0), -1)

            var_l_x = 0
            var_l_y = 0
            for idx in eye_l_landmark_idx:
                coordinates = face_landmarks.landmark[idx]
                x = coordinates.x * image_width
                y = coordinates.y * image_height
                z = coordinates.z
                var_l_x += x
                var_l_y += y

                # x, y 좌표 화면에 그리기
                cv2.circle(annotated_image, (int(x), int(y)), 1, (255, 0, 0), -1)

            var_l_x /= len(eye_l_landmark_idx)
            var_l_y /= len(eye_l_landmark_idx)

            coordinates = face_landmarks.landmark[RIGHT_IRIS_IDX]
            iris_r_x = coordinates.x * image_width
            iris_r_y = coordinates.y * image_height

            coordinates = face_landmarks.landmark[LEFT_IRIS_IDX]
            iris_l_x = coordinates.x * image_width
            iris_l_y = coordinates.y * image_height

            # cv2.circle(annotated_image, (int(var_x), int(var_y)), 3, (0, 255, 0), -1)

            diff_r_x = iris_r_x - var_r_x
            diff_r_y = iris_r_y - var_r_y

            diff_l_x = iris_l_x - var_l_x
            diff_l_y = iris_l_y - var_l_y

            target_r_x = var_r_x + diff_r_x * 3
            target_r_y = var_r_y + diff_r_y * 3

            target_l_x = var_l_x + diff_l_x * 3
            target_l_y = var_l_y + diff_l_y * 3

            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())

            cv2.line(annotated_image, (int(var_r_x), int(var_r_y)), (int(target_r_x), int(target_r_y)), [0, 255, 255],
                     2)
            cv2.line(annotated_image, (int(var_l_x), int(var_l_y)), (int(target_l_x), int(target_l_y)), [0, 255, 255],
                     2)

        cv2.imshow("DETECT", annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()