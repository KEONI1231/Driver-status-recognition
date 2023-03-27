import face_landmark
import gui_maker
import cv2
import filter

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Mediapipe 객체
    landmarker = face_landmark.Detector()
    # 그래프 시각화 객체
    monitor = gui_maker.monitor(900, 300)



    #casecaded lpf
    # Low Pass Filter 객체
    eye_l_seta_lpf = filter.CascadedLPF(0.7,0.7)
    eye_r_seta_lpf = filter.CascadedLPF(0.7,0.7)
    mouth_l_seta_lpf = filter.CascadedLPF(0.7,0.7)
    mouth_r_seta_lpf = filter.CascadedLPF(0.7,0.7)


    # eye_l_seta_lpf = filter.LPF(0.7)
    # eye_r_seta_lpf = filter.LPF(0.7)
    # mouth_l_seta_lpf = filter.LPF(0.7)
    # mouth_r_seta_lpf = filter.LPF(0.7)


    # Webcam 객체
    cam = cv2.VideoCapture(0)

    # 메인 루프
    while True:
        # Webcam으로부터 이미지 읽어오기
        check, image = cam.read()

        # 이미지 가져오기에 실패 시 재시도
        if check is None:
            continue
        # 이미지를 mediapipe에 넣기
        landmarker.processImg(image)
        # Mediapipe FACEMESH의 결과물 출력
        #monitor.DrawMediapipe(landmarker.getMaskedIMG(image))q
        # 인식 결과 이미지(보조선) 출력
        landmarker.getEyeResultIMG(image)
        landmarker.getMouthResultIMG(image)
        image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.imshow("MOUTH", image)

        # 얼굴이 감지 되지 않았다면 재탐색
        if not landmarker.isFaceDetected():
            continue
        # 눈의 각도 계산
        eye_l_seta, eye_r_seta = landmarker.getEyeSetas()
        # 1차 Low Pass Filter로 잡음 제거
        eye_l_seta, eye_r_seta = eye_l_seta_lpf.compute(eye_l_seta), eye_r_seta_lpf.compute(eye_r_seta)
        # 인식된 눈의 각도 데이터 넣기
        monitor.pushEyeSeta(eye_l_seta*3, eye_r_seta*3)
        # 눈의 각도 인식 결과 그래프 출력
        monitor.DrawMonitorEyeSeta()
        # 입 벌어짐 각도 계산
        mouth_l_seta, mouth_r_seta = landmarker.getMouthSetas()
        # 1차 Low Pass Filter로 잡음 제거
        mouth_l_seta, mouth_r_seta = mouth_l_seta_lpf.compute(mouth_l_seta), mouth_r_seta_lpf.compute(mouth_r_seta)
        # 인식된 입의 각도 데이터 넣기
        monitor.pushMouthSeta(mouth_l_seta, mouth_r_seta)
        # 입의 각도 인식 결과 그래프 출력
        monitor.DrawMonitorMouthSeta()
        #
        monitor.DrawStatus(eye_l_seta, eye_r_seta, mouth_l_seta, mouth_r_seta)


        # 'q' 입력 시 프로그램 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Webcam 해제
    cam.release()
    # Opencv 창 모두 종료
    cv2.destroyAllWindows()
