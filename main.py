import face_landmark
import fuzzy_logic
import gui_maker
import cv2
import filter
import server
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Press the green button in the gutter to run the script.
if __name__ == '__main__': 
    # Mediapipe 객체
    landmarker = face_landmark.Detector()
    # 그래프 시각화 객체
    monitor = gui_maker.monitor(900, 300)

    #udp_sock = server.UDP_socket('127.0.0.1', '9999')

    eye_l_seta_lpf = filter.CascadedLPF(0.7, 0.7)
    eye_l_seta_lpf = filter.CascadedLPF(0.7, 0.7)
    eye_r_seta_lpf = filter.CascadedLPF(0.7, 0.7)
    mouth_l_seta_lpf = filter.CascadedLPF(0.7, 0.7)
    mouth_r_seta_lpf = filter.CascadedLPF(0.7, 0.7)
    # Webcam 객체
    cam = cv2.VideoCapture(0)

    #프레임 계산을 위한 이전 시간 저장하는 변수
    prev_frame_t = time.time()
    #타겟 프레임
    TARGET_FPS = 30

    # 그래프 초기화
    plt.ion()
    fig, ax = plt.subplots()
    x = np.arange(0, 100)
    y = np.zeros_like(x)
    ax.set_ylim([-0.1, 1.1])
    ax.axhline(y=0.5, color='r', linestyle='--')
    line, = ax.plot(x, y)



    # 메인 루프
    while True:
        # Webcam으로부터 이미지 읽어오기
        check, image = cam.read()
        check, frame = cam.read()
        image = cv2.resize(image, (1920, 1080))

        # 이미지 가져오기에 실패 시 재시도
        if check is None:
            continue
        # 이미지를 mediapipe에 넣기
        landmarker.processImg(image)
        # Mediapipe FACEMESH의 결과물 출력
        #monitor.DrawMediapipe(landmarker.getMaskedIMG(image))
        # 인식 결과 이미지(보조선) 출력
        landmarker.getEyeResultIMG(image)
        landmarker.getMouthResultIMG(image)
        image = cv2.flip(image, 1)
        cv2.imshow("DETECT", image)

        # 얼굴이 감지 되지 않았다면 재탐색
        if not landmarker.isFaceDetected():
            continue
        # 눈의 각도 계산
        eye_l_seta, eye_r_seta = landmarker.getEyeSetas()
        # 1차 Low Pass Filter로 잡음 제거q
        eye_l_seta, eye_r_seta = eye_l_seta_lpf.compute(eye_l_seta), eye_r_seta_lpf.compute(eye_r_seta)


        #여긴 내가 작성한 퍼지 논리 그래프 시각화
        #eye_LR_seta = (eye_r_seta+eye_l_seta)/2.0
        # line = None  # 추가된 코드
        # def sleepiness_detection(x):
        #     y1 = np.where(x < 15, 1, np.where(x < 30, (30 - x) / 15, 0))
        #     y2 = np.where(x < 20, 0, np.where(x < 35, (x - 20) / 15, np.where(x < 50, (50 - x) / 15, 0)))
        #     y3 = np.where(x < 45, 0, np.where(x < 60, (x - 45) / 15, 1))
        #     return np.fmax(np.fmax(y1, y2), y3)
        #
        # eye_LR_seta = eye_l_seta
        #
        # # 현재 프레임에서의 졸음 상태 판단
        # sleepiness = sleepiness_detection(eye_LR_seta)
        #
        # # 그래프에 졸음 상태 추가
        # y = np.append(y, sleepiness)
        # print(y)
        # # 최근 100프레임까지의 졸음 상태 그래프 표시
        # if line is None:
        #     line, = ax.plot(x[-100:], y[-100:])
        # else:
        #     line.set_ydata(y[-100:])
        # ax.axhline(y=0.5, color='r', linestyle='--')
        # ax.set_ylim([-0.1, 1.1])
        # plt.draw()
        # plt.pause(0.01)
        #
        # # OpenCV 화면에 그래프 그리기
        # if len(plt.get_fignums()) > 0:  # 이미 창이 열려있으면 닫아줌
        #     plt.close()
        # fig, ax = plt.subplots()
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (800, 600))
        # plt_img = np.zeros_like(img)
        # plt_img[100:300, 500:600, :] = 255 * np.tile(np.expand_dims(y[-1], axis=-1), (200, 100, 3))
        # plt_img = cv2.cvtColor(plt_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # img = cv2.addWeighted(img, 0.8, plt_img, 0.2, 0)
        #
        #
        # # 종료
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        #
        #

        #여기까지

        # 인식된 눈의 각도 데이터 넣기
        monitor.pushEyeSeta(eye_l_seta, eye_r_seta)
        # 눈의 각도 인식 결과 그래프 출력
        monitor.DrawMonitorEyeSeta()

        monitor.DrawHistogramEyeSeta()
        # 입 벌어짐 각도 계산
        mouth_l_seta, mouth_r_seta = landmarker.getMouthSetas()
        # 1차 Low Pass Filter로 잡음 제거
        mouth_l_seta, mouth_r_seta = mouth_l_seta_lpf.compute(mouth_l_seta), mouth_r_seta_lpf.compute(mouth_r_seta)
        # 인식된 입의 각도 데이터 넣기
        monitor.pushMouthSeta(mouth_l_seta, mouth_r_seta)
        # 입의 각도 인식 결과 그래프 출력
        monitor.DrawMonitorMouthSeta()
        #타겟 프레임 이상 나올시 남은 시간 대기시켜 타겟프레임에 근사
        while (time.time() - prev_frame_t) <= 1./TARGET_FPS:
            continue

        #프레임 계산
        new_frame_t = time.time()
        fps = 1 / (new_frame_t - prev_frame_t)
        prev_frame_t = new_frame_t
        #결과 출력
        monitor.DrawStatus(eye_l_seta, eye_r_seta, mouth_l_seta, mouth_r_seta, fps)


        # 'q' 입력 시 프로그램 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Webcam 해제
    cam.release()
    # Opencv 창 모두 종료
    cv2.destroyAllWindows()