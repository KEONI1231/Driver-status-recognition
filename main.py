import face_landmark
import gui_maker
import cv2

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    landmarker = face_landmark.Detector()
    monitor = gui_maker.monitor(900, 300)
    cam = cv2.VideoCapture(0)

    while True:
        check, image = cam.read()
        if check is None:
            continue
        landmarker.processImg(image)
        seta_l, seta_r = landmarker.getEyeSetas()
        monitor.pushEyeSeta(seta_l, seta_r)
        monitor.DrawMonitorSeta()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cam.release()
    cv2.destroyAllWindows()
