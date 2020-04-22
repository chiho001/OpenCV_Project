#video 출력 소스
import cv2
capture = cv2.VideoCapture("Saved_Video_lookdown.raw")

while True:
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.open("Saved_Video_lookdown.raw")

    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

    if cv2.waitKey(33) > 0: break

capture.release() #VideoCapture의 장치를 닫고 메모리를 해제
cv2.destroyAllWindows()

# 구문오류 -> syntax error
# 런타임에러
# 논리오류