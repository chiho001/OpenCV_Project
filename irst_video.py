#video 출력 소스
# capture = cv2.VideoCapture("Star.mp4")
#
# while True:
#     if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
#         capture.open("Star.mp4")
#
#     ret, frame = capture.read()
#     cv2.imshow("VideoFrame", frame)
#
#     if cv2.waitKey(33) > 0: break
#
# capture.release() #VideoCapture의 장치를 닫고 메모리를 해제
# cv2.destroyAllWindows()
# a = [input().split(" ")]
