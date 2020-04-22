

## 크기조절
## 영상이나 이미지의 크기를 원하는 크기로 조절할 수 있습니다.
import cv2
import numpy as np

src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
dst = cv2.resize(src, dsize=(640, 512), interpolation=cv2.INTER_AREA)
dst2 = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)
cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 확대 축소
#pyrUp()과 pyrDown()함수는 결과 이미지 크기와 픽셀 외삽법은 기본갑승로 설정된 인수를 할당해야하므로 생략하여 사용.
#피라미드 함수에서 픽셀 외삽은 cv2.BORDER_DEFAULT만 사용 가능

# import cv2
# import numpy as np
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
#
# height, width, channel = src.shape
# dst = cv2.pyrUp(src, dstsize=(width*2, height*2), borderType=cv2.BORDER_DEFAULT)
# dst2 = cv2.pyrDown(src)
# dst3 = cv2.pyrDown(dst2)
#
# cv2.imshow("src", src)
# cv2.imshow("dst", dst)
# cv2.imshow("dst2", dst2)
# cv2.imshow("dst3", dst3)
# cv2.imshow("dst3", dst3)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 이미지 회전
# import cv2
# import numpy as np
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
#
# height, width, channel = src.shape
# np.matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1) # matrix에 회전 배열을 생성하여 저장합니다.
# #(중심점 X좌표, 중심점 Y좌표), 각도(회전할 각도), 스케일(확대비율)
# dst = cv2.warpAffine(src, np.matrix, (width, height)) #원본이미지, 배열, 결과이미지 너비, 결과 이미지 높이
#
# cv2.imshow("src", src)
# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 이미지 대칭화
# import cv2
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_ANYCOLOR)
# dst = cv2.flip(src, 1)
#
# cv2.imshow("페라리대칭전", src)
# cv2.imshow("페라리대칭후", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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

# image = cv2.imread("249422.jpg", cv2.IMREAD_COLOR)
# #cv2.imshow('Image Basic', image)
# #cv2.waitKey(0)
# #cv2.destroyAllWindows()
#
# height, width, channel = image.shape # 크기 , 넓이, 채널(이미지의 색상 정보)
# print(height, width , channel)



# cap = cv.VideoCapture(0)
#
#
# while(True):
#
#     ret, img_color = cap.read()
#
#     if ret == False:
#         continue;
#
#     cv.imshow('bgr', img_color)
#
#     # ESC 키누르면 종료
#     if cv.waitKey(1) & 0xFF == 27:
#         break
#
#
# cap.release()
# cv.destroyAllWindows()