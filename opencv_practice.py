

##############################################################
#흑색 혹은 흰색으로 변경
#이진화

import cv2

src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


##############################################################'
#이미지 색상 반전하기

# import cv2
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
#
# dst = cv2.bitwise_not(src)
# #cv2.bitwise_not(원본 이미지)를 이용하여 이미지의 색상을 반전할 수 있습니다.
# #비트 연산을 이용하여 색상을 반전시킵니다.
# #Tip : not 연산 이외에도 and, or, xor 연산이 존재합니다.
#
# cv2.imshow("src", src)
# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##############################################################
# 영상이나 이미지의 색상을 흑백색상으로 변환하기 위해서 사용합니다
# import cv2
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
#
# dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# # cv2.cvtcolor(원본 이미지, 색상 변환 코드)를 이용하여 이미지의 색상 공간을 변경할 수 있습니다.
# #
# # 색상 변환 코드는 원본 이미지 색상 공간2 결과 이미지 색상 공간을 의미합니다.
# #
# # 원본 이미지 색상 공간은 원본 이미지와 일치해야합니다.
# #
# # Tip : BGR은 RGB 색상 채널을 의미합니다. (Byte 역순)
# cv2.imshow("src", src)
# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##############################################################
# 영상 자르기
# import cv2
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
#
# dst = src.copy() # src.copy()를 이용하여 dst에 이미지를 복제합니다.
# dst = src[100:600, 200:700] # dst 이미지에 src[높이(행), 너비(열)]에서 잘라낼 영역을 설정합니다. List형식과 동일합니다.
#
# cv2.imshow("src", src)
# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##############################################################
# 영상 자르기 2
# import cv2
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
#
# dst = src.copy()
# roi = src[100:600, 200:700]
# dst[200:700, 100:600] = roi
# #roi를 생성하여 src[높이(행), 너비(열)]에서 잘라낼 영역을 설정합니다. List형식과 동일합니다.
# #이후, dst[높이(행), 너비(열)] = roi를 이용하여 dst 이미지에 해당 영역을 붙여넣을 수 있습니다.
#
# cv2.imshow("src", src)
# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





##############################################################
# 크기조절
# 영상이나 이미지의 크기를 원하는 크기로 조절할 수 있습니다.-
# 핵심 함수 cv2.resize()
# interpolation :
# ①새로운 점을 만들기 위해 수많은 점들을 평균화시키는 것.
# 이 방법은 샘플점들을 직선으로 연결하지 않고 곡선으로 연결함으로써 본래 신호파형에 대한 변형을 최소화시켜 준다.
# ②영상신호의 표준방식 변환시 기존의 정보로부터 새로운 정보를 만들어야 하는데 가령 525라인에서 625라인을 만들 때 처리되는 방식을 말한다.

#보간은 두 점을 연결하는 방법을 의미한다.
# 여기서 말하는 연결은 궤적을 생성한다는 뜻이다.
# 보간이 필요한 이유는 정보를 압축한 것을 다시 복원하기 위함이다.
#
# 특징점이라 불리는 선의 모양 복원에 꼭 필요한 점듦나 취해서 저장하는데 이 과정을 sampling이라 부른다.
# 일반적으로 sampling은 일정 시간 주기로 선의 점을 취하는 방식을 사용하는데 녹음 기술에서 많이 쓴다.


# import cv2
# import numpy as np
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
# dst = cv2.resize(src, dsize=(640, 512), interpolation=cv2.INTER_AREA)
# dst2 = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)
# cv2.imshow("src", src)
# cv2.imshow("dst", dst)
# cv2.imshow("dst2", dst2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


###############################################################
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
###############################################################
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
###############################################################
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