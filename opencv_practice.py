



##############################################################
#트랙바
#트랙 바는 일정 범위 내의 값을 변경할 때 사용하며, 적절한 임곗값을 찾거나 변경하기 위해 사용합니다.
#OpenCV의 트랙 바는 생성된 윈도우 창에 트랙바를 부착해 사용할 수 있습니다.
# import cv2
#
# def onChange(pos):
#     pass
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_GRAYSCALE)
#
# cv2.namedWindow("Trackbar Windows")
#
# cv2.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
# cv2.createTrackbar("maxValue", "Trackbar Windows", 0, 255, lambda x : x)
#
# cv2.setTrackbarPos("threshold", "Trackbar Windows", 127)
# cv2.setTrackbarPos("maxValue", "Trackbar Windows", 255)
#
# while cv2.waitKey(1) != ord('q'):
#
#     thresh = cv2.getTrackbarPos("threshold", "Trackbar Windows")
#     maxval = cv2.getTrackbarPos("maxValue", "Trackbar Windows")
#
#     _, binary = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)
#
#     cv2.imshow("Trackbar Windows", binary)
#
# cv2.destroyAllWindows()
##############################################################
#캡처 및 녹화
#영상이나 이미지를 캡쳐하거나 녹화하기 위해 사용합니다. 영상이나 이미지를 연속적 또는 순간적으로 캡쳐하거나 녹화할 수 있습니다.
import datetime
import cv2

capture = cv2.VideoCapture("Star.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

while True:
    if (capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        capture.open("Star.mp4")

    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    key = cv2.waitKey(33)

    if key == 27:
        break
    elif key == 26:
        print("캡쳐")
        cv2.imwrite("D:/" + str(now) + ".png", frame)
    elif key == 24:
        print("녹화 시작")
        record = True
        video = cv2.VideoWriter("D:/" + str(now) + ".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    elif key == 3:
        print("녹화 중지")
        record = False
        video.release()

    if record == True:
        print("녹화 중..")
        video.write(frame)

capture.release()
cv2.destroyAllWindows()

##############################################################
# 기하학적 변환(Warp Perspective)
# 영상이나 이미지 위에 기하학적으로 변환하기 위해 사용. 영상이나 이미지 펼치거나 좁힐 수 있다.
# Tip : WarpPerspective의 경우 4개의 점을 매핑합니다. (4개의 점을 이용한 변환)
# Tip : WarpAffine의 경우 3개의 점을 매핑합니다. (3개의 점을 이용한 변환)
# import numpy as np
# import cv2
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
# height, width, channel = src.shape
# srcPoint = np.array([[300, 200], [400, 200], [500, 500], [200, 500]], dtype=np.float32)
# dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
# # 원본 이미지에서 4점 변환할 srcPoint와 결과 이미지의 위치가 될 dstPoint를 선언합니다.
# # 좌표의 순서는 좌상, 우상, 우하, 좌하 순서입니다. numpy 형태로 선언하며, 좌표의 순서는 원본 순서와 결과 순서가 동일해야합니다.
# # Tip : dtype을 float32 형식으로 선언해야 사용할 수 있습니다.
# matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
# # 원근법 제거
# # 기하학적 변환을 위하여 cv2.getPerspectiveTransform(원본 좌표 순서, 결과 좌표 순서)를 사용하여 matrix를 생성합니다.
# # 다음과 같은 형식으로 매트릭스가 생성됩니다.
# dst = cv2.warpPerspective(src, matrix, (width, height))
# # 위 함수를 거쳐야만 변환행렬값을 적용하여 최종 결과 이미지 얻을 수 있다.
# # cv2.warpPerspective(원본 이미지, 매트릭스, (결과 이미지 너비, 결과 이미지 높이))를 사용하여 이미지를 변환할 수 있습니다.
# # 저장된 매트릭스 값을 사용하여 이미지를 변환합니다.
# # 이외에도, 보간법, 픽셀 외삽법을 추가적인 파라미터로 사용할 수 있습니다.
# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##############################################################
#그래픽 드로잉(Drawing)

#영상이나 이미지 위에 그래픽을 그리기 위해 사용합니다. 선, 원, 사각형 등을 그릴 수 있습니다.

# shift : 좌표를 Shift(비트 연산)만큼 이동시켜 표시합니다.
# offset : 좌표를 (x, y)만큼 이동시켜 표시합니다.#
#
# import numpy as np
# import cv2
#
# src = np.zeros((768, 1366, 3), dtype = np.uint8)
#
# cv2.line(src, (100, 100), (1200, 100), (255, 0, 0), 3, cv2.LINE_AA)
# #cv2.line(이미지, (x1, y1), (x2, y2), (B, G, R), 두께, 선형 타입)을 이용하여 선을 그릴 수 있습니다.
# # (x1, y1)과 (x2, y2)가 연결된 (B, G, R) 색상, 두께 굵기의 선을 그릴 수 있습니다.
# # 선형 타입은 선의 연결성을 의미합니다.
#
# cv2.circle(src, (300, 300), 50, (0, 255, 0), cv2.FILLED, cv2.LINE_4)
# # cv2.circle(이미지, (x, y), 반지름, (B, G, R), 두께, 선형 타입)을 이용하여 원을 그릴 수 있습니다.
# # (x, y) 중심점을 가지는 반지름 크기로 설정된 (B, G, R) 색상, 두께 굵기의 원을 그릴 수 있습니다.
# # Tip : 내부를 채우는 경우, 두께를 cv2.FILLED을 사용하여 내부를 채울 수 있습니다.
#
# cv2.rectangle(src, (500, 200), (1000, 400), (255, 0, 0), 5, cv2.LINE_8)
# # cv2.rectangle(이미지, (x1, y1), (x2, y2), (B, G, R), 두께, 선형 타입)을 이용하여 사각형을 그릴 수 있습니다.
# # (x1, y1)의 좌측 상단 모서리와 (x2, y2)의 우측 하단 모서리가 연결된 (B, G, R) 색상, 두께 굵기의 사각형을 그릴 수 있습니다.
#
# cv2.ellipse(src, (1200, 300), (100, 50), 0, 90, 180, (255, 255, 0), 2)
# # cv2.ellipse(이미지, (x, y), (lr, sr), 각도, 시작 각도, 종료 각도, (B, G, R), 두께, 선형 타입)을 이용하여 타원을 그릴 수 있습니다.
# # (x, y)의 중심점을 가지며 중심에서 가장 먼 거리를 가지는 lr과 가장 가까운 거리를 가지는 sr의 타원을 각도만큼 기울어진 타원를 생성합니다.
# # 시작 각도와 종료 각도를 설정하여 호의 형태로 그리며 (B, G, R) 색상, 두께 굵기의 타원을 그릴 수 있습니다.
# # Tip : 선형 타입은 설정하지 않아도 사용할 수 있습니다.
#
# pts1 = np.array([[100,50],[300,50],[200,600]])
#
# pts2 = np.array([[600, 500], [800, 500], [700, 600]])
#
# #poly 함수를 사용하는 경우, numpy 형태로 저장된 위치 좌표들이 필요합니다.
# #n개의 점이 저장된 경우, n각형을 그릴 수 있습니다.
#
#
# cv2.polylines(src, [pts1], True, (0, 255, 255), 2)
# # cv2.polylines(이미지, [위치 좌표], 닫힘 유/무, (B, G, R), 두께, 선형 타입 )을 이용하여 다각형을 그릴 수 있습니다.
# # [위치 좌표]들의 지점들을 가지며 시작점과 도착점이 연결되어있는지 닫힘 유/무를 설정하여 (B, G, R) 색상, 두께 굵기의 다각형을 그릴 수 있습니다.
# #cv2.fillPoly(src, [pts2], (255, 0, 255), cv2.LINE_AA)
# # cv2.fillPoly(이미지, [위치 좌표], (B, G, R), 두께, 선형 타입 )을 이용하여 내부가 채워진 다각형을 그릴 수 있습니다.
# # [위치 좌표]들의 지점들을 가지며 (B, G, R) 색상, 두께 굵기의 내부가 채워진 다각형을 그릴 수 있습니다.
# #cv2.putText(src, "CHIHO", (900,600), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
# # cv2.putText(이미지, 문자, (x, y), 글꼴, 글자 크기, (B, G, R), 두께, 선형 타입)을 이용하여 문자를 그릴 수 있습니다.
# # 문자 내용을 가지는 문자열을 (x, y) 위치에 표시합니다. 글꼴와 글자 크기를 가지며 (B, G, R) 색상, 두께 굵기의 문자를 그릴 수 있습니다.
# # Tip : 문자의 위치는 좌표의 좌측 하단을 기준으로 글자가 생성됩니다.
#
# cv2.imshow("Result", src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
##############################################################
# 채널 분리(Split) 및 병합(Merge)
# 영상이나 이미지를 채널을 나누고 합치기 위해 사용합니다. 채널을 B(Blue), G(Green), R(Red)로 분리하여 채널을 변환할 수 있습니다.
# 빈 공간(흑백 이미지)을 이용하여 한 채널 아예 지워버리기. Red 지워버리기 ㅋㅋ
# Tip : OpenCV의 가산혼합의 삼원색 기본 배열순서는 BGR입니다..
# b, g, r = cv2.split(이미지)를 이용하여 채널을 분리합니다.
# 채널에 순서의 맞게 각 변수에 대입됩니다.
# 분리된 채널들은 단일 채널이므로 흑백의 색상으로만 표현됩니다.
# cv2.merge((채널1, 채널2, 채널3))을 이용하여 나눠진 채널을 다시 병합할 수 있습니다.
# 채널을 변형한 뒤에 다시 합치거나 순서를 변경하여 병합할 수 있습니다.
# 순서가 변경될 경우, 원본 이미지와 다른 색상으로 표현됩니다.
# Additional Information
# numpy 형식 채널 분리
# b = src[:,:,0]
# g = src[:,:,1]
# r = src[:,:,2]
# 이미지[높이, 너비, 채널]을 이용하여 특정 영역의 특정 채널만 불러올 수 있습니다.
#
# :, :, n을 입력할 경우, 이미지 높이와 너비를 그대로 반환하고 n번째 채널만 반환하여 적용합니다.
# 빈 이미지
# height, width, channel = src.shape
# zero = np.zeros((height, width, 1), dtype = np.uint8)
# bgz = cv2.merge((b, g, zero))
# 검은색 빈 공간 이미지가 필요할 때는 np.zeros((높이, 너비, 채널), dtype=정밀도)을 이용하여 빈 이미지를 생성할 수 있습니다.
# Blue, Green, Zero이미지를 병합할 경우, Red 채널 영역이 모두 흑백이미지로 변경됩니다.
# Tip : import numpy as np가 포함된 상태여야합니다.>
# import cv2
# import numpy as np
#
# src = cv2.imread("ferrari.jpg",cv2.IMREAD_COLOR)
# height, width, channel = src.shape
# b, g, r = cv2.split(src)
# inversebgr = cv2.merge((r,g,b))
# zero = np.zeros((height,width,1),dtype=np.uint8)
# bgz = cv2.merge((b, g, zero))
#
# cv2.imshow("b",b)
# cv2.imshow("g",g)
# cv2.imshow("r",r)
# cv2.imshow("inverse", inversebgr)
# cv2.imshow("bgz", bgz)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##############################################################
#채널 범위 병합(addWeighted)
#영상이나 이미지를 색상을 검출 할 때, cv2.inRange()의 영역이 한정되어 색상을 설정하는 부분이 한정되어 있습니다.
#이 때 특정 범위들을 병합할 때 사용합니다.
# 빨간색 영역은 0 ~ 5, 170 ~ 180의 범위로 두부분으로 나뉘어 있습니다.
#
# 이 때, 두 부분을 합쳐서 한 번에 출력하기 위해서 사용합니다.
#
# cv2.inRange(다채널 이미지, (채널1 최솟값, 채널2 최솟값, 채널3 최솟값), (채널1 최댓값, 채널2 최댓값, 채널3 최댓값))을 통하여 다채널 이미지도 한 번에 범위를 설정할 수 있습니다.
#
# HSV 형식이므로 각각의 h, s, v 범위를 한 번에 설정합니다.
#
# 분리된 채널을 cv2.addWeighted(이미지1, 이미지1 비율, 이미지2, 이미지2 비율, 가중치)를 이용하여 채널을 하나로 합칠 수 있습니다.
#
# 두 이미지의 채널을 그대로 합칠 예정이므로 각각의 비율은 1.0으로 사용하고, 가중치는 사용하지 않으므로 0.0을 할당합니다.
#
# cv2.inRange()를 사용할 때, 단일 채널 이미지의 범위만 할당하여 병합할 수 도 있습니다.
# import cv2
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
# hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(hsv)
#
# lower_red = cv2.inRange(hsv, (0, 100, 100), (5, 255, 255))
# upper_red = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
# added_red = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)
#
# red = cv2.bitwise_and(hsv, hsv, mask = added_red)
# red = cv2.cvtColor(red, cv2.COLOR_HSV2BGR)
#
# cv2.imshow("red", red)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##############################################################
# HSV(Hue, Saturation, Value)
# 영상이나 이미지를 색상을 검출 하기 위해 사용합니다. 채널을 Hue, Saturation, Value로 분리하여 변환할 수 있습니다.
#
# 색상 (Hue) : 색의 질입니다. 빨강, 노랑, 파랑이라고 하는 표현으로 나타내는 성질입니다.
# 채도 (Saturation) : 색의 선명도입니다. 아무것도 섞지 않아 맑고 깨끗하며 원색에 가까운 것을 채도가 높다고 표현합니다.
# 명도 (Value) : 색의 밝기입니다. 명도가 높을수록 백색에, 명도가 낮을수록 흑색에 가까워집니다.
# Detailed Code
# hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(hsv)
# 초기 속성은 BGR이므로, cv2.cvtColor()를 이용하여 HSV채널로 변경합니다.
#
# 각각의 속성으로 분할하기 위해서 cv2.split()을 이용하여 채널을 분리합니다.
#
# Tip : 분리된 채널들은 단일 채널이므로 흑백의 색상으로만 표현됩니다.

#### Main code 1
# import cv2
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
# hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(hsv) # 분리된 채널은 단일 채널
#
# cv2.imshow("h", h)
# cv2.imshow("s", s)
# cv2.imshow("v", v)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#### Main code 2
#Hue의 범위를 조정하여 특정 색상만 출력할 수 있습니다.
#
# cv2.inRange(단일 채널 이미지, 최솟값, 최댓값)을 이용하여 범위를 설정합니다.
#
# 주황색은 약 8~20 범위를 갖습니다.
#
# 이 후, 해당 마스크를 이미지 위에 덧씌워 해당 부분만 출력합니다.
#
# cv2.bitwise_and(원본, 원본, mask = 단일 채널 이미지)를 이용하여 마스크만 덧씌웁니다.
#
# 이 후, 다시 HSV 속성에서 BGR 속성으로 변경합니다.
#
# 색상 (Hue) : 0 ~ 180의 값을 지닙니다.
# 채도 (Saturation) : 0 ~ 255의 값을 지닙니다.
# 명도 (Value) : 0 ~ 255의 값을 지닙니다.
# import cv2
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
# hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(hsv)
#
# # inRange(단일 채널 이미지, 최소값, 최대값)
# h = cv2.inRange(h, 8, 20)
# # bitwise_and(원본,원본, mask = 단일 채널 이미지)
# orange = cv2.bitwise_and(hsv, hsv, mask = h)
# # HSV 속성에서 BGR속성으로 변경
# orange = cv2.cvtColor(orange, cv2.COLOR_HSV2BGR)
#
# cv2.imshow("orange", orange)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##############################################################
# 가장자리 검출
# 영상이나 이미지를 가장자리를 검출 하기 위해 사용합니다.
# Canny Detailed Code
# canny = cv2.Canny(src, 100, 255)
# cv2.Canny(원본 이미지, 임계값1, 임계값2, 커널 크기, L2그라디언트)를 이용하여 가장자리 검출을 적용합니다.
#
# 임계값1은 임계값1 이하에 포함된 가장자리는 가장자리에서 제외합니다.
#
# 임계값2는 임계값2 이상에 포함된 가장자리는 가장자리로 간주합니다.
#
# 커널 크기는 Sobel 마스크의 Aperture Size를 의미합니다. 포함하지 않을 경우, 자동으로 할당됩니다.
#
# L2그라디언트는 L2방식의 사용 유/무를 설정합니다. 사용하지 않을 경우, 자동적으로 L1그라디언트 방식을 사용합니다.
# Sobel Detailed Code
# sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
# cv2.Sobel(그레이스케일 이미지, 정밀도, x방향 미분, y방향 미분, 커널, 배율, 델타, 픽셀 외삽법)를 이용하여 가장자리 검출을 적용합니다.
#
# 정밀도는 결과 이미지의 이미지 정밀도를 의미합니다. 정밀도에 따라 결과물이 달라질 수 있습니다.
#
# x 방향 미분은 이미지에서 x 방향으로 미분할 값을 설정합니다.
#
# y 방향 미분은 이미지에서 y 방향으로 미분할 값을 설정합니다.
#
# 커널은 소벨 커널의 크기를 설정합니다. 1, 3, 5, 7의 값을 사용합니다.
#
# 배율은 계산된 미분 값에 대한 배율값입니다.
#
# 델타는 계산전 미분 값에 대한 추가값입니다.
#
# 픽셀 외삽법은 이미지를 가장자리 처리할 경우, 영역 밖의 픽셀은 추정해서 값을 할당해야합니다.
#
# 이미지 밖의 픽셀을 외삽하는데 사용되는 테두리 모드입니다. 외삽 방식을 설정합니다.
#
# Tip : x방향 미분 값과 y방향의 미분 값의 합이 1 이상이여야 하며 각각의 값은 0보다 커야합니다.
# Laplacian Detailed Code
# laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
# cv2.Laplacian(그레이스케일 이미지, 정밀도, 커널, 배율, 델타, 픽셀 외삽법)를 이용하여 가장자리 검출을 적용합니다.
#
# 정밀도는 결과 이미지의 이미지 정밀도를 의미합니다. 정밀도에 따라 결과물이 달라질 수 있습니다.
#
# 커널은 2차 미분 필터의 크기를 설정합니다. 1, 3, 5, 7의 값을 사용합니다.
#
# 배율은 계산된 미분 값에 대한 배율값입니다.
#
# 델타는 계산전 미분 값에 대한 추가값입니다.
#
# 픽셀 외삽법은 이미지를 가장자리 처리할 경우, 영역 밖의 픽셀은 추정해서 값을 할당해야합니다.
#
# 이미지 밖의 픽셀을 외삽하는데 사용되는 테두리 모드입니다. 외삽 방식을 설정합니다.
#
# Tip : 커널의 값이 1일 경우, 3x3 Aperture Size를 사용합니다. (중심값 = -4)
#
# import cv2
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#
# canny = cv2.Canny(src, 100, 100)
# #100-> 임계값 이하 포함된 가장자리는 가장자리에서 제외
# #255-> 임계값 이상에 포함된 가장자리는 가장자리로 간주
#
# sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
# laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
#
# cv2.imshow("canny", canny)
# cv2.imshow("sobel", sobel)
# cv2.imshow("laplacian", laplacian)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






##############################################################
# blur처리 하기
# #영상이나 이미지를 흐림 효과를 주어 번지게 하기 위해 사용합니다. 해당 픽셀의 주변값들과 비교하고 계산하여 픽셀들의 색상 값을 재조정합니다.
# cv2.blur(원본 이미지, (커널 x크기, 커널 y크기), 앵커 포인트, 픽셀 외삽법)를 이용하여 흐림 효과를 적용합니다.
#
# 커널 크기는 이미지에 흐림 효과를 적용할 크기를 설정합니다. 크기가 클수록 더 많이 흐려집니다.
#
# 앵커 포인트는 커널에서의 중심점을 의미합니다. (-1, -1)로 사용할 경우, 자동적으로 커널의 중심점으로 할당합니다.
#
# 픽셀 외삽법은 이미지를 흐림 효과 처리할 경우, 영역 밖의 픽셀은 추정해서 값을 할당해야합니다.
#
# 이미지 밖의 픽셀을 외삽하는데 사용되는 테두리 모드입니다. 외삽 방식을 설정합니다.


# import cv2
#
# src = cv2.imread("ferrari.jpg",cv2.IMREAD_COLOR)
#
# dst = cv2.blur(src,(1, 1), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
#
# cv2.imshow("dst",dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



##############################################################
#이진화
#영상이나 이미지를 어느 지점을 기준으로 흑색 또는 흰색의 색상으로 변환하기 위해서 사용합니다.
#
# ret, dst를 이용하여 이진화 결과를 저장합니다. ret에는 임계값이 저장됩니다.
#
# cv2.threshold(그레스케일 이미지, 임계값, 최댓값, 임계값 종류)를 이용하여 이진화 이미지로 변경합니다.
#
# 임계값은 이미지의 흑백을 나눌 기준값을 의미합니다. 100으로 설정할 경우, 100보다 이하면 0으로, 100보다 이상이면 최댓값으로 변경합니다.
#
# 임계값 종류를 이용하여 이진화할 방법 설정합니다.
# 임계값 종류
# 속성	의미
# cv2.THRESH_BINARY	임계값 이상 = 최댓값 임계값 이하 = 0
# cv2.THRESH_BINARY_INV	임계값 이상 = 0 임계값 이하 = 최댓값
# cv2.THRESH_TRUNC	임계값 이상 = 임계값 임계값 이하 = 원본값
# cv2.THRESH_TOZERO	임계값 이상 = 원본값 임계값 이하 = 0
# cv2.THRESH_TOZERO_INV	임계값 이상 = 0 임계값 이하 = 원본값
# cv2.THRESH_MASK	흑색 이미지로 변경
# cv2.THRESH_OTSU	Otsu 알고리즘 사용
# cv2.THRESH_TRIANGLE	Triangle 알고리즘 사용

# import cv2
#
# src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)
# #cvtColor -> Convert Color 줄인 말 인듯
# gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
# ret, dst = cv2.threshold(gray, 100, 250, cv2.THRESH_BINARY) # cv2.threshold(그레이스케일이미지, 임계값, 최대값, 임계값 종류)
#
# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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