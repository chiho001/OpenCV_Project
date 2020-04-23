# 이미지 확대 축소
# pyrUp()과 pyrDown()함수는 결과 이미지 크기와 픽셀 외삽법은 기본갑승로 설정된 인수를 할당해야하므로 생략하여 사용.
# 피라미드 함수에서 픽셀 외삽은 cv2.BORDER_DEFAULT만 사용 가능

import cv2
import numpy as np
src = cv2.imread("ferrari.jpg", cv2.IMREAD_COLOR)

height, width, channel = src.shape
dst = cv2.pyrUp(src, dstsize=(width*2, height*2), borderType=cv2.BORDER_DEFAULT)
dst2 = cv2.pyrDown(src)
dst3 = cv2.pyrDown(dst2)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)
cv2.imshow("dst3", dst3)
cv2.imshow("dst3", dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()