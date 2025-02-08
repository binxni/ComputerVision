import cv2 as cv
import numpy as np

def image_process(image):
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab_clahe = cv.merge((l, a, b))
    enhanced_image = cv.cvtColor(lab_clahe, cv.COLOR_LAB2BGR)
    
    denoised_image = cv.bilateralFilter(enhanced_image, d=-1, sigmaColor=50, sigmaSpace=50)
    
    return denoised_image


def mouse_callback(event, x, y, flags, param): #마우스 콜백 함수
    global corners

    # 마우스 클릭 이벤트가 발생하면 좌표를 저장
    if event == cv.EVENT_LBUTTONDOWN:
        if len(corners) < 4:  # 최대 4개 좌표만 저장
            corners.append((x, y))
            print(f"좌표 추가: ({x}, {y})")
            # 선택한 좌표에 원 그리기
            cv.circle(param, (x, y), 5, (0, 255, 0), -1)
            cv.imshow("Checkerboard", param)

def perspective_transform_mouse(image, corners): #2.수동 투시변환 함수
    # 모서리 좌표를 정수형으로 변환
    corners = np.array(corners, dtype=np.float32)

    # 출력할 정방형의 모서리 정의 (시계 방향으로 정의)
    width, height = 300, 300  # 출력할 이미지의 크기
    dst_points = np.array([[0, 0],[width - 1, 0],[width - 1, height - 1],[0, height - 1]], dtype=np.float32)

    # 투시 변환 행렬 계산
    matrix = cv.getPerspectiveTransform(corners, dst_points)

    # 투시 변환 적용
    transformed_image = cv.warpPerspective(image, matrix, (width, height))

    return transformed_image

board = cv.imread('checker.png')
if board is None:
    print("image load failed")
    exit()
board_process = image_process(board)

#수동 투시 변환
image = np.copy(board_process)
corners = [] # 모서리 좌표를 저장할 리스트
cv.imshow("Checkerboard", image)
cv.setMouseCallback("Checkerboard", mouse_callback, image)
    
print("네 개의 모서리 좌표를 클릭하여 선택하세요. (좌상단부터 시계방향으로)")
cv.waitKey(0) 
    
# 모서리가 4개 선택되었는지 확인
if len(corners) == 4:
    # 투시 변환 적용
    transformed_image = perspective_transform_mouse(image, corners)
    
    # 결과 출력
    cv.imshow("Transformed mouse Checkerboard", transformed_image)
    cv.waitKey(0)
else:
    print("모서리 좌표가 4개가 아닙니다.")
cv.destroyAllWindows()