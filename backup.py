import cv2
import numpy as np
from skimage import io


def edge_get(img):
    # edge_result = img
    print('finally')
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0

    edge_result = cv2.GaussianBlur(img, (17, 17), 0)  # 가우시안 필터 적용

    grad_x = cv2.Sobel(edge_result, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(edge_result, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad


def SaltPepperNoise(edgeImg):
    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0
        count = count + 1
        if count > 70:
            break
        lastMedian = median
        median = cv2.medianBlur(edgeImg, 3)

    return edgeImg


def findSignificant_contour(edgeImg):
    print('contour')
    contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # Find level 1 contours
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)
        # From among them, find the contours with large surface area.
        contoursWithArea = []
        for tupl in level1Meta:
            contourIndex = tupl[0]
            contour = contours[contourIndex]
            area = cv2.contourArea(contour)
            contoursWithArea.append([contour, area, contourIndex])
    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largest_contour = contoursWithArea[0][0]
    print('out')

    return largest_contour


def final(contour, edges_):
    mask = np.zeros_like(edges_)  # given array 와 사이즈가 같은 배열을 모두 zero 로 초기화해서 리턴
    cv2.fillPoly(mask, [contour], 255)  # 다각형 만들기
    # calculate sure foreground area by dilating the mask
    kernel = np.ones((3, 3), np.uint8)  # 여백의 사이즈를 정하는 커널
    mapFg = cv2.erode(mask, kernel, iterations=10)
    # mark inital mask as "probably background"
    # and mapFg as sure foreground
    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD

    # visualize trimap
    trimap_print = np.copy(trimap)
    trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
    trimap_print[trimap_print == cv2.GC_FGD] = 255
    cv2.imwrite('trimap.jpg', trimap_print)
    # cv2.imshow('in here', trimap_print)
    # cv2.waitKey(0)

    return trimap_print


if __name__ == '__main__':
    img = cv2.imread('pants.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(500, 500), interpolation=cv2.INTER_AREA)
    # original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    original = cv2.imread('pants.jpg', cv2.IMREAD_COLOR)
    original = cv2.resize(original, dsize=(500, 500), interpolation=cv2.INTER_AREA)
    edges_ = edge_get(img)
    edges_ = SaltPepperNoise(edges_)

    # edges_.astype(np.uint8)
    edges_ = np.asarray(edges_, np.uint8)
    edge_result = findSignificant_contour(edges_)

    contourImg = np.copy(original)
    cv2.drawContours(contourImg, [edge_result], 0, (0, 255, 0), 1, cv2.LINE_AA, maxLevel=1)  # contour 두께 설정 및 등등

    cv2.imshow('result', edges_)
    cv2.waitKey(0)

    cv2.imshow('result 2', contourImg)

    output = final(edge_result, edges_)

    cv2.imshow('result 3', output)

    mask2 = np.where((output < 128), 0, 1).astype('uint8')
    print('min : ' + str(np.max(mask2)))
    cv2.imshow('here', mask2)  # edge_result
    cv2.imwrite('mask.jpg', mask2)
    original2 = img * mask2
    cv2.imshow('draft1', original2)

    mask2 = cv2.cvtColor(mask2, cv2.cv2.COLOR_GRAY2RGB)
    original1 = original * mask2
    cv2.imshow('draft2', original1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()