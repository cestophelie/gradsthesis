import cv2
import numpy as np


def edge_get(img):
    # edge_result = img
    print('finally')
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0

    edge_result = cv2.GaussianBlur(img, (5, 5), 0)  # 가우시안 필터 적용

    grad_x = cv2.Sobel(edge_result, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(edge_result, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad


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
    mask = np.zeros_like(edges_)
    cv2.fillPoly(mask, [contour], 255)
    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)
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

    return trimap_print


if __name__ == '__main__':
    img = cv2.imread('pants.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(500, 500), interpolation=cv2.INTER_AREA)
    # original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    original = cv2.imread('pants.jpg', cv2.IMREAD_COLOR)
    original = cv2.resize(original, dsize=(500, 500), interpolation=cv2.INTER_AREA)
    edges_ = edge_get(img)

    # edges_.astype(np.uint8)
    edges_ = np.asarray(edges_, np.uint8)
    edge_result = findSignificant_contour(edges_)

    contourImg = np.copy(original)
    cv2.drawContours(contourImg, [edge_result], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)

    cv2.imshow('result', edges_)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # contourImg = cv2.resize(contourImg, dsize=(500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow('result 2', contourImg)

    output = final(edge_result, edges_)

    cv2.imshow('result 3', output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
