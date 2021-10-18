import os

import cv2 as cv
import numpy as np

from digit_reckon import predict
from solver import solve


def find_max_area_contour(contours):
    areas = [cv.contourArea(contour) for contour in contours]
    if len(areas) != 0:
        max_area = max(areas)
        max_area_index = areas.index(max_area)
        return contours[max_area_index]
    return None


img = cv.imread("sudoku2.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
thresh = cv.adaptiveThreshold(gray, 250, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 3)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

max_contour = find_max_area_contour(contours)
approx = cv.approxPolyDP(max_contour, 0.1 * cv.arcLength(max_contour, True), True)


def clean_corners(contour):
    # Return:
    # [left-top right-top left-bottom right-bottom]
    result = [0, 0, 0, 0]
    corners = [x[0] for x in contour]
    if len(corners) != 4:
        print("invalid")
        os._exit(0)
    else:
        sums = [(corner[0] + corner[1]) for corner in corners]
        max_index = sums.index(max(sums))
        min_index = sums.index(min(sums))
        result[0] = contour[min_index]
        result[3] = contour[max_index]
        corners.pop(max_index)
        corners.pop(min_index)
        if corners[0][0] < corners[1][0]:
            result[2] = np.array([corners[0]])
            result[1] = np.array([corners[1]])
        else:
            result[2] = np.array([corners[1]])
            result[1] = np.array([corners[0]])
        return np.array(result)


def zoom_sudoku(img, corners):
    pts1 = np.float32([corner[0] for corner in corners])
    pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (400, 400))
    return dst, pts1, M


cleaned = clean_corners(approx)
zoomed, pts1, M = zoom_sudoku(img, cleaned)
zoomed_gray = cv.cvtColor(zoomed, cv.COLOR_BGR2GRAY)
zoomed_gray = cv.GaussianBlur(zoomed_gray, (5, 5), 0)
zoomed_thresh = cv.adaptiveThreshold(zoomed_gray, 250, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 3)


def extract_numbers(zoomed_thresh):
    cell_width = zoomed_thresh.shape[1] // 9
    cell_height = zoomed_thresh.shape[0] // 9
    result = np.empty((9, 9, cell_height, cell_width))
    for row in range(9):
        for col in range(9):
            a = zoomed_thresh[row * cell_width:(row + 1) * cell_width,
                col * cell_height:cell_height * (col + 1)]
            result[row, col] = a
    return result


extracted = extract_numbers(zoomed_thresh)


def clean_borders(extracted):
    result = np.empty((9, 9, 28, 28), np.uint8)
    for row in range(extracted.shape[0]):
        for col in range(extracted.shape[1]):
            img = np.uint8(extracted[row, col])
            contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                if (x < 6 or y < 6 or w < 6 or h < 6):
                    # Clean borders
                    cv.fillPoly(img, [contour], (0, 0, 0))
                img = cv.resize(img, (28, 28))
                result[row, col] = img
    return result


extracted = clean_borders(extracted)


def resize_images(images, size=28):
    result = np.zeros((images.shape[0], images.shape[1], size, size), np.uint8)
    for row in range(images.shape[0]):
        for col in range(images.shape[1]):
            img = images[row, col]
            resized_img = cv.resize(img, (size, size), interpolation=cv.INTER_AREA)
            result[row, col] = resized_img
    return result


resized = resize_images(extracted, 28)


def zoom_images(images):
    for row in range(images.shape[0]):
        for col in range(images.shape[1]):
            img = images[row, col]
            contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            max_contour = find_max_area_contour(contours)
            if max_contour is not None:
                if cv.contourArea(max_contour) > 20:
                    x, y, w, h = cv.boundingRect(max_contour)
                    if (x > 3 and y > 3):
                        offset = 1
                        x -= offset
                        y -= offset
                        w += 2 * offset
                        h += 2 * offset
                        pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
                        pts2 = np.float32([[0, 0], [28, 0], [0, 28], [28, 28]])
                        M = cv.getPerspectiveTransform(pts1, pts2)
                        img = cv.warpPerspective(img, M, (28, 28))
                        images[row, col] = img


zoom_images(resized)


def make_predictions(board):
    result = np.empty((9, 9))
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            result[row, col] = predict(board[row, col])
    return result


predicted_board = make_predictions(resized)
solved_board = solve(predicted_board)


def put_numbers_on_zoomed(board, zoomed):
    for r, row in enumerate(board):
        for c, el in enumerate(row):
            if el.block is False:
                font = cv.FONT_HERSHEY_SIMPLEX
                cell_width = zoomed.shape[1] // 9
                cell_height = zoomed.shape[0] // 9
                bottomLeftCornerOfText = (int(cell_width * (c + 0.25)), int(cell_height * (r + 1 - 0.15)))
                fontScale = 1
                fontColor = (0, 0, 255)
                lineType = 2

                cv.putText(zoomed, str(el.number),
                           bottomLeftCornerOfText,
                           font,
                           fontScale,
                           fontColor,
                           lineType)


put_numbers_on_zoomed(solved_board, zoomed)
zoomed = cv.warpPerspective(zoomed, M, (img.shape[1], img.shape[0])
                            , flags=cv.WARP_INVERSE_MAP)


def get_mask(zoomed, pts):
    mask = np.zeros((zoomed.shape[0], zoomed.shape[1]), dtype=np.uint8)
    tmp = pts[0, 2].copy()
    pts[0, 2] = pts[0, 3]
    pts[0, 3] = tmp
    cv.fillPoly(mask, pts, (255, 255, 255))
    ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    return thresh


mask = get_mask(zoomed, np.array([pts1], dtype=np.int32))


def get_only_object(img, mask, back_img):
    fg = cv.bitwise_or(img, img, mask=mask)

    mask_inv = cv.bitwise_not(mask)
    fg_back_inv = cv.bitwise_or(back_img, back_img, mask=mask_inv)

    final = cv.bitwise_or(fg, fg_back_inv)
    return final


result = get_only_object(zoomed, mask, img)
cv.imshow("a", result)
cv.waitKey(0)
cv.destroyAllWindows()
