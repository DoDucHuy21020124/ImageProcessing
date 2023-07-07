import cv2
import numpy as np

def resizeImage(image, height):
    h, w, _ = image.shape
    ratio = w / h
    width = int(ratio * height)
    image = cv2.resize(image, (width, height), interpolation= cv2.INTER_CUBIC)
    return image

def SpotTheDifferences(image1: str, image2: str):
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)
    print(image1.shape, image2.shape)

    image1 = resizeImage(image1, 480)
    image2 = resizeImage(image2, 480)
    
    clone1 = image1.copy()
    clone2 = image2.copy()

    clone1 = cv2.cvtColor(clone1, cv2.COLOR_BGR2GRAY)
    clone2 = cv2.cvtColor(clone2, cv2.COLOR_BGR2GRAY)

    # clone1 = cv2.GaussianBlur(clone1, ksize = (11, 11), sigmaX = 0)
    # clone2 = cv2.GaussianBlur(clone2, ksize = (11, 11), sigmaX = 0)

    diff = cv2.absdiff(clone1, clone2)
    print(diff.shape)
    cv2.imshow('diff', diff)
    thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('thresh', thresh)

    kernel = np.ones((3,3), np.uint8) 
    dilate = cv2.dilate(thresh, kernel, iterations=3)
    cv2.imshow('Dilate', dilate)

    mask = np.zeros(shape = image1.shape)

    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:
            cv2.drawContours(mask, [contour], -1, (0, 255, 255), 3)
            ((cx, cy), radius) = cv2.minEnclosingCircle(contour)
            cv2.circle(image1, (int(cx), int(cy)), int(radius), (0, 255, 255), 3)
            cv2.circle(image2, (int(cx), int(cy)), int(radius), (0, 255, 255), 3)

    cv2.imshow('contours', mask)
    cv2.imshow('image1', image1)
    cv2.imshow('image2', image2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    SpotTheDifferences('./images/church0.jpg', './game_data/church0.jpg')

if __name__ == '__main__':
    main()