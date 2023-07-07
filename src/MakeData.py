import cv2
import numpy as np
import random
import glob
import os

def resizeImage(image, height):
    h, w, _ = image.shape
    ratio = w / h
    width = int(ratio * height)
    image = cv2.resize(image, (width, height), interpolation= cv2.INTER_CUBIC)
    return image

def getDiameter(img):
    h, w = img.shape[:2]
    diameter = int(np.sqrt(h ** 2 + w ** 2)) + 1
    return diameter

def addBorders(img):
    h, w = img.shape[:2]
    diameter = getDiameter(img)
    top = bottom = diameter - h
    left = right = diameter - w
    img_border = cv2.copyMakeBorder(
        img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value = (0, 0, 0)
    )
    return img_border


def removeBorders(img):
    h, w = np.shape(img)[:2]

    B = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2GRAY)

    left = w
    right = 1
    top = h
    bottom = 1

    for i in range (1, h):
        for j in range (1, w):
            if B[i,j] > 0:

                if i < top:
                    top = i

                if i > bottom:
                    bottom = i

                if j < left:
                    left = j

                if j > right:
                    right = j

    C = img[top: bottom + 1, left: right + 1]

    return C

def fillHole(img, contour):
    col, row, w, h = contour[0], contour[1], contour[2], contour[3] # x: column, y: row
    for y in range(row, row + h):
        for x in range(col, col + w):
            if img[y][x][0] == 0 and img[y][x][1] == 0 and img[y][x][2] == 0:
                sum = np.zeros(shape = (3, ))
                count = 0
                if x - 1 >= 0:
                    if img[y][x - 1][0] > 0 or img[y][x - 1][1] > 0 or img[y][x - 1][2] > 0:
                        sum += img[y, x - 1, :]
                        count += 1
                    if y - 1 >= 0:
                        if img[y - 1][x - 1][0] > 0 or img[y - 1][x - 1][1] > 0 or img[y - 1][x - 1][2] > 0:
                            sum += img[y - 1, x - 1, :]
                            count += 1

                        if img[y - 1][x][0] > 0 or img[y - 1][x][1] > 0 or img[y - 1][x][2] > 0:
                            sum += img[y - 1, x, :]
                            count += 1

                    if y + 1 < img.shape[0]:
                        if img[y + 1][x - 1][0] > 0 or img[y + 1][x - 1][1] > 0 or img[y + 1][x - 1][2] > 0:
                            sum += img[y + 1, x - 1, :]
                            count += 1

                        if img[y + 1][x][0] > 0 or img[y + 1][x][1] > 0 or img[y + 1][x][2] > 0:
                            sum += img[y + 1, x, :]
                            count += 1

                if x + 1 < img.shape[1]:
                    if img[y][x + 1][0] > 0 or img[y][x + 1][1] > 0 or img[y][x + 1][2] > 0:
                        sum += img[y, x + 1, :]
                        count += 1
                    
                    if y - 1 >= 0:
                        if img[y - 1][x + 1][0] > 0 or img[y - 1][x + 1][1] > 0 or img[y - 1][x + 1][2]:
                            sum += img[y - 1, x + 1, :]
                            count += 1
                        
                    if y + 1 < img.shape[0]:
                        if img[y + 1][x + 1][0] > 0 or img[y + 1][x + 1][1] > 0 or img[y + 1][x + 1][2] > 0:
                            sum += img[y + 1, x + 1, :]
                            count += 1
                if count > 0:
                    img[y, x, :] = (sum / count).astype('uint8')
    return img

def deleteObject(image, contours):
    image = image.copy()
    maskList = []
    mask = np.zeros(shape = image.shape)
    for i in range(len(contours)):
        cv2.drawContours(mask, contours, i, (255, 255, 255), 3)
        cv2.imshow('contour', mask)
        cv2.fillPoly(mask, [contours[i]], color = (255, 255, 255))
        print(image.shape, mask.shape)
        (x, y, w, h) = cv2.boundingRect(contours[i])
        print(x, y, w, h)
        maskList.append([x, y, w, h])

    mask = mask.astype('uint8')
    cv2.imshow('mask', mask)
    mask1 = 255 - mask
    cv2.imshow('mask1', mask1)
    image = cv2.bitwise_and(image, mask1)
    cv2.imshow('bitwise_and', image)
    
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    cv2.imshow('inpaint', image)

    for i in range(len(maskList)):
        x, y, w, h = maskList[i]
        local = image[y: y + h, x: x + w]
        local = cv2.fastNlMeansDenoisingColored(local, None, 10, 10, 7, 21)
        image[y: y + h, x: x + w] = local
    cv2.imshow('denoise', image)

    return image

def rotateObject(image, contours):
    image = image.copy()
    mask = np.zeros(shape = image.shape)
    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        if not (x == 0 or x + w == image.shape[1] or y == 0 or y + h == image.shape[0]):
            cv2.drawContours(mask, contours, i, (255, 255, 255), 1)
            cv2.fillPoly(mask, [contours[i]], color = (255, 255, 255))

            local = image[y: y + h, x: x + w]
            local = addBorders(local)

            mask_local = mask[y: y + h, x: x + w]
            mask_local = 255 - mask_local
            mask_local = addBorders(mask_local)

            angle = 180
            M = cv2.getRotationMatrix2D((local.shape[1] // 2, local.shape[0] // 2), angle, 1)

            fake = cv2.warpAffine(local, M, (local.shape[1], local.shape[0]))
            fake = removeBorders(fake)
            image[y: y + h, x: x + w] = fake

            fake_mask = cv2.warpAffine(mask_local, M, (mask_local.shape[1], mask_local.shape[0]))
            fake_mask = removeBorders(fake_mask.astype('uint8'))

            mask[y: y + h, x: x + w] = fake_mask
    
    mask = mask.astype('uint8')
    mask = 255 - mask
    cv2.imshow('mask', mask)
    image = cv2.bitwise_and(image, mask)
    cv2.imshow('bitwise_and', image)
    mask = 255 - mask
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    print(mask.shape, image.shape)
    image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return image

def addObject(image, filepath, contours, amount):
    mask = np.zeros(shape = image.shape)
    cv2.imshow('mask', mask)
    pngfile = glob.glob(os.path.join(filepath, '*.png'))
    if amount < 0:
        amount = random.randint(1, len(pngfile))

    area = []
    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        area.append([x, y, w, h])
    
    for count in range(amount):
        index = random.randint(0, len(pngfile) - 1)
        add_image = cv2.imread(pngfile[index])
        cv2.imshow('add_image', add_image)
        h, w, _ = add_image.shape
        n = 0
        while (n < 20):
            x = random.randint(0, image.shape[1] - 1)
            y = random.randint(0, image.shape[0] - 1)
            check = True
            for i in range(len(area)):
                if area[i][0] < x and area[i][0] + area[i][2] > x and area[i][1] < y and area[i][1] + area[i][3] > y:
                    check = False
                    break
                if area[i][0] < x + w and area[i][0] + area[i][2] > x + w and area[i][1] < y + h and area[i][1] + area[i][3] > y + h:
                    check = False
                    break

                if (x + w) > image.shape[1]:
                    check = False
                    break
                if (y + h) > image.shape[0]:
                    check = False
                    break

            if check:
                mask = cv2.cvtColor(add_image, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(add_image, 0, 255, cv2.THRESH_BINARY)
                mask = 255 - mask
                cv2.imshow('add_mask', mask)
                local = image[y: y + h, x: x + w]
                local = cv2.bitwise_and(local, mask)
                cv2.imshow('bitwise_and', local)
                local = cv2.bitwise_or(local, add_image)
                cv2.imshow('bitwise_or', local)
                image[y: y + h, x: x + w] = local
                area.append([x, y, w, h])
                break
            else:
                n += 1
    return image
        
def changeColorObject(image, contours):
    image = image.copy()
    mask = np.zeros(shape = image.shape)
    for i in range(len(contours)):
        cv2.drawContours(mask, contours, i, (255, 255, 255), 1)
        cv2.fillPoly(mask, [contours[i]], color = (255, 255, 255))
        (x, y, w, h) = cv2.boundingRect(contours[i])
        for j in range(y, y + h):
            for k in range(x, x + w): 
                if mask[j, k, 0] == 255 and mask[j, k, 1] == 255 and mask[j, k, 2] == 255:
                    image[j, k, :] = image[j, k, ::-1]
    return image 

def saveImage(output_dir, filename, image):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return cv2.imwrite(os.path.join(output_dir, filename), image)


def main():
    filename = 'beach0.jpg'
    image = cv2.imread(os.path.join('./images', filename))
    
    image = resizeImage(image, 480)
    areaImage = image.shape[0] * image.shape[1]

    imageClone = image.copy()

    cv2.imshow('original image', image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edge = cv2.Canny(gray, 30, 180)
    cv2.imshow('edge', edge)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(shape = image.shape)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), 3)
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations= 3)
    # print(mask.dtype)
    # print('maskshape', mask)
    cv2.imshow('contours1', mask)

    # print(mask.shape)
    # mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)[1]
    # print(mask.shape)
    mask = mask.astype('uint8')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask2 = np.zeros(shape = image.shape)
    contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask2, contours2, -1, (255, 255, 255), 1)
    cv2.imshow('mask2', mask2)
    # for i in range(len(contours2)):
    #     print(contours2[i])
    # mask2 = mask2.astype('uint8')
    # mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('maskgray',  mask2)

    # kernel2 = np.ones(shape = (3, 3))
    # mask2_dilation = cv2.dilate(mask2, kernel2, iterations=3)
    # cv2.imshow('mask2_dilation', mask2_dilation)

    contour_test = []
    for i in range(len(contours2)):
        contourArea = cv2.contourArea(contours2[i])
        if contourArea > 0.0025 * areaImage and contourArea < areaImage * 0.2:
            contour_test.append(contours2[i])    
    contour_test = sorted(contour_test, key = cv2.contourArea)

    numOfContour = len(contour_test)
    part = int(numOfContour / 3)

    # imageClone = deleteObject(imageClone, contour_test[part: (2 * part)])
    # imageClone = rotateObject(imageClone, contour_test[0: part])
    imageClone = changeColorObject(imageClone, contour_test[(2 * part): numOfContour])
    # imageClone = addObject(imageClone, './add_images', contour_test, 1)
    cv2.imshow('change color object', imageClone)

    cv2.waitKey(0) 

    if saveImage('game_data', filename, imageClone):
        print('Save image successfully!')
    else:
        print('Save image failed!')

if __name__ == '__main__':
    main()