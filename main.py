import cv2
import numpy as np

def build_gaussian_pyramid(image, numberOfLevels):
    img = image.copy()
    gaussianPyramid = [img]

    for i in range(numberOfLevels):
        img = cv2.GaussianBlur(img, (5, 5), 1.6)
        img = cv2.resize(img, None, fx=1/2, fy=1/2)
        gaussianPyramid.append(np.float32(img))
    return gaussianPyramid


def build_laplacian_pyramid(gaussianPyramid):
    top = gaussianPyramid[-1]
    numberOfLevels = len(gaussianPyramid) - 1

    laplacianPyramid = [top]

    for i in range(numberOfLevels, 0, -1):
        size = (gaussianPyramid[i - 1].shape[1], gaussianPyramid[i - 1].shape[0])
        upSampled = cv2.resize(gaussianPyramid[i], size)
        laplacian = np.subtract(gaussianPyramid[i - 1], upSampled)
        laplacianPyramid.append(laplacian)

    return laplacianPyramid


def blend(laplacian1, laplacian2, maskPyramid_1):
    L = []

    for i in range(len(laplacian1)):
        l1 = laplacian1[i]
        l2 = laplacian2[i]
        mask = maskPyramid_1[i]
        l = l1 * mask + l2 * (1.0 - mask)
        L.append(l)

    return L


def reconstruct(laplacianPyramid):
    laplacian_top = laplacianPyramid[0]
    laplacian_lst = [laplacian_top]
    numberOfLevels = len(laplacianPyramid) - 1

    for i in range(numberOfLevels):
        size = (laplacianPyramid[i + 1].shape[1], laplacianPyramid[i + 1].shape[0])
        upSampled = cv2.resize(laplacian_top, size)
        laplacian_top = cv2.add(laplacianPyramid[i + 1], upSampled)
        laplacian_lst.append(laplacian_top)

    return laplacian_lst

def launch(img1, img2, numberOfLevels, mode = 0):
    mask = np.zeros(img1.shape, dtype='float32')
    x, y, w, h = cv2.selectROI("Select Area", img1)

    if mode == 0: # rectangle mask
        mask[int(y):int(y + h), int(x):int(x + w), :] = (1, 1, 1)
    else: # ellipse mask
        axes_length = (w // 2, h // 2)  # x,y
        center_coordinates = (x+(w // 2), y+(h // 2))
        cv2.ellipse(mask, center_coordinates, axes_length, 0, 0, 360, (1, 1, 1), -1)


    gaussianPyramid_1 = build_gaussian_pyramid(img1, numberOfLevels)

    laplacianPyramid_1 = build_laplacian_pyramid(gaussianPyramid_1)


    gaussianPyramid_2 = build_gaussian_pyramid(img2, numberOfLevels)

    laplacianPyramid_2 = build_laplacian_pyramid(gaussianPyramid_2)


    maskPyramid = build_gaussian_pyramid(mask, numberOfLevels)
    maskPyramid.reverse()

    laplacianPyramid_12 = blend(laplacianPyramid_1, laplacianPyramid_2, maskPyramid)

    result = reconstruct(laplacianPyramid_12)

    return result[numberOfLevels]



#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

# images have to be the same shape

img2 = cv2.imread("image2.jpg") # background image
#img2 = cv2.resize(img2, (690, 460))
img1 = cv2.imread("image.jpg")
#img1 = cv2.resize(img1, (690, 460))


# mode=0 rectangle mask
# mode=1 circular mask
result = launch(img1, img2, numberOfLevels=5, mode=1)

cv2.imwrite("final image.png", result)




