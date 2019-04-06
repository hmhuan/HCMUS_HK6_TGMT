import cv2 as cv
import numpy as np 
import math

# Hàm ép kiểu và kiểu uint8
def saturate_cast(val):
    """ Fix value of a pixel to saturate with image type """
    if int(val) < 0:
        return 0
    if int(val) > 255:
        return 255
    return int(val)

# Hàm phát hiện biên cạnh bằng toán tử Sobel
# Input: ảnh nguồn
# Output: ảnh đạo hàm theo x, y và ảnh đích
def Sobel(srcImage):
    """ Edge detector using Sobel operator
        Argument:
        @srcImage: source image
        Return: Destination image after applying Sobel operator
    """
    # Get height and width of @srcImage
    height, width = srcImage.shape
    # Create @dstImage with shape and type of @srcImage
    Gx = np.zeros((height, width), dtype = srcImage.dtype)
    Gy = np.zeros((height, width), dtype = srcImage.dtype)
    dstImage = np.zeros((height, width), dtype = srcImage.dtype)
    # Kernel of the derivative in x direction    
    Wx = np.array([[0.25, 0, -0.25], [0.5, 0, -0.5], [0.25, 0, -0.25]])
    # Kernel of the derivative in y direction    
    Wy = np.array([[-0.25, -0.5, -0.25], [0, 0, 0], [0.25, 0.5, 0.25]])
    for row in range(1, height-1):
        for col in range(1, width-1):
            # The derivative in x direction of pixel[rol, col]
            temp = srcImage[row - 1 : row + 2, col - 1 : col + 2]
            # The derivative in x direction of pixel[rol, col]
            fx = (temp * Wx).sum()
            Gx[row, col] = saturate_cast(fx)
            # The derivative in y direction of pixel[rol, col]
            fy = (temp * Wy).sum()
            Gy[row, col] = saturate_cast(fy)
            # Approximation of the gradient in that pixel
            dstImage[row, col] = saturate_cast(abs(fx + fy))
    return Gx, Gy, dstImage

# Hàm phát hiện biên cạnh bằng toán tử Prewitt
# Input: ảnh nguồn
# Output: ảnh đạo hàm theo x, y và ảnh đích
def Prewitt(srcImage, dstImage):
    """ Edge detector using Sobel operator
        Argument:
        @srcImage: source image
        Return: Destination image after applying Prewitt operator
    """
    # Get height and width of @srcImage
    height, width = srcImage.shape
    # Create @dstImage with shape and type of @srcImage
    Gx = np.zeros((height, width), dtype = srcImage.dtype)
    Gy = np.zeros((height, width), dtype = srcImage.dtype)
    dstImage = np.zeros((height, width), dtype = srcImage.dtype)
    # Kernel of the derivative in x direction  
    Wx = np.array([[-1/3, 0, 1/3], [-1/3, 0, 1/3], [-1/3, 0, 1/3]])
    # Kernel of the derivative in y direction  
    Wy = np.array([[1/3, 1/3, 1/3], [0, 0, 0], [-1/3, -1/3, -1/3]])
    for row in range(1, height-1):
        for col in range(1, width-1):
            temp = srcImage[row - 1 : row + 2, col - 1 : col + 2]
            # The derivative in x direction of pixel[rol, col]
            fx = (temp * Wx).sum()
            Gx[row, col] = saturate_cast(fx)
            # The derivative in y direction of pixel[rol, col]
            fy = (temp * Wy).sum()
            Gy[row, col] = saturate_cast(fy)
            # Approximation of the gradient in that pixel
            dstImage[row, col] = saturate_cast(abs(fx + fy))
    return Gx, Gy, dstImage

# Hàm phát hiện biên cạnh bằng toán tử Laplace
# Input: ảnh nguồn
# Output: ảnh đích
def Laplace(srcImage):
    """ Edge detector using Laplace operator
        Argument:
        @srcImage: source image
        Return: Destination image after applying Laplace operator
    """
    # Get height and width of @srcImage
    height, width = srcImage.shape
    # Create @dstImage with shape and type of @srcImage
    dstImage = np.zeros((height, width), dtype = srcImage.dtype)
    # Lapalce kernel
    WLap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    for row in range(1, height-1):
        for col in range(1, width-1):
            val = (srcImage[row-1:row+2, col-1:col+2] * WLap).sum()
            dstImage[row, col] = saturate_cast(val)
    return dstImage

# Hàm Canny detector
# Input: ảnh nguồn (ảnh xám), lowThreshold, highTheshold
# Output: ảnh đích
def Canny(srcImage, lowThreshold = 0, highThreshold = 0):

    (m, n) = srcImage.shape
    if (lowThreshold == 0):
        highThreshold = 30
        lowThreshold = 90
    
    # Initial Gradient X, Gradient Y and Edge
    Theta = np.zeros((m, n))
    G = np.zeros((m, n), dtype = srcImage.dtype)
    
    # Step 1: Preprocessing - Gaussian filter
    blurImg = cv.GaussianBlur(srcImage, (5, 5), 1.4)

    # Step 2: Find Orientation and Magnitude at each pixel and Non-maximum suppression
    # Mat Wx, Wy - 2 ma trận đạo hàm đã "ngược hóa" để thuận lới tính tích chập
    Wx = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]])
    Wy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            #Slice matrix - cắt kaays ma trận 3x3 để tính toán
            temp = blurImg[i - 1 : i + 2, j - 1 : j + 2]
            #Tính G, Theta
            # Nhân chập - bản chất là nhân với ma trận đã "ngược hóa" sau đó lấy tổng
            fx = (Wx * temp).sum()
            fy = (Wy * temp).sum()
            # G = (fx^2 + fy^2)^(1/2)
            G[i, j] = saturate_cast(math.sqrt(fx*fx + fy*fy))
            # Quy đổi về góc 0 -> 180 để dễ xét bin
            Theta[i, j] = np.arctan2(fy, fx) * 180 / math.pi
            if Theta[i, j] < 0: 
                Theta[i, j] += 180
    
    gradNms = np.zeros((m, n), dtype = G.dtype)
    #Non-maximum Supression
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            # Chia làm 4 bin với 2 nửa bin là 0 -> 22.5 và 157.5 -> 180 => bin 1
            if (Theta[i, j] < 22.5 or Theta[i, j] >= 157.5):
                Ga, Gb = G[i, j - 1], G[i, j + 1]
            # bin 2: 22.5 -> 67.5
            elif (Theta[i, j] >= 22.5 and Theta[i, j] < 67.5):
                Ga, Gb = G[i + 1, j - 1], G[i - 1, j + 1]
            # bin 3: 67.5 -> 112.5
            elif (Theta[i, j] >= 67.5 and Theta[i, j] < 112.5):
                Ga, Gb = G[i - 1, j], G[i + 1, j]
            else:
                #bin 4: từ 112.5 -> 157.5
                Ga, Gb = G[i - 1, j - 1], G[i + 1, j + 1]
            # Các điểm kề mà nhỏ hơn điểm đang xét thì không loại bỏ nó (ở đây ban đầu gán là 0 nên gán nó là giá trị cường độ gradient tại i, j)
            if (G[i, j] >= Ga) and (G[i, j] >= Gb):
                gradNms[i, j] = G[i, j]
    
    # Step 3: Hysteresis Thresholding  [L, H] Recursive
    #Khởi tạo ảnh đích với các giá trị lớn hơn ngưỡng trên là 255, ngược lại là 0
    dstImage = np.array(gradNms >= highThreshold, dtype = srcImage.dtype) * 255
    #Danh sách lưu trữ các điểm weak
    currentPixels = []
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if (gradNms[i, j] >= lowThreshold and dstImage[i, j] == 0):
                # Những điểm nằm giữa ngưỡng thấp và ngưỡng cao (weak)
                localMax = gradNms[i - 1: i + 2, j - 1: j + 2].max()
                if (localMax >= highThreshold):
                    currentPixels.append((i, j))
                    dstImage[i, j] = 255
                
    # Duyệt và tìm các điểm weak kề các điểm strong
    while len(currentPixels) > 0:
        currentPixel = currentPixels.pop()
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (i != 0  and j != 0):
                    x, y = currentPixel[0] + i, currentPixel[1] + j
                    if (x > 0 and x < gradNms.shape[0] and y > 0 and y < gradNms.shape[1] and gradNms[x, y] >= lowThreshold and dstImage[x, y] == 0):
                        dstImage[x, y] = 255
                        currentPixels.append((x, y))
    return dstImage, lowThreshold, highThreshold
