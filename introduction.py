import cv2
import numpy as np
import matplotlib.pyplot as plt
# I is an RGB-image
I = cv2.imread('kot_pixel.jpg')
# Number of histogram bins
histSize = 256
# Histogram range
# The upper boundary is exclusive
histRange = (0, 256)
# Split an image into color layers
# OpenCV stores RGB image as BGR
I_BGR = cv2.split(I)
 # Calculate a histogram for each layer
bHist = cv2.calcHist(I_BGR, [0], None, [histSize], histRange)
gHist = cv2.calcHist(I_BGR, [1], None, [histSize], histRange)
rHist = cv2.calcHist(I_BGR, [2], None, [histSize], histRange)

# Создаем график
plt.figure()
plt.title("Гистограммы RGB-каналов")
plt.xlabel("Значение интенсивности")
plt.ylabel("Количество пикселей")

# Отображаем гистограммы
plt.plot(bHist, color="blue", label="Blue")
plt.plot(gHist, color="green", label="Green")
plt.plot(rHist, color="red", label="Red")

# Добавляем легенду
plt.legend()

# Показываем график
plt.show()



# преобразуем изображение в формат float32 для точности вычислений
image_float = I.astype(np.float32)

# сдвигаем яркость на 50 уровней (в диапазоне 0-255)
image_shifted = image_float * 2.5

# ограничиваем значения, чтобы они оставались в диапазоне [0, 255]
image_shifted = np.clip(image_shifted, 0, 255).astype(np.uint8)

# показываем исходное и обработанное изображение
cv2.imshow('Исходное изображение', I)
cv2.imshow('Изображение со сдвигом гистограммы', image_shifted)

# ждем нажатия клавиши и закрываем окна
cv2.waitKey(0)
cv2.destroyAllWindows()



