import numpy as np
from matplotlib import pyplot as plt

file_path = "week11/full_numpy_bitmap_alarm clock.npy"
image_alarm = np.load(file_path).astype(np.float32)
print(image_alarm.shape)
train_images = image_alarm[:-10]
test_images = image_alarm[-10:]

avg_image = np.mean(train_images, axis=0)
avg_image = avg_image.reshape(28,28)

plt.imshow(avg_image)
plt.show()

index = 4
test_image = test_images[index].reshape(28,28)
score = np.dot(avg_image.flatten(), test_image.flatten())
print(score)
# #2
# images_ant = np.load("week11/full_numpy_bitmap_ant.npy")
# print(images_ant.shape)
# avg_image_ant = np.mean(images_ant, axis = 0)
# avg_image_ant = avg_image_ant.reshape(28,28)
# # plt.imshow(avg_image_ant)
# # plt.show()
# score2 = np.dot(avg_image_ant.flatten(), test_image.flatten())
# print(score2)
# #3
# images_banana = np.load("week11/full_numpy_bitmap_banana.npy")
# print(images_banana.shape)
# avg_image_banana = np.mean(images_banana, axis = 0)
# avg_image_banana = avg_image_banana.reshape(28,28)
# # plt.imshow(avg_image_banana)
# # plt.show()
# score3 = np.dot(avg_image_banana.flatten(), test_image.flatten())
# print(score3)
# #4
# images_bridge = np.load("week11/full_numpy_bitmap_bridge.npy")
# print(images_bridge.shape)
# avg_image_bridge = np.mean(images_bridge, axis = 0)
# avg_image_bridge = avg_image_bridge.reshape(28,28)
# # plt.imshow(avg_image_bridge)
# # plt.show()
# score4 = np.dot(avg_image_bridge.flatten(), test_image.flatten())
# print(score4)
# #5
# images_calculator = np.load("week11/full_numpy_bitmap_calculator.npy")
# print(images_calculator.shape)
# avg_image_calculator = np.mean(images_calculator, axis = 0)
# avg_image_calculator = avg_image_calculator.reshape(28,28)
# # plt.imshow(avg_image_calculator)
# # plt.show()
# score5 = np.dot(avg_image_calculator.flatten(), test_image.flatten())
# print(score5)
# #6
# images_candle = np.load("week11/full_numpy_bitmap_candle.npy")
# print(images_candle.shape)
# avg_image_candle = np.mean(images_candle, axis = 0)
# avg_image_candle = avg_image_candle.reshape(28,28)
# # plt.imshow(avg_image_candle)
# # plt.show()
# score6 = np.dot(avg_image_candle.flatten(), test_image.flatten())
# print(score6)
# #7
# images_car = np.load("week11/full_numpy_bitmap_car.npy")
# print(images_car.shape)
# avg_image_car = np.mean(images_car, axis = 0)
# avg_image_car = avg_image_car.reshape(28,28)
# # plt.imshow(avg_image_car)
# # plt.show()
# score7 = np.dot(avg_image_car.flatten(), test_image.flatten())
# print(score7)
# #8
# images_cloud = np.load("week11/full_numpy_bitmap_cloud.npy")
# print(images_cloud.shape)
# avg_image_cloud = np.mean(images_cloud, axis = 0)
# avg_image_cloud = avg_image_cloud.reshape(28,28)
# # plt.imshow(avg_image_cloud)
# # plt.show()
# score8 = np.dot(avg_image_cloud.flatten(), test_image.flatten())
# print(score8)
# #9
# images_cup = np.load("week11/full_numpy_bitmap_cup.npy")
# print(images_cup.shape)
# avg_image_cup = np.mean(images_cup, axis = 0)
# avg_image_cup = avg_image_cup.reshape(28,28)
# # plt.imshow(avg_image_cup)
# # plt.show()
# score9 = np.dot(avg_image_cup.flatten(), test_image.flatten())
# print(score9)
# #10
# images_flower = np.load("week11/full_numpy_bitmap_flower.npy")
# print(images_flower.shape)
# avg_image_flower = np.mean(images_flower, axis = 0)
# avg_image_flower = avg_image_flower.reshape(28,28)
# # plt.imshow(avg_image_flower)
# # plt.show()
# score10 = np.dot(avg_image_flower.flatten(), test_image.flatten())
# print(score10)

list = ["ant","banana","bridge","calculator","candle","car","cloud","cup","flower"]
scores = []
avg_images = []
for category in list:
    file_paths = f'week11/full_numpy_bitmap_{category}.npy'
    images = np.load(file_paths).astype(np.float32)
    avg_image = np.mean(images, axis = 0)
    avg_image = avg_image.reshape(28,28)
    avg_images.append(avg_image)
    scorel = np.dot(avg_image.flatten(), test_image.flatten())
    scores.append((category,scorel))

for category, score in scores:
    print(f"Score with {category.capitalize()}: {score}")

plt.figure(figsize=(10, 5))
for i, avg_image in enumerate(avg_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(avg_image)
    plt.title(list[i].capitalize())
    plt.axis("off")
plt.tight_layout()
plt.show()
"""
2622977.0
1581216.6600247168
1384111.3262626
1416932.8619878204 
2990439.066017527
1899284.2626797133
1988860.2525825654
1582593.5655677048
2064056.76613551
2540789.80731677
"""
# Bước 5: Các bạn hãy lặp lại bước 1 đến 3 cho tất cả các categories còn lại (chú ý tại bước 1 các bạn không cần phân
# ra train với test images nữa nhé, coi như là dùng tất cả cho train). Sau đó các bạn hãy tính tích vô hướng của từng ảnh
# trung bình của ảnh test các bạn chọn ở bước 4 với từng bức ảnh trung bình này.
#
# Cuối cùng các bạn xem là liệu trong 10 score này, score tương ứng với tích vô hướng của ảnh test này với
# ảnh trung bình của category của chính nó có phải là score lớn nhất không nhé. Các bức ảnh trung bình mà các bạn tính ra
# có thể xem như là weight cho từng category mà các bạn vừa học ở bài 1 (tất nhiên là weight của mô hình sau khi đã
# train xong)
# ==> Ảnh trung bình của category của chính nó không phải là score lớn nhất

# Bước 6 (optional): Các bạn thử visualize 10 weight (avg_image) này trong cùng 1 ảnh kích thước 2x5 hoặc 5x2 để so sánh xem,
# weight của các categories nào dễ nhìn và weight nào không nhé
