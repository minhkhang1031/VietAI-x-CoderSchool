import numpy as np
from ultralytics.data.dataset import YOLODataset

class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # Đếm số lượng object trong từng class
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Chọn cách tính trọng số (mặc định: np.mean)
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """ Đếm số lượng object trong từng class """
        self.counts = [0 for _ in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)  # Tránh chia cho 0

    def calculate_weights(self):
        """ Tính trọng số ảnh, chỉ chọn ảnh có chứa "ball" (class 1) """
        weights = []
        for i, label in enumerate(self.labels):
            cls = label['cls'].reshape(-1).astype(int)

            # Nếu ảnh không có class "ball" (1), bỏ qua
            if 1 not in cls:
                weights.append(0)  # Đặt trọng số bằng 0 để không chọn ảnh này
                continue

            # Tính trọng số trung bình của các class trong ảnh
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """ Chuyển trọng số thành xác suất sampling """
        total_weight = sum(self.weights)

        # Tránh lỗi chia cho 0 khi không có ảnh nào hợp lệ
        if total_weight == 0:
            probabilities = np.zeros(len(self.weights))  # Nếu không có ảnh "ball", đặt tất cả xác suất = 0
        else:
            probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """ Lấy ảnh theo trọng số khi training """
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            # Chỉ chọn ảnh có chứa "ball"
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))
