# Fibonacci-Net

Fibonacci-Net là một mô hình mạng neural đơn giản sử dụng dãy Fibonacci để xác định cấu trúc của mạng. Dự án này cung cấp một cách tiếp cận sáng tạo trong việc thiết kế kiến trúc mạng neural.

## Giới thiệu

Fibonacci-Net tận dụng đặc tính của dãy Fibonacci để thiết kế số lượng neuron trong mỗi lớp ẩn của mạng neural. Mô hình này được thiết kế để khám phá mối quan hệ giữa các cấu trúc toán học tự nhiên như dãy Fibonacci và hiệu suất của mạng neural.

## Cài đặt

pip install torch numpy matplotlib
git clone https://github.com/conbopk/Fibonacci-Net.git
cd Fibonacci-Net

## Cách sử dụng

from fibonacci_net import FibonacciNet

# Tạo một mô hình Fibonacci-Net với 2 đầu vào và 1 đầu ra
model = FibonacciNet(input_size=2, output_size=1, num_layers=5)

# Huấn luyện mô hình
model.train(X_train, y_train, learning_rate=0.01, epochs=1000)

# Thực hiện dự đoán
predictions = model.predict(X_test)

## Kiến trúc

FibonacciNet được xây dựng với các đặc điểm sau:

- **Cấu trúc dựa trên Fibonacci**: Số lượng neuron trong mỗi lớp ẩn tuân theo dãy Fibonacci
- **PyTorch Implementation**: Sử dụng thư viện PyTorch để xây dựng và huấn luyện mô hình
- **Khả năng tùy chỉnh**: Cho phép điều chỉnh số lượng lớp, kích thước đầu vào và đầu ra

## Hàm và Phương thức

- `__init__(input_size, output_size, num_layers)`: Khởi tạo mô hình với kích thước đầu vào/ra và số lớp xác định
- `forward(x)`: Thực hiện quá trình lan truyền xuôi
- `train(X, y, learning_rate, epochs)`: Huấn luyện mô hình với dữ liệu đầu vào
- `predict(X)`: Thực hiện dự đoán với dữ liệu đầu vào
- `evaluate(X, y)`: Đánh giá hiệu suất mô hình
- `plot_loss()`: Vẽ đồ thị hàm mất mát theo thời gian

## Ví dụ

import numpy as np
from fibonacci_net import FibonacciNet

# Tạo dữ liệu mẫu
X = np.random.rand(100, 2)
y = np.sum(X, axis=1).reshape(-1, 1)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Khởi tạo và huấn luyện mô hình
model = FibonacciNet(input_size=2, output_size=1, num_layers=4)
model.train(X_train, y_train, learning_rate=0.01, epochs=1000)

# Đánh giá mô hình
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")

# Vẽ đồ thị hàm mất mát
model.plot_loss()

## Yêu cầu

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib

## Đóng góp

Đóng góp cho dự án luôn được chào đón. Vui lòng tạo một issue hoặc pull request nếu bạn muốn cải thiện Fibonacci-Net.

## Giấy phép

Dự án này được cấp phép theo [MIT License](LICENSE).
