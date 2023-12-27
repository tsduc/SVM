import csv
import random
import math

# Đọc dữ liệu từ tệp CSV
def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

# Loại bỏ cột không cần thiết và dòng có giá trị NaN
def drop_columns_and_nan(data, columns_to_drop):
    # Lấy header
    header = data[0]
    
    # Tìm indices của các cột cần giữ
    indices_to_keep = [i for i in range(len(header)) if header[i] not in columns_to_drop]
    
    # Loại bỏ cột không cần thiết và giữ nguyên giá trị nếu không thể chuyển đổi sang số
    data = [[row[i] if i not in indices_to_keep or not str(row[i]).replace(".", "").replace("-", "").replace(" ", "").isdigit() else float(row[i]) for i in range(len(row))] for row in data[1:]]
    
    # Loại bỏ dòng có giá trị NaN
    data = [row for row in data if not any(math.isnan(float(cell)) if isinstance(cell, (int, float)) else False for cell in row)]
    
    # Thêm header vào lại data
    data = [header] + data
    
    return data

# Chuyển các biến thành biến nhãn loại (categorical)
def convert_to_categorical(data, binary_columns):
    # Lấy header
    header = data[0]

    # Tìm indices của các cột cần chuyển
    indices_to_convert = [header.index(col) for col in binary_columns]

    # Chuyển các biến thành biến nhãn loại
    for row in data[1:]:
        for i in indices_to_convert:
            row[i] = 'Yes' if str(row[i]) == '1' else 'No'

    return data

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
def train_test_split(data, test_size, random_state):
    random.seed(random_state)
    # Lấy header
    header = data[0]
    
    # Tính toán số lượng dòng cho tập kiểm tra
    test_size = int(len(data) * test_size)
    
    random.shuffle(data[1:])

    # Chia thành tập huấn luyện và tập kiểm tra
    train_data = [header] + data[test_size + 1:]
    test_data = [header] + data[1:test_size + 1]

    return train_data, test_data

# Chuẩn hóa dữ liệu sử dụng Min-Max Scaling
def min_max_scaling(data):
    # Lấy header
    header = data[0]

    # Lấy indices của các cột cần chuẩn hóa
    indices_to_scale = [i for i in range(len(header)) if header[i] not in ['Dt_Customer', 'Response']]

    for i in indices_to_scale:
        # Bỏ qua giá trị không phải số
        column_values = [float(row[i]) for row in data[1:] if str(row[i]).replace(".", "").replace("-", "").isdigit()]

        # Kiểm tra nếu column_values không rỗng
        if column_values:
            min_val = min(column_values)
            max_val = max(column_values)

            for row in data[1:]:
                # Bỏ qua giá trị không phải số
                if not str(row[i]).replace(".", "").replace("-", "").isdigit():
                    continue

                row[i] = (float(row[i]) - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0

    return data

# Tạo mô hình Logistic Regression với thay đổi solver và tăng max_iter
def logistic_regression(X_train, y_train, solver, max_iter, random_state):
    random.seed(random_state)
    # Khởi tạo trọng số
    weights = [random.uniform(0, 1) for _ in range(len(X_train[0]))]

    # Học trọng số thông qua gradient descent
    learning_rate = 0.01
    for epoch in range(max_iter):
        for i in range(len(X_train)):
            # Chỉ tính toán và cập nhật trọng số nếu giá trị dữ liệu là số
            if all(isinstance(x, (int, float)) for x in X_train[i]):
                prediction = predict(X_train[i], weights)
                error = float(y_train[i]) - prediction
                for j in range(len(weights)):
                    weights[j] = weights[j] + learning_rate * error * float(X_train[i][j])

    return weights

# Dự đoán trên tập kiểm tra
def predict(X_test, weights):
    # Chỉ chọn các giá trị số để tính toán
    numeric_values = [float(val) for val in X_test if isinstance(val, (int, float))]
    
    z = sum([numeric_values[i] * weights[i] for i in range(len(numeric_values))])
    probability = 1 / (1 + math.exp(-z))
    return 'Yes' if probability >= 0.5 else 'No'

# Đánh giá mô hình Logistic Regression
def evaluate(X_test, y_test, weights):
    predictions = [predict(row, weights) for row in X_test]

    # Tính các độ đo đánh giá
    accuracy = sum([1 for i in range(len(predictions)) if predictions[i] == y_test[i]]) / len(predictions)
    true_positive = sum([1 for i in range(len(predictions)) if predictions[i] == 'Yes' and y_test[i] == 'Yes'])
    false_positive = sum([1 for i in range(len(predictions)) if predictions[i] == 'Yes' and y_test[i] == 'No'])
    false_negative = sum([1 for i in range(len(predictions)) if predictions[i] == 'No' and y_test[i] == 'Yes'])
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print("Logistic Regression Metrics:")
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1_score}')

    # Hiển thị ma trận nhầm lẫn
    confusion_matrix_logistic = [[true_positive, false_negative], [false_positive, len(predictions) - true_positive - false_positive - false_negative]]
    print("\nConfusion Matrix:")
    print(confusion_matrix_logistic)

    # In báo cáo đánh giá
    print("\nClassification Report:")
    print(f'True Positive: {true_positive}')
    print(f'False Positive: {false_positive}')
    print(f'False Negative: {false_negative}')

# Đọc dữ liệu từ tệp CSV
cc = read_csv("./Data.csv")

# Loại bỏ cột không cần thiết và dòng có giá trị NaN
columns_to_drop = ["ID", "Z_CostContact", "Z_Revenue"]
cc = drop_columns_and_nan(cc, columns_to_drop)

# Chuyển các biến thành biến nhãn loại (categorical)
binary_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain', 'Response']
cc = convert_to_categorical(cc, binary_columns)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data, test_data = train_test_split(cc, test_size=0.3, random_state=0)

# Chuẩn hóa dữ liệu sử dụng Min-Max Scaling
train_data = min_max_scaling(train_data)
test_data = min_max_scaling(test_data)

# Tạo mô hình Logistic Regression với thay đổi solver và tăng max_iter
X_train = [row[:-1] for row in train_data[1:]]
y_train = [row[-1] for row in train_data[1:]]
weights = logistic_regression(X_train, y_train, solver='liblinear', max_iter=1000, random_state=0)

# Đánh giá mô hình
X_test = [row[:-1] for row in test_data[1:]]
y_test = [row[-1] for row in test_data[1:]]
evaluate(X_test, y_test, weights)
