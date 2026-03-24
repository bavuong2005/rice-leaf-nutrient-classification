# Rice Leaf Nutrient Deficiency Classification

  

## 1\. Tiêu đề và Mô tả ngắn (Title & Overview)

Hệ thống phân loại thiếu hụt dinh dưỡng (Nitơ - N, Photpho - P, Kali - K) trên lá lúa sử dụng Deep Learning. Dự án tập trung đánh giá các mô hình CNN tiền huấn luyện (MobileNetV3, EfficientNet) và tối ưu hóa kiến trúc hạng nhẹ tự thiết kế (MiniXception + ECA Attention) với mục tiêu triển khai trên thiết bị biên (Edge Devices) hỗ trợ chẩn đoán nông nghiệp nhanh chóng.

## 2\. Demo / Hình ảnh minh họa (Visuals)

  * **Hình ảnh dự đoán:**
  * **Kiến trúc MiniXception + ECA đề xuất:**

## 3\. Tập dữ liệu (Dataset)

Bộ dữ liệu được sử dụng là **"Nutrient-Deficiency-Symptoms-in-Rice"** thu thập từ nền tảng Kaggle.

  * **Số lượng:** 1,156 ảnh màu RGB chụp cận cảnh lá lúa.
  * **Phân bổ nhãn (3 classes):**
      * **Nitrogen (N - Thiếu Đạm):** 440 ảnh (\~38.1%) - biểu hiện vàng lá.
      * **Phosphorus (P - Thiếu Lân):** 333 ảnh (\~28.8%) - biểu hiện màu tím đỏ ở gốc lá.
      * **Potassium (K - Thiếu Kali):** 383 ảnh (\~33.1%) - biểu hiện đốm vàng nâu/cháy viền lá.
  * **Tải dữ liệu:** [Link Kaggle Dataset](https://www.google.com/search?q=https://www.kaggle.com/datasets/weeraphatraksarikon/nutrient-deficiency-symptoms-in-rice)

## 4\. Phương pháp & Công nghệ (Methodology & Tech Stack)

  * **Công cụ:** `Python`, `PyTorch`.
  * **Tiền xử lý & Augmentation:** Ảnh đầu vào không đồng nhất được chuẩn hóa về kích thước `1024x256` (tỷ lệ 4:1 phù hợp với hình dáng lá lúa), áp dụng Normalization theo ImageNet, và tăng cường dữ liệu (Random Horizontal Flip, Rotation ±20°, Affine) để giảm overfitting.
  * **Kiến trúc mô hình:**
      * **Transfer Learning:** Sử dụng MobileNetV3-Large, EfficientNet-B0 và Xception gốc để tận dụng khả năng trích xuất đặc trưng mạnh mẽ.
      * **Custom Models:** Đề xuất kiến trúc **MiniXception** sử dụng Depthwise Separable Convolutions giúp giảm thiểu tối đa tham số. Tích hợp thêm module **ECA (Efficient Channel Attention)** để mô hình tự động gán trọng số cao cho các vùng biểu hiện bệnh (vàng lá, đốm nâu) mà chỉ tiêu tốn thêm \~2K tham số.

## 5\. Kết quả thực nghiệm (Results & Benchmarking)

Kết quả đánh giá trên tập dữ liệu sử dụng kỹ thuật Stratified 5-Fold Cross-Validation:

| Mô hình | Số tham số (M) | Accuracy Trung Bình (%) | Thời gian Train 1 Epoch |
| :--- | :---: | :---: | :---: |
| **MobileNetV3-Large** | 4.21 | **96.21** | **17.53s** |
| EfficientNet-B0 | 4.01 | 95.69 | 17.87s |
| MiniXception (Custom) | 1.24 | 95.34 | 20.40s |
| Xception (Gốc) | 20.81 | 95.00 | 19.84s |
| MiniXception + ECA (Ours) | **1.24** | 94.83 | 21.34s |

> **Nhận xét:**
> \* **MobileNetV3-Large** là mô hình đạt độ chính xác cao nhất (96.21%) và có thời gian huấn luyện tối ưu nhất.
> \* Tuy nhiên, kiến trúc **MiniXception / MiniXception + ECA** tự thiết kế thể hiện tiềm năng cực kỳ lớn khi giữ được độ chính xác rất cạnh tranh (\~95%) trong khi kích thước mô hình (1.24M tham số) **nhỏ hơn gần 4 lần** so với MobileNetV3, vô cùng lý tưởng để tích hợp vào các thiết bị di động có cấu hình yếu.
> \* **Hạn chế chung:** Các mô hình đôi khi nhầm lẫn giữa nhãn Thiếu Photpho (P) và Thiếu Kali (K) do sự lấn át của các vết hoại tử (cháy mép lá) khi bệnh tiến triển nặng.

## 6\. Hướng dẫn cài đặt và sử dụng (Installation & Usage)

Dự án được tổ chức dưới dạng các file Jupyter Notebook (`.ipynb`) độc lập cho từng mô hình để tiện theo dõi quá trình huấn luyện và đánh giá.

**Bước 1: Clone repository**

```bash
git clone https://github.com/your-username/rice-leaf-nutrient-classification.git
cd rice-leaf-nutrient-classification
```

**Bước 2: Cài đặt thư viện**

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

**Bước 3: Chạy code**
Khởi động Jupyter Notebook hoặc mở các file `.ipynb` trên Kaggle/Google Colab để chạy tuần tự các cell:

```bash
jupyter notebook
```

## 7\. Cấu trúc thư mục (Repository Structure)

```text
rice-leaf-nutrient-classification/
│
├── 01_MiniXception.ipynb                # Huấn luyện và đánh giá mô hình MiniXception Custom
├── 02_MiniXception_ECA_Attention.ipynb  # Huấn luyện và đánh giá mô hình MiniXception tích hợp ECA
├── 03_Xception_Original.ipynb           # Transfer Learning với Xception gốc
├── 04_MobileNetV3_Large.ipynb           # Transfer Learning với MobileNetV3-Large
├── 05_EfficientNet_B0.ipynb             # Transfer Learning với EfficientNet-B0
│
├── README.md                            # Tài liệu mô tả dự án
└── requirements.txt                     # Danh sách thư viện phụ thuộc
```

## 8\. Hướng phát triển (Future Work)

  * Triển khai mô hình tốt nhất (MobileNetV3 hoặc Mini-Xception) lên ứng dụng di động để hỗ trợ nông dân chẩn đoán nhanh trực tiếp tại cánh đồng.
  * Mở rộng bài toán sang Object Detection (nhận diện vật thể) để khoanh vùng vị trí chính xác của các vết thương tổn trên lá do thiếu hụt dinh dưỡng.
  * Thu thập thêm dữ liệu hình ảnh đa dạng hơn trong môi trường thực tế để cải thiện khả năng phân biệt các triệu chứng gây nhầm lẫn (đặc biệt là giữa P và K).

-----

Bạn có muốn tôi hướng dẫn cách tạo nhanh file `requirements.txt` dựa trên môi trường Kaggle để commit luôn lên repo cùng với 5 file `.ipynb` này không?
