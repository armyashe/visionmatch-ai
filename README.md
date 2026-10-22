
# VisionMatch AI v0.1

Hệ thống tìm kiếm ảnh sản phẩm thời trang tương tự (Image Retrieval) sử dụng **PyTorch** và **CLIP (ViT-B/32)**.

## 📂 Cấu trúc thư mục dự án

Dự án được tổ chức theo chuẩn mô-đun, tách biệt rõ ràng giữa phần xử lý mô hình (ML pipeline), script tự động hóa, giao diện và dữ liệu:

```text
visionmatch-ai/
├── app.py                    # Giao diện Demo trực tuyến sử dụng Gradio (cho phép tải ảnh lên để tìm kiếm)
├── requirements.txt          # Danh sách các thư viện Python cần thiết (torch, transformers, gradio, numpy, pytest, v.v.)
├── README.md                 # Tài liệu hướng dẫn cài đặt, chạy dự án và giải thích cấu trúc
├── .gitignore                # Danh sách các file/thư mục bị bỏ qua bởi Git (như môi trường ảo, weights nặng, dữ liệu thô)
│
├── data/
│   ├── raw/                  # Thư mục chứa ảnh gốc (Footwear dataset)
│   └── processed/
│       └── catalog.csv       # File CSV chứa metadata (đường dẫn ảnh, nhãn, ID sản phẩm)
│
├── src/
│   └── visionmatch/
│       ├── __init__.py       # Đánh dấu thư mục là một Python package
│       ├── encoder.py        # Module chứa logic tải mô hình CLIP và trích xuất vector đặc trưng (embedding)
│       └── retrieval.py      # Module tính toán độ tương đồng Cosine Similarity và tìm kiếm Top-K kết quả
│
├── scripts/
│   ├── prepare_dataset.py    # Script kiểm tra tính hợp lệ của ảnh và tự động tạo file catalog.csv
│   ├── build_index.py        # Script chạy inference toàn bộ dataset để sinh ra ma trận embedding lưu vào artifacts/
│   └── search.py             # Script chạy kiểm thử tìm kiếm nhanh qua giao diện dòng lệnh (CLI)
│
├── artifacts/
│   └── clip_embeddings.npy   # File ma trận chứa các embedding vector của toàn bộ catalog (được sinh tự động)
│
└── tests/
    └── test_retrieval.py     # Các bài kiểm thử đơn vị (Unit tests) cho thuật toán retrieval
```
