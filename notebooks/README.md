# Quy ước chạy notebook

Chạy lần lượt notebook `01` đến `05` từ thư mục gốc của kho mã nguồn trong môi trường Clustomer. Trong giai đoạn khám phá, notebook chủ động không phụ thuộc vào `src/`; code triển khai là phần logic đã khóa được trích xuất sau đó.

```powershell
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks\01_EDA.ipynb --inplace
```

Lặp lại lệnh cho từng notebook theo thứ tự số. Một bản hợp lệ để đánh giá phải có số thứ tự thực thi ở mọi ô code, không có kết quả lỗi, lưu đầy đủ biểu đồ/bảng và có kết luận nhất quán với kết quả ngay phía trên. Chỉ notebook 05 được phép tổng hợp tập kiểm định cuối đã khóa.

Các dữ liệu trung gian và artifact mô hình được sinh dưới `outputs/` và không đưa vào Git. Các báo cáo nhỏ phục vụ truy xuất số liệu được lưu trong `outputs/reports/`. Có thể tái tạo artifact mô hình cuối bằng `scripts/train.py`.
