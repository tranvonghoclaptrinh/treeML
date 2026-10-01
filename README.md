# treeML — Decision Tree Regression Demo

Demo độc lập phục vụ phần thực nghiệm **cây quyết định trong bài toán dự đoán giá nhà** của Nhóm 9.

## Mục tiêu

- Chạy một pipeline `train/test` tái lập được trên bộ House Prices.
- Dùng một tập đặc trưng số nhỏ, dễ giải thích trên lớp.
- Hiển thị đúng kết quả của lần train: metrics train/test, mẫu dự đoán, cây đã học và biểu đồ actual-vs-predicted.
- Cho phép nhập đặc trưng trên giao diện tĩnh và duyệt đúng các node của cây đã train.

## Chạy lại thực nghiệm

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train_tree.py
```

Script đọc `data/train.csv`, chia Train/Test với `random_state=42`, điền missing bằng median của tập đặc trưng, huấn luyện `DecisionTreeRegressor(max_depth=4, min_samples_leaf=8)`, sau đó sinh:

- `artifacts/metrics.json`: thông số chia dữ liệu và MAE/RMSE/R² của Train/Test.
- `artifacts/test_predictions.csv`: dự đoán trên tập test.
- `artifacts/tree.json`: cấu trúc đúng của cây để giao diện duyệt node.
- `artifacts/decision_tree.png`: hình cây thực tế sau khi train.
- `artifacts/actual_vs_predicted.png`: biểu đồ test thực tế so với dự đoán.
- `artifacts/tree_rules.txt`: luật cây dạng văn bản.

## Chạy giao diện cục bộ

```bash
python -m http.server 8000 -d web
```

Mở `http://localhost:8000`. Giao diện đọc dữ liệu từ thư mục `web/artifacts`; khi chạy train, hãy copy artifacts:

```bash
cp artifacts/metrics.json artifacts/tree.json artifacts/defaults.json artifacts/test_predictions.csv web/artifacts/
cp artifacts/decision_tree.png artifacts/actual_vs_predicted.png web/artifacts/
```

## GitHub Pages

Repo có workflow tại `.github/workflows/pages.yml`. Sau khi push, vào **Settings → Pages**, chọn **GitHub Actions** nếu GitHub chưa tự chọn. Trang sẽ đọc các artifacts tĩnh đã commit trong `web/artifacts`.

> Đây là demo học thuật. Giá dự đoán không thay thế thẩm định giá bất động sản chuyên nghiệp.
