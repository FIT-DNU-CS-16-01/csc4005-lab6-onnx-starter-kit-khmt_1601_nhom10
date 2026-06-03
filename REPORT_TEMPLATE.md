# CSC4005 Lab 6 Report – Export ONNX + Consistency Test + Benchmark

## 1. Thông tin

- Họ tên: Lê Thị Ngọc Bích, Nguyễn Mạnh Duy, Lê Trọng Thanh Tùng
- Lớp: KHMT 16-01
- Link GitHub repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab6-onnx-starter-kit-khmt_1601_nhom10
- Link checkpoint hoặc mô tả checkpoint sử dụng: checkpoints/best_model.pt
- Link file ONNX nếu không commit trực tiếp:

## 2. Mô tả mô hình đầu vào

| Nội dung | Giá trị |
|---|---|
| Bài toán | Smart Campus Scene Classification |
| Dataset | MIT Indoor Scenes 67 subset |
| Số lớp | 5 |
| Classes | classroom, computerroom, library, corridor, office |
| Model PyTorch | Vision Transformer (vit_b_16) |
| Checkpoint | best_model.pt |
| Image size | 224x224 |
| Train mode từ lab trước | head_only |

## 3. Export ONNX

Điền thông tin:

| Thông số | Giá trị |
|---|---|
| ONNX path | outputs/vit_smartcampus.onnx |
| Opset | 17 |
| Dynamic batch | yes |
| Input name | input |
| Output name | logits |
| Model size | 106 KB + 335MB |

Lệnh đã chạy:

```bash
python -m src.export_onnx   --checkpoint checkpoints/best_model.pt   --onnx_path outputs/vit_smartcampus.onnx   --model_name vit_b_16   --img_size 224   --opset 17   --dynamic_batch                                        
```

## 4. Consistency Test

| Metric | Giá trị |
|---|---:|
| passed | true |
| num_samples | 32 |
| batch_size | 8 |
| max_abs_diff | 1.0251998901367188e-05 |
| mean_abs_diff | 2.2510299118039256e-06 |
| pred_match_rate | 1.0 |
| atol | 0.0001 |
| rtol | 0.001 |

Nhận xét:

- PyTorch và ONNX có nhất quán không?

Có nhất quán về mặt số học. Thử nghiệm đã vượt qua thành công (passed: true) trên cả 32 mẫu dữ liệu với cấu hình batch size bằng 8.

- Nếu có sai khác, sai khác lớn hay nhỏ?

Rất nhỏ. Độ sai lệch tuyệt đối lớn nhất (max_abs_diff) và sai lệch trung bình (mean_abs_diff) ở mức cực thấp. Cả hai chỉ số này đều nằm trong ngưỡng sai số cho phép rất nghiêm ngặt là atol và rtol .

- Sai khác này có làm thay đổi nhãn dự đoán không?

Không. Chỉ số pred_match_rate bằng 1.0 (tương đương 100%) khẳng định rằng bất chấp những sai số rất nhỏ do kiến trúc phần cứng/thư viện runtime thay đổi, cả hai mô hình PyTorch và ONNXRuntime đều đưa ra kết quả dự đoán nhãn giống nhau hoàn toàn trên toàn bộ các mẫu thử.

## 5. Benchmark

| Runtime | Batch size | Mean latency (ms) | Median latency (ms) | P95 latency (ms) | Throughput (img/s) | Model size (MB) |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch | 1 | 622.9843 | 674.7216 | 702.2115 | 1.6052 | 327.37 |
| ONNXRuntime | 1 | 419.6702 | 446.7269 | 633.6135 | 2.3828 | 0.1 |
| PyTorch | 4 | 2064.8917 | 2444.67 | 2579.6728 | 1.9371 | 327.37 |
| ONNXRuntime | 4 | 976.3739 | 864.9435 | 2271.5292 | 4.0968 | 0.1 |
| PyTorch | 8 | 1912.5603 | 1932.4885 | 1975.1519 | 4.1829 | 327.37 |
| ONNXRuntime | 8 | 1683.0986 | 1679.6079 | 1773.3729 | 4.7531 | 0.1 |

## 6. Phân tích kết quả

Trả lời:

1. ONNXRuntime có nhanh hơn PyTorch không?

Có, nhanh hơn đáng kể. Ở mọi cấu hình batch size, ONNXRuntime luôn có latency thấp hơn và throughput cao hơn PyTorch. Đặc biệt ở batch size 4, ONNXRuntime nhanh hơn PyTorch tới hơn 2 lần.

2. Batch size ảnh hưởng thế nào đến latency và throughput?

- Latency (Độ trễ): Tăng tỷ lệ thuận với batch size (batch size càng lớn, thời gian xử lý một lượt càng lâu).

- Throughput (Băng thông): Tăng mạnh khi tăng batch size nhờ khả năng tối ưu hóa tính toán song song phần cứng (ví dụ: throughput của ONNXRuntime tăng từ 2.38 lên 4.75 ảnh/giây khi tăng batch từ 1 lên 8).

3. Vì sao cần warm-up trước khi đo benchmark?

Để loại bỏ thời gian khởi tạo ban đầu (như cấp phát bộ nhớ, nạp thư viện, tối ưu graph lần đầu của ONNX). Quá trình này giúp phản ánh chính xác tốc độ xử lý thực tế của mô hình khi đã đi vào trạng thái ổn định.

4. Vì sao không nên chỉ đo một lần rồi kết luận?

Vì một lần chạy đơn lẻ rất dễ bị nhiễu bởi hệ điều hành (CPU bận đột xuất, xung đột luồng, nghẽn I/O). Việc đo lặp lại nhiều lần (như 50 lần trong bài) giúp lấy được giá trị trung bình (Mean) và các điểm phân vị (P95) nhằm đảm bảo tính khách quan và độ tin cậy của số liệu.

5. Nếu triển khai lên CPU/edge device, bạn chọn batch size nào? Vì sao?

Chọn Batch size = 1. Vì: CPU và các thiết bị edge thường có tài nguyên phần cứng (RAM, số nhân xử lý) rất hạn chế, không mạnh về tính toán song song quy mô lớn như GPU. Chạy batch size 1 giúp đảm bảo độ trễ (Latency) thấp nhất và phản hồi tức thì, điều vốn là ưu tiên hàng đầu của các ứng dụng edge thực tế (như camera giám sát, nhận diện thời gian thực).

## 7. Liên hệ triển khai thực tế

Viết 5–8 dòng:

- ONNX giúp gì trong triển khai mô hình?

Chuẩn hóa mô hình từ mọi framework (PyTorch, TensorFlow) thành một định dạng chung, giúp tăng tốc độ xử lý phần cứng đáng kể qua ONNXRuntime và dễ dàng triển khai đa nền tảng (Cloud, CPU, Edge).

- Consistency test giúp phát hiện lỗi gì?

Phát hiện sự sai lệch về mặt số học (sai số đầu ra) và lỗi lệch nhãn dự đoán (pred_match_rate thấp) giữa mô hình gốc và mô hình sau khi chuyển đổi sang ONNX.

- Benchmark giúp ra quyết định kỹ thuật như thế nào?

Cung cấp số liệu trực quan về độ trễ (Latency), băng thông (Throughput) và dung lượng bộ nhớ để kỹ sư lựa chọn phần cứng phù hợp và cấu hình batch_size tối ưu nhất cho bài toán thực tế.

- Nếu cần đưa mô hình vào hệ thống Smart Campus thật, còn cần kiểm thử thêm điều gì?

Kiểm thử khả năng chịu tải (Stress test) khi hàng trăm camera gửi dữ liệu đồng thời, kiểm thử độ ổn định khi chạy liên tục 24/7 (Long-run test) và kiểm thử độ an toàn bảo mật dữ liệu đầu vào.

## 8. Kết luận

Tóm tắt:

- Export ONNX thành công hay chưa?

Đã thành công. Mô hình vit_b_16 được chuyển đổi hoàn tất sang định dạng ONNX (Opset 17) và đã kích hoạt tính năng dynamic_batch giúp linh hoạt thay đổi kích thước đầu vào.

- Consistency test có pass không?

Có pass. Kết quả kiểm thử đạt trạng thái passed: true trên 32 mẫu thử với độ sai số cực kỳ nhỏ, đảm bảo nhãn dự đoán khớp nhau tuyệt đối (100%) so với PyTorch gốc.

- Runtime nào nhanh hơn?

ONNXRuntime nhanh hơn vượt trội. Trên mọi cấu hình batch size, ONNXRuntime đều tối ưu hơn PyTorch về cả độ trễ (Latency) lẫn băng thông (Throughput), đặc biệt đạt tốc độ xử lý nhanh gấp hơn 2 lần ở cấu hình batch size 4.

- Bài học chính rút ra từ lab này là gì?

Khi xuất mô hình phức tạp như Vision Transformer, bắt buộc phải cấu hình dynamic_axes và sử dụng dummy_input phù hợp để tránh lỗi định hình lại shape cố định. Đồng thời, việc tối ưu hóa qua ONNX là bước đi cốt lõi để giảm chi phí phần cứng và tăng tốc độ phản hồi khi đưa hệ thống Smart Campus vào vận hành thực tế.
