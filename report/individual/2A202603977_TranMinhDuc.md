# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603977
- **Họ và tên:** Trần Minh Đức
- **SEED:** 3881

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Cách làm | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.896 | 0.919 | 0.087 | 0 |
| 2. BoW + MLP | 0.951 | 0.974 | 0.693 | 0 |
| 3a. MiniLM + kNN | 0.871 | 1.000 | 101.582 | 0 |
| 3b. MiniLM + LogReg | 0.949 | 1.000 | 40.807 | 0 |
| 3c. MiniLM + MLP head | 0.971 | 1.000 | 35.194 | 0 |
| 4a. LLM nhỏ zero-shot | 0.884 | 0.884 | 2863.090 | 0  |
| 4b. LLM lớn zero-shot | 0.884 | 0.884 | 2494.669 | 0  |
| 4c. LLM nhỏ few-shot | 0.884 | 0.884 | 2485.820 | 0  |

† LLM em chỉ chạy trên SUB (40 tin) nên F1 ở cột test chính là F1 SUB.
‡ Notebook in $0 vì em chạy qua Groq, không có giá trong bảng. Theo phần tính tay (gpt-6-luna) thì khoảng $0.0105 / 1k tin.

Theo bảng, chỉ có cách 1, 2, 3b, 3c đạt p95 < 50 ms. SUB chỉ có 40 tin nên em coi chênh lệch dưới ~0.05 là hoà.

## 3. Dự đoán và thực tế

1. **`ml_lr`:** em đoán lr = 50 sẽ không hội tụ, nhưng loss vẫn giảm rất nhanh xuống khoảng 0.005. Sau khi tìm hiểu thì em hiểu là gradient của cross-entropy (p − y) luôn nằm trong [−1, 1] và sigmoid bão hoà nên bước nhảy không làm loss nổ. Tuy vậy loss train thấp chưa chắc mô hình tốt, cần xem thêm trên val.
2. **`llm_models`:** em đoán LLM sẽ phân loại tốt nhất, nhưng F1 SUB chỉ 0.884, thấp hơn các cách dùng encoder (1.000), và model 120B sai y hệt model 20B. Các tin sai là ham nhưng đọc lên giống gạ gẫm (ví dụ "send me some pix", "I will take care of financial problem"). Em nghĩ lỗi nằm ở ranh giới nhãn của dataset chứ không phải model thiếu năng lực.
3. **`llm_fewshot`:** em đoán thêm 4 ví dụ thì F1 tăng, nhưng vẫn 0.884. Em nghĩ 4 ví dụ đó quá dễ nên không giúp gì cho các tin biên đang sai, và SUB chỉ 40 tin nên 1 tin đã là 2.5 điểm.

Ngoài ra, MLP hơn LogReg khá rõ (F1 0.896 → 0.951, recall 0.812 → 0.918), rõ hơn em đã dự đoán. Em bỏ ReLU thì F1 vẫn 0.951, nên em đoán phần cải thiện đến từ cách huấn luyện (Adam, mini-batch) nhiều hơn là từ tính phi tuyến.

## 4. Phân tích 50 case khó

Model tụt nhiều nhất là BoW + LogReg: F1 từ 0.919 xuống 0.333 (BoW + MLP từ 0.974 xuống 0.400, encoder xuống 0.61–0.68, còn LLM gần như không tụt: 0.884 → 0.885). Nhóm BoW làm tệ nhất là spam giọng tự nhiên (đúng 10%), thao túng (12%) và né từ khoá (17% / 0%). BoW đúng 100% ở nhóm ham, nhưng em thấy đó là do nó chủ yếu đoán ham chứ không phải hiểu tốt. Encoder yếu nhất ở spam giọng tự nhiên (20–30%), còn LLM yếu nhất ở nhóm ham trông giống spam (60%).

Hai case cụ thể:
- **#34** "W.I.N a brand new iPh0ne 15 - just rep|y Y to this message": tokenizer của em tách thành `w`, `i`, `n`, `iph0ne`, `rep`, `y`, nên các từ `win`, `iphone`, `reply` không bao giờ xuất hiện trong vector. Thêm nữa `i` lại là từ kéo mạnh về ham, nên BoW bị lừa.
- **#13** "Hope you're well! Quick reminder that your photos from the event are ready… snap-prints.co/r/88": cả 7 model đều sai. Tin không có từ khoá spam, giọng thân thiện, dấu hiệu duy nhất là link bán hàng không ai yêu cầu.

Em cũng thấy prompt phòng thủ giúp nhóm prompt injection (83% → 100%) nhưng làm nhóm "nói về spam" tụt từ 100% xuống 75%, vì model trở nên nghi ngờ quá mức.

## 5. Thử thách cá nhân

**Đề #6:** sửa prompt để LLM trả JSON `{"label", "reason"}`, viết hàm parse, chạy lại trên 10 tin SUB rồi so token và chi phí với bản 1 từ.

**Cách làm:** em viết lại `build_classification_prompt` để chỉ trả về JSON, viết `parse_label_json` bằng `json.loads` (lỗi thì trả `None`), rồi chạy trên 10 tin đầu của SUB.

**Kết quả:** input khoảng 849 token cho cả hai bản, output của bản JSON khoảng 132 token (~13 token/tin), còn bản 1 từ khoảng 1 token/tin. Chi phí in ra là $0 do chạy Groq. Em cần nói thẳng là phép đo này có hạn chế: vì em ghi đè `build_classification_prompt` nên "bản 1 từ" thực ra cũng dùng prompt JSON, do đó input bằng nhau là do cách đo, và token chỉ là ước lượng.

**Giải thích:** giá output đắt gấp 5 lần input, nên nếu input như nhau thì bản JSON đắt hơn khoảng 68%. Với phân loại tự động quy mô lớn em thấy không đáng. JSON hợp để phân tích lỗi hoặc gán nhãn, vì trường `reason` cho biết model nghĩ gì. Em cũng nhận ra code của em đặt `label = 0` khi parse lỗi, nghĩa là lỗi định dạng sẽ làm spam lọt qua mà không ai biết; lẽ ra nên đếm riêng các phản hồi không hợp lệ.

## 6. Phương án cho sản phẩm thật

Em chọn **mô hình lai**, không đặt LLM trong đường xử lý thời gian thực:
- **BoW + MLP** cho mọi tin: p95 0.693 ms, F1 0.951, khoảng 28.000 tin/s.
- **MiniLM + MLP head** cho tin lưỡng lự (p trong [0.3, 0.7]): p95 35.2 ms, F1 0.971, nhưng chỉ ~46 tin/s mỗi bản sao nên chỉ nên nhận một phần nhỏ lưu lượng.
- **LLM chạy bất đồng bộ** cho tin khó và để gán nhãn retrain: p95 khoảng 2,5–2,9 giây, vượt SLA 50 ms khoảng 50 lần nên không thể đứng trước người dùng.

Về chi phí, nếu cho LLM xử lý toàn bộ 10 triệu tin thì khoảng $105/ngày (≈ $3.150/tháng) theo phép tính tay; nếu chỉ ~2% tin đi qua LLM (giả định của em) thì khoảng $63/tháng.

Cái em chấp nhận đánh đổi: vẫn bỏ sót spam giọng tự nhiên và thao túng ở đường thời gian thực (bù bằng LLM bất đồng bộ và retrain); tốn thêm hạ tầng (≥ 3 bản sao cho tầng encoder); và số liệu của em còn mỏng (SUB 40 tin, 50 case viết tay). Ở quy mô 10 triệu tin/ngày, FPR chỉ ~0,15% (em ước tính thô) vẫn có thể chặn nhầm cỡ chục nghìn tin hợp lệ mỗi ngày, nên cần danh sách trắng cho tin OTP.

## 7. AI log và điều đã học

**Điều em hiểu ra hôm nay:** F1 cao trên tập test thường chưa nói lên mô hình bền, vì BoW tụt từ 0.919 xuống 0.333 khi gặp case khó. Model to hơn hay thêm few-shot cũng không sửa được lỗi nằm ở ranh giới nhãn. Em dự đoán sai nhiều nhất ở learning rate: lr = 50 vẫn hội tụ vì gradient bị chặn, nhưng loss thấp chưa chắc là mô hình tốt. Với sản phẩm thật, độ trễ và chi phí quan trọng ngang F1, nên cách hợp lý là dùng mô hình rẻ cho phần lớn tin và để LLM lo phần khó.
