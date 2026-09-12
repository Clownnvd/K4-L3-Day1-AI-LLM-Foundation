# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Học viên:** NGUYỄN VĂN DUY — `2A202602729`
**Code nộp:** [`solution.py`](solution.py)

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Tôi đã chạy cùng prompt trên `gpt-4o`: temperature 0.0 trả danh sách tổng quát, 0.5 nhắc di sản/ẩm thực, 1.0 nói nhiều về lịch sử, còn 1.5 chuyển sang địa lý. Các chủ đề và cách diễn đạt khác nhau, nhưng từ một lượt mỗi mức chưa thể kết luận độ dài hay độ đúng tăng/giảm đều theo temperature. Temperature cao cho phép lấy mẫu đa dạng hơn; sự thật cụ thể vẫn phải được kiểm tra riêng.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi chọn khoảng `0.2` để câu trả lời hỗ trợ khách hàng ổn định, bám chính sách và ít biến thể không cần thiết. Temperature thấp không bảo đảm câu trả lời đúng, nên vẫn cần nguồn chính sách, kiểm thử câu khó và chuyển người phụ trách khi không chắc.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload có `10.000 × 3 × 350 = 10.500.000` output token/ngày. Dùng đơn giá **minh họa trong repo** (GPT-4o `0.010 USD/1K` output, mini `0.0006 USD/1K`), phần output tương ứng khoảng `105 USD` và `6,30 USD`/ngày, nên GPT-4o đắt hơn khoảng **16,7 lần**; phép tính này chưa gồm input token. Tôi dùng GPT-4o cho phân tích phức tạp cần người kiểm kết quả, còn mini cho FAQ/nghiệp vụ lặp lại đã có nguồn rõ; khi vận hành thật phải tra lại giá hiện hành.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Tôi đã chạy hai system prompt trên cùng câu hỏi blockchain. Prompt “giáo viên tiểu học” tạo cách giải thích bằng ví dụ cuốn sổ/trang ghi chép và ít thuật ngữ; prompt “chuyên gia tài chính” dùng các từ như sổ cái phân tán, mã băm, giao dịch và giải thích kỹ thuật dài hơn. Vậy system prompt đổi đối tượng, từ vựng và mức chi tiết dù user prompt không đổi. Nó định hướng cách diễn đạt chứ không tự bảo đảm độ chính xác của nội dung.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Tôi dùng đoạn tiếng Việt **106 từ** trong [`token_sample.txt`](token_sample.txt). `count_tokens` với `gpt-4o` đếm **126 token**, còn ước lượng `106 / 0,75 = 141,33 token`; kết quả thật thấp hơn ước lượng khoảng **10,85%** nếu lấy 141,33 làm mẫu số. Tiếng Việt có dấu và từ đa âm tiết nên cách tokenizer tách văn bản có thể khác tiếng Anh; không có tỷ lệ cố định cho mọi đoạn, và ngay ví dụ này cho thấy mẹo `số từ / 0,75` chỉ là ước lượng thô.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming hữu ích khi phản hồi dài hoặc model trả chậm: người dùng thấy chữ đầu tiên sớm và biết chương trình vẫn hoạt động, dù tổng thời gian sinh không nhất thiết giảm. Với output ngắn cần xử lý như một khối hoàn chỉnh, chẳng hạn kết quả JSON phải kiểm schema trước khi hiển thị, non-streaming đơn giản và dễ kiểm lỗi hơn. Khi stream, chương trình phải ghép các chunk thành reply cuối và chỉ cập nhật history sau khi hoàn tất.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giãn khoảng thử lại sau mỗi lỗi, giảm áp lực đúng lúc API đang quá tải và cho server thời gian phục hồi. Nếu hàng nghìn client đều chờ cố định một giây, chúng có thể cùng gọi lại và tạo một đợt quá tải mới. Trong hệ thống thật tôi sẽ thêm **jitter** ngẫu nhiên và giới hạn số lần retry để các client không đồng bộ với nhau vô hạn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona trong mini-project là: **“Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.”** “Trợ giảng” giới hạn vai trò vào hỗ trợ học tập, không giả làm người quyết định điểm số; “ngắn gọn bằng tiếng Việt” giúp người học nắm ý chính mà không phải đọc đoạn quá dài. Tôi vẫn cần kiểm nguồn cho câu trả lời mang tính quy định hoặc số liệu.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất hiện tại là trợ lý chỉ giữ **ba lượt gần nhất**, nên có thể quên yêu cầu quan trọng từ đầu cuộc trò chuyện. Tôi sẽ tạo một bản tóm tắt ngắn của các lượt cũ, giữ nó cùng ba lượt gần nhất trong `messages`, rồi cập nhật bản tóm tắt sau mỗi lượt mới. Cần thử bằng các hội thoại dài và kiểm xem tóm tắt có làm sai hoặc lộ thông tin nhạy cảm không.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — **100/100** trên máy; điểm exercises do giảng viên có thể điều chỉnh khi đọc nội dung.
- [x] Cả 4 checkpoint pytest đều pass (**35/35 test**).
- [x] Tất cả 9 câu trong file này đã được trả lời; các lượt gọi model ở Câu 1.1 và 2.1 được AI hỗ trợ chạy trên máy chung.
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang Lab 1 ở VLearn trước **00:01 ngày 13/09/2026** theo thông báo VLearn Support trên Discord ngày 12/09 (mốc trong README cũ đã được cập nhật).
