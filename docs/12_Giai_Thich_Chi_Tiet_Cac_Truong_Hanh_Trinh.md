# Giải Thích Ý Nghĩa Các Trường Dữ Liệu Hành Trình (Journey Field Definitions)

> **Tài liệu hướng dẫn trực quan, dễ hiểu và liền mạch về các chỉ số phân tích hành vi, phân khúc hành trình, phát hiện ma sát UX và dự đoán bước tiếp theo trong hệ thống HiFPT Journey Clustering.**

---

## 1. Bức Tranh Tổng Thể (The Big Picture)

Trong hệ thống phân tích hành vi người dùng HiFPT, một chuỗi log clickstream thô không được xử lý nguyên khối mà được chia nhỏ thành các **Hành trình (Journeys)** độc lập. Mỗi hành trình đại diện cho **một nhiệm vụ hoặc một ý định duy nhất** của người dùng (ví dụ: *Thanh toán cước*, *Đổi mật khẩu Wi-Fi*, *Đăng nhập bằng OTP*, *Gửi yêu cầu hỗ trợ*).

Các trường dữ liệu dưới đây đóng vai trò như các "chỉ số sinh tồn" (Vital Signs) để đo lường:
1. **Hình thái & Quy mô**: Hành trình bắt đầu thế nào, dài bao nhiêu bước, qua bao nhiêu màn hình?
2. **Nhịp điệu & Hành vi**: Người dùng thao tác nhanh hay chậm, bấm nhiều hay xem nhiều, có đi thẳng hay quay lui?
3. **Sức khỏe & Ma sát (Friction)**: Người dùng có bị bế tắc, lạc đường, lặp vòng hay gặp lỗi bất thường không?
4. **Dự báo tương lai**: Bước tiếp theo người dùng chuẩn bị làm là gì?

---

## 2. Nhóm 1: Ranh Giới & Quy Mô Hành Trình (Journey Boundaries & Scale)

```
[Dòng sự kiện liên tục] ──► [Điểm cắt (boundary)] ──► [Làm sạch & Rút gọn] ──► [Hành trình hoàn chỉnh]
                                                      (raw ──► final)
```

---

### 🔹 `boundary` (hoặc `boundary_reason`) — Lý do cắt hành trình

- **Ý nghĩa là gì?**
  Là lý do cụ thể khiến hệ thống quyết định **kết thúc hành trình cũ và mở ra một hành trình mới**. Người dùng không mở app lên để làm một việc duy nhất suốt cả ngày, do đó hệ thống cần các "điểm ngắt" tự nhiên.
- **Các giá trị thường gặp**:
  - `session_start`: Người dùng vừa mới mở app hoặc khởi động phiên truy cập mới.
  - `idle_gap`: Người dùng dừng thao tác (không chạm vào màn hình) quá **90 giây** ($\tau = 90$s). Khi họ chạm lại, đó thường là một việc khác.
  - `root_return`: Người dùng quay trở về màn hình Trang chủ / Hub chính (`HomeVC`, `HOME`) sau khi đã đi qua ít nhất 6 bước. Đây là tín hiệu hoàn tất nhiệm vụ trước đó và chuẩn bị làm việc mới.
  - `auth_change`: Người dùng vừa thực hiện đăng nhập hoặc đăng xuất thành công. Trạng thái người dùng đã thay đổi hoàn toàn.
  - `length_cap`: Hành trình chạm ngưỡng tối đa 80 sự kiện (ngăn chặn các chuỗi quá dài gây nhiễu).
- **Ứng dụng thực tế**: Giúp phân tích nguyên nhân người dùng rời bỏ giữa chừng hay chuyển đổi mục tiêu.

---

### 🔹 `n_events_raw` & `n_events_final` — Độ dài thô vs. Độ dài mô hình hoá

- **Ý nghĩa là gì?**
  - `n_events_raw`: **Tổng số sự kiện log thô** được ghi nhận trong hành trình trước khi xử lý làm sạch.
  - `n_events_final`: **Số sự kiện thực sự còn lại** sau khi đã loại bỏ các thao tác trùng lặp liên tiếp (Spam click), rút gọn các vòng lặp vô nghĩa và lọc bỏ màn hình hệ thống. Đây là độ dài thực tế đưa vào mô hình học máy.
- **Ví dụ so sánh**:
  - Người dùng bấm nút "Thanh toán" 5 lần liên tiếp trong 1 giây do mạng chậm: `n_events_raw = 8`, nhưng sau khi gộp lại `n_events_final = 4`.
- **Ứng dụng thực tế**:
  - Khoảng cách giữa `n_events_raw` và `n_events_final` càng lớn chứng tỏ người dùng thao tác cuống cuồng (Rage clicking/Spamming) hoặc ứng dụng đang bị phản hồi chậm.

---

### 🔹 `n_unique_tokens` — Số màn hình / hành động độc nhất

- **Ý nghĩa là gì?**
  Là số lượng màn hình hoặc nút bấm **hoàn toàn khác nhau** mà người dùng đã chạm qua trong suốt hành trình.
- **Ví dụ so sánh**:
  - *Hành trình A*: Dài 12 bước, đi qua 10 màn hình khác nhau $\to$ `n_unique_tokens = 10` (Luồng nghiệp vụ sâu, quy trình gồm nhiều bước).
  - *Hành trình B*: Dài 12 bước, nhưng chỉ lặp đi lặp lại giữa 2 màn hình $\to$ `n_unique_tokens = 2` (Người dùng đang bế tắc hoặc phân vân giữa 2 lựa chọn).
- **Ứng dụng thực tế**: Đánh giá độ sâu và độ phong phú về tính năng trong một lần sử dụng.

---

### 🔹 `n_dropped_screens` — Số màn hình hệ thống đã lọc bỏ

- **Ý nghĩa là gì?**
  Là số lượng sự kiện thuộc về **vỏ bọc hệ thống hoặc màn hình chờ** (như `SplashVC`, `LaunchScreen`, `MainTabBarController`, `BaseNavigation`) đã được chủ động lọc bỏ khỏi hành trình để tránh làm nhiễu dữ liệu.
- **Vì sao phải lọc?**
  Người dùng không mở app để "ngắm" màn hình Splash hay thanh điều hướng hệ thống. Việc lọc bỏ giúp mô hình tập trung $100\%$ vào các màn hình tính năng mang giá trị nghiệp vụ thực sự.

---

## 3. Nhóm 2: Nhịp Điệu & Hành Vi Thao Tác (Interaction & Behavioral Dynamics)

```
        Thao tác người dùng
       ┌─────────┴─────────┐
       ▼                   ▼
  action_rate           back_rate
 (Bấm / Thao tác)    (Quay lui / Hủy)
       │                   │
       └─────────┬─────────┘
                 ▼
            revisit_rate
      (Vòng vèo / Ghé thăm lại)
```

---

### 🔹 `action_rate` (hoặc `action_ratio`) — Tỷ lệ tương tác chủ động

- **Công thức**:
  $$\text{action\_rate} = \frac{\text{Số sự kiện dạng Action (Bấm, Chạm, Vuốt)}}{n\_events\_final}$$
- **Ý nghĩa là gì?**
  Đo lường mức độ **chủ động thao tác** của người dùng so với việc chỉ xem thụ động:
  - $\text{action\_rate} \ge 0.7$ (Rất cao): Người dùng thao tác liên tục, cấu hình thiết bị, gõ bàn phím, chọn lựa.
  - $\text{action\_rate} \le 0.3$ (Thấp): Người dùng chủ yếu lướt đọc tin tức, tra cứu hóa đơn hoặc xem hướng dẫn (Viewing/Browsing).
- **Ứng dụng thực tế**: Phân loại tính chất hành trình là *Thực thi nghiệp vụ (Transactional)* hay *Tìm kiếm thông tin (Informational)*.

---

### 🔹 `back_rate` — Tỷ lệ quay lui / Thoát luồng

- **Công thức**:
  $$\text{back\_rate} = \frac{\text{Số thao tác Back, Close, Dismiss, Cancel}}{n\_events\_final}$$
- **Ý nghĩa là gì?**
  Đo lường mức độ người dùng phải **lùi lại, đóng popup hoặc hủy bỏ** trong hành trình.
- **Cảnh báo UX**:
  - `back_rate` bình thường: $0.05 - 0.15$.
  - `back_rate` bất thường ($> 0.30$): Người dùng đi nhầm đường, không tìm thấy thông tin cần thiết, gặp giao diện khó hiểu hoặc cố gắng hủy bỏ quy trình vì gặp lỗi.

---

### 🔹 `revisit_rate` (hoặc `revisit_ratio`) — Tỷ lệ ghé thăm lại màn hình cũ

- **Công thức**:
  $$\text{revisit\_rate} = 1 - \frac{n\_unique\_tokens}{n\_events\_final}$$
- **Ý nghĩa là gì?**
  Phản ánh mức độ "đi vòng vèo" của hành trình:
  - $\text{revisit\_rate} \approx 0$: **Luồng đi thẳng (Linear Flow)**. Người dùng đi tuần tự từ bước 1 $\to$ 2 $\to$ 3 $\to$ Đích mà không cần quay lại bước nào.
  - $\text{revisit\_rate} > 0.5$: **Luồng quanh co (Non-linear Flow)**. Người dùng phải đi tới rồi lùi lui, chuyển qua chuyển lại giữa các tab nhiều lần.
- **Ứng dụng thực tế**: Nhận diện các giao diện phức tạp buộc người dùng phải so sánh thông tin qua lại giữa nhiều màn hình.

---

### 🔹 `median_gap` (hoặc `median_gap_s`) — Nhịp độ thao tác trung bình

- **Ý nghĩa là gì?**
  Là **khoảng thời gian trung vị (tính bằng giây)** giữa 2 thao tác liên tiếp của người dùng.
- **Các khoảng giá trị điển hình**:
  - $1 - 3$ giây: **Thao tác nhanh, dứt khoát**. Người dùng đã quen thuộc với giao diện, biết chính xác mình cần bấm gì.
  - $4 - 10$ giây: **Nhịp độ bình thường**. Người dùng vừa đọc vừa thao tác.
  - $> 15 - 30$ giây: **Dừng lại ngập ngừng (Dwell/Hesitation)**. Người dùng gặp khó khăn trong việc hiểu nội dung, mất thời gian tìm nút bấm, hoặc đang phải đợi hệ thống tải dữ liệu.

---

## 4. Nhóm 3: Phát Hiện Bất Thường & Điểm Nghẽn UX (Anomaly & Friction Intelligence)

Hệ thống kết hợp **2 lăng kính độc lập** để bắt trọn mọi sự cố:
1. **Lăng kính Hình học (Geometric)**: Nhìn từ trên cao — *Toàn bộ hình thái hành trình này có xa lạ so với khuôn mẫu chuẩn không?*
2. **Lăng kính Sinh chuỗi (Generative Markov)**: Soi từng bước đi — *Bước chuyển từ màn hình A sang màn hình B có kỳ quặc, trái quy luật không?*

```
                     ┌────────────────────────────────┐
                     │   Hành trình của người dùng    │
                     └───────────────┬────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
         [Khoảng cách Hình học]            [Mô hình Chuỗi Markov]
         (Geometric Distance)              (Generative Log-prob)
                    │                                 │
            > 95th Percentile?                < 5th Percentile?
                    │                                 │
                    ▼                                 ▼
           geometric_anomaly                 generative_anomaly
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
                             Cả 2 cùng xảy ra?
                                     ▼
                               severe_anomaly
                       (Bất thường đặc biệt nghiêm trọng)
```

---

### 🔹 `generative_anomaly` — Bất thường về chuỗi chuyển bước (Markov Anomaly)

- **Ý nghĩa là gì?**
  Là cờ cảnh báo (`True`/`False`) cho biết hành trình chứa những **bước nhảy màn hình vô cùng hiếm gặp hoặc phi logic**.
- **Cách tính**:
  Mỗi cụm hành trình có một ma trận xác suất chuyển bước Markov $P(t_{i+1} \mid t_i)$. Điểm log-likelihood được chuẩn hóa theo độ dài. Nếu điểm này rơi vào nhóm $5\%$ thấp nhất (`markov_p05`), hành trình bị gắn cờ `generative_anomaly = True`.
- **Ví dụ thực tế**:
  - *Bình thường*: `Chọn gói cước` $\to$ `Xem chi tiết` $\to$ `Nhập OTP` $\to$ `Xác nhận`.
  - *Bất thường*: Đang ở `Màn hình hỗ trợ kỹ thuật` bỗng nhảy thẳng sang `Thanh toán thành công` mà không qua bước chọn cổng thanh toán.

---

### 🔹 `severe_anomaly` — Bất thường nghiêm trọng đa chiều

- **Ý nghĩa là gì?**
  Là cờ cảnh báo nguy hiểm cấp cao nhất (`True`/`False`), kích hoạt khi hành trình **vi phạm đồng thời cả 2 chuẩn mực**:
  $$\text{severe\_anomaly} = (\text{geometric\_anomaly} == \text{True}) \ \land \ (\text{generative\_anomaly} == \text{True})$$
- **Ý nghĩa cảnh báo**:
  Hành trình này vừa có cấu trúc tổng thể xa lạ với tất cả các nhóm người dùng bình thường, vừa có chuỗi bước đi bên trong cực kỳ dị biệt.
- **Hành động cần làm**: Đây là nhóm dữ liệu cần chuyển ngay cho đội ngũ Kỹ thuật / QA kiểm tra vì có nguy cơ cao là **Lỗi ứng dụng (App Crash, Deadlock, UI Loop Bug)** hoặc hành vi gian lận (Fraud/Abuse).

---

### 🔹 `friction_rate` — Tỷ lệ ma sát trải nghiệm

- **Ý nghĩa là gì?**
  Là tỷ lệ phần trăm số hành trình gặp phải trở ngại trong một cụm tính năng hoặc phân khúc người dùng:
  $$\text{friction\_rate} = \frac{\text{Số hành trình có ít nhất 1 cờ ma sát}}{\text{Tổng số hành trình trong nhóm}}$$
- **Ứng dụng thực tế**:
  - Dùng để **xếp hạng mức độ "nhức nhối"** giữa các tính năng.
  - Ví dụ: Tính năng *Đổi mật khẩu Wi-Fi* có `friction_rate = 12%` (tốt), nhưng tính năng *Ký hợp đồng điện tử VNeID* có `friction_rate = 48%` $\to$ Cần ưu tiên nguồn lực để tối ưu lại luồng VNeID trước.

---

### 🔹 `behavioral_friction_flags` — Danh sách cờ ma sát hành vi cụ thể

- **Ý nghĩa là gì?**
  Là chuỗi văn bản ghi lại **chính xác những kiểu khó khăn** mà người dùng đã trải qua trong hành trình, phân tách nhau bằng dấu `|`.
- **Các cờ ma sát chính**:

| Tên cờ (Flag) | Tiêu chí kích hoạt | Ý nghĩa thực tế đối với người dùng |
|---|---|---|
| `excessive_back` | `back_rate > p90` | Người dùng bấm quay lại liên tục vì lạc đường hoặc không tìm thấy mục tiêu. |
| `navigation_loop` | `n_loop_removed > 0` | Người dùng bị kẹt trong vòng lặp $A \to B \to A \to B$ (đi đi lại lại giữa 2 màn hình). |
| `screen_thrash` | `revisit_ratio > p90` | Người dùng bối rối, nhảy qua nhảy lại giữa rất nhiều màn hình khác nhau. |
| `slow_journey` | `span_seconds > p95` | Hành trình tốn thời gian lâu bất thường so với thời gian trung bình của người khác. |

---

## 5. Nhóm 4: Dự Báo Bước Tiếp Theo (Next-Action Prediction)

```
[Vị trí hiện tại của khách hàng] ──► [Mô hình Markov của Cụm] ──► [Dự báo: effective_next_action]
                                                                        │
                                                                        ▼
                                                             [Kích hoạt gợi ý / Smart Nudge]
```

---

### 🔹 `next_action` & `effective_next_action` — Dự báo hành động tiếp theo

- **Ý nghĩa là gì?**
  Là màn hình hoặc nút bấm mà mô hình chuỗi Markov dự báo **người dùng có xác suất thực hiện cao nhất tiếp theo**, dựa trên vị trí hiện tại và loại hành trình họ đang tham gia.
  - `next_action`: Dự báo trực tiếp từ mô hình cụm cấp cơ sở.
  - `effective_next_action`: Dự báo tin cậy cuối cùng sau khi đã xử lý qua cây phân cấp fallback (áp dụng khi cụm cơ sở chưa có đủ dữ liệu lịch sử).
- **Ứng dụng kinh doanh & sản phẩm**:
  1. **Hỗ trợ chủ động (Proactive Assistance)**: Nếu khách hàng đang trong hành trình "Báo hỏng mạng" và bước tiếp theo thường là "Gọi tổng đài", app có thể hiển thị sẵn nút hỗ trợ nhanh hoặc kiểm tra modem tự động.
  2. **Tải trước dữ liệu (Prefetching)**: Tải ngầm dữ liệu của màn hình tiếp theo để app phản hồi ngay tức thì khi người dùng bấm vào.
  3. **Gợi ý thông minh (Smart Nudge)**: Hiển thị hướng dẫn ngắn giúp người dùng hoàn thành bước tiếp theo dễ dàng hơn.

---

## 6. Bảng Tra Cứu Nhanh Tổng Hợp (Quick Reference Table)

| Tên trường | Đơn vị / Kiểu | Mục đích chính | Giá trị lý tưởng | Dấu hiệu cảnh báo |
|---|---|---|---|---|
| `boundary` | Text | Biết lý do bắt đầu hành trình | `session_start`, `root_return` | Cắt vì `length_cap` quá nhiều |
| `n_events_raw` | Số nguyên | Tổng số log thô ban đầu | $4 - 25$ sự kiện | $> 50$ sự kiện |
| `n_events_final` | Số nguyên | Số bước thực tế sau làm sạch | $4 - 20$ bước | Chênh lệch quá xa so với `raw` |
| `n_unique_tokens` | Số nguyên | Độ phong phú màn hình/nút bấm | $3 - 10$ màn hình | $= 1$ hoặc $= 2$ trên chuỗi dài |
| `n_dropped_screens` | Số nguyên | Số màn hình rác hệ thống đã lọc | Tuỳ cấu hình | Không ảnh hưởng người dùng |
| `action_rate` | Số thập phân $[0, 1]$ | Đo độ chủ động tương tác | Tuỳ nghiệp vụ ($0.3 - 0.7$) | $= 0$ trên luồng cần thanh toán |
| `back_rate` | Số thập phân $[0, 1]$ | Đo mức độ quay lui / thoát luồng | $< 0.10$ | $> 0.25$ (Lạc đường/Hủy luồng) |
| `revisit_rate` | Số thập phân $[0, 1]$ | Đo độ quanh co của hành trình | $< 0.20$ (Đi thẳng) | $> 0.50$ (Phân vân/Quay lại nhiều) |
| `median_gap` | Giây | Đo nhịp độ thao tác | $2 - 6$ giây | $> 20$ giây (Khựng lại/Chờ tải) |
| `generative_anomaly` | Boolean (`T/F`) | Phát hiện nhảy bước kỳ lạ | `False` | `True` (Chuyển bước trái quy luật) |
| `severe_anomaly` | Boolean (`T/F`) | Cảnh báo bất thường nghiêm trọng | `False` | `True` (Nghi vấn lỗi phần mềm/Bug) |
| `friction_rate` | Phần trăm $\%$ | Đo tỷ lệ gặp ma sát toàn luồng | $< 15\%$ | $> 35\%$ (Luồng UX có vấn đề lớn) |
| `effective_next_action` | Text Token | Dự báo bước đi tiếp theo | Tên màn hình / Action | Không có dự báo (`null`) |
| `behavioral_friction_flags` | Text (`\|`) | Tên các loại lỗi ma sát cụ thể | Rỗng (Trơn tru) | Chứa `navigation_loop`, `screen_thrash` |

---

## 7. Kết Luận

Bộ chỉ số trên tạo thành một **hệ thống giám sát trải nghiệm người dùng toàn diện**:
- Giúp **Đội ngũ Phát triển Sản phẩm (Product Owners / UX Designers)** nhìn thấy rõ những "hòn đá cản đường" trên giao diện để cải tiến luồng màn hình.
- Giúp **Đội ngũ Kỹ thuật (Dev / QA)** khoanh vùng nhanh các lỗi tiềm ẩn (deadlock, lag, vòng lặp) trước khi người dùng gửi khiếu nại.
- Giúp **Đội ngũ Vận hành & Chăm sóc Khách hàng (CS / Operations)** triển khai các kịch bản trợ giúp tự động, đúng người, đúng thời điểm.
