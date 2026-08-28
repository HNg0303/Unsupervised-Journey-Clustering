# Từ Clickstream đến Hành trình Người dùng
### Giải pháp phân cụm hành vi không giám sát cho ứng dụng HiFPT

> Tài liệu này mô tả toàn bộ lời giải, ưu tiên **trực giác** trước, **kỹ thuật** sau.
> Mỗi phần có 3 lớp: *Vấn đề* → *Cách giải* → *Bằng chứng trên dữ liệu thật*.
> Mọi con số đều lấy từ lần chạy đã commit trong `outputs/` (dữ liệu 01–07/07/2026).

---

## 0. Bài toán và giá trị

### Chúng ta đang có gì

Ứng dụng ghi lại mọi thao tác của người dùng dưới dạng **clickstream**: mỗi lần
mở một màn hình, mỗi lần chạm một nút đều sinh ra một dòng log. Bảy ngày dữ liệu
cho ra **114.534 sự kiện** từ 154 khách hàng.

Nhưng log thô là một dòng chảy không có ý nghĩa. Nó giống như nghe được từng
tiếng bước chân trong một tòa nhà mà không biết ai đang đi đâu, để làm gì.

### Chúng ta muốn có gì

Biến dòng chảy đó thành các **hành trình (journey) có tên**:

```
… 47 dòng log rời rạc …
        ↓
"Khách này đang thanh toán hóa đơn trả sau"
"Khách này đang đăng nhập bằng OTP"
"Khách này đang nâng gói Internet — và đang gặp khó khăn"
```

### Vì sao điều đó đáng giá

| Năng lực | Ý nghĩa kinh doanh |
|---|---|
| **Hiểu ý định** | Biết người dùng *đến để làm gì*, không chỉ *đã bấm gì* |
| **Phân loại tự động** | Mọi phiên mới được gán nhãn hành trình mà không cần con người dán nhãn |
| **Phát hiện bất thường** | Hành trình lệch khỏi khuôn mẫu = dấu hiệu người dùng đang mắc kẹt |
| **Hỗ trợ chủ động** | Biết bước tiếp theo người dùng *thường* làm → gợi ý đúng lúc, hoặc bật trợ giúp trước khi họ bỏ cuộc |
| **Xếp hạng điểm nghẽn UX** | Biết luồng nào tốn nhiều bước nhất, quay lui nhiều nhất → ưu tiên sửa |

Điểm mấu chốt: đây là bài toán **không giám sát (unsupervised)**. Không ai ngồi
dán nhãn 10.000 hành trình. Hệ thống phải tự tìm ra các khuôn mẫu, và con người
chỉ đặt tên cho những gì nó tìm được.

### Toàn cảnh pipeline

```
Log thô  →  Chuẩn hóa  →  Token hóa  →  Cắt thành hành trình  →  Làm sạch
                                                                    │
                                        ┌───────────────────────────┴────────┐
                              LỘ TRÌNH A│                                    │LỘ TRÌNH B
                            TF-IDF n-gram▼                                   ▼ PrefixSpan
                                    SVD(64)                              SVD(64)
                                        └────────────┬───────────────────────┘
                                          + 10 đặc trưng hành vi (trọng số 0.35)
                                                     │
                                                 HDBSCAN
                                                     │
                                       Markov chain theo từng cụm
                                                     │
                             Cụm · Điểm bất thường · Dự đoán bước tiếp theo
```

Ta chạy **hai lộ trình song song** (A và B) trên cùng một đầu vào. Nếu hai cách
mã hóa hoàn toàn khác nhau vẫn cho ra cùng một cấu trúc, cấu trúc đó là thật —
đây là phép kiểm chứng rẻ nhất mà pipeline này có.

---

## 1. Khám phá dữ liệu — và ba phát hiện đã đổi hướng thiết kế

### Dữ liệu trông như thế nào

| Chỉ số | Giá trị |
|---|---|
| Sự kiện | 114.534 |
| Phiên (`session_id`) | 1.425 |
| Thiết bị / Khách hàng | 182 / 154 |
| Khoảng thời gian | 01–07/07/2026 |
| Loại sự kiện | 2 — `View` (86.690) và `Action` (27.844) |
| `segment_name` khác nhau | 1.617 |
| `screen_name` khác nhau | 611 |

Các trường quan trọng:

- `session_id` — nhóm sự kiện theo phiên
- `OS` / `segmentation.segment` — nền tảng **iOS** hay **Android**
- `event_type` — **View** (xem màn hình) hay **Action** (thao tác)
- `segment_name`, `screen_name` — màn hình / đường dẫn thao tác
- `timestamp` — mốc thời gian (đơn vị **ms**, đã kiểm chứng bằng đối chiếu với
  `created_at`: sai lệch trung vị 1,03 giây)

### Phát hiện 1 — Vai trò của các trường **đảo ngược** theo `event_type`

Đây là phát hiện quan trọng nhất, và nó suýt làm hỏng toàn bộ mô hình.

| `event_type` | `segment_name` chứa | `screen_name` chứa |
|---|---|---|
| **View** (86.690 dòng) | **màn hình** (`HomeVC`, `android/Home`, URL webview) | NULL — **thiếu 100%** |
| **Action** (27.844 dòng) | **đường dẫn thao tác** (`home/.../click_modem_control`) | **màn hình** — có 98,4% |

Nghĩa là cùng một cột chứa hai thứ khác nhau tùy dòng. Nếu ghép thẳng
`event_type + segment_name + screen_name`, ta được hai "vũ trụ" token rời nhau:
mô hình **không thể** học được rằng `View HomeVC` và `Action Home/Nav_profile`
xảy ra trên **cùng một màn hình**.

Hậu quả: không có trục màn hình chung → không dựng được đồ thị điều hướng →
không đo được "quay về Home", "nhảy qua nhảy lại giữa hai màn" — đúng những tín
hiệu ma sát mà ta cần.

**Cách sửa** — chuẩn hóa về bộ ba đúng vai trò trước khi token hóa:

```
View    →  (View,   screen = segment_name,  target = ∅)
Action  →  (Action, screen = screen_name,   target = segment_name)
```

Sau khi sửa, **542 / 611** màn hình của Action khớp sạch với màn hình của View.
Trục màn hình xuất hiện.

### Phát hiện 2 — `session_id` **không phải** là một phiên có ý nghĩa

| Chỉ số của `session_id` | Giá trị |
|---|---|
| Độ dài thời gian | trung vị **3,2 phút** · p75 18,9 phút · **tối đa 30,5 giờ** |
| Số sự kiện | trung vị 34 · **tối đa 1.524** |
| Khoảng cách giữa 2 sự kiện | trung vị **0,003 giây** |

Một `session_id` kéo dài 30 giờ chứa hàng chục mục đích khác nhau. Nếu phân cụm
theo `session_id`, ta đang phân cụm những **hỗn hợp** mục đích — kết quả sẽ vô
nghĩa.

→ Vì vậy **cắt phiên thành hành trình** không phải một bước phụ, nó là bước
*chịu lực* của cả pipeline (Phần 4).

### Phát hiện 3 — Hai nền tảng phải tách riêng

`HomeVC` (iOS) và `android/Home` (Android) là **cùng một ý định người dùng**
nhưng **không có một ký tự nào trùng nhau**. Nếu gộp chung, mô hình sẽ tiêu tốn
phần lớn năng lực chỉ để học lại ranh giới iOS/Android thay vì học hành vi.

→ Mọi màn hình được gắn tiền tố nền tảng (`iOS::HomeVC`, `Android::HOME`), và
toàn bộ pipeline chạy **độc lập cho iOS và Android**.

---

## 2. Chuẩn hóa (Canonization) — dọn nhiễu định danh, giữ nguyên hành vi

### Vấn đề

**575 trong 821** màn hình View là URL webview. Và đuôi dài của chúng gần như
hoàn toàn là **định danh phiên bản**, không phải hành vi:

```
.../spin-wheel/home?utm_source=banner
.../spin-wheel/home?env=testing&navigationType=previous
.../spin-wheel/home?env=testing&navigationType=previous&test=2003344
.../update-package/home?merchant_id=BH_UINT&contractNo=SGABP2073
```

Bốn dòng trên là **cùng một màn hình**, nhưng máy đọc thành bốn màn hình khác
nhau. 11.046 dòng log mang query string kiểu này. Kết quả là từ vựng phình to,
và mỗi token chỉ xuất hiện một lần — vô dụng với mọi thuật toán đo khoảng cách.

### Cách giải

Ba phép biến đổi, tất cả đều **thuần cấu trúc** — không có phép gộp ngữ nghĩa nào:

1. **Bỏ query param không mang ngữ nghĩa** — `contractNo`, `orderId`,
   `timestamp`, `utm_*` bị loại. Nhưng có **danh sách trắng** giữ lại những
   param *thực sự đổi nội dung người dùng nhìn thấy*: `tab`, `cat_id`,
   `orderType`, `step`, `mode`, `view`, `status`.
2. **Che các đoạn định danh trong đường dẫn** — `/order/8734` → `/order/{id}`.
3. **Gắn tiền tố nền tảng** — như Phát hiện 3.

Nguyên tắc: ta xóa **ai/cái nào**, giữ nguyên **đang làm gì**.

### Bằng chứng

| Giai đoạn | Kích thước từ vựng | Token chỉ xuất hiện 1 lần | Top-100 phủ được |
|---|---|---|---|
| Bộ ba thô | 2.494 | 918 (36,8%) | 57,0% |
| Sau chuẩn hóa `(type, screen, target)` | **1.640** | **364 (22,2%)** | **75,3%** |
| Chỉ riêng trục màn hình | 423 | 20 (4,7%) | 91,5% |

Từ vựng giảm **34%**, đuôi nhiễu giảm gần **40%**, và 100 token phổ biến nhất
giờ phủ được 3/4 lưu lượng thay vì hơn nửa. Đây là bước có tỉ lệ
lợi-ích/chi-phí cao nhất trong cả pipeline.

---

## 3. Token hóa — biến một sự kiện thành một "từ"

### Ý tưởng

Nếu coi hành trình là một **câu**, thì mỗi sự kiện phải là một **từ**. Từ đó
được ghép từ ba mảnh: *loại sự kiện* + *màn hình* + *thao tác*.

```
View  trên HomeVC                    →  V@iOS::HomeVC
Action "Home/Nav_profile" tại HomeVC  →  A@iOS::HomeVC#Home/Nav_profile
```

### Ba độ phân giải, không phải một

Một độ phân giải không thể phục vụ cả mô hình hóa lẫn diễn giải, nên ta phát ra
ba mức cho mỗi sự kiện:

| Mức | Dạng | Từ vựng | Token xuất hiện 1 lần | Dùng cho |
|---|---|---|---|---|
| **L1** | `V@iOS::HomeVC` | 636 | 7,2% | đặt tên cụm, dự phòng |
| **L2** | `A@iOS::HomeVC#Home/Nav_profile` | 1.259 | 15,5% | **mặc định cho mô hình** |
| **L3** | đường dẫn thao tác đầy đủ | 1.640 | 22,2% | điều tra chi tiết, dựng lại chính xác |

**Vì sao chọn L2.** L3 trung thực nhất nhưng quá thưa — 22% token chỉ xuất hiện
đúng một lần, và với mọi thước đo khoảng cách thì đó là nhiễu thuần túy. L2 cắt
đường dẫn thao tác còn 2–3 tầng (`home/home_service_management/*`), đúng chỗ tín
hiệu hữu ích nằm.

**Xử lý token hiếm (rare-token backoff).** Token xuất hiện trong dưới 3 *hành
trình* sẽ lùi về dạng L1, rồi về `<rare>`. Điểm tinh tế: tần suất được đếm theo
**hành trình**, không theo **sự kiện** — một token bắn 200 lần trong một hành
trình vẫn chỉ là *một* bằng chứng. Thực tế bước này chỉ chạm 1,2% token, vì
chuẩn hóa ở Phần 2 đã làm phần nặng.

---

## 4. Cắt phiên thành hành trình — bước chịu lực

Đây là bước quyết định **đơn vị phân tích**. Cắt sai thì mọi thứ sau đó đều sai.
Ta làm theo hai cách và đối chiếu.

---

### 4.1 Cách 1 — Rule-based: cắt theo logic sản phẩm

#### Trực giác

Con người biết khi nào một việc kết thúc. Nếu bạn quay về màn hình chủ, hoặc
đăng xuất, hoặc để máy nằm im 2 phút — việc trước đó đã xong. Ta mã hóa chính
những trực giác đó thành luật.

#### Năm luật cắt

| Luật | Lý do |
|---|---|
| **Đổi phiên** | Ranh giới hiển nhiên |
| **Nghỉ quá lâu** — khoảng cách > **90 giây** | Người dùng đã rời đi rồi quay lại với mục đích khác |
| **Quay về màn hình chủ** sau ≥ 6 sự kiện | Về Home = "việc vừa rồi xong" |
| **Đăng nhập / đăng xuất** (chỉ tính *thao tác*) | Đổi danh tính = đổi ngữ cảnh |
| **Trần độ dài** 80 sự kiện | Chặn các phiên bệnh lý 1.500 sự kiện |

#### Hai luật đã sai ngay lần đầu — và dữ liệu bắt được

Phần này đáng kể lại vì nó cho thấy quy trình có tự kiểm chứng:

1. **`MainTabBarController` trông giống màn hình gốc nhưng không phải.** iOS đẩy
   qua nó trên gần như *mọi* thao tác điều hướng (7.049 sự kiện). Coi nó là
   "màn hình chủ" khiến luật "quay về gốc" bắn **8.544 lần** và băm mọi phiên
   thành các mảnh 3 sự kiện. → Màn hình chủ phải là màn hình **người dùng nhận
   ra là trang chủ**, không phải cái container chứa nó.
2. **Nhận diện đăng nhập bằng cách tìm chữ `login` trong token** đã bắn trên mọi
   `View LoginVC` trong lúc khách vãng lai chỉ đang xem. → Chỉ nhận diện trên
   **đích của thao tác** (`login/continue_login`, `Home/Account/Log_out`).

#### Ngưỡng 90 giây được chọn thế nào

Không đoán — quét:

| τ (giây) | Số hành trình | Độ dài trung bình | Tỉ lệ quá ngắn (<3) |
|---|---|---|---|
| 30 | 12.314 | 9,30 | 13,8% |
| 60 | 10.824 | 10,58 | 10,2% |
| **90** | **10.262** | **11,16** | **8,9%** |
| 300 | 9.227 | 12,41 | 6,5% |
| 900 | 8.679 | 13,20 | 4,8% |

Sau ~90 giây, số hành trình gần như đi ngang. Chọn ngưỡng ở **vùng phẳng**, không
phải trên vách dốc — nghĩa là kết quả không nhạy cảm với việc chỉnh nhẹ tham số.

Phân bố lý do cắt cuối cùng: quay về gốc 4.987 · nghỉ lâu 2.517 · đổi phiên
1.425 · đổi trạng thái đăng nhập 1.264 · chạm trần 69.

#### Điểm mạnh / điểm yếu

| ✅ Mạnh | ⚠️ Yếu |
|---|---|
| **Giải thích được 100%** — mỗi lát cắt có một lý do bằng tiếng người | **Phải biết trước** đâu là màn hình chủ, đâu là hành động auth |
| Không cần huấn luyện, chạy tức thì, xác định (deterministic) | Luật gắn với **thiết kế app hiện tại** — app đổi điều hướng thì luật phải sửa |
| Đội sản phẩm có thể **tranh luận và sửa** trực tiếp | Bỏ sót các ranh giới không có tín hiệu rõ ràng |
| Cho hành trình **liên tục, không chồng lấn**, phủ 100% log | Ngưỡng thời gian là một lựa chọn, dù đã dựa trên dữ liệu |

---

### 4.2 Cách 2 — PrefixSpan: để dữ liệu tự nói

#### Trực giác

Cách 1 hỏi *"hành trình này kết thúc ở đâu?"*. PrefixSpan hỏi một câu khác:
*"chuỗi thao tác nào lặp đi lặp lại ở nhiều người dùng khác nhau?"* — và trả lời
mà **không cần biết trước bất cứ giả định nào** về Home, về auth, về thời gian.

Khác biệt cốt lõi so với n-gram: n-gram đòi hỏi **liền kề**, PrefixSpan thì
không.

```
Hành trình : HOME → SUPPORT → BACK → PAY → CONFIRM

n-gram   : "HOME PAY CONFIRM" KHÔNG khớp — đoạn rẽ ngang đã phá vỡ chuỗi
PrefixSpan: HOME → PAY → CONFIRM KHỚP — nó là một chuỗi con (subsequence)
```

Người dùng thật rẽ ngang liên tục. Nên n-gram nhìn thấy **đường đi chính xác**,
còn PrefixSpan nhìn thấy **mục tiêu**.

#### Thuật toán, ngắn gọn

1. Bắt đầu từ mẫu rỗng; hình chiếu của nó là mọi hành trình tại vị trí 0.
2. Với mỗi token ứng viên, tìm lần xuất hiện **đầu tiên** từ vị trí hiện tại trở
   đi. *Chỉ lần đầu tiên* — điều này khiến độ hỗ trợ (support) đếm **số hành
   trình**, không phải số lần xuất hiện. Một người lặp 200 lần không thể tự chế
   ra một mẫu.
3. Giữ token nếu nó xuất hiện trong ≥ `min_support` hành trình. Phát ra mẫu mở
   rộng, rồi đệ quy.
4. Dừng ở độ dài 5.

**Lọc mẫu đóng (closed patterns).** Một mẫu là *đóng* khi không có mẫu dài hơn
nào có cùng độ hỗ trợ. Không lọc thì đầu ra ngập trong mọi tiền tố của cùng một
hình dạng (`A`, `A→B`, `A→B→C` đều support 79). Chỉ mẫu cực đại mang thông tin.

Tham số: `min_support = 2%` (sàn 20 hành trình), tối đa độ dài 5, giới hạn 300 mẫu.

| Nền tảng | Số mẫu | Mẫu/hành trình (TB) | Hành trình không khớp mẫu nào |
|---|---|---|---|
| Android | 300 | 16,7 | 165 (9,2%) |
| iOS | 300 | 16,0 | 870 (21,4%) |

> **Chi tiết chỉnh tham số đáng biết:** hạn mức 300 mẫu được chia **theo từng độ
> dài**, không phải toàn cục. Nếu xếp hạng thuần theo độ dài, đầu ra chỉ còn mẫu
> dài 5 và độ phủ sụp xuống 30%/9%. Chia hạn mức theo độ dài đưa nó lên
> **90,8% Android / 78,6% iOS**.

#### Hai lối ra từ PrefixSpan

- **B-1 — Đặc trưng.** Mỗi mẫu thành một cột nhị phân "hành trình này có chứa mẫu
  đó không". 300 cột, đưa thẳng vào bước phân cụm.
- **B-2 — Nhãn nguyên mẫu không cần phân cụm.** Gán cho mỗi hành trình mẫu *đặc
  thù nhất* mà nó chứa (dài nhất, hòa thì lấy hiếm nhất). Đây là một baseline
  hoàn toàn diễn giải được, **không dùng clustering** — và là phép kiểm tra tỉnh
  táo cho các cụm ta thực sự khớp.

#### Điểm mạnh / điểm yếu

| ✅ Mạnh | ⚠️ Yếu |
|---|---|
| **Không cần giả định trước** — phát hiện được cả những luồng ta chưa biết để tìm | Không tự sinh ra ranh giới cắt — nó tìm *khuôn mẫu*, vẫn cần Cách 1 tạo đơn vị |
| **Chịu được đoạn rẽ ngang** — bắt mục tiêu chứ không bắt đường đi | Nhạy với `min_support`: đặt thấp thì bùng nổ tổ hợp, đặt cao thì mất luồng ngách |
| Mẫu **đọc được như một câu**: `HOME → payment → chọn hóa đơn → thanh toán` | **9–21% hành trình không khớp mẫu nào** → vector toàn số 0, không có bằng chứng gì |
| Độ hỗ trợ là con số kiểm chứng được, không phải điểm trừu tượng | Chi phí tính toán tăng nhanh theo độ dài mẫu |

> **Cái bẫy, và cách xử lý.** Hành trình không khớp mẫu nào có vector toàn 0. Mọi
> hành trình như vậy nằm chồng lên đúng một điểm, và HDBSCAN sẽ ngoan ngoãn báo
> chúng là **một cụm khổng lồ** — 165 trên Android, 870 trên iOS, đúng bằng số
> không khớp. Đó không phải nguyên mẫu, đó là *"không có bằng chứng"*. Pipeline
> gắn lại chúng thành nhiễu và chấm điểm lại.

---

### 4.3 Làm sạch sau khi cắt — và một sai lầm suýt xóa mất biến mục tiêu

Có **ba** loại lặp trong dữ liệu, và chỉ **hai** trong số đó là nhiễu:

| Loại | Khối lượng | Xử lý |
|---|---|---|
| Lặp liên tiếp y hệt `A A A` | 12.528 sự kiện | **Gộp** — framework bắn trùng, nhiễu thuần |
| **Lặp chu kỳ `A B A B A B`** | 2.731 sự kiện, 765 hành trình | **Gộp nhưng GIỮ LẠI SỐ ĐẾM** |
| Màn hình hệ thống / khởi động | 31.758 sự kiện (27,7%) | **Bỏ khỏi chuỗi**, giữ lại số đếm |

Loại thứ hai là điểm mấu chốt. Một người dùng nhảy `ModemSchedule ⇄ ManageModem`
sáu lần **chính là hành vi ta muốn phát hiện**. Gộp nó mà không ghi lại số lần
tức là **xóa mất chính biến mục tiêu**.

Nguyên tắc: **không có gì bị vứt đi**. Mỗi sự kiện bị gỡ khỏi chuỗi đều trở
thành một đặc trưng số (`n_loop_removed`, `n_dedup_removed`, `n_chrome_events`,
`max_run_length`). Nhiễu chỉ *chuyển kênh* — từ kênh chuỗi sang kênh số.

**Kết quả:** 114.534 → 67.517 sự kiện (tỉ lệ nén 0,59).

Đầu vào cuối cùng cho bước phân cụm:

| | Hành trình | Sự kiện (đã sạch) | Độ dài trung vị |
|---|---|---|---|
| Android | 1.795 | 18.804 | 7 |
| iOS | 4.063 | 42.639 | 6 |

---

## 5. Vector đặc trưng — biến một hành trình thành một điểm trong không gian

Máy không so sánh được hai chuỗi text. Nó cần mỗi hành trình là một **vector
số**. Bước này làm việc đó, qua hai kênh ghép lại.

### 5.1 Kênh chuỗi — TF-IDF trên n-gram

#### Vì sao TF-IDF?

Coi mỗi hành trình là một **văn bản ngắn** mà "từ" là các token. TF-IDF trả lời:
*"token nào thực sự đặc trưng cho hành trình này?"*

Hai thành phần:

- **TF (tần suất trong hành trình)** — token xuất hiện nhiều trong hành trình
  này thì quan trọng. Nhưng dùng **sub-linear TF** (`1 + log tf`) để một token
  lặp 20 lần không nuốt chửng cả vector.
- **IDF (nghịch đảo tần suất tài liệu)** — token xuất hiện ở *mọi* hành trình
  thì **không nói lên điều gì**. `View@iOS::HomeVC` có mặt trong 1/4 số hành
  trình — nó vô giá trị để phân biệt, và IDF tự động dìm nó xuống.

> Đây chính là trực giác quan trọng nhất của phần này: **thứ phân biệt các hành
> trình không phải là những gì ai cũng làm, mà là những gì chỉ nhóm này làm.**

#### Vì sao n-gram (n = 1…4)?

TF-IDF trên từng token đơn sẽ **mất hoàn toàn thứ tự**. Mà thứ tự chính là hành
vi:

```
HOME → PAY  → CONFIRM     (thanh toán thành công)
HOME → SUPPORT → CHAT     (đi tìm trợ giúp)
```

Hai hành trình này có thể chồng lấn ở mức token đơn, nhưng **không chung một
bigram nào**. n-gram là thứ khiến thứ tự trở thành đặc trưng.

Cách làm: nối chuỗi với **sentinel** `<bos> A B C <eos>` — sentinel quan trọng vì
chúng biến *"bắt đầu từ Home"* và *"kết thúc ở Payment"* thành đặc trưng hạng
nhất, thay vì chỉ là hệ quả ngẫu nhiên của vị trí. Sau đó liệt kê mọi n-gram
liền kề với n = 1…4: `A B C` cho ra `A`, `B`, `C`, `A B`, `B C`, `A B C`.

`min_df = 3` loại các n-gram xuất hiện ở dưới 3 hành trình — chủ yếu là đuôi
nhiễu.

#### n = 4 có thực sự đáng không?

Đo, không giả định:

| Nền tảng | n-gram | Từ vựng | Phương sai SVD giữ được | Cụm | Nhiễu | Silhouette ↑ | Davies-Bouldin ↓ |
|---|---|---|---|---|---|---|---|
| Android | 1–2 | 1.540 | 0,546 | 23 | 34,0% | 0,336 | 1,376 |
| Android | 1–3 | 2.849 | 0,457 | 26 | 38,3% | 0,354 | 1,344 |
| **Android** | **1–4** | 3.968 | 0,417 | **29** | 38,7% | **0,357** | **1,227** |
| iOS | 1–2 | 2.847 | 0,471 | 50 | 41,3% | **0,389** | 1,089 |
| iOS | 1–3 | 5.665 | 0,375 | 53 | 41,1% | 0,379 | 1,099 |
| **iOS** | **1–4** | 8.005 | 0,334 | 50 | **39,1%** | 0,377 | 1,098 |

Đọc thẳng thắn: **4-gram giúp Android một chút và không giúp gì cho iOS.**
Android được thêm 3 cụm và Davies-Bouldin tốt nhất; iOS trả gấp 2,8 lần từ vựng
để đổi lấy silhouette *giảm* 0,012. Mặc định để `(1, 4)` vì chi phí chỉ là bộ
nhớ, nhưng `(1, 3)` hoàn toàn bảo vệ được.

*Lưu ý:* phương sai SVD giữ được **giảm** khi n tăng (0,55 → 0,42 trên Android).
Đó là điều **dự kiến, không phải lỗi** — n-gram dài thêm vào các cột hiếm, gần
trực giao mà 64 thành phần không hấp thụ nổi. Hãy đánh giá bằng chất lượng cụm,
không bằng con số đó.

#### SVD — nén 8.000 chiều xuống 64

Ma trận TF-IDF có **8.005 cột** (iOS) nhưng mỗi hành trình chỉ "bật" vài chục
trong số đó. Đó là ma trận cực kỳ thưa, và trong không gian 8.000 chiều thì mọi
điểm đều cách xa nhau như nhau — hiện tượng *curse of dimensionality*. Khoảng
cách mất hết ý nghĩa, và mọi thuật toán phân cụm sẽ thất bại.

**Truncated SVD** tìm ra 64 "chủ đề" (hướng biến thiên chính) trong dữ liệu và
chiếu mỗi hành trình lên chúng. Sau đó chuẩn hóa L2 để mọi vector có cùng độ dài
— khiến so sánh trở thành so sánh **hình dạng**, không phải độ dài hành trình.

> Vì sao SVD chứ không phải PCA? PCA cần trừ giá trị trung bình, việc đó biến ma
> trận thưa thành ma trận đặc và làm nổ bộ nhớ. Truncated SVD làm việc trực tiếp
> trên ma trận thưa.

### 5.2 Kênh số — 10 đặc trưng hành vi

Chuỗi cho biết người dùng *đi qua đâu*. Nó không cho biết họ *đi như thế nào*.
Mười đặc trưng sau bổ sung đúng phần đó — và chúng chính là nơi thông tin đã bị
gỡ khỏi chuỗi ở Phần 4.3 quay trở lại:

| Đặc trưng | Ý nghĩa hành vi |
|---|---|
| `n_events_final` | Hành trình dài bao nhiêu bước |
| `n_unique_tokens` | Bao nhiêu màn hình *khác nhau* được chạm tới |
| `action_ratio` | Tỉ lệ thao tác / xem — chủ động hay chỉ lướt |
| `back_rate` | **Tỉ lệ bấm quay lui — tín hiệu bối rối trực tiếp** |
| `revisit_ratio` | Tỉ lệ màn hình bị quay lại — dấu hiệu đi vòng |
| `n_loop_removed` | **Số vòng lặp A⇄B đã phát hiện — tín hiệu ma sát** |
| `n_dedup_removed` | Số sự kiện trùng do framework |
| `span_seconds` | Tổng thời gian hành trình |
| `total_dwell_s` | Tổng thời gian dừng trên màn hình |
| `median_gap_s` | Nhịp thao tác — nhanh dứt khoát hay chậm do dừng nghĩ |

Xử lý: mọi cột đếm/thời gian đều có **đuôi phải rất nặng**, nên `log1p` trước;
sau đó chuẩn hóa (standardize); rồi **hạ trọng số xuống 0,35** để 10 con số vô
hướng không thể lấn át 64 chiều ngữ nghĩa của chuỗi.

### 5.3 Kết quả

```
64 chiều (SVD của chuỗi)  +  10 chiều (hành vi × 0.35)  =  74 chiều / hành trình
```

Điểm quan trọng về mặt kỹ thuật vận hành: **cả hai lộ trình A và B đều dùng
chung khối 10 đặc trưng này**. Nhờ vậy, mọi khác biệt giữa hai lộ trình đều do
**cách mã hóa chuỗi** gây ra — đó là toàn bộ lý do chạy song song hai lộ trình.

---

## 6. HDBSCAN — tìm ra các nguyên mẫu

### Vì sao là HDBSCAN, không phải K-Means

| | K-Means | **HDBSCAN** |
|---|---|---|
| Phải biết trước số cụm `k` | ✅ Có — mà ta không biết | ❌ Không cần |
| Hình dạng cụm | Chỉ hình cầu | Bất kỳ hình dạng nào |
| Với điểm lạ | **Ép** vào cụm gần nhất | **Từ chối** gán nhãn → `-1` |

Cột cuối cùng là lý do quyết định. Với bài toán này, **việc thuật toán từ chối
gán nhãn chính là sản phẩm**, không phải khuyết điểm — một hành trình không
giống bất cứ nguyên mẫu nào chính là ứng viên bất thường đầu tiên. K-Means sẽ
phá hủy tín hiệu đó.

Bằng chứng ủng hộ: khi chạy K-Means để đối chứng, silhouette **tăng đơn điệu**
đến k=30 (0,088 → 0,278) mà **không có điểm khuỷu**. Đó là dấu hiệu cấu trúc ở
đây có dạng **mật độ**, không phải dạng **tâm cụm**.

### Thuật toán, mức khái niệm

HDBSCAN xem dữ liệu như một cảnh quan có đồi và thung lũng mật độ:

1. Ước lượng mật độ quanh mỗi điểm (dựa trên khoảng cách tới `min_samples` láng
   giềng gần nhất).
2. Xây cây phân cấp: hạ dần ngưỡng mật độ, quan sát các cụm tách ra rồi tan rã.
3. Ở mỗi mức, giữ lại những cụm **bền vững nhất** — cụm tồn tại qua nhiều ngưỡng
   mật độ là cụm thật.
4. Mọi điểm không thuộc cụm bền vững nào → nhãn **`-1` (nhiễu)**.

Tham số: `min_cluster_size = 15` (một nguyên mẫu phải có ít nhất 15 hành trình
mới đáng gọi tên), `min_samples = 5`.

### Đầu vào / Đầu ra

```
VÀO :  ma trận 74 chiều  (1.795 hành trình Android · 4.063 hành trình iOS)
RA  :  một số nguyên cho mỗi hành trình  (0, 1, 2, … hoặc −1 = nhiễu)
       + tâm cụm (centroid) của từng cụm
       + Markov chain riêng cho từng cụm  (dùng ở Phần 7)
```

### Kết quả

| Nền tảng | Lộ trình | Cụm | Nhiễu | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|---|---|---|
| Android | A (tf-idf) | 29 | 38,7% | **0,357** | 1,227 | 61,9 |
| Android | B (PrefixSpan) | 34 | **35,5%** | 0,337 | **1,158** | **99,8** |
| iOS | A (tf-idf) | 50 | **39,1%** | 0,377 | **1,098** | 122,3 |
| iOS | B (PrefixSpan) | 56 | 43,6% | **0,402** | 1,109 | **217,6** |

**Về tỉ lệ nhiễu 35–44%.** Con số này cao, và nó là **thật**. Các hành trình
nhiễu không phân biệt được với hành trình đã gán cụm về độ dài, tỉ lệ quay lui
hay số vòng lặp (trung vị y hệt) — chúng hiếm ở **hình dạng chuỗi**, đúng thứ mà
HDBSCAN nên từ chối ép vào cụm. Với dữ liệu hành vi clickstream, đây là kết quả
hợp lý, không phải thất bại.

### Hai lộ trình có đồng ý với nhau không?

| Nền tảng | ARI (toàn bộ) | ARI (chỉ phần đã gán cụm) |
|---|---|---|
| Android | 0,131 | **0,577** |
| iOS | 0,072 | **0,513** |

Đây là con số giàu thông tin nhất trong cả báo cáo. Mức đồng thuận tổng thể
trông tệ, nhưng nó bị chi phối bởi bất đồng về *cái gì là nhiễu*. Xét riêng
những hành trình mà **cả hai lộ trình đều gán cụm**, mức đồng thuận là
**0,51–0,58** — đáng kể.

Diễn giải: **hai lộ trình đồng ý về cấu trúc lõi và bất đồng ở phần rìa.** Hành
trình được cả hai gán vào cụm tương ứng là chắc chắn. Hành trình hai lộ trình
bất đồng chính là những cái đáng xem lại bằng tay.

> Và đó là **phép kiểm tra hồi quy rẻ nhất** ta có: nếu hai lộ trình bỗng ngừng
> đồng ý về phần lõi, chắc chắn có gì đó ở phía thượng nguồn đã hỏng.

### Đặt tên cụm bằng LLM (hậu xử lý)

HDBSCAN trả về số nguyên. Số nguyên không dùng để họp được. Mỗi cụm được đặt tên
dựa trên **ba nguồn bằng chứng**:

1. **Đường đi medoid** — hành trình đại diện nhất của cụm
2. **Token có lift cao nhất** — token đặc trưng riêng cho cụm này
3. **Thống kê hành vi** — độ dài, tỉ lệ quay lui, tỉ lệ quay lại, tỉ lệ vòng lặp

Bằng chứng được lưu lại trong trường `note`, nên **mọi cái tên đều có thể bị
chất vấn** thay vì phải tin.

Tổng cộng **169 cụm** đã được đặt tên, và được gom về **13 nhóm (family)** —
đây mới là mức nên đưa vào slide:

| Nhóm | Android (lộ trình A) | iOS (lộ trình A) |
|---|---|---|
| payment | **264 (14,7%)** | 226 (5,6%) |
| auth | 209 (11,6%) | 357 (8,8%) |
| contract | 158 (8,8%) | 324 (8,0%) |
| service | 119 (6,6%) | **365 (9,0%)** |
| support | 92 (5,1%) | 167 (4,1%) |
| econtract | 66 (3,7%) | 184 (4,5%) |
| account | 70 (3,9%) | 114 (2,8%) |
| shop | — | 222 (5,5%) |
| guest | 21 (1,2%) | 195 (4,8%) |
| *(nhiễu)* | *694 (38,7%)* | *1.588 (39,1%)* |

**Khác biệt nền tảng trong một câu:** nhóm lớn nhất của Android là **thanh toán**
(14,7%); của iOS là **quản lý dịch vụ** (9,0%) với **auth** và **hợp đồng** bám
sát. Trong tuần dữ liệu này, người dùng Android vào app để trả hóa đơn; người
dùng iOS vào để quản lý gói Internet và chuyển hợp đồng.

### Nguyên mẫu thực tế trông như thế nào

Android (lộ trình A) — tám cụm lớn nhất:

| # | n | dài | thời gian | quay lui | vòng lặp | Đây là gì |
|---|---|---|---|---|---|---|
| 25 | 129 | 11 | 42 s | 0,05 | 16% | **Thanh toán hóa đơn (khách vãng lai)** |
| 8 | 121 | 7 | 8 s | 0,12 | 2% | **Chuyển hợp đồng** — nhanh, cơ học |
| 19 | 90 | 21 | 88 s | 0,16 | **26%** | **Nâng gói Internet (web)** — dài, đi vòng nhiều. *Ứng viên ma sát* |
| 26 | 75 | 8 | 35 s | 0,05 | 11% | **Thanh toán hóa đơn trả sau** |
| 18 | 60 | 7 | 7 s | 0,09 | 8% | **Vào trả trước rồi thoát** — ngắn, bỏ dở |
| 20 | 47 | 24 | 126 s | 0,10 | 19% | **Ký hợp đồng điện tử qua VNeID** |
| 1 | 44 | 4 | 13 s | **0,00** | **0%** | **Đăng nhập OAuth** — không quay lui, không lặp: sạch |
| 24 | 44 | 6 | 7 s | 0,18 | 2% | **Mở rồi đóng yêu cầu hỗ trợ** |

**Xếp hạng ma sát — kết quả có thể hành động ngay.** Sắp theo tỉ lệ quay lại và
tỉ lệ vòng lặp *xét cùng nhau*:

| Nền tảng | Tên cụm | n | dài | quay lại | vòng lặp |
|---|---|---|---|---|---|
| iOS | **E-contract list thrash** | 17 | 38 | 0,64 | 53% |
| iOS | **Package/product webview thrash** | 28 | 18 | 0,57 | 68% |
| iOS | **Pay on behalf** (trả hộ) | 27 | 20 | 0,52 | 56% |
| iOS | **E-contract identity input** | **81** | 17 | 0,39 | **52%** |
| Android | **Change address (e-counter)** | 15 | 34 | 0,49 | 47% |
| iOS | **Internet package upgrade (web)** | **188** | 21 | 0,44 | 20% |
| Android | **Internet package upgrade (web)** | 90 | 21 | 0,41 | 26% |

Hai kết luận:

1. **Các luồng webview thống trị danh sách ma sát.** `update-package`,
   `web/shop`, và host PDF hợp đồng điện tử xuất hiện trong gần như mọi cụm ma
   sát cao trên cả hai nền tảng. Các màn hình native sạch hơn hẳn.
2. **iOS #39 "E-contract identity input" là cái cần leo thang trước** — 81 hành
   trình, hơn một nửa chứa vòng lặp điều hướng, lớn gấp ba lần cụm ma sát cao
   kế tiếp.

Để công bằng: các cụm **sạch nhất** là các luồng đăng nhập. Android "OAuth
login" và iOS "OTP login" đều có tỉ lệ quay lui **bằng 0**. Đăng nhập không phải
chỗ có vấn đề.

---

## 7. Chạy trên dữ liệu mới — từ mô tả sang giám sát

Sáu phần trên **mô tả** dữ liệu quá khứ. Phần này là chỗ nó trở thành một hệ
thống chạy được.

### 7.1 Nguyên tắc: không huấn luyện lại

Đối tượng `JourneyScorer` đóng gói **toàn bộ trạng thái đã khớp**: từ vựng
TF-IDF, cơ sở SVD, bộ chuẩn hóa số, tâm cụm, các Markov chain, và các ngưỡng đã
hiệu chỉnh.

Khi có dữ liệu mới, nó **tự chạy lại đúng pipeline đó** — chuẩn hóa → token hóa
→ cắt hành trình → làm sạch → vector hóa — chứ không phải một bản sao chép tay.
Không có gì được khớp lại. Nhờ vậy, huấn luyện và suy luận **không thể trôi lệch
khỏi nhau**.

```bash
python scripts/score_partitioned_events.py --platform android --input data/new_events.csv
```

### 7.2 Kết quả trả về

Một dòng cho mỗi hành trình:

| Trường | Ý nghĩa |
|---|---|
| `cluster` | Nguyên mẫu được gán (hoặc −1) |
| `distance_to_centroid` | Cách tâm cụm bao xa |
| `markov_logprob` | Chuỗi này "hợp lý" đến đâu dưới nguyên mẫu của nó |
| `geometric_anomaly` | Bất thường **hình dạng** |
| `generative_anomaly` | Bất thường **chuyển tiếp** |
| `severe_anomaly` | Cả hai |
| `friction_flags` | Lý do bằng chữ, không phải điểm số |
| `next_action` / `next_action_share` | Bước tiếp theo dự đoán + tỉ lệ |

### 7.3 Hai kênh bất thường — cố ý giữ riêng

| Kênh | Câu hỏi nó trả lời | Cách tính |
|---|---|---|
| **Hình học (Geometric)** | *"Hành trình này có **hình dạng** lạ không?"* | Xa mọi tâm cụm, vượt ngưỡng p95 của tập huấn luyện |
| **Sinh mẫu (Generative)** | *"Các **bước chuyển bên trong** nó có bất thường không?"* | Log-xác suất Markov thấp hơn ngưỡng p05 |

Ví dụ để thấy tại sao phải tách:

- Một hành trình có hình dạng hoàn toàn bình thường (đúng số bước, đúng loại màn
  hình) nhưng đi qua một cặp chuyển tiếp mà **chưa ai từng đi** → chỉ kênh sinh
  mẫu bắt được.
- Một hành trình dài bất thường với các bước chuyển đều rất phổ biến → chỉ kênh
  hình học bắt được.

Hai kênh bất đồng đủ thường xuyên để việc **trộn chúng thành một con số duy nhất
sẽ hủy mất thông tin**. `severe_anomaly` = cả hai cùng bắn.

### 7.4 Kết quả kiểm chứng

Đo trên **ngày cuối cùng được giữ lại khỏi tập huấn luyện**:

| | Android | iOS |
|---|---|---|
| Hành trình được chấm | 219 | 642 |
| **Khớp một nguyên mẫu đã biết** | **95,9%** | **93,3%** |
| Bất thường hình học | 4,1% | 6,7% |
| Bất thường sinh mẫu | 20,5% | 18,4% |
| Nghiêm trọng (cả hai) | 0 | 1 |

**93–96% hành trình chưa từng thấy rơi vào một nguyên mẫu đã biết.** Đây là con
số nói rằng các cụm **tổng quát hóa được**, chứ không phải học thuộc lòng dữ
liệu cũ.

### 7.5 Cờ ma sát — giải thích bằng chữ, không bằng điểm

`friction_flags` trả lời câu hỏi *"vì sao hành trình này trông xấu?"* theo cách
người đọc được ngay:

`excessive_back` · `navigation_loop` · `screen_thrash` · `slow_journey` ·
`unknown_archetype` · `improbable_transitions`

Trên tập holdout Android, phổ biến nhất: `improbable_transitions` (45),
`screen_thrash` (27), `navigation_loop` (21), `excessive_back` (12),
`slow_journey` (10).

Đây chính là đầu vào cho **can thiệp chủ động**: khi một hành trình đang chạy bật
lên 2–3 cờ, đó là lúc bật gợi ý hoặc mời chat hỗ trợ — trước khi người dùng bỏ
cuộc.

### 7.6 Dự đoán bước tiếp theo

Cùng một Markov chain đã dùng để chấm bất thường, nhưng đọc **xuôi chiều**:

```
Đang ở    : Action@iOS::/dkol/update-package/home#webHeader/BackButton
  →  40,0%  View@iOS::staging-hi.fpt.vn/dkol/update-package/home
  →  20,0%  Action@iOS::.../product-management-v920?cat_id=3#web/LocationButton
  →  20,0%  View@iOS::nointent_pop-up
```

**Điều kiện hóa theo cụm là thứ khiến việc này hữu ích.** `P(bước sau | màn hình
hiện tại)` rất khác nhau giữa "đang trả hóa đơn" và "đang tìm hỗ trợ". Một
Markov chain toàn cục sẽ trộn chúng thành thứ vô nghĩa; một chain riêng cho mỗi
nguyên mẫu thì không.

Hai giới hạn cần nói rõ:

- **Bậc một** — chỉ token cuối cùng quyết định dự đoán. Đủ để đoán màn hình kế
  tiếp, **không đủ** để suy luận ý định nhiều bước.
- Có hai loại xác suất trong đầu ra. `smoothed_probability` là thứ dùng cho điểm
  bất thường, nhưng với ~700 token thì khối lượng làm mịn lấn át mẫu số và ép mọi
  giá trị về gần 0. **Khi trình bày cho người, hãy dùng `observed_share`** — nó
  là câu nói thẳng: *"trong số người đến màn hình này, 40% đi tiếp sang X"*.

### 7.7 Vòng khép kín cho sản phẩm

```
Sự kiện mới đến
      ↓
Chấm theo nguyên mẫu  ──→  Khớp?  ──Không──→  Bất thường: hành vi chưa từng thấy
      ↓ Có                                     (hoặc app vừa đổi luồng — cần kiểm tra)
Kiểm tra cờ ma sát
      ↓
Có cờ?  ──Có──→  "Người dùng đang mắc kẹt ở bước X của luồng Y"
      ↓ Không            → gợi ý chủ động / mời hỗ trợ / ghi nhận cho UX review
Dự đoán bước tiếp theo
      ↓
"Người dùng có 40% khả năng đi sang màn hình Z"  → gợi ý lối tắt, preload, cá nhân hóa
```

### 7.8 Khi nào phải huấn luyện lại

Không có gì ở đây tự thích nghi trực tuyến. Hai điều kiện kích hoạt huấn luyện
lại:

1. App phát hành thay đổi điều hướng (màn hình mới, luồng mới).
2. Tỉ lệ **"khớp nguyên mẫu đã biết"** trên dữ liệu mới tụt xuống dưới ~85%.

Chính tỉ lệ đó là **đồng hồ đo trôi dạt (drift monitor)** của hệ thống.

---

## 8. Chọn lộ trình nào, và những giới hạn cần biết

### Khuyến nghị: **Triển khai lộ trình A, giám sát bằng lộ trình B**

**Lộ trình A (TF-IDF)** là mặc định trong scorer đã khớp, vì ba lý do vận hành:
nó **đặc** (mọi hành trình đều có biểu diễn), nó **suy giảm êm** khi gặp dữ liệu
lạ, và nó **không cần bảng mẫu** lúc suy luận. Lộ trình B để 9–21% hành trình
hoàn toàn không có bằng chứng — đó là rủi ro trong môi trường sản xuất.

**Lộ trình B (PrefixSpan)** xứng đáng có mặt như **lớp diễn giải và kiểm chứng**:

- Mẫu của nó đọc được nguyên văn — `HOME → payment → chọn hóa đơn → thanh toán`
  là một câu; một thành phần SVD thì không.
- Nó cho một baseline nguyên mẫu **không cần clustering** để đối chiếu.
- ARI giữa hai lộ trình (0,51–0,58 trên phần đã gán cụm) là phép kiểm tra tính
  hợp lệ cấu trúc mà không silhouette đơn lẻ nào cung cấp được.

### Những giới hạn phải nói rõ

Nêu thẳng, vì đây là giới hạn của **dữ liệu**, không phải của phương pháp:

1. **Mẫu nhỏ và nhiều khả năng không phải lưu lượng sản xuất.** 154 khách hàng,
   182 thiết bị, 1.425 phiên trong 7 ngày; **13.357 sự kiện nằm trên
   `staging-hi.fpt.vn`**. Danh mục cụm này mô tả *đúng* lưu lượng này. Coi nó là
   phân loại hành trình sản xuất, hoặc hiệu chỉnh ngưỡng bất thường ở đây rồi
   đem đi triển khai, thì **chưa an toàn** — lưu lượng QA có hình dạng khác
   người dùng thật.

2. **43% hành trình bị loại trước khi phân cụm** (4.478 / 10.262) vì dưới 4
   token sau khi làm sạch. Chúng chủ yếu là các bước nhảy vặt
   (`Home → chạm → màn hình`) không còn thông tin thứ tự để phân cụm. Chúng vẫn
   được ghi ra đĩa, nhưng **danh mục cụm chỉ mô tả các hành trình thực chất**.

3. **iOS phân mảnh quá mức ở một số nhóm.** Lộ trình A tìm ra 9 cụm auth trên
   iOS, trong đó **7 là các biến thể của "đăng xuất"** — 158 hành trình bị chia
   7 cách theo những khác biệt không ai đi báo cáo. Đây là phát hiện **về tham
   số, không phải về người dùng**: với 4.063 hành trình,
   `min_cluster_size = 15` là quá nhỏ. Hai lựa chọn — báo cáo ở **mức nhóm
   (family)** và coi id cụm là chi tiết, hoặc khớp lại iOS với
   `min_cluster_size` 30–40.

4. **`screen_name` NULL 100% trên các dòng View là vấn đề nên nêu với đội
   mobile.** Mọi thứ ở trên đã đi vòng qua nó, nhưng nếu sự kiện View cũng mang
   màn hình trong `screen_name`, sự nhập nhằng vai trò sẽ biến mất **ngay tại
   nguồn** — và pipeline đơn giản đi một bậc.

### Các bước tiếp theo, theo thứ tự ưu tiên

1. **Lấy thêm dữ liệu** — lý tưởng là 4+ tuần lưu lượng sản xuất. Gần như mọi
   cảnh báo ở trên đều là vấn đề kích thước mẫu.
2. **Để đội sản phẩm đọc và xác nhận** 20 đường đi medoid hàng đầu. Nguyên mẫu
   được **người** đặt tên là thứ biến cái này từ một phép phân cụm thành một hệ
   thống giám sát.
3. **Chạy `--entropy`** và đối chiếu ranh giới của nó với ranh giới rule-based —
   mỗi chỗ bất đồng là một luật còn thiếu hoặc một ngưỡng đặt sai.
4. **Chỉ sau đó** mới hiệu chỉnh ngưỡng bất thường, trên dữ liệu sản xuất, với
   phép chia theo thời gian.

---

## Phụ lục — Tóm tắt một trang

| Bước | Vào | Ra | Quyết định then chốt |
|---|---|---|---|
| **1. Khám phá** | 114.534 log thô | Hiểu cấu trúc | Vai trò trường đảo theo `event_type`; `session_id` ≠ phiên |
| **2. Chuẩn hóa** | Log thô | Bộ ba `(type, screen, target)` | Bỏ định danh, giữ hành vi → từ vựng ↓34% |
| **3. Token hóa** | Bộ ba chuẩn | Token L1/L2/L3 | Dùng L2; đếm tần suất theo *hành trình* |
| **4. Cắt hành trình** | Chuỗi theo phiên | 5.858 hành trình | Rule-based (chịu lực) + PrefixSpan (kiểm chứng); giữ số vòng lặp |
| **5. Vector hóa** | Chuỗi token | 74 chiều | TF-IDF n-gram(1–4) → SVD 64 + 10 đặc trưng × 0,35 |
| **6. Phân cụm** | Ma trận 74 chiều | 29–56 cụm + nhiễu | HDBSCAN vì nó **được phép từ chối** gán nhãn |
| **7. Suy luận** | Sự kiện mới | Cụm + bất thường + bước kế | Hai kênh bất thường tách riêng; Markov theo từng cụm |

**Một câu tóm tắt toàn bộ giải pháp:**

> Chuẩn hóa log thô để loại định danh mà giữ hành vi, cắt phiên thành các hành
> trình có mục đích, mã hóa chúng thành vector giữ được cả *thứ tự* lẫn *cách đi*,
> để thuật toán mật độ tự tìm ra nguyên mẫu và **tự từ chối** những gì không
> giống nguyên mẫu nào — rồi dùng chính sự từ chối đó làm tín hiệu phát hiện
> người dùng đang gặp khó khăn.

---

### Tài liệu chi tiết

| Tài liệu | Nội dung |
|---|---|
| [1_Data_Exploration.md](1_Data_Exploration.md) | Thống kê dữ liệu, kiểu dữ liệu, giá trị thiếu, timestamp |
| [2_Session_Based_EDA.md](2_Session_Based_EDA.md) | Phân bố phiên, ngưỡng tách phiên |
| [3_Exact_Tokenization_and_Sequences.md](3_Exact_Tokenization_and_Sequences.md) | Từ điển token, chuỗi mẫu |
| [4_Clustering_Two_Routes.md](4_Clustering_Two_Routes.md) | Chi tiết hai lộ trình, sweep tham số, suy luận |
| [5_Cluster_Names.md](5_Cluster_Names.md) | Toàn bộ 169 tên cụm + bằng chứng |
| [Journey_Approach_Validation.md](Journey_Approach_Validation.md) | Đối chiếu các thuật toán ứng viên, lý do chọn |
