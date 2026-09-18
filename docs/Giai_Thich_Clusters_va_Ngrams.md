# Hướng dẫn kiểm tra và đặt tên Cluster

> Tài liệu dành cho người am hiểu nghiệp vụ, không yêu cầu kiến thức kỹ thuật.  
> Mục tiêu: giúp anh/chị hiểu kết quả, kiểm tra tên hệ thống gợi ý và sửa lại tên cluster khi cần.

## 1. Dữ liệu ban đầu: những click thô

Ứng dụng ghi lại từng việc người dùng làm, ví dụ:

```text
Mở Trang chủ → Bấm Thanh toán → Chọn hóa đơn → Bấm Xác nhận
```

Mỗi thao tác là một dòng **click thô**. Dòng log có thể chứa tên kỹ thuật như `PaymentHomeVC`, đường dẫn màn hình hoặc tên nút bấm. Một dòng đứng riêng thường chưa cho biết người dùng muốn làm gì.

Hãy hình dung click thô giống như từng dấu chân. Ta cần nối nhiều dấu chân theo đúng thứ tự mới thấy người dùng đang đi đâu.

## 2. Hệ thống làm gì với dữ liệu?

Hệ thống xử lý theo một chuỗi bước đơn giản:

```text
Click thô
  → làm sạch tên màn hình và nút bấm
  → nối các click thành một hành trình
  → biến hành trình thành các mẫu chuyển bước
  → gom hành trình giống nhau thành cluster
  → gợi ý tên cho từng cluster
```

Hiểu nôm na:

1. **Làm sạch:** giảm bớt ID, mã động và chi tiết gây nhiễu.
2. **Tạo journey:** ghép các click liên tiếp thành một chuyến đi ngắn của người dùng.
3. **Tìm mẫu:** xem những bước nào thường xuất hiện và đi liền nhau.
4. **Tạo cluster:** gom các journey có “dáng đi” tương tự vào cùng một nhóm.
5. **Đặt tên:** đọc các bằng chứng nổi bật để đoán mục đích nghiệp vụ của nhóm.

Máy làm tốt việc tìm mẫu lặp lại. Người nghiệp vụ làm tốt hơn việc xác nhận **mục đích thật sự** của người dùng. Vì vậy bước kiểm tra tên thủ công rất quan trọng.

## 3. Output chính nói cho ta điều gì?

| Output | Hiểu đơn giản | Cách sử dụng |
|---|---|---|
| **Journey / sequence** | Chuỗi sự kiện của một lần sử dụng | Đọc để hiểu người dùng đã đi qua những bước nào |
| **Cluster ID** | Số nhóm như `0`, `1`, `2` | Chỉ là mã kỹ thuật, không tự mang ý nghĩa nghiệp vụ |
| **Cluster size / mass** | Nhóm lớn hay nhỏ | Biết mẫu hành vi này phổ biến đến đâu |
| **Representative journey / medoid** | Một hành trình tiêu biểu của nhóm | Xem ví dụ đầy đủ, gần với “trung tâm” cluster |
| **Top n-grams** | Những đoạn chuyển bước đặc trưng | Bằng chứng chính để hiểu và đặt tên cluster |
| **Cluster name** | Tên nghiệp vụ được gợi ý | Cần người nghiệp vụ kiểm tra và có thể sửa |
| **Confidence** | Mức độ chắc chắn của tên gợi ý | Thấp/medium cần xem kỹ hơn; high vẫn nên kiểm tra |
| **Cluster `-1` / noise** | Các journey chưa thuộc nhóm ổn định | Có thể giữ “Chưa phân loại”, không nên ép tên |

### Các cluster liên quan với nhau như thế nào?

- Hai cluster có thể cùng thuộc **Thanh toán**, nhưng một nhóm là **Thanh toán hóa đơn**, nhóm kia là **Xem lịch sử thanh toán**.
- Hai cluster có vài màn hình giống nhau chưa chắc cùng mục đích. Thứ tự chuyển bước mới giúp phân biệt.
- Cluster ID không phải thứ hạng. Cluster `2` không tốt hơn cluster `8`.

### Khi chạy inference trên dữ liệu mới

Với một journey mới, hệ thống sẽ:

```text
Journey mới → so với các mẫu đã học → gán cluster gần nhất → lấy tên đã được duyệt
```

Kết quả cuối cùng có thể hiểu là: **“Journey này giống nhất với nhóm nào, và nhóm đó đang được business gọi tên là gì?”** Nếu journey quá khác các nhóm đã biết, hệ thống có thể đánh dấu là bất thường hoặc chưa phân loại.

> Tên cluster đã được duyệt phải được lưu cùng đúng phiên bản model, vì Cluster ID có thể đổi khi huấn luyện lại.

## 4. Vì sao đặt tên cluster là phần khó nhất?

Máy nhìn thấy rất nhiều token kỹ thuật, ví dụ:

```text
view@PaymentHomeVC
action@android/Home#Home/Nav_payement
view@android/Home/Payment/payment_history
```

Người trong nghiệp vụ có thể nhận ra đây là khu vực Thanh toán. Nhưng LLM hoặc người ngoài dự án có thể gặp khó vì:

- tên màn hình giữa Android và iOS khác nhau;
- token chứa đường dẫn dài, mã kỹ thuật hoặc từ viết tắt;
- một cluster có cả màn hình chung như Trang chủ, popup, đăng nhập;
- cùng một màn hình có thể xuất hiện trong nhiều mục đích khác nhau;
- hàng trăm token rải rác dễ làm bằng chứng chính bị chìm.

Vì vậy, tên do hệ thống hoặc LLM tạo ra chỉ là **gợi ý ban đầu**. Người nghiệp vụ cần trả lời:

> “Nhìn toàn bộ luồng này, người dùng thực sự đang muốn làm việc gì?”

## 5. N-gram, bigram và cluster naming

### N-gram là gì?

N-gram là một đoạn gồm **N sự kiện đứng liền nhau**.

Với journey:

```text
Trang chủ → Thanh toán → Chọn hóa đơn → Xác nhận
```

| Loại | Ví dụ | Ý nghĩa |
|---|---|---|
| 1-gram | `Thanh toán` | Biết một bước có xuất hiện |
| **Bigram (2-gram)** | `Thanh toán → Chọn hóa đơn` | Biết người dùng chuyển từ bước nào sang bước nào |
| 3-gram | `Thanh toán → Chọn hóa đơn → Xác nhận` | Mô tả một đoạn dài và cụ thể hơn |

### Vì sao bigram đặc biệt hữu ích?

Bigram giống một cụm hai từ: ngắn nhưng đã có ngữ cảnh.

- `Thanh toán` đứng riêng còn khá chung chung.
- `Thanh toán → Chọn hóa đơn` cho thấy hướng đi rõ hơn.
- `Lịch sử thanh toán → Chi tiết giao dịch` gợi ý mục đích xem lại giao dịch.

Bigram giữ được **thứ tự**. `Trang chủ → Thanh toán` có nghĩa là đi vào Thanh toán; `Thanh toán → Trang chủ` có thể là quay ra.

Chuỗi 3-4 bước cung cấp thêm chi tiết nhưng thường ít lặp lại giống hệt. Vì vậy bigram thường là điểm cân bằng tốt giữa **dễ hiểu**, **đủ phổ biến** và **có ý nghĩa**. Hệ thống vẫn có thể xem thêm 1-gram và 3-gram; không đặt tên chỉ từ một bigram duy nhất.

### Bigram giúp đặt tên cluster như thế nào?

Nếu nhiều journey trong cùng cluster thường có:

```text
Thanh toán → Lịch sử thanh toán
Lịch sử thanh toán → Chi tiết giao dịch
```

thì đây là bằng chứng mạnh rằng cluster liên quan đến **Xem lịch sử thanh toán**.

Ta nên chọn tên dựa trên phần giao nhau của nhiều bằng chứng, không chọn theo một token hiếm hoặc một màn hình chung như Trang chủ.

## 6. Cách đặt tên hiện tại và trường hợp confidence thấp

Hệ thống hiện đọc kết hợp:

1. **Top n-grams:** các đoạn chuyển bước đặc trưng nhất.
2. **Representative journey:** một journey tiêu biểu để thấy toàn bộ câu chuyện.
3. **Nhóm nghiệp vụ:** ví dụ Thanh toán, Hỗ trợ, Quản lý dịch vụ, Tài khoản.
4. **Mức đồng thuận:** các bằng chứng có cùng chỉ về một mục đích hay không.

Tên thường có cấu trúc:

```text
Nhóm nghiệp vụ | Mục đích chính | Chi tiết (nếu đủ chắc chắn)
```

```text
Thanh toán | Lịch sử thanh toán | Xem danh sách và chi tiết lịch sử
```

### Khi nào confidence không cao?

Confidence thấp hoặc medium thường xảy ra khi:

- top n-grams nói về nhiều mục đích khác nhau;
- popup, Trang chủ, đăng nhập hoặc token chung chiếm quá nhiều;
- journey tiêu biểu không khớp rõ với top n-grams;
- cluster chứa cả Thanh toán, Loyalty và Quản lý dịch vụ;
- chỉ có một bằng chứng nghiệp vụ rõ, còn lại là nhiễu.

Ví dụ cluster lộn xộn:

```text
Trang chủ → Thanh toán
Trang chủ → Loyalty
Popup → Quản lý thiết bị
```

Không nên cố chọn một tên rất cụ thể. Có thể:

- chọn tên rộng hơn, ví dụ **“Trang chủ - Điều hướng chức năng”**;
- ghi chú **“Cần kiểm tra thêm”**;
- hoặc giữ **“Chưa phân loại / hành trình hỗn hợp”**.

> Confidence thấp không có nghĩa model chắc chắn sai. Nó có nghĩa bằng chứng hiện tại chưa kể cùng một câu chuyện, nên cần người nghiệp vụ quyết định.

## 7. Ví dụ để người nghiệp vụ tự đặt tên

Giả sử hệ thống đưa ra cluster `12`:

```text
Representative journey:
Trang chủ → Thanh toán → Lịch sử thanh toán → Chi tiết giao dịch

Top bigrams:
1. Thanh toán → Lịch sử thanh toán
2. Lịch sử thanh toán → Chi tiết giao dịch
3. Trang chủ → Thanh toán
```

Hãy đặt tên theo 3 câu hỏi:

1. **Khu vực nào?** - Thanh toán.
2. **Người dùng muốn làm gì?** - Xem lại lịch sử thanh toán.
3. **Chi tiết nào lặp lại đủ rõ?** - Xem danh sách và mở chi tiết giao dịch.

Tên đề xuất:

```text
Thanh toán | Lịch sử thanh toán | Xem danh sách và chi tiết giao dịch
```

### Checklist trước khi duyệt tên

- [ ] Tên có nói được **mục đích của người dùng** không?
- [ ] Ít nhất 2-3 bằng chứng có cùng ủng hộ tên này không?
- [ ] Representative journey có hợp với tên không?
- [ ] Tên có tránh dựa vào Trang chủ, popup hoặc đăng nhập chung chung không?
- [ ] Người không xem token kỹ thuật có hiểu tên không?
- [ ] Nếu bằng chứng lộn xộn, tên đã đủ rộng hoặc được đánh dấu cần kiểm tra chưa?

## 8. Bộ file người nghiệp vụ sẽ nhận

Các file có vai trò khác nhau, nhưng người nghiệp vụ **không cần sửa tất cả**.

```text
ios_cluster_catalog.json  ─┐
                           ├→ bằng chứng để hiểu cluster
ios_cluster_ngrams.csv    ─┘
                                  ↓
ios_scores_named.shareholder_catalog.json
                                  ↓
ios_scores_named.cluster_mapping.csv  ← file cần mở bằng Excel và chỉnh tên
```

### 8.1 `ios_cluster_catalog.json` - cluster thô với ID

File này là “hồ sơ gốc” do model tạo ra cho từng cluster. Mỗi khối JSON tương ứng với một `cluster`.

| Trường | Hiểu đơn giản |
|---|---|
| `cluster` | ID của cluster, dùng để nối với các file khác |
| `size` | Số journey trong cluster |
| `share` | Tỷ lệ cluster trong toàn bộ dữ liệu |
| `median_length` | Một journey điển hình có khoảng bao nhiêu sự kiện |
| `top_entry_token` | Sự kiện thường bắt đầu journey |
| `top_exit_token` | Sự kiện thường kết thúc journey |
| `medoid_path` | Journey tiêu biểu nhất của cluster |
| `label_kind` | `cluster` là nhóm bình thường; `noise` là nhóm hỗn hợp `-1` |

Ví dụ dễ hiểu:

```text
cluster: 738
size: 3.621 journey
medoid_path:
Trang chủ → Popup → đóng Popup
```

Ta mới biết **cluster 738 thường là một luồng popup từ Trang chủ**. Đây vẫn là dữ liệu thô, chưa phải tên nghiệp vụ cuối cùng.

### 8.2 `ios_cluster_ngrams.csv` - các mẫu đặc trưng của từng cluster

Mỗi dòng là một n-gram nổi bật mà model học được cho một cluster.

| Cột | Hiểu đơn giản |
|---|---|
| `cluster` | ID để ghép với cluster catalog và mapping |
| `rank` | Thứ tự của n-gram trong danh sách bằng chứng |
| `ngram` | Một hoặc nhiều sự kiện đi liền nhau |
| `lift` | Mẫu này đặc trưng cho cluster đến mức nào so với cluster khác |
| `cluster_mass` | Mẫu này có sức nặng/độ phổ biến trong cluster đến đâu |

Ví dụ:

```text
cluster = 0
ngram = tab thiết bị đang chặn → tải lại danh sách
```

Nếu nhiều n-gram của cluster `0` đều nói về **thiết bị kết nối, thiết bị bị chặn và tải lại danh sách**, ta có cơ sở đặt tên liên quan đến **Quản lý thiết bị**.

Không nên chỉ chọn n-gram có `lift` cao nhất. Một mẫu có thể rất đặc trưng nhưng xuất hiện ít. Hãy đọc vài dòng đầu, ưu tiên các mẫu có câu chuyện giống nhau và đối chiếu với `medoid_path`.

### 8.3 `ios_scores_named.shareholder_catalog.json` - hồ sơ đã được diễn giải

File này gom dữ liệu thô và kết quả đặt tên thành một hồ sơ đầy đủ cho từng cluster. Nó phục vụ hệ thống, dashboard hoặc kiểm tra chi tiết.

Các trường dễ đọc nhất:

- `cluster_id`: ID cluster.
- `mapping_name`: tên hệ thống đang đề xuất.
- `business_family`, `business_submodule`, `business_detail`: tên theo từng cấp nghiệp vụ.
- `confidence`: mức chắc chắn của tên.
- `reasoning`: lý do hệ thống chọn tên.
- `supporting_evidence`: các bằng chứng ủng hộ.
- `alternative_submodules`: những tên khác cũng có khả năng đúng.
- `top_mass_ngrams`: các n-gram quan trọng.
- `medoid_path_support_only`: journey tiêu biểu dùng để tham khảo.

File JSON này nhiều thông tin nhưng không thuận tiện để chỉnh bằng tay. Hãy dùng nó như **hồ sơ tham khảo**, không phải file Excel cần bàn giao lại.

### 8.4 `ios_scores_named.cluster_mapping.csv` - file cần chỉnh bằng Excel

Đây là **đầu ra chính dành cho người nghiệp vụ**. Mỗi dòng tương ứng với một Cluster ID. Có thể mở trực tiếp bằng Excel, lọc theo `needs_review` hoặc `naming_confidence`, sửa tên rồi gửi lại.

Các cột quan trọng khi review:

| Cột | Người nghiệp vụ cần làm gì? |
|---|---|
| `cluster_id` | Giữ nguyên; đây là khóa nối với kết quả inference |
| `cluster_name` | Kiểm tra và sửa thành tên đầy đủ, dễ hiểu |
| `business_family` | Chọn nhóm lớn như Thanh toán, Hỗ trợ, Quản lý dịch vụ |
| `business_submodule` | Chọn mục đích cụ thể hơn |
| `business_detail` | Chỉ điền khi bằng chứng đủ rõ |
| `naming_confidence` | Ưu tiên review `low`, sau đó `medium` |
| `needs_review` | `true` nghĩa là hệ thống đề nghị con người kiểm tra |
| `reasoning` | Đọc lý do tên hiện tại được chọn |
| `top_ngrams` | Đọc các mẫu chuyển bước làm bằng chứng |
| `medoid_path` | Đọc một journey tiêu biểu để kiểm tra lại |

### Quy trình review trên Excel

1. Mở `ios_scores_named.cluster_mapping.csv` bằng Excel.
2. Lọc `needs_review = true` hoặc `naming_confidence = low`.
3. Với từng `cluster_id`, đọc `medoid_path` và 3-5 mẫu đầu trong `top_ngrams`.
4. Nếu cần thêm bối cảnh, tra cùng ID trong `ios_cluster_catalog.json` và `ios_cluster_ngrams.csv`.
5. Sửa `cluster_name` và các cột nghiệp vụ; **không sửa `cluster_id`**.
6. Nếu chưa đủ bằng chứng, đặt tên rộng hoặc giữ “Chưa phân loại / hành trình hỗn hợp”.
7. Lưu lại dưới định dạng CSV UTF-8 để giữ đúng tiếng Việt.

Ví dụ một dòng sau khi được duyệt:

```text
cluster_id: 738
cluster_name: Home | Popup thông báo
business_family: Home
business_submodule: Popup thông báo
business_detail: Đóng popup từ Trang chủ
```

> Tóm lại: ba file catalog/n-gram cung cấp **bằng chứng**; file `ios_scores_named.cluster_mapping.csv` là nơi người nghiệp vụ **chốt tên theo ID** và gửi lại cho đội kỹ thuật.

---

### Chỉnh sửa và xuất PDF

Chỉnh trực tiếp file Markdown này, sau đó chạy từ thư mục gốc của repo:

```bash
python scripts/export_cluster_ngram_doc.py
```

PDF được tạo tại `output/pdf/Giai_Thich_Clusters_va_Ngrams.pdf`.
