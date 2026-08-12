# Android clickstream simulator

MVP Android Studio app để replay clickstream từ `app/src/main/assets/test_data_android.csv`.

## Chức năng

- Đọc raw CSV `test_data_android.csv`; vẫn hỗ trợ Extended JSON có dạng `_id.$oid` và `created_at.$date`.
- Group theo `session_id` và sort theo `timestamp`, giữ thứ tự gốc khi timestamp bằng nhau.
- Chọn session, tốc độ `0.25x / 1x / 5x / 10x`.
- `Start`, `Pause`, `Stop`, progress và event log trực tiếp trên màn hình.
- Giữ nguyên `customer_id = null` để mô phỏng trạng thái anonymous trước đăng nhập.
- Chạy model local bằng ONNX + Markov: input JSON events, output cluster, friction flags và next action.

## Chạy bằng Android Studio

1. Mở thư mục `mobile/android` (không mở riêng thư mục `app`).
2. Chờ Gradle Sync hoàn tất.
3. Chọn emulator Android API 26 trở lên.
4. Run configuration `app`.

File dữ liệu đã được copy vào assets khi tạo project. Nếu thay data nguồn, copy lại:

```text
data/test_data/test_data_android.csv
→ mobile/android/app/src/main/assets/test_data_android.csv
```

MVP này phát event cục bộ trong app ngay sau khi bấm `Start`. Có hai route để so sánh:

- `Fixed duration`: gom event theo event timestamp và phát kết quả sau mỗi duration (mặc định
  30 giây). Journey đang mở được giữ xuyên qua các window, không bị cắt giả tại mốc thời gian.
- `Journey complete`: mỗi khi event mới tạo boundary, journey trước boundary được finalize và
  chạy preprocessing + ONNX + Markov ngay lập tức. Các boundary hiện có là session change,
  idle gap 90 giây, root return, auth change và length cap 80 event. Khi replay kết thúc, buffer
  cuối được finalize bằng `finish()`.

Simulator cho phép chọn route và duration trên màn hình trước khi bấm `Start`. Duration chỉ có
ý nghĩa với route `Fixed duration`; với route `Journey complete`, model chạy theo boundary.

Nút `Analyze rule-based journeys` dùng cùng bộ rule để phân tích toàn bộ session được chọn,
phục vụ đối chiếu với kết quả streaming.

## Mobile inference module

`MobileJourneyModel` là entry point:

```kotlin
val model = MobileJourneyModel(context)
val result = model.analyzeJson(jsonText)
val outputJson = result.toJson().toString()
model.close()
```

`MobileJourneyStream` dùng cho replay từng event:

```kotlin
val stream = MobileJourneyStream(model)
val update = stream.append(event, scoreProvisional = false)
val completedJourneys = update.finalized
val finalPrediction = stream.flush()
```

Code production nên dùng facade `MobileClickstreamProcessor`, không gọi `analyzeEvents` cho từng
event:

```kotlin
val processor = MobileClickstreamProcessor(
    model = MobileJourneyModel(context),
    route = MobileProcessingRoute.JOURNEY_COMPLETE,
    windowDurationSeconds = 30L
)
processor.start()

// Gọi từ callback clickstream, trên background executor.
processor.accept(event).let { update ->
    if (update.emitted) sendToUiOrBackend(update.toJson().toString())
}

// Gọi khi logout/session stream kết thúc/app bị stop.
processor.finish()?.let { sendToUiOrBackend(it.toJson().toString()) }
```

Nếu callback đang trả raw JSON, dùng `processor.acceptJson(rawJson)`. Hàm này nhận một event JSON
hoặc một JSON array và trả về các update tương ứng.

Module thực hiện canonize, journey segmentation, consecutive/cycle cleanup, vector hóa
TF-IDF/SVD + numeric features, ONNX nearest-cluster assignment, Markov next-action và
friction flags. Journey dưới 4 event sau cleanup trả trạng thái `collecting`.

Output JSON có các trường chính:

```json
{
  "journey_id": "J000001",
  "state": "final",
  "cluster": 10,
  "cluster_name": "Đăng nhập OTP",
  "sequence": "view@Home -> action@Home/payment -> view@PaymentInfo",
  "event_sequence": ["view@Home", "action@Home/payment", "view@PaymentInfo"],
  "class_code": "guest_otp_login",
  "friction_flags": "",
  "next_action": "View@Android::android/login_screen/OTP_screen"
}
```

Các asset model nằm trong `app/src/main/assets/`:

- `journey_classifier.onnx` — cluster/distance/geometric anomaly.
- `preprocessing.json` — vocabulary, IDF, SVD, scaler và feature contract.
- `markov.json` — sparse transition tables cho log-probability/next action.
- `friction_config.json` — các threshold friction/anomaly.
- `class_mapping.json` — cluster → business label; cluster `-1` là unknown.

## Runtime dependency

```kotlin
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
}
```

Put `journey_classifier.onnx`, `preprocessing.json` and `class_mapping.json`
from `output/mobile/android/` in the app's assets directory. Load them as byte
arrays/UTF-8 strings and construct `JourneyOnnxClassifier`.

The runner accepts a finalized token sequence plus the ten numeric journey
features. The app must use the same canonicalization and journey rules recorded
under `journey_contract` in `preprocessing.json`.

Keep an event buffer per `(customer_id, session_id)`. For `JOURNEY_COMPLETE`,
finalize and score the buffered journey as soon as the next event triggers root
return, auth change, max length, session change, or a 90-second idle gap. For
`FIXED_DURATION`, emit a snapshot every configured event-time duration while
keeping open journeys in the buffer. Flush the remaining buffer when the stream
ends.

ONNX returns nearest cluster, centroid distance, and geometric anomaly. The
Kotlin runner joins the cluster with `class_mapping.json` to return the
meaningful Vietnamese class. Markov anomaly and next-action predictions are
included in the local processor output.

## Bundle gửi cho team Android

Một bundle Android phải giữ nguyên các file của cùng một platform/run:

| File | Vai trò |
|---|---|
| `journey_classifier.onnx` | ONNX nearest-cluster model |
| `preprocessing.json` | vocabulary, IDF, SVD, scaler và journey contract |
| `class_mapping.json` | cluster → class tiếng Việt; cluster `-1` là unknown |
| `markov.json` | transition probabilities, next action và Markov anomaly |
| `friction_config.json` | thresholds friction/anomaly |
| `manifest.json` | platform, schema và danh sách output |

Nguồn bundle đã export nằm ở `output/mobile/android/`. Team Android copy sáu file trên vào
`app/src/main/assets/`, giữ nguyên tên file, thêm dependency ONNX Runtime rồi gọi
`MobileClickstreamProcessor`. `test_data_android.csv` chỉ dành cho simulator, không cần đưa vào
production app.

Không trộn bundle Android với iOS hoặc bundle khác run: `preprocessing.json`, ONNX và
`class_mapping.json` phải được phát hành cùng nhau. Khi model train lại, gửi lại cả bundle mới.
