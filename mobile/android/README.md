# Android clickstream simulator

MVP Android Studio app để replay clickstream từ `app/src/main/assets/test_data_july.json`.

## Chức năng

- Đọc dữ liệu Extended JSON có dạng `_id.$oid` và `created_at.$date`.
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
data/test_data/test_data_july.json
→ mobile/android/app/src/main/assets/test_data_july.json
```

MVP này phát event cục bộ trong app. Trong lúc replay, app thu event theo các transport window
30 giây và cập nhật provisional output trực tiếp. Nút `Analyze 30s windows` chạy lại cùng logic
cho một session đã chọn. Mốc 30 giây không reset journey; journey chỉ finalize theo idle gap
90 giây, root return, auth change, length cap hoặc `flush`.

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
val update = stream.append(event)
val finalPrediction = stream.flush()
```

Module thực hiện canonize, journey segmentation, consecutive/cycle cleanup, vector hóa
TF-IDF/SVD + numeric features, ONNX nearest-cluster assignment, Markov next-action và
friction flags. Journey dưới 4 event sau cleanup trả trạng thái `collecting`.

Output JSON có các trường chính:

```json
{
  "journey_id": "J000001",
  "state": "final",
  "cluster": 10,
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

Do not reset state every 30 seconds. Keep an event buffer per
`(customer_id, session_id)`. A 30-second upload is a transport window, while
the trained journey boundary uses a 90-second idle gap. Return a provisional
prediction for an open journey and a final prediction only on root return,
auth change, max length, or 90 seconds of inactivity.

ONNX returns nearest cluster, centroid distance, and geometric anomaly. The
Kotlin runner joins the cluster with `class_mapping.json` to return the
meaningful Vietnamese class. Markov anomaly and next-action predictions are
not part of this first mobile graph.
