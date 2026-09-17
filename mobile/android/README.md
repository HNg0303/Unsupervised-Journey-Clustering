# Android Clickstream Journey Clustering & Scorer

Android application & SDK module để replay clickstream và chấm điểm hành trình (Journey Scoring) sử dụng bundle mobile export `v2.1.0` từ `output/mobile/android`.

## Chức năng

- Đọc dữ liệu production CSV `test_data_android.csv` với định dạng:
  `_id,device_id,device_created_at,session_id,customer_id,platform,key,segmentation_name,client_time`
- Tự động chuẩn hoá đường dẫn URL, masking dynamic identifiers (`{id}`, `{uuid}`, `{code}`) và trích xuất ngữ nghĩa đa cấp độ (`family`, `module`, `object`, `operation`).
- Nhận diện các boundary phân đoạn journey: `session_start`, `idle_gap` (90s), `root_return` (>= 6 events), `auth_change`, `length_cap` (80 events).
- Làm sạch chuỗi event: khử lặp liên tiếp (dedup) và khử chu trình lặp (cycles period 2..4).
- Vector hoá đa kênh bản địa (Native Kotlin):
  - Primary TF-IDF + SVD folded projection (weight 1.0)
  - Coarse semantic channel `family/module` (weight 0.45)
  - Intent semantic channel `family/module/object/operation` (weight 0.30)
  - Operation channel `stage:operation` (weight 0.20)
  - 11 behavioural numeric features log1p-damped & scaled (weight 0.35)
  - Global PCA projection đưa về không gian embedding 48 chiều L2-normalized.
- Gán nhãn Archetype gần nhất trong 734 cụm centroids với ngưỡng khoảng cách calibrated `p95 = 0.7073`.
- Mô hình Markov bậc 1 tính xác suất log-prob và dự đoán hành động tiếp theo (`next_action`).
- Cảnh báo Anomaly (Geometric & Generative) và cờ ma sát (Friction Flags: `excessive_back`, `navigation_loop`, `screen_thrash`, `slow_journey`, `unknown_archetype`, `improbable_transitions`).
- Giao diện trực quan: chọn session, điều chỉnh tốc độ (0.25x - 10x), chọn route (`Journey complete` / `Fixed duration`), xem chi tiết hành trình, cờ bất thường và copy JSON output.

## Cấu trúc bundle assets (`app/src/main/assets/`)

Toàn bộ các module đã export từ `output/mobile/android` được tích hợp:

| File | Vai trò |
|---|---|
| `manifest.json` | Master manifest định nghĩa bundle version 2.1.0, 734 clusters, weights và danh sách file |
| `config.json` | Cấu hình segmentation, tokenization và canonization rules |
| `thresholds.json` | Các ngưỡng phân vị calibrated: distance p95, markov p05/p01, back rate, loop count |
| `vocabularies.json` | Bộ từ vựng 1-gram & 2-gram cho 4 kênh (`primary`, `coarse`, `intent`, `operation`) |
| `vectorizer_metadata.json` | Cấu hình scaler, numeric mean/scale, PCA components |
| `projection_weights.float16.bin` | Trọng số ma trận folded SVD*IDF, Global PCA và Scaler (FP16/FP32) |
| `centroids_matrix.bin` | Ma trận 734 centroids x 48 dimensions (FP16) |
| `centroids.json` | Centroids dự phòng dạng JSON |
| `markov.json` & `markov_graph.bin` | Bảng xác suất chuyển trạng thái Markov và dự đoán next action |
| `cluster_mapping.json` | Bảng ánh xạ 734 cụm sang tên nghiệp vụ tiếng Việt, phân nhóm và medoid path |
| `test_data_android.csv` | Mẫu dữ liệu clickstream production để simulator replay |

## Tích hợp vào Production SDK

### 1. Khởi tạo Model
```kotlin
val model = MobileJourneyModel(context)
```

### 2. Phân tích chuỗi sự kiện hoặc JSON
```kotlin
// Từ danh sách ClickstreamEvent
val result = model.analyzeEvents(events)

// Hoặc từ chuỗi JSON
val resultFromJson = model.analyzeJson(rawJsonString)
val outputJson = result.toJson().toString(2)
```

### 3. Xử lý Streaming Clickstream
```kotlin
val processor = MobileClickstreamProcessor(
    model = model,
    route = MobileProcessingRoute.JOURNEY_COMPLETE, // hoặc MobileProcessingRoute.FIXED_DURATION
    windowDurationSeconds = 30L
)
processor.start()

// Đưa event vào processor trên background thread
val update = processor.accept(event)
if (update.emitted) {
    update.finalized.forEach { prediction ->
        Log.d("Journey", "${prediction.journeyId}: ${prediction.clusterName}")
    }
}

// Khi kết thúc session hoặc logout
val finalUpdate = processor.finish()
```

## Chạy trên Android Studio

1. Mở thư mục `mobile/android` trong Android Studio.
2. Chờ Gradle Sync hoàn tất.
3. Chọn thiết bị Android (API 26+).
4. Run configuration `app`.
