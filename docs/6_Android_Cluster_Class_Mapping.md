# Android và iOS cluster class mapping

Mỗi mapping thuộc đúng model/run được mô tả trong
`output/<platform>_run_config.json`. ID của HDBSCAN có thể thay đổi sau mỗi lần
train, vì vậy cần duyệt và sinh lại mapping sau khi refit model.

## Hai cấp nhãn

- `class_group_code`, `class_group`: nhóm nghiệp vụ lớn để tổng hợp báo cáo.
- `class_code`, `class_name`: use case/hành trình chi tiết của cluster.

`class_code` là khóa ổn định nên dùng trong code và dashboard; `class_name` là
nhãn tiếng Việt dùng để hiển thị. `naming_confidence=low` cần được product owner
review trước khi dùng làm KPI chính thức.

Cluster `-1` là noise. Cluster không có trong mapping cũng được đưa về
`unknown_journey`; tuyệt đối không ép một hành trình lạ vào một nhãn nghiệp vụ.

## Áp dụng cho inference cuối cùng

Sau khi `score_partitioned_events.py` tạo output journey, dùng mapping nghiệp vụ
đã review thủ công để tạo file `*_all_named.csv`. Đây là anchor duy nhất cho
dashboard và post-analysis:

```bash
python scripts/score_partitioned_events.py --input data/giga_data/android_events_t3-2026.csv --platform android \
  --single-run output/partitioned_runs/latest/android --output-root output/scores/current
python scripts/apply_mapping_name.py \
  --input output/scores/current/android/model_version=latest/platform=android \
  --platform android \
  --mapping output/scores/current/Cluster_naming.csv \
  --output output/scores/current/android_all_named.csv
```

The mapping is joined on `(platform, cluster_id)`. Cluster `-1` is required for
both platforms and unresolved/new clusters fall back to that row instead of
receiving a guessed business label.

## Artifact

- `output/android_cluster_class_mapping.json`: nguồn mapping có version và metadata.
- `output/ios_cluster_class_mapping.json`: mapping cho model iOS.
- `output/android_cluster_class_mapping.csv`: bảng phẳng để join SQL/BI.
- `output/android_cluster_catalog_named.csv`: catalog gốc đã gắn tên.
