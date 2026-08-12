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

## Áp dụng cho event mới

Đầu tiên dùng fitted scorer để biến raw event thành journey và gán `cluster`.
Sau đó gắn class mapping vào file kết quả:

```powershell
python scripts/score_new_events.py --platform android --input data/new_events.csv --output output/android_scored.csv
python scripts/apply_cluster_mapping.py --platform android --input output/android_scored.csv --output output/android_scored_named.csv
python scripts/apply_cluster_mapping.py --platform ios --input output/ios_scored.csv --output output/ios_scored_named.csv
```

Hoặc áp dụng trong Python ngay sau `scorer.score(raw_events)`:

```python
from Rule_based.cluster_mapping import apply_cluster_mapping

scored = scorer.score(raw_events)
scored = apply_cluster_mapping(scored, "output/android_cluster_class_mapping.json")
```

Kết quả có thêm sáu cột: `class_group_code`, `class_group`, `class_code`,
`class_name`, `class_description`, `naming_confidence`.

Nếu cần dùng tên diễn giải trong catalog dành cho cổ đông/lãnh đạo, dùng
catalog `output/clusters/shareholder_cluster_catalog_vi.json`:

```powershell
python scripts/apply_cluster_name_mapping.py --platform android --input output/android_scored.csv
python scripts/apply_cluster_name_mapping.py --platform ios --input output/ios_scored.csv
```

Kết quả thêm `business_family`, `cluster_name` và `naming_confidence`. Catalog
chứa cả Android và iOS nên `--platform` là bắt buộc.

## Artifact

- `output/android_cluster_class_mapping.json`: nguồn mapping có version và metadata.
- `output/ios_cluster_class_mapping.json`: mapping cho model iOS.
- `output/android_cluster_class_mapping.csv`: bảng phẳng để join SQL/BI.
- `output/android_cluster_catalog_named.csv`: catalog gốc đã gắn tên.
