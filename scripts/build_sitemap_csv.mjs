import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const rows = [
  ["Home", "Avatar + Fgold", ""],
  ["Home", "Noti + search + Quét QR", ""],
  ["Home", "Số hợp đồng", ""],
  ["Home", "Quản lý dịch vụ", "Connection Status"],
  ["Home", "Quản lý dịch vụ", "Quản lý thiết bị"],
  ["Home", "Quản lý dịch vụ", "Báo lỗi Internet"],
  ["Home", "Quản lý dịch vụ", "Sức khoẻ mạng"],
  ["Home", "Quản lý dịch vụ", "Khởi động Modem"],
  ["Home", "Quản lý dịch vụ", "Báo hỏng nhanh"],
  ["Home", "Banner", ""],
  ["Home", "Ưu đãi hot", "Loyalty"],
  ["Home", "Đăng ký dịch vụ", "Internet"],
  ["Home", "Đăng ký dịch vụ", "FPT Play"],
  ["Home", "Đăng ký dịch vụ", "Camera"],
  ["Home", "Đăng ký dịch vụ", "Combo"],
  ["Home", "Đăng ký dịch vụ", "Ultra Fast"],
  ["Home", "Đăng ký dịch vụ", "FPT Wifi"],
  ["Home", "Đăng ký dịch vụ", "Thiết bị mạng"],
  ["Home", "Đăng ký dịch vụ", "Tất cả"],
  ["Home", "Chỉ có tại Hi FPT", "Gói thuê bao bán"],
  ["Home", "Chỉ có tại Hi FPT", "Giới thiệu Hi FPT"],
  ["Home", "Chỉ có tại Hi FPT", "Tarot"],
  ["Home", "Chỉ có tại Hi FPT", "Nhà trọ thông minh"],
  ["Home", "Taskbar", "Trang chủ"],
  ["Home", "Taskbar", "Thanh toán"],
  ["Home", "Taskbar", "Hỗ trợ"],
  ["Home", "Taskbar", "Ưu đãi"],
  ["Home", "Taskbar", "Tài khoản"],

  ["Tài khoản", "Thông tin khách hàng", "Thông tin người dùng"],
  ["Tài khoản", "Thông tin khách hàng", "Khách hàng thân thiết"],
  ["Tài khoản", "Thông tin khách hàng", "Đánh giá của tôi"],
  ["Tài khoản", "Ưu đãi của tôi", ""],
  ["Tài khoản", "Người giới thiệu", ""],
  ["Tài khoản", "Hợp đồng và dịch vụ", "Thông tin hợp đồng"],
  ["Tài khoản", "Hợp đồng và dịch vụ", "Thay đổi giấy tờ tuỳ thân"],
  ["Tài khoản", "Hợp đồng và dịch vụ", "Chuyển địa điểm"],
  ["Tài khoản", "Hợp đồng và dịch vụ", "Đăng quyền quản lý HĐ"],
  ["Tài khoản", "Hợp đồng và dịch vụ", "Hợp đồng PLHĐ"],
  ["Tài khoản", "Biên bản nghiệm thu", ""],
  ["Tài khoản", "Quản lý thanh toán", ""],
  ["Tài khoản", "Lan toả cùng Hi FPT", "Giới thiệu bạn bè"],
  ["Tài khoản", "Lan toả cùng Hi FPT", "Trở thành đối tác"],
  ["Tài khoản", "Lan toả cùng Hi FPT", "Bạn bè Hi FPT từ đâu?"],
  ["Tài khoản", "Nhân viên FPT", "Thẻ nhân viên"],
  ["Tài khoản", "Nhân viên FPT", "Mã QR nhân viên"],
  ["Tài khoản", "Khách hàng tiềm năng", ""],
  ["Tài khoản", "Cài đặt", "Tài khoản và bảo mật"],
  ["Tài khoản", "Cài đặt", "Thiết lập giọng nói"],
  ["Tài khoản", "Cài đặt", "Thiết lập tài khoản"],
  ["Tài khoản", "Cài đặt", "Ngôn ngữ"],
  ["Tài khoản", "Cài đặt", "Điều khoản sử dụng"],
  ["Tài khoản", "Cài đặt", "Đăng xuất"],
  ["Tài khoản", "Tuyên bố bản quyền Hi FPT", "Kênh chăm sóc khách hàng trên các FPT Telecom"],

  ["Quản lý dịch vụ", "Số hợp đồng", ""],
  ["Quản lý dịch vụ", "Internet (Hợp đồng)", "Gói dịch vụ sử dụng"],
  ["Quản lý dịch vụ", "Internet (Hợp đồng)", "Nâng cấp dịch vụ"],
  ["Quản lý dịch vụ", "Quản lý Modem", ""],
  ["Quản lý dịch vụ", "Khởi động Modem", ""],
  ["Quản lý dịch vụ", "Quản lý Wi-Fi", ""],
  ["Quản lý dịch vụ", "Mô hình mạng", ""],
  ["Quản lý dịch vụ", "Quản lý thiết bị", ""],
  ["Quản lý dịch vụ", "Báo lỗi Internet", ""],
  ["Quản lý dịch vụ", "Sức khoẻ mạng", ""],
  ["Quản lý dịch vụ", "Dịch vụ đi kèm (Ultra fast)", ""],
  ["Quản lý dịch vụ", "FPT Play (Hợp đồng)", "Gói dịch vụ sử dụng"],
  ["Quản lý dịch vụ", "FPT Play (Hợp đồng)", "Nâng cấp dịch vụ"],
  ["Quản lý dịch vụ", "FPT Play (Hợp đồng)", "Lịch phát sóng"],
  ["Quản lý dịch vụ", "Nhắc lịch", ""],
  ["Quản lý dịch vụ", "Camera", "Thông tin Camera"],
  ["Quản lý dịch vụ", "Camera", "Gia hạn Cloud"],
  ["Quản lý dịch vụ", "Camera", "Khôi phục dịch vụ"],

  ["Trung tâm hỗ trợ", "Tạo yêu cầu hỗ trợ", ""],
  ["Trung tâm hỗ trợ", "Lịch sử hỗ trợ", ""],
  ["Trung tâm hỗ trợ", "Câu hỏi thường gặp", ""],
  ["Trung tâm hỗ trợ", "Điểm giao dịch gần nhất", ""],
  ["Trung tâm hỗ trợ", "Chat CSKH", ""],

  ["Thanh toán", "Thanh toán hoá đơn/khoản thu", ""],
  ["Thanh toán", "Lịch sử thanh toán", ""],
  ["Thanh toán", "Tiện ích", "Trả tự động"],
  ["Thanh toán", "Tiện ích", "Trả trước"],
  ["Thanh toán", "Tiện ích", "Hoá đơn điện tử"],
  ["Thanh toán", "Tiện ích", "Quản lý ví thẻ"],
  ["Thanh toán", "Tiện ích", "QR thanh toán"],
  ["Thanh toán", "Tiện ích", "Gia hạn thanh toán"],
  ["Thanh toán", "Tiện ích", "Kênh nhận thông báo cước"],
  ["Thanh toán", "Tiện ích", "Lịch nhắc thanh toán"],

  ["Loyalty", "Thông tin KHTT", ""],
  ["Loyalty", "Đăng ký thành viên", ""],
  ["Loyalty", "Ưu đãi của tôi", ""],
  ["Loyalty", "Tích điểm/Đổi quà", ""],
  ["Loyalty", "Hot Deal/Độc quyền", "Đặc quyền ưu tiên"],

  ["Đăng ký dịch vụ", "Menu dịch vụ", ""],
  ["Đăng ký dịch vụ", "Địa chỉ", ""],
  ["Đăng ký dịch vụ", "Banner", ""],
  ["Đăng ký dịch vụ", "Gợi ý cho bạn", ""],
  ["Đăng ký dịch vụ", "Đăng ký dịch vụ Internet", "Đăng ký dịch vụ truyền hình"],
  ["Đăng ký dịch vụ", "Đăng ký dịch vụ Internet", "Camera"],
  ["Đăng ký dịch vụ", "Đăng ký dịch vụ Internet", "Ultra-fast"],
  ["Đăng ký dịch vụ", "Đăng ký dịch vụ Internet", "F-Safe"],
  ["Đăng ký dịch vụ", "Đăng ký dịch vụ Internet", "Thiết bị mạng"],
  ["Đăng ký dịch vụ", "Đăng ký dịch vụ Internet", "FPT Wifi"],
  ["Đăng ký dịch vụ", "Đăng ký dịch vụ Internet", "Thiết bị đo đường huyết"],
  ["Đăng ký dịch vụ", "Đăng ký dịch vụ Internet", "Smart Home"],
];

const headers = ["Business Family", "Business Submodule", "Business Details"];
const escapeCsv = (value) => `"${String(value).replaceAll('"', '""')}"`;
const csv = "\uFEFF" + [headers, ...rows].map((row) => row.map(escapeCsv).join(",")).join("\r\n") + "\r\n";

const outputDir = new URL("../output/", import.meta.url);
await fs.mkdir(outputDir, { recursive: true });
const outputPath = new URL("sitemap_3_levels.csv", outputDir);
await fs.writeFile(outputPath, csv, "utf8");

// Parse through artifact-tool to validate the final CSV as a spreadsheet artifact.
const workbook = await Workbook.fromCSV(csv.replace(/^\uFEFF/, ""), { sheetName: "Sitemap" });
const topCheck = await workbook.inspect({
  kind: "table",
  range: "Sitemap!A1:C8",
  include: "values",
  tableMaxRows: 8,
  tableMaxCols: 3,
  maxChars: 3000,
});
const bottomCheck = await workbook.inspect({
  kind: "table",
  range: `Sitemap!A${rows.length - 3}:C${rows.length + 1}`,
  include: "values",
  tableMaxRows: 5,
  tableMaxCols: 3,
  maxChars: 3000,
});

console.log(JSON.stringify({
  outputPath: outputPath.pathname,
  rowCount: rows.length,
  topCheck: topCheck.ndjson,
  bottomCheck: bottomCheck.ndjson,
}, null, 2));
