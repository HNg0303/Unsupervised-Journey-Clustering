import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const [sourcePath, outputPath] = process.argv.slice(2);
if (!sourcePath || !outputPath) {
  throw new Error("usage: node build_taxonomy_features_csv.mjs <taxonomy.md> <output.csv>");
}

const markdown = await fs.readFile(sourcePath, "utf8");
const pattern = /^\|\s*(M\d{2}\.\d{2}\.\d{2})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|/gm;
const rows = [];
for (const match of markdown.matchAll(pattern)) {
  rows.push(match.slice(1, 5).map((value) => value.trim()));
}

const ids = new Set(rows.map((row) => row[0]));
const families = new Set(rows.map((row) => row[1]));
const submodules = new Set(rows.map((row) => `${row[1]}\u0000${row[2]}`));
if (rows.length !== 486 || ids.size !== rows.length || families.size !== 16 || submodules.size !== 85) {
  throw new Error(JSON.stringify({
    message: "taxonomy shape did not match the expected detailed table",
    rows: rows.length,
    uniqueIds: ids.size,
    families: families.size,
    submodules: submodules.size,
  }));
}

const headers = [
  "taxonomy_id",
  "business_family",
  "business_submodule",
  "business_detail",
];
const escapeCsv = (value) => {
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
const csvText = [headers, ...rows]
  .map((row) => row.map(escapeCsv).join(","))
  .join("\r\n") + "\r\n";

await fs.mkdir(new URL(".", `file://${outputPath}`).pathname, { recursive: true });
await fs.writeFile(outputPath, `\uFEFF${csvText}`, "utf8");

// Re-import the authored CSV with the spreadsheet runtime to validate the
// tabular artifact, Unicode content, headers, and representative rows.
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Taxonomy" });
workbook.recalculate();
const inspection = await workbook.inspect({
  kind: "table",
  range: "Taxonomy!A1:D8",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 4,
});
const preview = await workbook.render({
  sheetName: "Taxonomy",
  range: "A1:D20",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  "/tmp/hifpt_taxonomy_3_levels_preview.png",
  new Uint8Array(await preview.arrayBuffer()),
);

console.log(JSON.stringify({
  outputPath,
  rows: rows.length,
  businessFamilies: families.size,
  businessSubmodules: submodules.size,
  businessDetails: rows.length,
  inspection: inspection.ndjson,
}));
