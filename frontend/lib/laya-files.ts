export type LayaRole = "knowledge" | "data";

export type FileCheck =
  | { ok: true; role: LayaRole; detail: string }
  | { ok: false; detail: string };

// The proxy caps both files together at 5 MB once they are wrapped in JSON.
export const MAX_FILE_BYTES = 2 * 1024 * 1024;

const plural = (count: number, word: string) => `${count} ${word}${count === 1 ? "" : "s"}`;

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function describeRows(rows: unknown[]): FileCheck {
  if (rows.length === 0) return { ok: false, detail: "No observations" };
  if (!rows.every(isObject)) {
    return { ok: false, detail: "Not a list of observations" };
  }
  return { ok: true, role: "data", detail: plural(rows.length, "row") };
}

// One JSON object per line, as in data/test_observations.jsonl.
function parseLines(text: string): unknown[] | null {
  const rows: unknown[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      rows.push(JSON.parse(line));
    } catch {
      return null;
    }
  }
  return rows.length > 0 && isObject(rows[0]) ? rows : null;
}

/**
 * Tells a Laya knowledge file from a data file by its shape. This only labels
 * the file in the Library; the pipeline does the real validation on a run.
 */
export function inspectFile(text: string): FileCheck {
  if (!text.trim()) return { ok: false, detail: "Empty file" };

  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    const rows = parseLines(text);
    if (rows) return describeRows(rows);
    const reason = error instanceof Error ? error.message : "";
    return {
      ok: false,
      detail: reason ? `Invalid JSON: ${reason}` : "Invalid JSON",
    };
  }

  if (Array.isArray(value)) return describeRows(value);
  if (isObject(value)) {
    if ("center_frequency_hz" in value) return describeRows([value]);
    const questions = Object.values(value);
    if (
      questions.length > 0 &&
      questions.every((item) => isObject(item) && isObject(item.criteria))
    ) {
      const types = questions.reduce<number>(
        (total, item) => total + Object.keys((item as { criteria: object }).criteria).length,
        0,
      );
      return {
        ok: true,
        role: "knowledge",
        detail: `${plural(questions.length, "question")}, ${plural(types, "type")}`,
      };
    }
  }
  return { ok: false, detail: "Not a Laya knowledge or data file" };
}
