import type { Json, JsonObject } from "../../core/types";

/** What the `clock` channel answers once: the wall clock, and the user's time zone and language (from their browser). */
export interface ClockInfo {
  epochMs: number;
  timeZone?: string;
  locale?: string;
}

export function clockInfo(data: Json): ClockInfo | undefined {
  const d = data as JsonObject | null;
  if (!d || typeof d.epochMs !== "number") return undefined;
  return {
    epochMs: d.epochMs,
    ...(typeof d.timeZone === "string" ? { timeZone: d.timeZone } : {}),
    ...(typeof d.locale === "string" ? { locale: d.locale } : {}),
  };
}

const validZone = (tz: string | undefined) => {
  if (!tz) return undefined;
  try {
    new Intl.DateTimeFormat("en-GB", { timeZone: tz });
    return tz;
  } catch {
    return undefined;
  }
};

/**
 * The live context for the system prompt: when the call started, in the user's time zone, and their
 * language. Written once per session, so the prompt (and its cache) stays the same for every request.
 */
export function contextBlock(c: ClockInfo): string {
  const timeZone = validZone(c.timeZone) ?? "UTC";
  const at = new Date(c.epochMs);
  const part = (o: Intl.DateTimeFormatOptions) =>
    new Intl.DateTimeFormat("en-GB", { timeZone, ...o }).format(at);
  const date = part({ weekday: "long", day: "numeric", month: "long", year: "numeric" });
  const time = part({ hour: "2-digit", minute: "2-digit", hour12: false });
  const offset =
    new Intl.DateTimeFormat("en-GB", { timeZone, timeZoneName: "longOffset" })
      .formatToParts(at)
      .find((p) => p.type === "timeZoneName")?.value ?? "GMT";
  const lines = [
    "Live context:",
    `- The call started on ${date} at ${time}, the user's local time (time zone ${timeZone}, ${offset}).`,
  ];
  if (c.locale) lines.push(`- The user's browser language is ${c.locale}.`);
  return lines.join("\n");
}
