// TEMPORARY test branch only (never merged): can Vercel reach the feeds?
import { NextResponse } from "next/server";
import { usdCalendar, usdHeadlines } from "@/lib/news";
export const dynamic = "force-dynamic";
export async function GET() {
  const [c, h] = await Promise.all([usdCalendar(), usdHeadlines()]);
  return NextResponse.json({
    calendar: c.data.length, calendarError: c.error, firstEvent: c.data[0]?.title ?? null,
    headlines: h.data.length, headlinesError: h.error, latest: h.data[0]?.title ?? null,
  });
}
