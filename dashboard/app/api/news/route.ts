import { NextResponse } from "next/server";

import { needLogin } from "@/lib/db";
import { usdCalendar, usdHeadlines } from "@/lib/news";

export const dynamic = "force-dynamic";

export async function GET() {
  const denied = await needLogin();
  if (denied) return denied;
  const [cal, heads] = await Promise.all([usdCalendar(), usdHeadlines()]);
  return NextResponse.json(
    {
      calendar: cal.data, calendarError: cal.error,
      headlines: heads.data, headlinesError: heads.error,
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
