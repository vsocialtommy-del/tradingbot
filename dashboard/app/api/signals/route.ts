import { NextResponse } from "next/server";

import { rowToSignal, Signal, SignalRow, supabase } from "@/lib/supabase";

// Vercel edge cache headers: tell the CDN to never cache this route
// since the page polls every 10 s and we always want the freshest
// row from Supabase.
export const dynamic = "force-dynamic";
export const revalidate = 0;

interface ApiResponse {
  buy: Signal | null;
  sell: Signal | null;
  fetchedAt: string;
}

export async function GET(): Promise<NextResponse<ApiResponse | { error: string }>> {
  const { data, error } = await supabase
    .from("signals")
    .select("*")
    .eq("is_active", true)
    .order("updated_at", { ascending: false });

  if (error) {
    return NextResponse.json(
      { error: error.message },
      { status: 502 },
    );
  }

  const rows = (data ?? []) as SignalRow[];
  // The bot guarantees at most one active row per direction, but the
  // dashboard is defensive: take the freshest row of each direction
  // in case a write race ever leaves duplicates.
  const buyRow = rows.find((r) => r.direction === "BUY") ?? null;
  const sellRow = rows.find((r) => r.direction === "SELL") ?? null;

  return NextResponse.json({
    buy: buyRow ? rowToSignal(buyRow) : null,
    sell: sellRow ? rowToSignal(sellRow) : null,
    fetchedAt: new Date().toISOString(),
  });
}
