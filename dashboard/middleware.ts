import { NextResponse, type NextRequest } from "next/server";

import { COOKIE, validToken } from "./lib/session";

// Everything needs the PIN except the login page itself and the files the
// phone needs to show the home-screen icon.
export async function middleware(req: NextRequest) {
  if (await validToken(req.cookies.get(COOKIE)?.value)) return NextResponse.next();
  if (req.nextUrl.pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "Log in again" }, { status: 401 });
  }
  const url = req.nextUrl.clone();
  url.pathname = "/login";
  url.search = "";
  return NextResponse.redirect(url);
}

export const config = {
  matcher: [
    "/((?!login|api/login|api/news-probe|_next/static|_next/image|icons/|icon.png|manifest.webmanifest|favicon.ico).*)",
  ],
};
