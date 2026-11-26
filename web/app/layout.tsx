import type { Metadata, Viewport } from "next";
import { Public_Sans } from "next/font/google";
import "./globals.css";

const body = Public_Sans({ subsets: ["latin"], weight: ["400", "500", "600", "700"], variable: "--f-body" });

const base = process.env.NEXT_PUBLIC_BASE_PATH || "";
// Public address of the site, for absolute share-preview links. Without it the preview has no image (not localhost).
const site = process.env.NEXT_PUBLIC_SITE_URL;
const description = "Short-range dust-storm forecasts for Interstate 10 in Arizona and New Mexico, 15 to 60 minutes ahead. Research prototype.";

export const metadata: Metadata = {
  title: "DustAhead · I-10 dust alerts",
  description,
  applicationName: "DustAhead",
  icons: { icon: [{ url: `${base}/icon.svg`, type: "image/svg+xml" }], apple: `${base}/icons/apple-touch-icon.png` },
  appleWebApp: { capable: true, title: "DustAhead", statusBarStyle: "default" },
  ...(site ? { metadataBase: new URL(site) } : {}),
  openGraph: { title: "DustAhead · I-10 dust alerts", description, type: "website", ...(site ? { images: [`${base}/icons/icon-512.png`] } : {}) },
  twitter: { card: "summary", title: "DustAhead · I-10 dust alerts", description },
};

export const viewport: Viewport = { themeColor: "#ffffff", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={body.variable}>
      <body>{children}</body>
    </html>
  );
}
