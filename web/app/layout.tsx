import type { Metadata, Viewport } from "next";
import { Public_Sans } from "next/font/google";
import "./globals.css";

const body = Public_Sans({ subsets: ["latin"], weight: ["400", "500", "600", "700"], variable: "--f-body" });

export const metadata: Metadata = {
  title: "DustAhead · I-10 dust alerts",
  description: "Short-range dust-storm nowcasts for Interstate 10 in Arizona and New Mexico (research prototype).",
};

export const viewport: Viewport = { themeColor: "#ffffff", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={body.variable}>
      <body>{children}</body>
    </html>
  );
}
