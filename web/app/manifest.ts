import type { MetadataRoute } from "next";

/** Lets phones add DustAhead to the home screen and open it like an app (the blueprint's web fallback). */
export const dynamic = "force-static";

const base = process.env.NEXT_PUBLIC_BASE_PATH || "";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "DustAhead: I-10 dust alerts",
    short_name: "DustAhead",
    description: "Short-range dust forecasts for Interstate 10 in Arizona and New Mexico (research prototype).",
    start_url: `${base}/`,
    scope: `${base}/`,
    display: "standalone",
    background_color: "#f4f5f4",
    theme_color: "#ffffff",
    icons: [
      { src: `${base}/icons/icon-192.png`, sizes: "192x192", type: "image/png" },
      { src: `${base}/icons/icon-512.png`, sizes: "512x512", type: "image/png" },
      { src: `${base}/icons/icon-maskable-512.png`, sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
