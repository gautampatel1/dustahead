/** Static export: the whole site is files on R2 / any static host. No server, no API routes.
 *  NEXT_PUBLIC_BASE_PATH lets the export live under a sub-path (e.g. /dustahead). */
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";
export default {
  output: "export",
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
};
