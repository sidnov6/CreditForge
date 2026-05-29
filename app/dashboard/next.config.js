/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Static HTML export — all pages are client-rendered and fetch from /api at
  // runtime, so the cockpit ships as static files served by FastAPI.
  output: "export",
  trailingSlash: true,            // -> /portfolio/index.html, served via StaticFiles(html=True)
  images: { unoptimized: true },
  // Same-origin API in the container; override for split local dev.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "",
  },
};
module.exports = nextConfig;
