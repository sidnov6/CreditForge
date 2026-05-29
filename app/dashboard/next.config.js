/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // API base for the FastAPI scoring service (override at deploy time)
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001",
  },
};
module.exports = nextConfig;
