/** @type {import('next').NextConfig} */
const nextConfig = {
  // Disable image optimization — we don't ship images and Vercel's
  // image-optimization runtime adds startup time we don't need.
  images: { unoptimized: true },
  // Strict TypeScript / React in dev to catch issues early.
  reactStrictMode: true,
};

module.exports = nextConfig;
