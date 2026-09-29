/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The desktop release ships the frontend as static files served by the
  // local FastAPI process. Node/Next.js is build-time only, never a runtime
  // dependency of the offline desktop application.
  output: "export",
  trailingSlash: true,
};

module.exports = nextConfig;
