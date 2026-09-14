/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "media.api-sports.io",
      },
    ],
  },
};

module.exports = nextConfig;