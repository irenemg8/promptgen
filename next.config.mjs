/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  basePath: process.env.GITHUB_ACTIONS ? '/promptgen' : '',
  output: 'export',
  distDir: 'out',
}

export default nextConfig