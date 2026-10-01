import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allows a second, independent build (e.g. a preview pointed at another API) next to the dev server.
  distDir: process.env.NEXT_DIST_DIR || ".next",
};

export default nextConfig;
