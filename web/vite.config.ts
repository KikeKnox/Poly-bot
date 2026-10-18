import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  // The SPA is served nested under /ui (see static_router in src/api/mod.rs).
  // Without this, Vite references assets as /assets/... (domain root), which
  // the ServeDir mounted at /ui never serves → 404 → blank screen.
  base: "/ui/",
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "./src") },
  },
  build: {
    outDir: path.resolve(import.meta.dirname, "../src/api/static/dist"),
    emptyOutDir: true,
  },
});
