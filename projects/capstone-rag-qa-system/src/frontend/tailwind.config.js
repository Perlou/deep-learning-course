/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0f172a",
        card: {
          DEFAULT: "#1e293b",
          foreground: "#f8fafc",
        },
        border: "#334155",
        primary: {
          DEFAULT: "#3b82f6",
          foreground: "#ffffff",
          hover: "#60a5fa",
        },
        accent: "#8b5cf6",
        muted: {
          DEFAULT: "#64748b",
          foreground: "#94a3b8",
        },
        foreground: "#f8fafc",
        success: "#22c55e",
        warning: "#eab308",
        destructive: "#ef4444",
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
      },
      borderRadius: {
        "2xl": "16px",
        "3xl": "24px",
      },
      boxShadow: {
        card: "0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.3)",
        modal: "0 25px 50px -12px rgba(0, 0, 0, 0.5)",
        glow: "0 0 20px rgba(59, 130, 246, 0.3)",
      },
    },
  },
  plugins: [],
}
