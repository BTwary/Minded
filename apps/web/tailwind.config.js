/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        amber: {
          50: "#fffbeb",
          100: "#fef3c7",
          200: "#fde68a",
          300: "#fcd34d",
          400: "#fbbf24",
          500: "#f5a623",
          600: "#d97706",
          700: "#b45309",
        },
        copper: {
          light: "#FFCC70",
          DEFAULT: "#F5A623",
          dark: "#B87333",
        },
        espresso: {
          DEFAULT: "#1A1A1A",
          dark: "#0E0E0E",
        },
        ivory: {
          DEFAULT: "#F8F9F7",
          surface: "#FFFFFF",
          muted: "#ECEFEA",
        },
        emerald: {
          50: "#ecfdf5",
          100: "#d1fae5",
          200: "#a7f3d0",
          300: "#6ee7b7",
          400: "#34d399",
          500: "#10b981",
          600: "#00a86b",
          700: "#047857",
        },
      },
      fontFamily: {
        sans: ["var(--font-body)", "IBM Plex Sans", "Source Sans 3", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        heading: ["Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
};
