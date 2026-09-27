import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/pages/**/*.{js,ts,jsx,tsx,mdx}", "./src/components/**/*.{js,ts,jsx,tsx,mdx}", "./src/app/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: { extend: { colors: { background: "hsl(var(--background))", foreground: "hsl(var(--foreground))", card: "hsl(var(--card))", "card-foreground": "hsl(var(--card-foreground))", border: "hsl(var(--border))", muted: "hsl(var(--muted))", "muted-foreground": "hsl(var(--muted-foreground))", primary: "hsl(var(--primary))", "primary-foreground": "hsl(var(--primary-foreground))" } } },
  plugins: [],
};
export default config;
