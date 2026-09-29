import "./globals.css";
import React from "react";
import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import MiningCursorTrail from "../components/MiningCursorTrail";
import { ThemeProvider } from "../components/ThemeProvider";

export const metadata = {
  title: "MindEd AA-OS — Autonomous Analytical Intelligence Operating System",
  description: "Local-first autonomous analytical workspace for real, traceable data investigation, statistical analysis, forecasting, causal reasoning, verification, and human review.",
  icons: {
    icon: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        <link rel="alternate icon" href="/favicon.ico" type="image/x-icon" />
        <link rel="apple-touch-icon" href="/favicon.svg" />
      </head>
      <body className="bg-slate-950 text-slate-100 flex min-h-screen">
        <ThemeProvider>
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <Header />
          <main className="flex-1 p-8 max-w-7xl w-full mx-auto">{children}</main>
        </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
