import "./globals.css";
import React from "react";
import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import MiningCursorTrail from "../components/MiningCursorTrail";
import { ThemeProvider } from "../components/ThemeProvider";

export const metadata = {
  applicationName: "MindEd AA-OS",
  title: "MindEd AA-OS",
  description: "Local-first autonomous data analyst companion",
  manifest: "/site.webmanifest",
  icons: {
    icon: [
      { url: "/icons/favicon.svg", type: "image/svg+xml" },
      { url: "/favicon.ico", sizes: "any" },
    ],
    apple: "/icons/icon-192.png",
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
        <link rel="icon" href="/icons/favicon.svg" type="image/svg+xml" />
        <link rel="alternate icon" href="/favicon.ico" type="image/x-icon" />
        <link rel="apple-touch-icon" href="/icons/icon-192.png" />
        <link rel="manifest" href="/site.webmanifest" />
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
