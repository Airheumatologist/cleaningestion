import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Elixir AI | Premium Medical RAG",
  description: "Advanced medical research assistant grounded in PMC Open Access literature",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
