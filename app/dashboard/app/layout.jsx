import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata = {
  title: "CreditForge · Risk Cockpit",
  description: "Bank-grade credit-risk modeling, validation, and monitoring (BONITAS).",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <Sidebar />
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
