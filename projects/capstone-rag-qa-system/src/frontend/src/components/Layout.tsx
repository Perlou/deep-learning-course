import { Outlet, NavLink } from "react-router-dom";
import {
  Database,
  FileText,
  MessageSquare,
  Settings,
  Brain,
} from "lucide-react";
import { cn } from "../lib/utils";

const navItems = [
  { to: "/", icon: Database, label: "知识库", exact: true },
  { to: "/settings", icon: Settings, label: "设置", exact: true },
];

export function Layout() {
  return (
    <div className="flex min-h-screen bg-background">
      {/* 背景装饰 */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-accent/10 rounded-full blur-3xl" />
      </div>

      {/* Sidebar */}
      <aside className="w-16 lg:w-64 bg-card/50 backdrop-blur-xl border-r border-border/50 flex flex-col shrink-0 relative z-10">
        {/* Logo */}
        <div className="h-16 flex items-center gap-3 px-3 lg:px-4 border-b border-border/50">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-glow shrink-0">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div className="hidden lg:block">
            <span className="font-bold text-lg">DocuMind</span>
            <span className="text-primary text-lg ml-1">AI</span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-2 lg:p-4 space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                cn(
                  "flex items-center justify-center lg:justify-start gap-3 px-3 py-3 rounded-xl transition-all",
                  isActive
                    ? "bg-primary/20 text-primary shadow-sm"
                    : "text-muted-foreground hover:bg-card hover:text-foreground",
                )
              }
            >
              <Icon className="w-5 h-5 shrink-0" />
              <span className="hidden lg:block text-sm font-medium">
                {label}
              </span>
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-3 lg:p-4 border-t border-border/50">
          <div className="flex items-center justify-center lg:justify-start gap-3">
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-green-400 to-emerald-600 flex items-center justify-center shrink-0">
              <span className="text-sm text-white font-medium">U</span>
            </div>
            <div className="hidden lg:block flex-1 min-w-0">
              <p className="text-sm font-medium truncate">用户</p>
              <p className="text-xs text-muted-foreground">在线</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden relative z-10">
        <Outlet />
      </main>
    </div>
  );
}
