/**
 * DocuMind AI - 侧边栏导航组件
 */

import { Link, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  FileText,
  Database,
  Settings,
  Brain,
} from "lucide-react";

interface NavItem {
  icon: React.ElementType;
  label: string;
  path: string;
}

const navItems: NavItem[] = [
  { icon: LayoutDashboard, label: "Dashboard", path: "/" },
  { icon: FileText, label: "Documents", path: "/documents" },
  { icon: Database, label: "Knowledge Bases", path: "/" },
  { icon: Settings, label: "Settings", path: "/settings" },
];

export function Sidebar() {
  const location = useLocation();

  return (
    <aside className="w-16 lg:w-64 bg-card/50 backdrop-blur-xl border-r border-border/50 flex flex-col py-6 shrink-0">
      {/* Logo */}
      <div className="px-4 mb-8 flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center">
          <Brain className="w-6 h-6 text-white" />
        </div>
        <span className="hidden lg:block text-xl font-semibold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
          DocuMind AI
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;

          return (
            <Link
              key={item.path + item.label}
              to={item.path}
              className={`
                flex items-center gap-3 px-3 py-3 rounded-xl mb-1 transition-all
                ${
                  isActive
                    ? "bg-primary/20 text-primary"
                    : "text-muted-foreground hover:bg-card hover:text-foreground"
                }
              `}
            >
              <Icon className="w-5 h-5 shrink-0" />
              <span className="hidden lg:block text-sm font-medium">
                {item.label}
              </span>
            </Link>
          );
        })}
      </nav>

      {/* User */}
      <div className="px-4 pt-4 border-t border-border/50">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-gradient-to-br from-green-400 to-emerald-600 flex items-center justify-center text-white font-medium">
            U
          </div>
          <div className="hidden lg:block">
            <div className="text-sm font-medium">User</div>
            <div className="text-xs text-muted-foreground">Admin</div>
          </div>
        </div>
      </div>
    </aside>
  );
}
