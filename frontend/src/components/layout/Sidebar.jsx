import { NavLink } from "react-router-dom";
import { Plus, Activity, History, FlaskConical, Settings } from "lucide-react";
import "./Sidebar.css";

const navigationItems = [
  { name: "New Run", path: "/new-run", icon: Plus },
  { name: "Active Run", path: "/pipeline", icon: Activity },
  { name: "History", path: "/history", icon: History },
  { name: "Experiments", path: "/experiments", icon: FlaskConical }
];

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">DS</div>

      <nav className="sidebar-navigation" aria-label="Main navigation">
        {navigationItems.map((item) => {
          const Icon = item.icon;

          return (
            <NavLink
              key={item.name}
              to={item.path}
              end={item.path === "/pipeline"}
              title={item.name}
              aria-label={item.name}
              className={({ isActive }) =>
                isActive ? "sidebar-link active" : "sidebar-link"
              }
            >
              <Icon size={20} strokeWidth={1.8} />
            </NavLink>
          );
        })}
      </nav>

      <div className="sidebar-bottom">
        <NavLink
          to="/settings"
          title="Settings"
          aria-label="Settings"
          className={({ isActive }) =>
            isActive ? "sidebar-link active" : "sidebar-link"
          }
        >
          <Settings size={20} strokeWidth={1.8} />
        </NavLink>
      </div>
    </aside>
  );
}

export default Sidebar;