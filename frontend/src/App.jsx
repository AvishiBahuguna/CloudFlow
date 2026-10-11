import {
  Routes,
  Route,
  Navigate
} from "react-router-dom";

import AppLayout from "./components/layout/AppLayout.jsx";

import NewRunPage from "./pages/NewRunPage.jsx";
import LivePipelinePage from "./pages/LivePipelinePage.jsx";
import HistoryPage from "./pages/HistoryPage.jsx";
import ExperimentsPage from "./pages/ExperimentsPage.jsx";
import SettingsPage from "./pages/SettingsPage.jsx";

function App() {
  return (
    <Routes>

      <Route element={<AppLayout />}>

        <Route
          path="/"
          element={
            <Navigate to="/new-run" replace />
          }
        />

        <Route
          path="/new-run"
          element={<NewRunPage />}
        />

        <Route
          path="/pipeline"
          element={<LivePipelinePage />}
        />

        <Route
          path="/history"
          element={<HistoryPage />}
        />

        <Route
          path="/experiments"
          element={<ExperimentsPage />}
        />

        <Route
          path="/settings"
          element={<SettingsPage />}
        />

        <Route
          path="*"
          element={
            <Navigate to="/new-run" replace />
          }
        />

        <Route
          path="/pipeline/:runId"
          element={<LivePipelinePage />}
        />
      </Route>

    </Routes>
  );
}

export default App;
