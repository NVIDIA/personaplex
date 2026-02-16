import { useState, useCallback, useMemo } from "react";
import ReactDOM from "react-dom/client";
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";
import "./index.css";
import { Queue } from "./pages/Queue/Queue";
import { I18nContext, Language, translate } from "./i18n";

const App = () => {
  const [language, setLanguage] = useState<Language>("en");

  const t = useCallback(
    (key: string) => translate(language, key),
    [language],
  );

  const i18nValue = useMemo(
    () => ({ language, setLanguage, t }),
    [language, t],
  );

  const router = useMemo(
    () =>
      createBrowserRouter([
        {
          path: "/",
          element: <Queue />,
        },
      ]),
    [],
  );

  return (
    <I18nContext.Provider value={i18nValue}>
      <RouterProvider router={router} />
    </I18nContext.Provider>
  );
};

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <App />
);
