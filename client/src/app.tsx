import ReactDOM from "react-dom/client";
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";
import "./index.css";
import { Queue } from "./pages/Queue/Queue";
import { SimpleTest } from "./pages/SimpleTest/SimpleTest";

const router = createBrowserRouter([
  {
    path: "/",
    element: <SimpleTest />,
  },
  {
    path: "/full",
    element: <Queue />,
  },
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <RouterProvider router={router}/>
);
