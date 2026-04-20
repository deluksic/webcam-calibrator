import { Loading, render } from "@solidjs/web";
import { App } from "./components/App";
import "./styles/global.css";
import "./version";

const root = document.getElementById("root");
if (!root) throw new Error("#root element not found");

render(
  () => (
    <Loading>
      <App />
    </Loading>
  ),
  root,
);
