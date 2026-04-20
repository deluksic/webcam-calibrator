const fs = require("fs");
const path = require("path");

const BUILD_LOG = "/tmp/ship-watch.log";
const ALERT_LOG = "/tmp/gpu-build-errors.log";

const WATCHER_PID = process.pid;

console.log(`[GPU Watcher ${WATCHER_PID}] watching ${BUILD_LOG} for errors...`);

let lastLine = "";

setInterval(() => {
  if (!fs.existsSync(BUILD_LOG)) return;

  const content = fs.readFileSync(BUILD_LOG, "utf8");
  const lines = content.split("\n");

  // Check for new lines
  for (const line of lines) {
    if (line === lastLine) continue;
    if (line.includes("[GPU Watcher")) continue;

    if (
      line.includes("Type checking warnings:") ||
      line.includes("error TS") ||
      line.includes("Cannot find name")
    ) {
      const alert = `[${new Date().toISOString()}] GPU Build Error:\n${line}\n`;
      fs.appendFileSync(ALERT_LOG, alert);
      console.error(alert.trim());
    }
  }

  lastLine = lines[lines.length - 1] || "";
}, 1000);
