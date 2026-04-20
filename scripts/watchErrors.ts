import { watch } from "node:fs";
import { join } from "node:path";

const BUILD_LOG = "/tmp/ship-watch.log";
const ALERT_FILE = "/tmp/gpu-build-alert.txt";

watch(BUILD_LOG, (eventType) => {
  if (eventType === "change") {
    const fs = require("fs");
    const content = fs.readFileSync(BUILD_LOG, "utf8");
    // Check for TypeGPU-specific errors/warnings
    if (content.includes("Type checking warnings:") || content.includes("error TS")) {
      fs.writeFileSync(ALERT_FILE, new Date().toISOString() + "\n" + content);
      console.log("GPU build has errors/warnings");
    } else {
      fs.unlinkSync(ALERT_FILE);
    }
  }
});

console.log("Watching " + BUILD_LOG + " for GPU errors...");
