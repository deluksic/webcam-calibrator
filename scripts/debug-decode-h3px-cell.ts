/**
 * Forensics: homography-mismatch stress decode — vote tallies, sample dots, and a **PNG** of per-pixel
 * votes **into one 8×8 module** (`targetMi`). Votes match **`decodeTagPattern`** (**radial** `g·(p−c)`).
 *
 * Usage:
 * `pnpm exec tsx scripts/debug-decode-h3px-cell.ts [wh=96] [targetMi=10] [supersample=4] [hMaxAxisPx=3]`
 *
 * `hMaxAxisPx` — max |Δx|,|Δy| per decode corner vs raster (~px), via
 * `decodeStressHomographyMismatchScaleForMaxAxisPx`. **0** = matched homography.
 *
 * PNG: `output/decode-stress/forensics/h{h}px-w{wh}-ss{ss}-mi{target}-votes-on-raster-rgba.png`
 *
 * Inner 6×6 (row=0,col=1) → pattern index 1 → mx=2,my=1 → mi=10.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";

import { computeHomography } from "../src/lib/geometry";
import { imagePixelToUnitSquareUv } from "../src/lib/aprilTagRaycast";
import {
  buildTagGrid,
  decodeEdgeDistanceUv,
  decodeTagPatternWithVoteMaps,
  decodeTriangleFromLocalUv,
  decodeVoteBinRadialDot,
  decodeVoteModuleIndices,
  imageSobelToTagGradient,
  minQuadEdgeLengthPx,
  primaryModuleFromUv,
} from "../src/lib/grid";
import {
  decodeStressCornersGridOrder,
  decodeStressFitPerspectiveStrip,
  decodeStressHomographyMismatchScaleForMaxAxisPx,
  decodeStressRasterSobel,
  decodeStressStripWithHomographyMismatchOffsetsPx,
} from "../src/lib/decodeStressHarness";
import { TAG36H11_CODES, codeToPattern } from "../src/lib/tag36h11";

const TAG = 8;
const DOT_EPS = 1e-8;
const MIN_VOTE_TOTAL = 3;
const SPECKLE = 0;
const TAG_ID = 0;

const scriptDir = dirname(fileURLToPath(import.meta.url));
const FORENSICS_OUT = join(scriptDir, "..", "output", "decode-stress", "forensics");

function homoMaxAxisTagForFilename(maxAxisPx: number): string {
  return String(maxAxisPx).replace(/\./g, "p");
}

function writeVotesOnRasterPng(
  wh: number,
  targetMi: number,
  ss: number,
  hMaxAxisPx: number,
  voteKind: Uint8Array,
  intensity: Float32Array,
): string {
  mkdirSync(FORENSICS_OUT, { recursive: true });
  const rgba = Buffer.alloc(wh * wh * 4);
  const voteAlpha = 0.72;
  for (let i = 0; i < wh * wh; i++) {
    const o = i * 4;
    const k = voteKind[i]!;
    const grey = Math.round(Math.min(255, Math.max(0, intensity[i]! * 255)));
    if (k === 0) {
      rgba[o] = grey;
      rgba[o + 1] = grey;
      rgba[o + 2] = grey;
      rgba[o + 3] = 255;
      continue;
    }
    let vr = 0;
    let vg = 0;
    let vb = 0;
    if (k === 1) {
      vr = 40;
      vg = 220;
      vb = 90;
    } else if (k === 2) {
      vr = 255;
      vg = 70;
      vb = 70;
    } else {
      vr = 255;
      vg = 220;
      vb = 40;
    }
    const a = voteAlpha;
    const ia = 1 - a;
    rgba[o] = Math.round(vr * a + grey * ia);
    rgba[o + 1] = Math.round(vg * a + grey * ia);
    rgba[o + 2] = Math.round(vb * a + grey * ia);
    rgba[o + 3] = 255;
  }
  const hTag = homoMaxAxisTagForFilename(hMaxAxisPx);
  const name = `h${hTag}px-w${wh}-ss${ss}-mi${targetMi}-votes-on-raster-rgba.png`;
  const path = join(FORENSICS_OUT, name);
  writeFileSync(path, PNG.sync.write({ width: wh, height: wh, data: rgba }));
  return path;
}

function classify(pos: number, neg: number): string {
  const sum = pos + neg;
  if (sum < MIN_VOTE_TOTAL) return `-1 (sum=${sum} < ${MIN_VOTE_TOTAL})`;
  if (pos > neg) return `1 black (pos ${pos} > neg ${neg})`;
  if (neg > pos) return `0 white (neg ${neg} > pos ${pos})`;
  return `-2 tie (pos=${pos} neg=${neg})`;
}

type Sample = {
  ix: number;
  iy: number;
  u: number;
  v: number;
  tri: string;
  dot: number;
  sign: "pos" | "neg" | "skip";
};

function signFromDot(dot: number): Sample["sign"] {
  if (dot > DOT_EPS) return "pos";
  if (dot < -DOT_EPS) return "neg";
  return "skip";
}

function main() {
  const wh = Number(process.argv[2]) || 96;
  const miArg = process.argv[3] !== undefined ? Number(process.argv[3]) : NaN;
  const target = Number.isFinite(miArg) ? miArg : 10;
  const ssArg = process.argv[4] !== undefined ? Number(process.argv[4]) : NaN;
  const ss = Number.isFinite(ssArg) ? Math.max(1, Math.min(16, Math.floor(ssArg))) : 4;
  const hMaxArg = process.argv[5] !== undefined ? Number(process.argv[5]) : NaN;
  const hMaxAxisPx = Number.isFinite(hMaxArg) ? Math.max(0, hMaxArg) : 3;

  const truthPat = codeToPattern(TAG36H11_CODES[TAG_ID]!);
  const scaleDiag = decodeStressHomographyMismatchScaleForMaxAxisPx(hMaxAxisPx);
  const rasterStrip = decodeStressFitPerspectiveStrip(wh, wh);
  const decodeStrip = decodeStressStripWithHomographyMismatchOffsetsPx(rasterStrip, scaleDiag);
  const { intensity, sobel } = decodeStressRasterSobel(
    wh,
    wh,
    rasterStrip,
    truthPat,
    ss,
    TAG_ID,
    SPECKLE,
  );

  const grid = buildTagGrid(decodeStressCornersGridOrder(decodeStrip), 6);
  const { pattern, posM, negM, uvProximityMax } = decodeTagPatternWithVoteMaps(
    grid,
    sobel,
    wh,
    undefined,
    wh,
  );

  const oc = grid.outerCorners;
  const strip = [oc[0], oc[1], oc[3], oc[2]] as const;
  const h = computeHomography([...strip]);
  const lMin = minQuadEdgeLengthPx(oc);
  const uvHalf = 0.5 / TAG;
  const uvMax = Math.max(0.1 / TAG, 2 / lMin, uvHalf);

  const row = 0;
  const col = 1;
  const mxInner = col + 1;
  const myInner = row + 1;
  const miInner = myInner * TAG + mxInner;

  console.log(
    `wh=${wh} homography max-axis≈${hMaxAxisPx}px (scale=${scaleDiag.toFixed(6)}), tagId=${TAG_ID}, ss=${ss}, speckle=${SPECKLE}`,
  );
  console.log(
    `uvProximityMax=${uvProximityMax.toFixed(6)} (TAU/Lmin/half) grid lMin≈${lMin.toFixed(2)}px`,
  );
  console.log("");

  const pi = row * 6 + col;
  console.log(
    `--- inner 6×6 (row=${row},col=${col}) pattern[${pi}] → 8×8 mi=${miInner} (mx=${mxInner},my=${myInner}) ---`,
  );
  console.log(`truth[${pi}]=${truthPat[pi]} decoded[${pi}]=${pattern[pi]}`);
  console.log(
    `posM[${miInner}]=${posM[miInner]} negM[${miInner}]=${negM[miInner]} → ${classify(posM[miInner]!, negM[miInner]!)}`,
  );
  console.log("");
  if (target !== miInner) {
    console.log(`--- module mi=${target} (mx=${target % TAG}, my=${(target / TAG) | 0}) ---`);
    console.log(
      `posM[${target}]=${posM[target]} negM[${target}]=${negM[target]} → ${classify(posM[target]!, negM[target]!)}`,
    );
    console.log("");
  }

  const samples: Sample[] = [];
  let x0 = Math.min(oc[0].x, oc[1].x, oc[2].x, oc[3].x);
  let y0 = Math.min(oc[0].y, oc[1].y, oc[2].y, oc[3].y);
  let x1 = Math.max(oc[0].x, oc[1].x, oc[2].x, oc[3].x);
  let y1 = Math.max(oc[0].y, oc[1].y, oc[2].y, oc[3].y);
  const ix0 = Math.max(0, Math.floor(x0));
  const iy0 = Math.max(0, Math.floor(y0));
  const ix1 = Math.min(wh - 1, Math.ceil(x1));
  const iy1 = Math.min(wh - 1, Math.ceil(y1));

  const voteKind = new Uint8Array(wh * wh);
  let scriptPos = 0;
  let scriptNeg = 0;

  for (let iy = iy0; iy <= iy1; iy++) {
    for (let ix = ix0; ix <= ix1; ix++) {
      const { u, v, inside } = imagePixelToUnitSquareUv(h, ix + 0.5, iy + 0.5);
      if (!inside) continue;
      const gx = sobel[(iy * wh + ix) * 2]!;
      const gy = sobel[(iy * wh + ix) * 2 + 1]!;
      const mag = Math.hypot(gx, gy);
      if (mag <= 1e-12) continue;
      const { gu, gv } = imageSobelToTagGradient(h, u, v, gx, gy);
      const { mx, my } = primaryModuleFromUv(u, v);
      const fu = u * TAG - mx;
      const fv = v * TAG - my;
      const tri = decodeTriangleFromLocalUv(fu, fv);
      const dEdge = decodeEdgeDistanceUv(fu, fv);
      if (dEdge > uvMax) continue;

      for (const mi of decodeVoteModuleIndices(mx, my, tri)) {
        if (mi !== target) continue;
        const mx2 = mi % TAG;
        const my2 = (mi / TAG) | 0;
        const cu = (mx2 + 0.5) / TAG;
        const cv = (my2 + 0.5) / TAG;
        const ru = u - cu;
        const rv = v - cv;
        if (ru * ru + rv * rv < 1e-10) continue;
        const dot = decodeVoteBinRadialDot(u, v, cu, cv, gu, gv);
        const sign = signFromDot(dot);
        if (sign === "pos") scriptPos++;
        else if (sign === "neg") scriptNeg++;
        const p = iy * wh + ix;
        voteKind[p] = sign === "pos" ? 1 : sign === "neg" ? 2 : 3;
        samples.push({ ix, iy, u, v, tri, dot, sign });
      }
    }
  }

  if (scriptPos !== posM[target] || scriptNeg !== negM[target]) {
    console.warn(
      `script tallies mi=${target} pos/neg ${scriptPos}/${scriptNeg} vs decode posM/negM ${posM[target]}/${negM[target]} (should match)`,
    );
    console.log("");
  }

  const pngPath = writeVotesOnRasterPng(wh, target, ss, hMaxAxisPx, voteKind, intensity);
  console.log(`Wrote votes on intensity raster: ${pngPath}`);
  console.log("");

  samples.sort((a, b) => Math.abs(b.dot) - Math.abs(a.dot));
  console.log(`--- up to 40 strongest radial vote samples into mi=${target} ---`);
  for (const s of samples.slice(0, 40)) {
    console.log(
      `${s.ix},${s.iy} tri=${s.tri} dot=${s.dot.toExponential(4)} ${s.sign} u=${s.u.toFixed(5)} v=${s.v.toFixed(5)}`,
    );
  }
  const nPos = samples.filter((s) => s.sign === "pos").length;
  const nNeg = samples.filter((s) => s.sign === "neg").length;
  const nSkip = samples.filter((s) => s.sign === "skip").length;
  console.log(
    `total pixel-bin contributions into mi=${target}: ${samples.length} (pos ${nPos} neg ${nNeg} deadband ${nSkip})`,
  );
}

main();
