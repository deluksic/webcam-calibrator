import { runIdentityStress } from '../src/lib/pointIdentity/stress'

const args = new Map<string, string>()
for (const arg of process.argv.slice(2)) {
  const [key, value] = arg.replace(/^--/, '').split('=')
  if (key !== undefined) {
    args.set(key, value ?? 'true')
  }
}

const numberArg = (name: string, fallback: number): number => {
  const value = args.get(name)
  return value === undefined ? fallback : Number(value)
}

const booleanArg = (name: string): boolean => {
  const value = args.get(name)
  return value === '1' || value === 'true' || value === 'yes'
}

const options = {
  runs: numberArg('runs', 2000),
  depth: numberArg('depth', 12),
  pointsPerEdge: numberArg('pointsPerEdge', 4),
  columns: numberArg('columns', 8),
  baseSeed: numberArg('baseSeed', 1),
  yawDeg: numberArg('yaw', 0),
  pitchDeg: numberArg('pitch', 0),
  cameraDistance: numberArg('distance', 2200),
  sphericalDistortion: numberArg('distortion', 0),
  dropoutPercent: numberArg('dropout', 0),
  dropoutSeed: numberArg('dropoutSeed', 1),
  cropLeft: numberArg('cropLeft', 0),
  cropRight: numberArg('cropRight', 0),
  cropTop: numberArg('cropTop', 0),
  cropBottom: numberArg('cropBottom', 0),
  jitterPercent: numberArg('jitter', 0),
  recovery: booleanArg('recover'),
  minProofPower: numberArg('minProofPower', 7),
  maxLocalIterations: numberArg('localIterations', 3),
  hopRadius: numberArg('hopRadius', 8),
}

const start = Date.now()
const report = runIdentityStress(options)
const elapsed = (Date.now() - start) / 1000

console.log(
  `runs=${report.runs} depth=${report.depth} N=${report.pointsPerEdge} view=${report.view.yawDeg}/${report.view.pitchDeg} distortion=${report.view.sphericalDistortion}`,
)
console.log(
  `accuracy mean=${(report.meanAccuracy * 100).toFixed(3)}% min=${(report.minAccuracy * 100).toFixed(2)}% max=${(report.maxAccuracy * 100).toFixed(2)}%`,
)
console.log(`perfect=${report.perfectRuns}/${report.runs} >=99%=${report.atLeast99Runs}/${report.runs}`)
console.log(`falseMatchRuns=${report.falseMatchRuns} totalFalseMatches=${report.totalFalseMatches}`)
console.log(`designCollisions mean=${report.meanDesignCollisions.toFixed(2)} max=${report.maxDesignCollisions}`)
console.log(`detectedPoints mean=${report.meanDetectedPoints.toFixed(1)}`)

if (report.recovery !== undefined) {
  const recovery = report.recovery
  console.log(
    `refined accuracy mean=${(recovery.meanRefinedAccuracy * 100).toFixed(3)}% min=${(recovery.minRefinedAccuracy * 100).toFixed(2)}%`,
  )
  console.log(
    `accepted accuracy mean=${(recovery.meanAcceptedRefinedAccuracy * 100).toFixed(3)}% min=${(recovery.minAcceptedRefinedAccuracy * 100).toFixed(2)}%`,
  )
  console.log(
    `refined accepted=${recovery.acceptedRuns}/${report.runs} rejected=${recovery.rejectedRuns}/${report.runs} falseMatches=${recovery.refinedFalseMatches}`,
  )
  console.log(
    `recovered mean=${recovery.meanRecovered.toFixed(1)} predictedMissing mean=${recovery.meanPredictedMissing.toFixed(1)} predictedMissingFalse=${recovery.predictedMissingFalse}`,
  )
  console.log(
    `anchors mean=${recovery.meanAnchors.toFixed(1)} neighborhoods mean=${recovery.meanInliers.toFixed(1)} pairs mean=${recovery.meanPairs.toFixed(1)} rms mean=${recovery.meanDesignRms.toFixed(3)} conflicts mean=${recovery.meanConflicts.toFixed(2)}`,
  )
  const reasons = Object.entries(recovery.rejectionReasons)
    .map(([reason, count]) => `${reason}:${count}`)
    .join(', ')
  console.log(`rejectionReasons=${reasons || 'none'}`)
}

console.log(`elapsed=${elapsed.toFixed(2)}s`)
