#!/usr/bin/env node
import { spawn } from "node:child_process";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  statSync,
  unlinkSync,
} from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("..", import.meta.url));
const argv = process.argv.slice(2);
const args = new Set(argv);
const includeProcessed = args.has("--processed") || args.has("--all");
const includeStimuli = args.has("--stimuli") || args.has("--all");
const includeHelpers = args.has("--helpers") || args.has("--all");
const jobsArg = argv.find((arg) => arg.startsWith("--jobs="));
const concurrency = Number(jobsArg?.slice("--jobs=".length) || process.env.TRIPLEN_DOWNLOAD_JOBS || 4);

if (!includeProcessed && !includeStimuli && !includeHelpers) {
  console.error("Usage: node scripts/download_scidb_subset.mjs [--processed] [--stimuli] [--helpers] [--all] [--jobs=8]");
  process.exit(2);
}

if (!Number.isInteger(concurrency) || concurrency < 1) {
  console.error("--jobs must be a positive integer");
  process.exit(2);
}

const processedMeta = JSON.parse(
  readFileSync(path.join(root, "data/metadata/scidb_processed_files.json"), "utf8"),
);
const othersMeta = JSON.parse(
  readFileSync(path.join(root, "data/metadata/scidb_others_files.json"), "utf8"),
);

const jobs = [];

if (includeProcessed) {
  for (const item of processedMeta.data ?? []) {
    if (!item.dir && item.fileName?.endsWith(".mat")) {
      jobs.push({
        id: item.id,
        name: item.fileName,
        size: item.size,
        dest: path.join(root, "data/processed", item.fileName),
      });
    }
  }
}

if (includeStimuli) {
  const item = (othersMeta.data ?? []).find((x) => x.fileName === "StimuliNNN.zip");
  if (item) {
    jobs.push({
      id: item.id,
      name: item.fileName,
      size: item.size,
      dest: path.join(root, "data/stimuli", item.fileName),
    });
  }
}

if (includeHelpers) {
  const helperNames = new Set(["AreaXYZ.xlsx", "exclude_area.xls", "ClusInfo.mat"]);
  for (const item of othersMeta.data ?? []) {
    if (helperNames.has(item.fileName)) {
      jobs.push({
        id: item.id,
        name: item.fileName,
        size: item.size,
        dest: path.join(root, "data/metadata", item.fileName),
      });
    }
  }
}

const totalBytes = jobs.reduce((sum, job) => sum + (job.size || 0), 0);
console.log(
  `Planned downloads: ${jobs.length} files, ${(totalBytes / 1024 / 1024).toFixed(1)} MB, ${concurrency} jobs`,
);

function isComplete(job) {
  return existsSync(job.dest) && statSync(job.dest).size === job.size;
}

function download(job) {
  return new Promise((resolve, reject) => {
    mkdirSync(path.dirname(job.dest), { recursive: true });
    if (isComplete(job)) {
      console.log(`skip ${job.name}`);
      resolve();
      return;
    }

    const url = `https://china.scidb.cn/download?fileId=${job.id}`;
    if ((job.size || 0) > 32 * 1024 * 1024) {
      downloadSegmented(job, url).then(resolve, reject);
      return;
    }

    console.log(`download ${job.name}`);
    const curl = spawn("curl", ["-L", "--fail", "--retry", "3", "--silent", "--show-error", "--continue-at", "-", "-o", job.dest, url], {
      stdio: "inherit",
    });
    curl.on("exit", (code) => {
      if (code === 0 && isComplete(job)) {
        resolve();
      } else {
        reject(new Error(`download failed for ${job.name} with exit code ${code}`));
      }
    });
  });
}

function downloadRange(url, partFile, start, end) {
  return new Promise((resolve, reject) => {
    if (existsSync(partFile) && statSync(partFile).size === end - start + 1) {
      resolve();
      return;
    }

    const curl = spawn(
      "curl",
      ["-L", "--fail", "--retry", "3", "--silent", "--show-error", "--range", `${start}-${end}`, "-o", partFile, url],
      { stdio: "inherit" },
    );
    curl.on("exit", (code) => {
      if (code === 0 && existsSync(partFile) && statSync(partFile).size === end - start + 1) {
        resolve();
      } else {
        reject(new Error(`range download failed for bytes ${start}-${end} with exit code ${code}`));
      }
    });
  });
}

async function downloadSegmented(job, url) {
  console.log(`download ${job.name} in ${concurrency} byte ranges`);
  const parts = [];
  const partSize = Math.ceil(job.size / concurrency);

  await Promise.all(
    Array.from({ length: concurrency }, async (_, index) => {
      const start = index * partSize;
      const end = Math.min(job.size - 1, start + partSize - 1);
      if (start > end) {
        return;
      }

      const partFile = `${job.dest}.part-${index}`;
      parts[index] = partFile;
      await downloadRange(url, partFile, start, end);
    }),
  );

  const tmp = `${job.dest}.tmp`;
  if (existsSync(tmp)) {
    unlinkSync(tmp);
  }

  for (const partFile of parts.filter(Boolean)) {
    appendFileSync(tmp, readFileSync(partFile));
  }
  renameSync(tmp, job.dest);

  if (!isComplete(job)) {
    throw new Error(`download failed for ${job.name}: combined file has wrong size`);
  }

  for (const partFile of parts.filter(Boolean)) {
    unlinkSync(partFile);
  }
}

let next = 0;
let failed = false;

async function worker() {
  while (!failed && next < jobs.length) {
    const job = jobs[next++];
    try {
      await download(job);
    } catch (error) {
      failed = true;
      console.error(error.message);
      process.exitCode = 1;
    }
  }
}

await Promise.all(Array.from({ length: Math.min(concurrency, jobs.length) }, () => worker()));

if (!failed) {
  console.log("Downloads complete.");
}
