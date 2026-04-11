import path from "node:path";
import { readFile } from "node:fs/promises";
import type { CliArgs } from "../types";

const DEFAULT_MODEL = "google/nano-banana-2";
const SYNC_WAIT_SECONDS = 60;
const POLL_INTERVAL_MS = 2000;
const MAX_POLL_MS = 300_000;
const SIZE_PRESET_PATTERN = /^\d+K$/i;
const SEEDREAM_45_SIZES = new Set(["2K", "4K"]);
const SEEDREAM_5_LITE_SIZES = new Set(["2K", "3K"]);
const WAN_PRO_PRESET_SIZES = new Set(["1K", "2K", "4K"]);
const WAN_PRESET_SIZES = new Set(["1K", "2K"]);

export function getDefaultModel(): string {
  return process.env.REPLICATE_IMAGE_MODEL || DEFAULT_MODEL;
}

function getApiToken(): string | null {
  return process.env.REPLICATE_API_TOKEN || null;
}

function getBaseUrl(): string {
  const base = process.env.REPLICATE_BASE_URL || "https://api.replicate.com";
  return base.replace(/\/+$/g, "");
}

export function parseModelId(model: string): { owner: string; name: string; version: string | null } {
  const [ownerName, version] = model.split(":");
  const parts = ownerName!.split("/");
  if (parts.length !== 2 || !parts[0] || !parts[1]) {
    throw new Error(
      `Invalid Replicate model format: "${model}". Expected "owner/name" or "owner/name:version".`
    );
  }
  return { owner: parts[0], name: parts[1], version: version || null };
}

function isNanoBananaModel(model: string): boolean {
  return model.startsWith("google/nano-banana");
}

function isSeedreamModel(model: string): boolean {
  return model.startsWith("bytedance/seedream-4.5") || model.startsWith("bytedance/seedream-5-lite");
}

function isSeedream45Model(model: string): boolean {
  return model.startsWith("bytedance/seedream-4.5");
}

function isSeedream5LiteModel(model: string): boolean {
  return model.startsWith("bytedance/seedream-5-lite");
}

function isWanModel(model: string): boolean {
  return model.startsWith("wan-video/wan-2.7-image");
}

function isWanProModel(model: string): boolean {
  return model.startsWith("wan-video/wan-2.7-image-pro");
}

function parsePixelSize(size: string): { width: number; height: number } | null {
  const match = size.trim().match(/^(\d+)\s*[xX*]\s*(\d+)$/);
  if (!match) return null;

  const width = Number.parseInt(match[1]!, 10);
  const height = Number.parseInt(match[2]!, 10);

  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }

  return { width, height };
}

function normalizePixelSize(size: string): string {
  const parsed = parsePixelSize(size);
  if (!parsed) return size;
  return `${parsed.width}*${parsed.height}`;
}

function isPresetSize(size: string): boolean {
  return SIZE_PRESET_PATTERN.test(size.trim());
}

function getSeedreamDefaultSize(model: string, quality: CliArgs["quality"]): string | null {
  if (!isSeedreamModel(model) || !quality) return null;
  return "2K";
}

function getWanDefaultSize(quality: CliArgs["quality"]): string | null {
  if (quality === "normal") return "1K";
  if (quality === "2k") return "2K";
  return null;
}

function getAllowedSeedreamSizes(model: string): Set<string> {
  return isSeedream45Model(model) ? SEEDREAM_45_SIZES : SEEDREAM_5_LITE_SIZES;
}

function getAllowedWanSizes(model: string): Set<string> {
  return isWanProModel(model) ? WAN_PRO_PRESET_SIZES : WAN_PRESET_SIZES;
}

function normalizePresetSize(size: string): string {
  return size.trim().toUpperCase();
}

function buildNanoBananaInput(prompt: string, args: CliArgs, referenceImages: string[]): Record<string, unknown> {
  const input: Record<string, unknown> = { prompt };

  if (args.aspectRatio) {
    input.aspect_ratio = args.aspectRatio;
  } else if (referenceImages.length > 0) {
    input.aspect_ratio = "match_input_image";
  }

  if (args.quality === "normal") {
    input.resolution = "1K";
  } else if (args.quality === "2k") {
    input.resolution = "2K";
  }

  if (args.n > 1) {
    input.number_of_images = args.n;
  }

  input.output_format = "png";

  if (referenceImages.length > 0) {
    input.image_input = referenceImages;
  }

  return input;
}

function buildSeedreamInput(prompt: string, model: string, args: CliArgs, referenceImages: string[]): Record<string, unknown> {
  const input: Record<string, unknown> = { prompt };
  const requestedSize = args.size || getSeedreamDefaultSize(model, args.quality);

  if (requestedSize) {
    if (isSeedream45Model(model)) {
      const parsed = parsePixelSize(requestedSize);
      if (parsed) {
        input.size = "custom";
        input.width = parsed.width;
        input.height = parsed.height;
      } else {
        input.size = normalizePresetSize(requestedSize);
      }
    } else {
      input.size = normalizePresetSize(requestedSize);
    }
  }

  if (args.aspectRatio && input.size !== "custom") {
    input.aspect_ratio = args.aspectRatio;
  }

  if (args.n > 1) {
    input.sequential_image_generation = "auto";
    input.max_images = args.n;
  }

  if (referenceImages.length > 0) {
    input.image_input = referenceImages;
  }

  if (isSeedream5LiteModel(model)) {
    input.output_format = "png";
  }

  return input;
}

function buildWanInput(prompt: string, model: string, args: CliArgs, referenceImages: string[]): Record<string, unknown> {
  const input: Record<string, unknown> = { prompt };
  const requestedSize = args.size || getWanDefaultSize(args.quality);

  if (requestedSize) {
    input.size = parsePixelSize(requestedSize) ? normalizePixelSize(requestedSize) : normalizePresetSize(requestedSize);
  }

  if (args.n > 1) {
    input.num_outputs = args.n;
  }

  if (referenceImages.length > 0) {
    input.images = referenceImages;
  }

  // thinking_mode only applies to pure text-to-image.
  // image_set_mode is not exposed by the current CLI, so switch only on input-image presence for now.
  input.thinking_mode = referenceImages.length === 0;

  return input;
}

export function getDefaultOutputExtension(model: string): ".png" | ".jpg" {
  if (isSeedream45Model(model)) return ".jpg";
  if (isSeedream5LiteModel(model)) return ".png";
  return ".png";
}

export function validateArgs(model: string, args: CliArgs): void {
  if (isNanoBananaModel(model) && args.n > 1) {
    throw new Error("Nano Banana models on Replicate do not support --n yet because their current schema does not expose a multi-image count field.");
  }

  if (isSeedreamModel(model)) {
    const referenceCount = args.referenceImages.length;

    if (args.size) {
      if (isSeedream45Model(model)) {
        const normalizedSize = normalizePresetSize(args.size);
        if (!getAllowedSeedreamSizes(model).has(normalizedSize) && !parsePixelSize(args.size)) {
          throw new Error(
            `Seedream 4.5 on Replicate requires --size to be one of ${Array.from(getAllowedSeedreamSizes(model)).join(", ")} or custom dimensions like 1536x1024. Received: ${args.size}`
          );
        }
      } else {
        const normalizedSize = normalizePresetSize(args.size);
        if (!getAllowedSeedreamSizes(model).has(normalizedSize)) {
          throw new Error(
            `Seedream on Replicate requires --size to be one of ${Array.from(getAllowedSeedreamSizes(model)).join(", ")}. Received: ${args.size}`
          );
        }
      }
    }

    if (referenceCount > 14) {
      throw new Error("Seedream on Replicate supports at most 14 reference images per request.");
    }

    if (args.n < 1 || args.n > 15) {
      throw new Error("Seedream on Replicate supports --n values from 1 to 15.");
    }

    if (referenceCount + args.n > 15) {
      throw new Error(
        `Seedream on Replicate allows at most 15 total images per request (reference images + generated images). Received ${referenceCount} reference images and --n ${args.n}.`
      );
    }
  }

  if (isWanModel(model)) {
    if (args.aspectRatio && !args.size) {
      throw new Error("Wan image models on Replicate require --size when using --ar. This provider does not infer size from aspect ratio.");
    }

    if (args.referenceImages.length > 9) {
      throw new Error("Wan image models on Replicate support at most 9 reference images per request.");
    }

    if (args.size) {
      const normalizedSize = parsePixelSize(args.size) ? normalizePixelSize(args.size) : normalizePresetSize(args.size);
      if (!parsePixelSize(args.size) && !getAllowedWanSizes(model).has(normalizedSize)) {
        throw new Error(
          `Wan image models on Replicate require --size to be one of ${Array.from(getAllowedWanSizes(model)).join(", ")} or custom dimensions like 1920x1080. Received: ${args.size}`
        );
      }
    }

    if (args.n < 1 || args.n > 4) {
      throw new Error("Wan image models on Replicate support --n values from 1 to 4 in standard mode.");
    }

    if (args.size && normalizePresetSize(args.size) === "4K" && args.referenceImages.length > 0) {
      throw new Error("Wan 2.7 Image Pro on Replicate only supports 4K for text-to-image requests without input images.");
    }
  }
}

export function buildInput(
  prompt: string,
  model: string,
  args: CliArgs,
  referenceImages: string[]
): Record<string, unknown> {
  if (isSeedreamModel(model)) {
    return buildSeedreamInput(prompt, model, args, referenceImages);
  }

  if (isWanModel(model)) {
    return buildWanInput(prompt, model, args, referenceImages);
  }

  // Fall back to nano-banana schema for unknown Replicate models.
  // This preserves backward compatibility; unsupported models will fail
  // at API validation time if they reject nano-banana-style fields.
  return buildNanoBananaInput(prompt, args, referenceImages);
}

async function readImageAsDataUrl(p: string): Promise<string> {
  const buf = await readFile(p);
  const ext = path.extname(p).toLowerCase();
  let mimeType = "image/png";
  if (ext === ".jpg" || ext === ".jpeg") mimeType = "image/jpeg";
  else if (ext === ".gif") mimeType = "image/gif";
  else if (ext === ".webp") mimeType = "image/webp";
  return `data:${mimeType};base64,${buf.toString("base64")}`;
}

type PredictionResponse = {
  id: string;
  status: string;
  output: unknown;
  error: string | null;
  urls?: { get?: string };
};

async function createPrediction(
  apiToken: string,
  model: { owner: string; name: string; version: string | null },
  input: Record<string, unknown>,
  sync: boolean
): Promise<PredictionResponse> {
  const baseUrl = getBaseUrl();

  let url: string;
  const body: Record<string, unknown> = { input };

  if (model.version) {
    url = `${baseUrl}/v1/predictions`;
    body.version = model.version;
  } else {
    url = `${baseUrl}/v1/models/${model.owner}/${model.name}/predictions`;
  }

  const headers: Record<string, string> = {
    Authorization: `Bearer ${apiToken}`,
    "Content-Type": "application/json",
  };

  if (sync) {
    headers["Prefer"] = `wait=${SYNC_WAIT_SECONDS}`;
  }

  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Replicate API error (${res.status}): ${err}`);
  }

  return (await res.json()) as PredictionResponse;
}

async function pollPrediction(apiToken: string, getUrl: string): Promise<PredictionResponse> {
  const start = Date.now();

  while (Date.now() - start < MAX_POLL_MS) {
    const res = await fetch(getUrl, {
      headers: { Authorization: `Bearer ${apiToken}` },
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Replicate poll error (${res.status}): ${err}`);
    }

    const prediction = (await res.json()) as PredictionResponse;

    if (prediction.status === "succeeded") return prediction;
    if (prediction.status === "failed" || prediction.status === "canceled") {
      throw new Error(`Replicate prediction ${prediction.status}: ${prediction.error || "unknown error"}`);
    }

    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }

  throw new Error(`Replicate prediction timed out after ${MAX_POLL_MS / 1000}s`);
}

export function extractOutputUrl(prediction: PredictionResponse): string {
  const output = prediction.output;

  if (typeof output === "string") return output;

  if (Array.isArray(output)) {
    const first = output[0];
    if (typeof first === "string") return first;
  }

  if (output && typeof output === "object" && "url" in output) {
    const url = (output as Record<string, unknown>).url;
    if (typeof url === "string") return url;
  }

  throw new Error(`Unexpected Replicate output format: ${JSON.stringify(output)}`);
}

async function downloadImage(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download image from Replicate: ${res.status}`);
  const buf = await res.arrayBuffer();
  return new Uint8Array(buf);
}

export async function generateImage(
  prompt: string,
  model: string,
  args: CliArgs
): Promise<Uint8Array> {
  const apiToken = getApiToken();
  if (!apiToken) throw new Error("REPLICATE_API_TOKEN is required. Get one at https://replicate.com/account/api-tokens");

  validateArgs(model, args);

  const parsedModel = parseModelId(model);

  const refDataUrls: string[] = [];
  for (const refPath of args.referenceImages) {
    refDataUrls.push(await readImageAsDataUrl(refPath));
  }

  const input = buildInput(prompt, model, args, refDataUrls);

  console.log(`Generating image with Replicate (${model})...`);

  let prediction = await createPrediction(apiToken, parsedModel, input, true);

  if (prediction.status !== "succeeded") {
    if (!prediction.urls?.get) {
      throw new Error("Replicate prediction did not return a poll URL");
    }
    console.log("Waiting for prediction to complete...");
    prediction = await pollPrediction(apiToken, prediction.urls.get);
  }

  console.log("Generation completed.");

  const outputUrl = extractOutputUrl(prediction);
  return downloadImage(outputUrl);
}
