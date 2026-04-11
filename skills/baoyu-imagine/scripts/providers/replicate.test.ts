import assert from "node:assert/strict";
import test from "node:test";

import type { CliArgs } from "../types.ts";
import {
  buildInput,
  extractOutputUrl,
  generateImage,
  getDefaultOutputExtension,
  parseModelId,
  validateArgs,
} from "./replicate.ts";

function makeArgs(overrides: Partial<CliArgs> = {}): CliArgs {
  return {
    prompt: null,
    promptFiles: [],
    imagePath: null,
    provider: null,
    model: null,
    aspectRatio: null,
    size: null,
    quality: null,
    imageSize: null,
    referenceImages: [],
    n: 1,
    batchFile: null,
    jobs: null,
    json: false,
    help: false,
    ...overrides,
  };
}

test("Replicate model parsing accepts official formats and rejects malformed ones", () => {
  assert.deepEqual(parseModelId("google/nano-banana-pro"), {
    owner: "google",
    name: "nano-banana-pro",
    version: null,
  });
  assert.deepEqual(parseModelId("owner/model:abc123"), {
    owner: "owner",
    name: "model",
    version: "abc123",
  });

  assert.throws(
    () => parseModelId("just-a-model-name"),
    /Invalid Replicate model format/,
  );
});

test("Replicate input builder keeps nano-banana mapping for compatible models", () => {
  assert.deepEqual(
    buildInput(
      "A robot painter",
      "google/nano-banana-2",
      makeArgs({
        aspectRatio: "16:9",
        quality: "2k",
      }),
      ["data:image/png;base64,AAAA"],
    ),
    {
      prompt: "A robot painter",
      aspect_ratio: "16:9",
      resolution: "2K",
      output_format: "png",
      image_input: ["data:image/png;base64,AAAA"],
    },
  );

  assert.deepEqual(
    buildInput("A robot painter", "google/nano-banana-pro", makeArgs({ quality: "normal" }), ["ref"]),
    {
      prompt: "A robot painter",
      aspect_ratio: "match_input_image",
      resolution: "1K",
      output_format: "png",
      image_input: ["ref"],
    },
  );
});

test("Replicate fallback preserves --n for unknown models", () => {
  assert.deepEqual(
    buildInput(
      "A robot painter",
      "unknown-owner/unknown-model",
      makeArgs({
        aspectRatio: "16:9",
        quality: "2k",
        n: 4,
      }),
      ["ref"],
    ),
    {
      prompt: "A robot painter",
      aspect_ratio: "16:9",
      resolution: "2K",
      number_of_images: 4,
      output_format: "png",
      image_input: ["ref"],
    },
  );
});

test("Replicate input builder maps Seedream models to size-based schema", () => {
  assert.deepEqual(
    buildInput(
      "A robot painter",
      "bytedance/seedream-4.5",
      makeArgs({
        quality: "2k",
        aspectRatio: "16:9",
        n: 4,
      }),
      ["data:image/png;base64,AAAA"],
    ),
    {
      prompt: "A robot painter",
      size: "2K",
      aspect_ratio: "16:9",
      sequential_image_generation: "auto",
      max_images: 4,
      image_input: ["data:image/png;base64,AAAA"],
    },
  );

  assert.deepEqual(
    buildInput(
      "A robot painter",
      "bytedance/seedream-5-lite",
      makeArgs({
        size: "3K",
        aspectRatio: "4:3",
      }),
      [],
    ),
    {
      prompt: "A robot painter",
      size: "3K",
      aspect_ratio: "4:3",
      output_format: "png",
    },
  );
});

test("Replicate input builder maps Wan models to their native schema", () => {
  assert.deepEqual(
    buildInput(
      "A robot painter",
      "wan-video/wan-2.7-image-pro",
      makeArgs({
        quality: "2k",
        n: 2,
      }),
      ["data:image/png;base64,AAAA"],
    ),
    {
      prompt: "A robot painter",
      size: "2K",
      num_outputs: 2,
      images: ["data:image/png;base64,AAAA"],
      thinking_mode: false,
    },
  );

  assert.deepEqual(
    buildInput(
      "A robot painter",
      "wan-video/wan-2.7-image",
      makeArgs({
        size: "2048x1152",
      }),
      [],
    ),
    {
      prompt: "A robot painter",
      size: "2048*1152",
      thinking_mode: true,
    },
  );
});

test("Replicate input builder falls back to nano-banana schema for unknown models", () => {
  assert.deepEqual(
    buildInput(
      "A robot painter",
      "unknown-owner/unknown-model",
      makeArgs({
        aspectRatio: "16:9",
        quality: "2k",
      }),
      ["ref"],
    ),
    {
      prompt: "A robot painter",
      aspect_ratio: "16:9",
      resolution: "2K",
      output_format: "png",
      image_input: ["ref"],
    },
  );
});

test("Replicate validation catches unsupported Seedream and Wan argument combinations", () => {
  assert.throws(
    () => validateArgs("bytedance/seedream-4.5", makeArgs({ size: "large" })),
    /Seedream on Replicate requires --size/,
  );

  assert.throws(
    () => validateArgs("bytedance/seedream-5-lite", makeArgs({ size: "4K" })),
    /Seedream on Replicate requires --size to be one of 2K, 3K/,
  );

  assert.throws(
    () => validateArgs("bytedance/seedream-4.5", makeArgs({ referenceImages: Array.from({ length: 15 }, () => "ref.png") })),
    /supports at most 14 reference images/,
  );

  assert.throws(
    () => validateArgs("bytedance/seedream-5-lite", makeArgs({ referenceImages: Array.from({ length: 10 }, () => "ref.png"), n: 10 })),
    /allows at most 15 total images per request/,
  );

  assert.throws(
    () => validateArgs("google/nano-banana-2", makeArgs({ n: 2 })),
    /Nano Banana models on Replicate do not support --n yet/,
  );

  assert.throws(
    () => validateArgs("wan-video/wan-2.7-image-pro", makeArgs({ aspectRatio: "16:9" })),
    /Wan image models on Replicate require --size when using --ar/,
  );

  assert.throws(
    () => validateArgs("wan-video/wan-2.7-image", makeArgs({ size: "wide" })),
    /Wan image models on Replicate require --size/,
  );

  assert.throws(
    () => validateArgs("wan-video/wan-2.7-image", makeArgs({ size: "4K" })),
    /Wan image models on Replicate require --size to be one of/,
  );

  assert.throws(
    () => validateArgs("wan-video/wan-2.7-image-pro", makeArgs({ size: "4K", referenceImages: ["ref"] })),
    /only supports 4K for text-to-image requests without input images/,
  );

  assert.throws(
    () => validateArgs("wan-video/wan-2.7-image-pro", makeArgs({ n: 5 })),
    /support --n values from 1 to 4/,
  );
});

test("Replicate output extraction supports string, array, and object URLs", () => {
  assert.equal(
    extractOutputUrl({ output: "https://example.com/a.png" } as never),
    "https://example.com/a.png",
  );
  assert.equal(
    extractOutputUrl({ output: ["https://example.com/b.png"] } as never),
    "https://example.com/b.png",
  );
  assert.equal(
    extractOutputUrl({ output: { url: "https://example.com/c.png" } } as never),
    "https://example.com/c.png",
  );

  assert.throws(
    () => extractOutputUrl({ output: { invalid: true } } as never),
    /Unexpected Replicate output format/,
  );
});

test("Replicate default output extension matches model family behavior", () => {
  assert.equal(getDefaultOutputExtension("bytedance/seedream-4.5"), ".jpg");
  assert.equal(getDefaultOutputExtension("bytedance/seedream-5-lite"), ".png");
  assert.equal(getDefaultOutputExtension("google/nano-banana-2"), ".png");
});

test("Replicate generateImage validates arguments before making API requests", async () => {
  const previousToken = process.env.REPLICATE_API_TOKEN;
  process.env.REPLICATE_API_TOKEN = "test-token";

  try {
    await assert.rejects(
      generateImage(
        "A robot painter",
        "wan-video/wan-2.7-image-pro",
        makeArgs({ aspectRatio: "16:9" }),
      ),
      /Wan image models on Replicate require --size when using --ar/,
    );
  } finally {
    if (previousToken === undefined) {
      delete process.env.REPLICATE_API_TOKEN;
    } else {
      process.env.REPLICATE_API_TOKEN = previousToken;
    }
  }
});
