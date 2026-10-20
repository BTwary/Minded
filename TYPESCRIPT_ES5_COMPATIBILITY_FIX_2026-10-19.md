# MindEd TypeScript ES5 Compatibility Fix

## Issue
The Windows production build failed during TypeScript checking at `apps/web/src/lib/api.ts` because spreading a `Uint8Array` requires iterator support that is not enabled by the app's ES5 TypeScript target.

## Change
The byte-to-binary-string conversion now builds a bounded `number[]` for each 32 KiB chunk and calls `String.fromCharCode.apply(null, charCodes)`. This avoids spreading/iterating the typed array and leaves the project's compiler target unchanged. The chunk size remains 32 KiB to stay below common argument-count limits.

## Verification
- A standalone TypeScript 5.8.3 check passed using the project's relevant compiler flags (`--target es5 --lib dom,dom.iterable,esnext --strict`).
- The full Next.js production build was not run in this environment; rebuild on Windows to verify the integrated app and desktop bundle.
