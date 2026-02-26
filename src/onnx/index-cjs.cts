import path from 'node:path';

// this one has a -cjs.cts suffix, so it will override the
// module at src/onnx/index.ts in the CJS build,
// and be excluded from the esm build.
const PACKAGE_ROOT = path.dirname(require.resolve('../../../package.json'));
export const ONNX_DIR = path.join(PACKAGE_ROOT, 'onnx');
