import path from 'node:path';

const PACKAGE_ROOT = path.dirname(require.resolve('../../../package.json'));
export const ONNX_DIR = path.join(PACKAGE_ROOT, 'onnx');
