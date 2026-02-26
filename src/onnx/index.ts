import path from 'node:path';
import { fileURLToPath } from 'node:url';

// @ts-ignore - CJS build will use index-cjs.cts instead
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PACKAGE_ROOT = path.resolve(__dirname, '../../..');
export const ONNX_DIR = path.join(PACKAGE_ROOT, 'onnx');
