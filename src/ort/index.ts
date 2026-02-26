import * as _ort from 'onnxruntime-node';

/**
 * CommonJS/ESM bridge for onnxruntime-node.
 */
const _actualOrt = (_ort as any).default || _ort;

export { _actualOrt as ort };

