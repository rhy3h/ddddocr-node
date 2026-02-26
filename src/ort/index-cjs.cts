import * as _ort from 'onnxruntime-node';

/**
 * CommonJS export for onnxruntime-node.
 * tshy will use this for the CJS build.
 */
export const ort = (_ort as any).default || _ort;
