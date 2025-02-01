import * as ort from 'onnxruntime-node';

import { DetectionBase } from 'ddddocr-core';

class Detection extends DetectionBase {
    private _ocrDetectionOrtSessionPending!: Promise<ort.InferenceSession>;

    constructor(onnxPath: string) {
        super(onnxPath);
    }

    private _loadDetectionOrtSession() {
        if (!this._ocrDetectionOrtSessionPending) {
            const ocrOnnxPromise = ort.InferenceSession.create(this._ortOnnxPath);
            this._ocrDetectionOrtSessionPending = ocrOnnxPromise;
        }

        return this._ocrDetectionOrtSessionPending;
    }

    private async _runDetection(inputTensor: ort.Tensor) {
        if (!this._ocrDetectionOrtSessionPending) {
            this._loadDetectionOrtSession();
        }

        const ortSession = await this._ocrDetectionOrtSessionPending;
        const result = await ortSession.run({ images: inputTensor });

        const onnxValue = result['output'];

        return {
            cpuData: onnxValue.data as Float32Array,
            dims: onnxValue.dims as number[],
        };
    }

    /**
     * Detects objects in an image by extracting bounding boxes and optionally 
     * visualizing the detection results on the image.
     * 
     * This method reads the image, retrieves bounding boxes, and then optionally 
     * draws the detected bounding boxes on the image if debugging is enabled.
     */
    public async detection(url: string) {
        const { inputArray, inputSize, width, height, ratio } = await this._preProcessImage(url);

        const inputTensor = new ort.Tensor('float32', inputArray, [1, 3, inputSize[0], inputSize[1]]);

        const { cpuData, dims } = await this._runDetection(inputTensor) as any;

        const result = this._postProcess(cpuData, dims, inputSize, width, height, ratio);

        return result;
    }
}

export { Detection };