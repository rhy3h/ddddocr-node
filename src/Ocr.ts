import fsm from 'node:fs/promises';

import * as ort from 'onnxruntime-node';

import { OCRBase } from 'ddddocr-core';

class OCR extends OCRBase {
    private _ocrOrtSessionPending!: Promise<[ort.InferenceSession, string[]]>;

    constructor(onnxPath: string, charsetPath: string) {
        super(onnxPath, charsetPath);
    }

    private async _loadCharset(charsetPath: string): Promise<string[]> {
        return fsm.readFile(charsetPath, { encoding: 'utf-8' })
            .then((result) => {
                return JSON.parse(result);
            });
    }

    private _loadOcrOrtSession() {
        if (!this._ocrOrtSessionPending) {
            const ocrOnnxPromise = ort.InferenceSession.create(this._ocrOnnxPath);
            const charsetPromise = this._loadCharset(this._charsetPath);
            this._ocrOrtSessionPending = Promise.all([ocrOnnxPromise, charsetPromise]);
        }

        return this._ocrOrtSessionPending;
    }

    private async _run(inputTensor: ort.Tensor) {
        if (!this._ocrOrtSessionPending) {
            this._loadOcrOrtSession();
        }

        const [ortSession, charset] = await this._ocrOrtSessionPending;
        const result = await ortSession.run({ input1: inputTensor });

        const onnxValue = result['387'];

        return {
            cpuData: onnxValue.data as Float32Array,
            dims: onnxValue.dims as number[],
            charset
        };
    }

    /**
     * Classifies an image by running it through an OCR model.
     */
    public async classification(url: string) {
        const { floatData, targetHeight, targetWidth } = await this._preProcessImage(url);

        const inputTensor = new ort.Tensor('float32', floatData, [1, 1, targetHeight, targetWidth]);

        const { cpuData, dims, charset } = await this._run(inputTensor);

        const result = await this.postProcess(cpuData, dims, charset);

        return result;
    }
}

export { OCR };