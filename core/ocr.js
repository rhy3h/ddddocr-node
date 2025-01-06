const fsm = require('node:fs/promises');

const ort = require('onnxruntime-node');

const { OCRBase } = require('ddddocr-core');

class OCR extends OCRBase {
    /**
     * @type {Promise<ort.InferenceSession>|null} A promise for loading the OCR ONNX inference session.
     * @private
     */
    _ocrOrtSessionPending = null;

    constructor(onnxPath, charsetPath) {
        super(onnxPath, charsetPath);
    }

    /**
     * Loads a character set from a specified file path.
     * 
     * @private
     * @param {string} charsetPath - The file path to the character set. 
     * @returns {Promise<string[]>}
     */
    async _loadCharset(charsetPath) {
        return fsm.readFile(charsetPath, { encoding: 'utf-8' })
            .then((result) => {
                return JSON.parse(result);
            });
    }

    /**
     * Loads the OCR ONNX model and charset asynchronously, storing the promises to avoid redundant loading.
     * 
     * @private
     * @returns {Promise<Array<ort.InferenceSession, string[]>>} A promise that resolves with an array containing 
     *          the OCR ONNX inference session and the charset.
     */
    _loadOcrOrtSession() {
        if (!this._ocrOrtSessionPending) {
            const ocrOnnxPromise = ort.InferenceSession.create(this._ocrOnnxPath);
            const charsetPromise = this._loadCharset(this._charsetPath);
            this._ocrOrtSessionPending = Promise.all([ocrOnnxPromise, charsetPromise]);
        }

        return this._ocrOrtSessionPending;
    }

    /**
     * Runs OCR processing on the provided input tensor using an ORT session.
     * 
     * This method waits for the OCR session to be loaded (if not already loaded) and then processes the input tensor to extract OCR results.
     * It returns the processed data including the `cpuData`, `dims`, and the associated `charset`.
     * 
     * @private
     * @param {ort.Tensor} inputTensor - The input tensor to process for OCR.
     * @returns {Promise<{cpuData: Float32Array, dims: number[], charset: string}>} A promise that resolves to the OCR result containing:
     *   - `cpuData`: The raw OCR data as a `Float32Array`.
     *   - `dims`: The dimensions of the result tensor.
     *   - `charset`: The character set used for OCR processing.
     */
    async _run(inputTensor) {
        if (!this._ocrOrtSessionPending) {
            this._loadOcrOrtSession();
        }

        const [ortSession, charset] = await this._ocrOrtSessionPending;
        const result = await ortSession.run({ input1: inputTensor });

        const { cpuData, dims } = result['387'];

        return {
            cpuData,
            dims,
            charset
        };
    }

    /**
     * Classifies an image by running it through an OCR model.
     * 
     * @param {string | Buffer | ArrayBuffer} url - The image to classify. It can be a file path (string) or image data (Buffer).
     * @returns {Promise<string>} A promise that resolves to the OCR result, represented as a string of recognized characters.
     */
    async classification(url) {
        const { floatData, targetHeight, targetWidth } = await this._preProcessImage(url);

        const inputTensor = new ort.Tensor('float32', floatData, [1, 1, targetHeight, targetWidth]);

        const { cpuData, dims, charset } = await this._run(inputTensor);

        const result = await this.postProcess(cpuData, dims, charset);

        return result;
    }
}

module.exports = {
    OCR
}