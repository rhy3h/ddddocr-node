const ort = require('onnxruntime-node');

const { DetectionBase } = require('ddddocr-core');

class Detection extends DetectionBase {
    /**
     * @type {Promise<ort.InferenceSession>|null} A promise for loading the OCR detection ONNX inference session.
     * @private
     */
    _ocrDetectionOrtSessionPending = null;

    constructor(onnxPath) {
        super(onnxPath);
    }

    /**
     * Loads the OCR detection ONNX model asynchronously, storing the promise to avoid redundant loading.
     * 
     * @private
     * @returns {Promise<ort.InferenceSession>} A promise that resolves with the OCR detection ONNX inference session.
     */
    _loadDetectionOrtSession() {
        if (!this._ocrDetectionOrtSessionPending) {
            const ocrOnnxPromise = ort.InferenceSession.create(this._ortOnnxPath);
            this._ocrDetectionOrtSessionPending = ocrOnnxPromise;
        }

        return this._ocrDetectionOrtSessionPending;
    }

    async _runDetection(inputTensor) {
        if (!this._ocrDetectionOrtSessionPending) {
            this._loadDetectionOrtSession();
        }

        const ortSession = await this._ocrDetectionOrtSessionPending;
        const result = await ortSession.run({ images: inputTensor });

        return result['output'];
    }

    /**
     * Detects objects in an image by extracting bounding boxes and optionally 
     * visualizing the detection results on the image.
     * 
     * This method reads the image, retrieves bounding boxes, and then optionally 
     * draws the detected bounding boxes on the image if debugging is enabled.
     * 
     * @public
     * @param {string | Buffer | ArrayBuffer} url - The image to classify. It can be a file path (string) or image data (Buffer).
     * @returns {Promise<Array<Array<number>>>} A promise that resolves to an array of bounding boxes, 
     *          where each box is represented as [x1, y1, x2, y2], adjusted to the image size.
     */
    async detection(url) {
        const { inputArray, inputSize, width, height, ratio } = await this._preProcessImage(url);

        const inputTensor = new ort.Tensor('float32', inputArray, [1, 3, inputSize[0], inputSize[1]]);

        const { cpuData, dims } = await this._runDetection(inputTensor);

        const result = this._postProcess(cpuData, dims, inputSize, width, height, ratio);

        return result;
    }
}

module.exports = {
    Detection
}