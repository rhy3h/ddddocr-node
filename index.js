const path = require('node:path');
const fs = require('node:fs');

const { OCR } = require('./core/ocr');
const { Detection } = require('./core/detection');
const { CHARSET_RANGE } = require('ddddocr-core');

/**
 * Model type constants representing different OCR models.
 * 
 * @readonly
 * @enum {number}
 */
const MODEL_TYPE = {
    /**
     * OCR model type.
     * @type {number}
     */
    OCR: 0,
    /**
     * OCR Beta model type.
     * @type {number}
     */
    OCR_BETA: 1
}

/**
 * Class representing an OCR (Optical Character Recognition) model.
 * @class DdddOcr
 */
class DdddOcr {
    /**
     * @type {string} Path to the ONNX model for standard OCR.
     * @private
     */
    _ocrOnnxPath = path.join(__dirname, './onnx/common_old.onnx');
    /**
     * @type {string} Path to the charset file for standard OCR.
     * @private
     */
    _charsetPath = path.join(__dirname, './onnx/common_old.json');

    /**
     * @type {string} Path to the ONNX model for beta OCR.
     * @private
     */
    _ocrBetaOnnxPath = path.join(__dirname, './onnx/common.onnx');
    /**
     * @type {string} Path to the charset file for beta OCR.
     * @private
     */
    _charsetBetaPath = path.join(__dirname, './onnx/common.json');

    /**
     * @type {string} Path to the ONNX model for OCR detection.
     * @private
     */
    _ocrDetectionOnnxPath = path.join(__dirname, './onnx/common_det.onnx');

    /**
     * @type {OCR}
     * @private
     */
    _ocr = null;

    /**
     * @type {OCR}
     * @private
     */
    _ocrBeta = null;

    /**
     * @type {MODEL_TYPE}
     * @private
     */
    _ocrMode = MODEL_TYPE.OCR;

    /**
     * @type {Detection}
     * @private
     */
    _detection = null;

    constructor() {
        this._ocr = new OCR(this._ocrOnnxPath, this._charsetPath);
        this._ocrBeta = new OCR(this._ocrBetaOnnxPath, this._charsetBetaPath);

        this._detection = new Detection(this._ocrDetectionOnnxPath);
    }

    /**
     * Enables the debug mode and prepares the debug folder.
     * 
     * @returns {DdddOcr} The current instance for method chaining.
     */
    enableDebug() {
        this._detection.enableDebug();

        const debugFolderPath = 'debug';

        if (fs.existsSync(debugFolderPath)) {
            fs.rmSync(debugFolderPath, { recursive: true, force: true });
        }

        fs.mkdirSync(debugFolderPath);

        return this;
    }

    /**
     * Sets the OCR mode for the instance.
     *
     * @param {MODEL_TYPE} mode - The OCR mode to set. Must be one of the values from `MODEL_TYPE`.
     * @returns {DdddOcr} The current instance for method chaining.
     */
    setOcrMode(mode) {
        switch (mode) {
            case MODEL_TYPE.OCR: 
            case MODEL_TYPE.OCR_BETA: {
                this._ocrMode = mode;
                break;
            }
            default: {
                throw new Error('Not support mode');
            }
        }

        return this;
    }

    /**
     * Sets the range restriction for OCR results.
     * 
     * This method restricts the characters returned by OCR based on the input:
     * - For `number` input, it applies a predefined character set. See the `CHARSET_RANGE` type for the available options.
     * - For `string` input, each character in the string is treated as a valid OCR result.
     * 
     * @public
     * @param {CHARSET_RANGE} charsetRange - A number for predefined character sets or a string for a custom character set.
     * @see https://rhy3h.github.io/ddddocr-core/global.html#CHARSET_RANGE
     * @returns {DdddOcr} The current instance for method chaining.
     */
    setRanges(charsetRange) {
        this._ocr.setRanges(charsetRange);
        this._ocrBeta.setRanges(charsetRange);

        return this;
    }

    /**
     * Classifies an image by running it through an OCR model.
     * 
     * @param {string | Buffer | ArrayBuffer} url - The image to classify. It can be a file path (string) or image data (Buffer).
     * @returns {Promise<string>} A promise that resolves to the OCR result, represented as a string of recognized characters.
     */
    async classification(url) {
        if (this._ocrMode === MODEL_TYPE.OCR_BETA) {
            return this._ocrBeta.classification(url);
        } else {
            return this._ocr.classification(url);
        }
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
        return this._detection.detection(url);
    }
}

module.exports = {
    DdddOcr,
    CHARSET_RANGE,
    MODEL_TYPE
};