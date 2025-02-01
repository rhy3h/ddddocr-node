import path from 'node:path';
import fs from 'node:fs';

import { CHARSET_RANGE } from 'ddddocr-core';

import { OCR } from './Ocr';
import { Detection } from './Detection';

/**
 * Model type constants representing different OCR models.
 */
enum MODEL_TYPE {
    /**
     * OCR model type.
     */
    OCR = 0,
    /**
     * OCR Beta model type.
     */
    OCR_BETA = 1
}

class DdddOcr {
    private _ocrOnnxPath: string;
    private _charsetPath: string;

    private _ocrBetaOnnxPath: string;
    private _charsetBetaPath: string;

    private _ocrDetectionOnnxPath: string;

    private _ocr: OCR;
    private _ocrBeta: OCR;
    private _ocrMode: MODEL_TYPE = MODEL_TYPE.OCR;

    private _detection: Detection;

    /**
     * Class representing an OCR (Optical Character Recognition) model.
     */
    constructor() {
        const root = path.join(__dirname, '..');

        this._ocrOnnxPath = `${root}/onnx/common_old.onnx`;
        this._charsetPath = `${root}/onnx/common_old.json`;

        this._ocrBetaOnnxPath = `${root}/onnx/common.onnx`;
        this._charsetBetaPath = `${root}/onnx/common.json`;

        this._ocrDetectionOnnxPath = `${root}/onnx/common_det.onnx`;

        this._ocr = new OCR(this._ocrOnnxPath, this._charsetPath);
        this._ocrBeta = new OCR(this._ocrBetaOnnxPath, this._charsetBetaPath);

        this._detection = new Detection(this._ocrDetectionOnnxPath);
    }

    /**
     * Enables the debug mode and prepares the debug folder.
     */
    public enableDebug(): this {
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
     */
    public setOcrMode(mode: MODEL_TYPE): this {
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
     * - For `number` input, it applies a predefined character set. See the [CHARSET_RANGE](https://rhy3h.github.io/ddddocr-core/enums/CHARSET_RANGE.html) type for the available options.
     * - For `string` input, each character in the string is treated as a valid OCR result.
     */
    public setRanges(charsetRange: CHARSET_RANGE | string): this {
        this._ocr.setRanges(charsetRange);
        this._ocrBeta.setRanges(charsetRange);

        return this;
    }

    /**
     * Classifies an image by running it through an OCR model.
     */
    async classification(url: string) {
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
     */
    public async detection(url: string) {
        return this._detection.detection(url);
    }
}

export {
    DdddOcr,
    CHARSET_RANGE,
    MODEL_TYPE
};