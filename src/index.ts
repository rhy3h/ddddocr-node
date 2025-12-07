import { existsSync, rmSync, mkdirSync, isSupportDebug } from './file-ops/index.js';

import { OCR, CHARSET_RANGE } from './Ocr.js';
import { Detection } from './Detection.js';
import { LogSeverityLevel } from './type.js';

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
    private _ocr: OCR;
    private _ocrBeta: OCR;
    private _ocrMode: MODEL_TYPE = MODEL_TYPE.OCR;

    private _detection: Detection;

    /**
     * Class representing an OCR (Optical Character Recognition) model.
     */
    constructor() {
        this._ocr = new OCR('common_old.onnx', 'common_old.json');
        this._ocrBeta = new OCR('common.onnx', 'common.json');

        this._detection = new Detection('common_det.onnx');
    }

    /**
     * Enables the debug mode and prepares the debug folder.
     */
    public enableDebug(): this {
        isSupportDebug();

        this._detection.enableDebug();

        const debugFolderPath = 'debug';

        if (existsSync(debugFolderPath)) {
            rmSync(debugFolderPath, { recursive: true, force: true });
        }

        mkdirSync(debugFolderPath);

        return this;
    }

    public setPath(root: string) {
        this._ocr.setPath(root);
        this._ocrBeta.setPath(root);
        this._detection.setPath(root);

        return this;
    }

    /**
     * 
     * @param logSeverityLevel Log severity level. See https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/common/logging/severity.h
     * @returns 
     */
    public setLogSeverityLevel(logSeverityLevel: LogSeverityLevel) {
        this._ocr.setLogSeverityLevel(logSeverityLevel);
        this._ocrBeta.setLogSeverityLevel(logSeverityLevel);
        this._detection.setLogSeverityLevel(logSeverityLevel);

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