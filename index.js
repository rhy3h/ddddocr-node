const path = require('node:path');
const fs = require('node:fs');
const fsm = require('node:fs/promises');

const tf = require('@tensorflow/tfjs');
const ort = require('onnxruntime-node');
const { Jimp, cssColorToHex } = require('jimp');

const { argSort } = require('./array-utils');
const { drawRectangle } = require('./image-utils');

const { tensorflowToImage, arrayToImage } = require('./debug-utils');

/**
 * Charset range constants that define different character sets for OCR.
 * These constants represent various combinations of character types (lowercase, uppercase, numeric).
 * 
 * @readonly
 * @enum {number}
 */
const CHARSET_RANGE = {
    /**
     * Character set containing only numeric characters.
     * @type {number}
     */
    NUM_CASE: 0,
    /**
     * Character set containing only lowercase characters.
     * @type {number}
     */
    LOWWER_CASE: 1,
    /**
     * Character set containing only uppercase characters.
     * @type {number}
     */
    UPPER_CASE: 2,
    /**
     * Character set containing both lowercase and uppercase characters.
     * @type {number}
     */
    MIX_LOWWER_UPPER_CASE: 3,
    /**
     * Character set containing both lowercase and numeric characters.
     * @type {number}
     */
    MIX_LOWWER_NUM_CASE: 4,
    /**
     * Character set containing both uppercase and numeric characters.
     * @type {number}
     */
    MIX_UPPER_NUM_CASE: 5,
    /**
     * Character set containing lowercase, uppercase, and numeric characters.
     * @type {number}
     */
    MIX_LOWWER_UPPER_NUM_CASE: 6,
    /**
     * Character set containing neither lowercase, uppercase, nor numeric characters.
     * @type {number}
     */
    NO_LOWEER_UPPER_NUM_CASE: 7,
}

const NUM_CASE = '0123456789';
const LOWWER_CASE = 'abcdefghijklmnopqrstuvwxyz';
const UPPER_CASE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const MIX_LOWWER_UPPER_CASE = LOWWER_CASE + UPPER_CASE;
const MIX_LOWWER_NUM_CASE = LOWWER_CASE + NUM_CASE;
const MIX_UPPER_NUM_CASE = UPPER_CASE + NUM_CASE;
const MIX_LOWER_UPPER_NUM_CASE = LOWWER_CASE + UPPER_CASE + NUM_CASE;

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
     * @type {Promise<ort.InferenceSession>|null} A promise for loading the OCR ONNX inference session.
     * @private
     */
    _ocrOrtSessionPending = null;
    /**
     * @type {Promise<ort.InferenceSession>|null} A promise for loading the OCR Beta ONNX inference session.
     * @private
     */
    _ocrBetaOrtSessionPending = null;

     /**
     * @type {Promise<ort.InferenceSession>|null} A promise for loading the OCR detection ONNX inference session.
     * @private
     */
    _ocrDetectionOrtSessionPending = null;

    /**
     * @type {Set<string>} A set of valid characters for OCR recognition.
     * @private
     */
    _validCharSet = new Set([]);
    /**
     * @type {Set<string>} A set of invalid characters for OCR recognition.
     * @private
     */
    _inValidCharSet = new Set([]);

    /**
     * @type {boolean} Flag indicating whether beta OCR is enabled.
     * @private
     */
    _isBetaOcrEnable = false;

    /**
     * @type {boolean} Flag indicating whether debugging is enabled.
     * @private
     */
    _isDebug = false;

    constructor() {}

    /**
     * Enables or disables the beta OCR feature.
     * 
     * @param {boolean} value A boolean value indicating whether to enable (true) or disable (false) the beta OCR feature.
     * @returns {DdddOcr} The current instance for method chaining.
     */
    enableBetaOcr(value) {
        this._isBetaOcrEnable = value;

        return this;
    }

    /**
     * Enables the debug mode and prepares the debug folder.
     * 
     * @returns {DdddOcr} The current instance for method chaining.
     */
    enableDebug() {
        this._isDebug = true;

        const debugFolderPath = 'debug';

        if (fs.existsSync(debugFolderPath)) {
            fs.rmSync(debugFolderPath, { recursive: true, force: true });
        }

        fs.mkdirSync(debugFolderPath);

        return this;
    }

    /**
     * Sets the mode of the OCR model.
     * 
     * @deprecated This method is deprecated. Please use `enableBetaOcr` to enable or disable the beta OCR mode.
     * 
     */
    setMode(mode) {
        switch (mode) {
            case MODEL_TYPE.OCR: {
                this.enableBetaOcr(false);
                break;
            }
            case MODEL_TYPE.OCR_BETA: {
                this.enableBetaOcr(true);
                break;
            }
            default: {
                throw new Error('Not support mode');
            }
        }

        return this;
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
     * Loads the OCR beta ONNX model and charset asynchronously, storing the promises to avoid redundant loading.
     * 
     * @private
     * @returns {Promise<Array<ort.InferenceSession, string[]>>} A promise that resolves with an array containing 
     *          the OCR beta ONNX inference session and the charset.
     */
    _loadBetaOcrOrtSession() {
        if (!this._ocrBetaOrtSessionPending) {
            const ocrOnnxPromise = ort.InferenceSession.create(this._ocrBetaOnnxPath);
            const charsetPromise = this._loadCharset(this._charsetBetaPath);
            this._ocrBetaOrtSessionPending = Promise.all([ocrOnnxPromise, charsetPromise]);
        }

        return this._ocrBetaOrtSessionPending;
    }

    /**
     * Loads the OCR detection ONNX model asynchronously, storing the promise to avoid redundant loading.
     * 
     * @private
     * @returns {Promise<ort.InferenceSession>} A promise that resolves with the OCR detection ONNX inference session.
     */
    _loadDetectionOrtSession() {
        if (!this._ocrDetectionOrtSessionPending) {
            const ocrOnnxPromise = ort.InferenceSession.create(this._ocrDetectionOnnxPath);
            this._ocrDetectionOrtSessionPending = ocrOnnxPromise;
        }

        return this._ocrBetaOrtSessionPending;
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
     * Sets the valid character set.
     * 
     * @public
     * @param {string} charset - A string containing the characters to define the valid character set.
     * @returns {DdddOcr} The current instance for method chaining.
     */
    setValidCharSet(charset) {
        this._validCharSet = new Set(charset);

        return this;
    }

    /**
     * Sets the invalid character set.
     * 
     * @public
     * @param {string} charset - A string containing the characters to define the invalid character set.
     * @returns {DdddOcr} The current instance for method chaining.
     */
    setInValidCharset(charset) {
        this._inValidCharSet = new Set(charset);

        return this;
    }

    /**
     * Checks if a character is valid based on the defined valid and invalid character sets.
     * 
     * @private
     * @param {string} char - The character to validate.
     * @returns {boolean} `true` if the character is valid, `false` otherwise.
     */
    _isValidChar(char) {
        if (this._inValidCharSet.has(char)) {
            return false;
        }

        if (this._validCharSet.size == 0) {
            return true;
        }

        return this._validCharSet.has(char);
    }

    /**
     * Sets the range restriction for OCR results.
     * 
     * This method restricts the characters returned by OCR based on the input:
     * - For `number` input, it applies a predefined character set. Supported values are:
     *   - `0`: Digits (0-9)
     *   - `1`: Lowercase letters (a-z)
     *   - `2`: Uppercase letters (A-Z)
     *   - `3`: Lowercase + Uppercase letters
     *   - `4`: Lowercase letters + Digits
     *   - `5`: Uppercase letters + Digits
     *   - `6`: Lowercase + Uppercase letters + Digits
     *   - `7`: Default set (a-z, A-Z, 0-9)
     * - For `string` input, each character in the string is treated as a valid OCR result.
     * 
     * @public
     * @param {number|string} charsetRange - A number for predefined character sets or a string for a custom character set.
     * @returns {DdddOcr} The current instance for method chaining.
     * @throws {Error} Throws an error if the input type or value is not supported.`
     */
    setRanges(charsetRange) {
        switch (typeof(charsetRange)) {
            case 'number': {
                switch (charsetRange) {
                    case CHARSET_RANGE.NUM_CASE: {
                        this.setValidCharSet(NUM_CASE);
                        break;
                    }
                    case CHARSET_RANGE.LOWWER_CASE: {
                        this.setValidCharSet(LOWWER_CASE);
                        break;
                    }
                    case CHARSET_RANGE.UPPER_CASE: {
                        this.setValidCharSet(UPPER_CASE);
                        break;
                    }
                    case CHARSET_RANGE.MIX_LOWWER_UPPER_CASE: {
                        this.setValidCharSet(MIX_LOWWER_UPPER_CASE);
                        break;
                    }
                    case CHARSET_RANGE.MIX_LOWWER_NUM_CASE: {
                        this.setValidCharSet(MIX_LOWWER_NUM_CASE);
                        break;
                    }
                    case CHARSET_RANGE.MIX_UPPER_NUM_CASE: {
                        this.setValidCharSet(MIX_UPPER_NUM_CASE);
                        break;
                    }
                    case CHARSET_RANGE.MIX_LOWWER_UPPER_NUM_CASE: {
                        this.setValidCharSet(MIX_LOWER_UPPER_NUM_CASE);
                        break;
                    }
                    case CHARSET_RANGE.NO_LOWEER_UPPER_NUM_CASE: {
                        this.setInValidCharset(MIX_LOWER_UPPER_NUM_CASE);
                        break;
                    }
                    default: {
                        throw new Error('Not support type');
                    }
                }
                break;
            }
            case 'string': {
                this.setValidCharSet(charsetRange);
                break
            }
            default: {
                throw new Error('Not support type');
            }
        }

        return this;
    }

    /**
     * Runs OCR processing on the provided input tensor using an ORT session.
     * 
     * This method waits for the OCR session to be loaded (if not already loaded) and then processes the input tensor to extract OCR results.
     * It returns the processed data including the `cpuData`, `dims`, and the associated `charset`.
     * 
     * @private
     * @param {ort.Tensor} inputTensor - The input tensor to process for OCR.
     * @returns {Promise<{cpuData: Float32Array, dims: number[], charset: string}>} A promise that resolves to an object containing:
     *   - `cpuData`: The raw OCR data as a `Float32Array`.
     *   - `dims`: The dimensions of the result tensor.
     *   - `charset`: The character set used for OCR processing.
     */
    async _runOcr(inputTensor) {
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
     * Runs OCR Beta processing on the provided input tensor using an ORT session.
     * 
     * This method waits for the OCR Beta session to be loaded (if not already loaded) and then processes the input tensor to extract OCR results.
     * It returns the processed data including the `cpuData`, `dims`, and the associated `charset`.
     * 
     * @private
     * @param {ort.Tensor} inputTensor - The input tensor to process for OCR Beta.
     * @returns {Promise<{cpuData: Float32Array, dims: number[], charset: string}>} A promise that resolves to an object containing:
     *   - `cpuData`: The raw OCR Beta data as a `Float32Array`.
     *   - `dims`: The dimensions of the result tensor.
     *   - `charset`: The character set used for OCR Beta processing.
     */
    async _runOcrBeta(inputTensor) {
        if (!this._ocrBetaOrtSessionPending) {
            this._loadBetaOcrOrtSession();
        }

        const [ortSession, charset] = await this._ocrBetaOrtSessionPending;
        const result = await ortSession.run({ input1: inputTensor });

        const { cpuData, dims } = result['387'];

        return {
            cpuData,
            dims,
            charset
        };
    }

    /**
     * Runs OCR processing (either standard or beta) on the provided input tensor based on the OCR mode.
     * 
     * If the beta OCR mode is enabled, it runs the OCR Beta processing. Otherwise, it runs the standard OCR processing.
     * 
     * @private
     * @param {ort.Tensor} inputTensor - The input tensor to process for OCR.
     * @returns {Promise<{cpuData: Float32Array, dims: number[], charset: string}>} A promise that resolves to the OCR result containing:
     *   - `cpuData`: The raw OCR data as a `Float32Array`.
     *   - `dims`: The dimensions of the result tensor.
     *   - `charset`: The character set used for OCR processing.
     */
    async _run(inputTensor) {
        if (this._isBetaOcrEnable) {
            return this._runOcrBeta(inputTensor);
        }

        return this._runOcr(inputTensor);
    }

    /**
     * Parses the given `argmaxData` array into a string using the provided character set.
     * 
     * The method iterates through the `argmaxData`, ensuring consecutive repeated items are skipped, 
     * and converts the data into valid characters based on the provided `charset`. 
     * The valid characters are checked using `_isValidChar`, and only valid characters are included in the final result.
     * 
     * @private
     * @param {number[]} argmaxData - An array of indices representing the OCR output. Each element corresponds to a character index in the `charset`.
     * @param {string[]} charset - An array of characters corresponding to the indices in `argmaxData`.
     * @returns {string} The parsed string formed by valid characters from `argmaxData` based on `charset`.
     */
    _parseToChar(argmaxData, charset) {
        const result = [];

        let lastItem = 0;
        for (let i = 0; i < argmaxData.length; i++) {
            if (argmaxData[i] == lastItem) {
                continue;
            } else {
                lastItem = argmaxData[i];
            }

            const char = charset[argmaxData[i]];
            if (argmaxData[i] != 0 && this._isValidChar(char)) {
                result.push(char);
            }
        }

        return result.join('');
    }

    /**
     * Classifies an image by running it through an OCR model.
     * 
     * @param {string | Buffer | ArrayBuffer} url - The image to classify. It can be a file path (string) or image data (Buffer).
     * @returns {Promise<string>} A promise that resolves to the OCR result, represented as a string of recognized characters.
     */
    async classification(url) {
        const image = await Jimp.read(url);

        const { width, height } = image.bitmap;
        const targetHeight = 64;
        const targetWidth = Math.floor(width * (targetHeight / height));
        image.resize({
            w: targetWidth, 
            h: targetHeight
        });
        image.greyscale();

        const { data } = image.bitmap;
        const floatData = new Float32Array(targetWidth * targetHeight);
        for (let i = 0, j = 0; i < data.length; i += 4, j++) {
            floatData[j] = (data[i] / 255.0 - 0.5) / 0.5;
        }
        const inputTensor = new ort.Tensor('float32', floatData, [1, 1, targetHeight, targetWidth]);

        const { cpuData, dims, charset } = await this._run(inputTensor);

        const tensor = tf.tensor(cpuData);
        const reshapedTensor = tf.reshape(tensor, dims);
        const argmaxResult = tf.argMax(reshapedTensor, 2);
        const argmaxData = await argmaxResult.data();

        const result = this._parseToChar(argmaxData, charset);

        return result;
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
     * Pre-processes an image by resizing and converting it into a tensor format.
     * 
     * @private
     * @param {Jimp} image - The image to pre-process. It is assumed to be a `Jimp` image object.
     * @param {number[]} inputSize - The target input size [height, width] for the model.
     * @returns {{inputTensor: ort.Tensor, ratio: number}} An object containing:
     *   - `inputTensor`: The pre-processed image converted into a tensor for model input.
     *   - `ratio`: The resizing ratio used to scale the image dimensions.
     */
    _preProcessImage(image, inputSize) {
        const grayBgImage = tf.fill([inputSize[1], inputSize[0], 3], 114, 'float32');

        if (this._isDebug) {
            tensorflowToImage(grayBgImage, inputSize, 'debug/pre-process-step-1.jpg');
        }

        const { width, height } = image.bitmap;
        const ratio = Math.min(inputSize[1] / width, inputSize[0] / height);

        const resizedWidth = Math.round(width * ratio);
        const resizedHeight = Math.round(height * ratio);

        image.resize({
            w: resizedWidth, 
            h: resizedHeight
        });

        const floatData = grayBgImage.arraySync();
        let index = 0;
        for (let h = 0; h < resizedHeight; h++) {
            for (let w = 0; w < resizedWidth; w++) {
                floatData[h][w][0] = image.bitmap.data[index + 2];
                floatData[h][w][1] = image.bitmap.data[index + 1];
                floatData[h][w][2] = image.bitmap.data[index + 0];
                index += 4;
            }
        }

        if (this._isDebug) {
            arrayToImage(floatData, inputSize, 'debug/pre-process-step-2.jpg');
        }

        const backToTensorImg = tf.tensor(floatData, [inputSize[1], inputSize[0], 3], 'float32');

        if (this._isDebug) {
            tensorflowToImage(backToTensorImg, inputSize, 'debug/pre-process-step-3.jpg');
        }

        const transposedImg = backToTensorImg.transpose([2, 0, 1]);
        let flatArray = transposedImg.dataSync();

        const inputTensor = new ort.Tensor('float32', flatArray, [1, 3, inputSize[0], inputSize[1]]);

        return {
            inputTensor,
            ratio
        }
    }

    /**
     * Post-processes the output tensor.
     * 
     * @private
     * @param {ort.Tensor} outputTensor - The output tensor from the model, containing raw bounding box predictions.
     * @param {number[]} imageSize - The original image size [height, width].
     * @returns {tf.Tensor} A tensor containing the post-processed bounding box coordinates.
     */
    _demoPostProcess(outputTensor, imageSize) {
        const grids = [];
        const expandedStrides = [];

        const strides = [8, 16, 32];

        const hSizes = strides.map(stride => Math.floor(imageSize[0] / stride));
        const wSizes = strides.map(stride => Math.floor(imageSize[1] / stride));

        for (let i = 0; i < strides.length; i++) {
            const hsize = hSizes[i];
            const wsize = wSizes[i];
            const stride = strides[i];

            const xv = Array.from({ length: wsize }, (_, x) => x);
            const yv = Array.from({ length: hsize }, (_, y) => y);

            const grid = [];
            for (let y = 0; y < yv.length; y++) {
                for (let x = 0; x < xv.length; x++) {
                    grid.push([xv[x], yv[y]]);
                }
            }
            grids.push(grid);

            const expandedStride = Array.from({ length: grid.length }, () => [stride]);
            expandedStrides.push(expandedStride);
        }

        const flatGrids = grids.flat();
        const flatExpandedStrides = expandedStrides.flat();

        const { cpuData, dims } = outputTensor;

        const tensor = tf.tensor(cpuData);
        const reshapedTensor = tf.reshape(tensor, dims);

        const outputs = reshapedTensor.arraySync();
        const [h, w, _c] = dims;

        function adjustBoundingBoxCoordinates(output, grid, expandedStride) {
            return (output + grid) * expandedStride;
        }

        for (let i = 0; i < h; i++) {
            for (let j = 0; j < w; j++) {
                const grid = flatGrids[j];
                const expandedStride = flatExpandedStrides[j][0];

                outputs[i][j][0] = adjustBoundingBoxCoordinates(outputs[i][j][0], grid[0], expandedStride);
                outputs[i][j][1] = adjustBoundingBoxCoordinates(outputs[i][j][1], grid[1], expandedStride);

                outputs[i][j][2] = Math.exp(outputs[i][j][2]) * expandedStride;
                outputs[i][j][3] = Math.exp(outputs[i][j][3]) * expandedStride;
            }
        }

        return tf.tensor(outputs[0]);
    }

    /**
     * Calculates bounding box coordinates.
     * 
     * @private
     * @param {tf.Tensor} boxes - A tensor containing bounding box coordinates in the format [centerX, centerY, width, height].
     * @param {number} ratio - The ratio by which to adjust the bounding box coordinates.
     * @returns {tf.Tensor} A tensor containing the calculated bounding box coordinates in the format [xMin, yMin, xMax, yMax].
     */
    _calcBbox(boxes, ratio) {
        const boxesOutput = boxes.arraySync();

        const results = [];
        for (let i = 0; i < boxesOutput.length; i++) {
            const boxes = boxesOutput[i];

            const result = [
                (boxes[0] - boxes[2] / 2) / ratio,
                (boxes[1] - boxes[3] / 2) / ratio,
                (boxes[0] + boxes[2] / 2) / ratio,
                (boxes[1] + boxes[3] / 2) / ratio,
            ];
            results.push(result);
        }

        const resultTensor = tf.tensor(results);

        return resultTensor;
    }
    
    /**
     * Extracts the bounding box coordinates (x1, y1, x2, y2) from the given boxes based on the specified order.
     * 
     * @private
     * @param {Array} boxes - An array of bounding boxes, where each box is represented as [x1, y1, x2, y2].
     * @param {Array<number>} [orders=undefined] - An optional array of indices specifying the order in which to extract the boxes. If not provided, the boxes are extracted in the original order.
     * @returns {Array<tf.Tensor>} An array of tensors representing the extracted coordinates: [x1, y1, x2, y2].
     */
    _getCurrentBox(boxes, orders = undefined) {
        if (orders == undefined) {
            orders = Array.from({ length: boxes.length }, (_, i) => i);
        }

        const x1 = [];
        const y1 = [];
        const x2 = [];
        const y2 = [];

        for (let i = 0; i < orders.length; i++) {
            const order = orders[i];

            x1.push(boxes[order][0]);
            y1.push(boxes[order][1]);
            x2.push(boxes[order][2]);
            y2.push(boxes[order][3]);
        }

        return [
            tf.tensor(x1),
            tf.tensor(y1),
            tf.tensor(x2),
            tf.tensor(y2)
        ];
    }

    /**
     * Filters the indices of an array based on the given threshold. Only indices where the corresponding value is less than or equal to the threshold are returned.
     * 
     * @private
     * @param {Array<number>} ovr - An array of values to be compared against the threshold.
     * @param {number} nmsThr - The threshold value. Indices with values less than or equal to this threshold will be included in the result.
     * @returns {Array<number>} An array of indices where the corresponding value in `ovr` is less than or equal to `nmsThr`.
     */
    _where(ovr, nmsThr) {
        const result = [];

        for (let i = 0; i < ovr.length; i++) {
            if (ovr[i] <= nmsThr) {
                result.push(i);
            }
        }

        return result;
    }

    /**
     * Single class NMS implemented in JS.
     * 
     * @private
     * @param {Array<Array<number>>} boxes - An array of bounding boxes, where each box is represented as [x1, y1, x2, y2].
     * @param {Array<number>} scores - An array of scores corresponding to each bounding box.
     * @param {number} nmsThr - The threshold for the overlap ratio. Boxes with an overlap greater than this threshold will be suppressed.
     * @returns {Array<number>} An array of indices representing the boxes that are kept after NMS.
     */
    _nms(boxes, scores, nmsThr) {
        let order = argSort(scores);
        const keep = [];

        const [x1, y1, x2, y2] = this._getCurrentBox(boxes);
        const areas = tf.mul(x2.sub(x1).add(1), y2.sub(y1).add(1));

        while (order.length > 0) {
            const orderTensor = tf.tensor(order, [order.length], 'int32');

            const i = order[0];
            keep.push(i);

            const [x1, y1, x2, y2] = this._getCurrentBox(boxes, order);

            const xx1 = tf.maximum(x1.slice(0, 1), x1.slice(1));
            const yy1 = tf.maximum(y1.slice(0, 1), y1.slice(1));
            const xx2 = tf.minimum(x2.slice(0, 1), x2.slice(1));
            const yy2 = tf.minimum(y2.slice(0, 1), y2.slice(1));

            const w = tf.maximum(0.0, xx2.sub(xx1).add(1));
            const h = tf.maximum(0.0, yy2.sub(yy1).add(1));

            const inter = w.mul(h);

            const ovr = inter.div(tf.add(areas.slice(0, 1), tf.gather(areas, orderTensor).slice(1)).sub(inter))
            
            const inds = this._where(ovr.arraySync(), nmsThr);

            order = tf.gather(orderTensor, tf.tensor(inds, undefined, 'int32').add(1).toInt()).arraySync();
        }
        
        return keep;
    }

    /**
     * Multiclass NMS implemented in JS. Class-agnostic version.
     * 
     * @private
     * @param {Array<Array<number>>} boxes - An array of bounding boxes, where each box is represented as [x1, y1, x2, y2].
     * @param {Array<Array<number>>} scores - An array of scores for each bounding box, typically representing object detection confidence.
     * @param {number} nmsThr - The threshold for the overlap ratio. Boxes with an overlap greater than this threshold will be suppressed.
     * @param {number} scoreThr - The threshold for the score. Only boxes with a score greater than this threshold will be considered for NMS.
     * @returns {Array<Array<number>>} An array of bounding boxes after NMS, where each box is represented as [x1, y1, x2, y2].
     */
    _multiclassNmsClassAgnostic(boxes, scores, nmsThr, scoreThr) {
        const clsScores = scores.flatten().arraySync();
        const clsBoxes = boxes.arraySync();

        const validScores = [];
        const validBoxes = [];
        for (let i = 0; i < clsScores.length; i++) {
            if (clsScores[i] > scoreThr) {
                validScores.push(clsScores[i]);
                validBoxes.push(clsBoxes[i]);
            }
        }

        const detections = [];

        const keep = this._nms(validBoxes, validScores, nmsThr);
        for (let i = 0; i < keep.length; i++) {
            const targetIdx = keep[i];

            const [x1, y1, x2, y2] = validBoxes[targetIdx];

            detections.push([x1, y1, x2, y2]);
        }

        return detections;
    }

    /**
     * Multiclass NMS implemented in JS.
     * 
     * @private
     * @param {Array<Array<number>>} boxes - An array of bounding boxes, where each box is represented as [x1, y1, x2, y2].
     * @param {Array<Array<number>>} scores - An array of scores for each bounding box, typically representing object detection confidence.
     * @param {number} nmsThr - The threshold for the overlap ratio. Boxes with an overlap greater than this threshold will be suppressed.
     * @param {number} scoreThr - The threshold for the score. Only boxes with a score greater than this threshold will be considered for NMS.
     * @returns {Array<Array<number>>} An array of bounding boxes after NMS, where each box is represented as [x1, y1, x2, y2].
     */
    _multiclassNms(boxes, scores, nmsThr, scoreThr) {
        return this._multiclassNmsClassAgnostic(boxes, scores, nmsThr, scoreThr);
    }

    /**
     * Processes an image to detect bounding boxes and returns the adjusted bounding boxes 
     * based on the detection results.
     * 
     * This method preprocesses the image, runs detection, post-processes the results, 
     * applies Non-Maximum Suppression (NMS), and adjusts the bounding box coordinates 
     * according to the original image size.
     *
     * @private
     * @param {Object} image - The image object to process. It should contain the bitmap data.
     * @returns {Promise<Array<Array<number>>>} A promise that resolves to an array of bounding boxes, 
     *          where each box is represented as [x1, y1, x2, y2], adjusted to the image size.
     */
    async _getBbox(image) {
        const inputSize = [416, 416];

        const { width, height } = image.bitmap;

        const { inputTensor, ratio } = this._preProcessImage(image, inputSize);

        const outputTensor = await this._runDetection(inputTensor);

        const predictions = this._demoPostProcess(outputTensor, inputSize);

        const boxes = predictions.slice([0, 0], [-1, 4]);
        const scores = predictions.slice([0, 4], [-1, 1]).mul(predictions.slice([0, 5], [-1, 1]));

        const boxesXyxy = this._calcBbox(boxes, ratio);

        const prediction = this._multiclassNms(boxesXyxy, scores, 0.45, 0.1);

        const result = [];
        let minX, maxX, minY, maxY;
        for (let i = 0; i < prediction.length; i++) {
            const [x1, y1, x2, y2] = prediction[i];

            if (x1 < 0) minX = 0;
            else minX = parseInt(x1);

            if (y1 < 0) minY = 0;
            else minY = parseInt(y1);

            if (x2 > width) maxX = width;
            else maxX = parseInt(x2);

            if (y2 > height) maxY = height;
            else maxY = parseInt(y2);

            result.push([minX, minY, maxX, maxY]);
        }

        return result;
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
        const image = await Jimp.read(url);

        const result = await this._getBbox(image.clone());

        if (this._isDebug) {
            const color = cssColorToHex('#ff0000');

            for (let i = 0; i < result.length; i++) {
                const [x1, y1, x2, y2] = result[i];

                const points = [
                    { x: x1, y: y1 },
                    { x: x2, y: y1 },
                    { x: x2, y: y2 },
                    { x: x1, y: y2 }
                ];
                drawRectangle(image, points, color);
            }

            image.write('debug/detection-result.jpg');
        }

        return result;
    }
}

module.exports = {
    DdddOcr,
    CHARSET_RANGE,
    MODEL_TYPE
};