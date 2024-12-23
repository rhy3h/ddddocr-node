const fsm = require('node:fs/promises');

const tf = require('@tensorflow/tfjs');
const ort = require('onnxruntime-node');
const { Jimp } = require('jimp');

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

class OCR {
    /**
     * @type {string} Path to the ONNX model for standard OCR.
     * @private
     */
    _ocrOnnxPath = '';
    /**
     * @type {string} Path to the charset file for standard OCR.
     * @private
     */
    _charsetPath = '';

    /**
     * @type {Promise<ort.InferenceSession>|null} A promise for loading the OCR ONNX inference session.
     * @private
     */
    _ocrOrtSessionPending = null;

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

    constructor(onnxPath, charsetPath) {
        this._ocrOnnxPath = onnxPath;
        this._charsetPath = charsetPath;
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
}

module.exports = {
    CHARSET_RANGE,
    OCR
}