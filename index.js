const path = require('node:path');
const fsm = require('node:fs/promises');

const tf = require('@tensorflow/tfjs');
const ort = require('onnxruntime-node');
const { Jimp } = require('jimp');

const CHARSET_RANGE = {
    NUM_CASE: 0,
    LOWWER_CASE: 1,
    UPPER_CASE: 2,
    MIX_LOWWER_UPPER_CASE: 3,
    MIX_LOWWER_NUM_CASE: 4,
    MIX_UPPER_NUM_CASE: 5,
    MIX_LOWWER_UPPER_NUM_CASE: 6,
    NO_LOWEER_UPPER_NUM_CASE: 7,
}

const NUM_CASE = '0123456789';
const LOWWER_CASE = 'abcdefghijklmnopqrstuvwxyz';
const UPPER_CASE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const MIX_LOWWER_UPPER_CASE = LOWWER_CASE + UPPER_CASE;
const MIX_LOWWER_NUM_CASE = LOWWER_CASE + NUM_CASE;
const MIX_UPPER_NUM_CASE = UPPER_CASE + NUM_CASE;
const MIX_LOWER_UPPER_NUM_CASE = LOWWER_CASE + UPPER_CASE + NUM_CASE;

const MODEL_TYPE = {
    OCR: 0,
    OCR_BETA: 1
}

class DdddOcr {
    _ocrOnnxPath = path.join(__dirname, './onnx/common_old.onnx');
    _charsetPath = path.join(__dirname, './onnx/common_old.json');

    _ocrBetaOnnxPath = path.join(__dirname, './onnx/common.onnx');
    _charsetBetaPath = path.join(__dirname, './onnx/common.json');

    _ocrOrtSessionPending = null;
    _ocrBetaOrtSessionPending = null;
    _charsetPending = null;

    _validCharSet = new Set([]);
    _inValidCharSet = new Set([]);

    _mode = MODEL_TYPE.OCR;

    constructor() {}

    async preload() {
        switch (this._mode) {
            case MODEL_TYPE.OCR: {
                await this._loadOcrOrtSession();
                break;
            }
            case MODEL_TYPE.OCR_BETA: {
                await this._loadBetaOcrOrtSession();
                break;
            }
        }

        return this;
    }

    setMode(mode) {
        switch (mode) {
            case MODEL_TYPE.OCR: 
            case MODEL_TYPE.OCR_BETA: {
                this._mode = mode;
                break;
            }
            default: {
                throw new Error('Not support mode');
            }
        }

        return this;
    }

    _loadOcrOrtSession() {
        if (!this._ocrOrtSessionPending) {
            const ocrOnnxPromise = ort.InferenceSession.create(this._ocrOnnxPath);
            const charsetPromise = this._loadCharset(this._charsetPath);
            this._ocrOrtSessionPending = Promise.all([ocrOnnxPromise, charsetPromise]);
        }

        return this._ocrOrtSessionPending;
    }

    _loadBetaOcrOrtSession() {
        if (!this._ocrBetaOrtSessionPending) {
            const ocrOnnxPromise = ort.InferenceSession.create(this._ocrBetaOnnxPath);
            const charsetPromise = this._loadCharset(this._charsetBetaPath);
            this._ocrBetaOrtSessionPending = Promise.all([ocrOnnxPromise, charsetPromise]);
        }

        return this._ocrBetaOrtSessionPending;
    }

    _loadCharset(charsetPath = this._charsetPath) {
        if (!this._charsetPending) {
            this._charsetPending = fsm.readFile(charsetPath, { encoding: 'utf-8' })
                .then((result) => {
                    return JSON.parse(result);
                });
        }

        return this._charsetPending;
    }

    setValidCharSet(charset) {
        this._validCharSet = new Set(charset);
    }

    setInValidCharset(charset) {
        this._inValidCharSet = new Set(charset);
    }

    _isValidChar(char) {
        if (this._inValidCharSet.has(char)) {
            return false;
        }

        if (this._validCharSet.size == 0) {
            return true;
        }

        return this._validCharSet.has(char);
    }

    async setRanges(charsetRange) {
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

    async _run(inputTensor) {
        if (this._mode == MODEL_TYPE.OCR_BETA) {
            return this._runOcrBeta(inputTensor);
        }

        return this._runOcr(inputTensor);
    }

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

    async classification(img) {
        const image = await Jimp.read(img);

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
    DdddOcr,
    CHARSET_RANGE,
    MODEL_TYPE
};