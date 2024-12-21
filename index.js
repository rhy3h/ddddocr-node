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
    OCR: 0
}

class DdddOcr {
    _ocrOnnxPath = path.join(__dirname, './onnx/common_old.onnx');
    _charsetPath = path.join(__dirname, './onnx/common_old.json');

    _ocrOrtSessionPending = null;
    _charsetPending = null;

    validCharsetRangeSet = new Set([]);
    deleteRangeSet = new Set([]);

    constructor() {}

    async preload(mode = MODEL_TYPE.OCR) {
        switch (mode) {
            case MODEL_TYPE.OCR: {
                await this._loadOcrOrtSession(this._ocrOnnxPath, this._charsetPath);
                break;
            }
        }

        return this;
    }

    _loadOcrOrtSession(onnxPath, charsetPath) {
        if (!this._ocrOrtSessionPending) {
            const ocrOnnxPromise = ort.InferenceSession.create(onnxPath);
            const charsetPromise = this._loadCharset(charsetPath);
            this._ocrOrtSessionPending = Promise.all([ocrOnnxPromise, charsetPromise]);
        }

        return this._ocrOrtSessionPending;
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

    setCharsetRange(charsetRange) {
        this.validCharsetRangeSet = new Set(charsetRange);
    }

    setDeleteCharsetRange(charsetRange) {
        this.deleteRangeSet = new Set(charsetRange);
    }

    isValidChar(char) {
        if (this.deleteRangeSet.has(char)) {
            return false;
        }

        if (this.validCharsetRangeSet.size == 0) {
            return true;
        }

        return this.validCharsetRangeSet.has(char);
    }

    async setRanges(charsetRange) {
        switch (typeof(charsetRange)) {
            case 'number': {
                switch (charsetRange) {
                    case CHARSET_RANGE.NUM_CASE: {
                        this.setCharsetRange(NUM_CASE);
                        break;
                    }
                    case CHARSET_RANGE.LOWWER_CASE: {
                        this.setCharsetRange(LOWWER_CASE);
                        break;
                    }
                    case CHARSET_RANGE.UPPER_CASE: {
                        this.setCharsetRange(UPPER_CASE);
                        break;
                    }
                    case CHARSET_RANGE.MIX_LOWWER_UPPER_CASE: {
                        this.setCharsetRange(MIX_LOWWER_UPPER_CASE);
                        break;
                    }
                    case CHARSET_RANGE.MIX_LOWWER_NUM_CASE: {
                        this.setCharsetRange(MIX_LOWWER_NUM_CASE);
                        break;
                    }
                    case CHARSET_RANGE.MIX_UPPER_NUM_CASE: {
                        this.setCharsetRange(MIX_UPPER_NUM_CASE);
                        break;
                    }
                    case CHARSET_RANGE.MIX_LOWWER_UPPER_NUM_CASE: {
                        this.setCharsetRange(MIX_LOWER_UPPER_NUM_CASE);
                        break;
                    }
                    case CHARSET_RANGE.NO_LOWEER_UPPER_NUM_CASE: {
                        this.setDeleteCharsetRange(MIX_LOWER_UPPER_NUM_CASE);
                        break;
                    }
                    default: {
                        throw new Error('Not support type');
                    }
                }
                break;
            }
            case 'string': {
                this.setCharsetRange(charsetRange);
                break
            }
            default: {
                throw new Error('Not support type');
            }
        }

        return this;
    }

    async _run(inputTensor) {
        const [ortSession, charset] = await this._ocrOrtSessionPending;
        const result = await ortSession.run({ input1: inputTensor });

        const { cpuData, dims } = result['387'];

        return {
            cpuData,
            dims,
            charset
        };
    }

    parseToChar(argmaxData, charset) {
        const result = [];

        let lastItem = 0;
        for (let i = 0; i < argmaxData.length; i++) {
            if (argmaxData[i] == lastItem) {
                continue;
            } else {
                lastItem = argmaxData[i];
            }

            const char = charset[argmaxData[i]];
            if (argmaxData[i] != 0 && this.isValidChar(char)) {
                result.push(char);
            }
        }

        return result.join('');
    }

    async classification(img) {
        if (!this._ocrOrtSessionPending) {
            await this.preload(MODEL_TYPE.OCR);
        }

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

        const result = this.parseToChar(argmaxData, charset);

        return result;
    }
}

module.exports = {
    DdddOcr,
    CHARSET_RANGE,
    MODEL_TYPE
};