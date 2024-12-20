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

class DdddOcr {
    _onnxPath = path.join(__dirname, './onnx/common_old.onnx');
    _charsetPath = path.join(__dirname, './onnx/common_old.json');

    _ortSessionPending = null;
    _charsetPending = null;

    _charsetRange = '';
    _deleteRange = '';
    _validCharsetRangeIndex = {};

    constructor(_onnxPath = undefined, _charsetPath = undefined) {
        this._loadOrtSession(_onnxPath);
        this._loadCharset(_charsetPath);
    }

    _loadOrtSession(onnxPath = this._onnxPath) {
        if (!this._ortSessionPending) {
            this._ortSessionPending = ort.InferenceSession.create(onnxPath);
        }

        return this._ortSessionPending;
    }

    _loadCharset(charsetPath = this._charsetPath) {
        if (!this._charsetPending) {
            this._charsetPending = fsm.readFile(charsetPath, { encoding: 'utf-8' })
                .then((result) => {
                    return JSON.parse(result);
                })
                .then((charset) => {
                    for (let i = 0; i < charset.length; i++) {
                        this._validCharsetRangeIndex[i] = charset[i];
                    }
                    return charset;
                });
        }

        return this._charsetPending;
    }

    async setRanges(charsetRange) {
        switch (typeof(charsetRange)) {
            case 'number': {
                switch (charsetRange) {
                    case CHARSET_RANGE.NUM_CASE: {
                        this._charsetRange = NUM_CASE;
                        break;
                    }
                    case CHARSET_RANGE.LOWWER_CASE: {
                        this._charsetRange = LOWWER_CASE;
                        break;
                    }
                    case CHARSET_RANGE.UPPER_CASE: {
                        this._charsetRange = UPPER_CASE;
                        break;
                    }
                    case CHARSET_RANGE.MIX_LOWWER_UPPER_CASE: {
                        this._charsetRange = MIX_LOWWER_UPPER_CASE;
                        break;
                    }
                    case CHARSET_RANGE.MIX_LOWWER_NUM_CASE: {
                        this._charsetRange = MIX_LOWWER_NUM_CASE;
                        break;
                    }
                    case CHARSET_RANGE.MIX_UPPER_NUM_CASE: {
                        this._charsetRange = MIX_UPPER_NUM_CASE;
                        break;
                    }
                    case CHARSET_RANGE.MIX_LOWWER_UPPER_NUM_CASE: {
                        this._charsetRange = MIX_LOWER_UPPER_NUM_CASE;
                        break;
                    }
                    case CHARSET_RANGE.NO_LOWEER_UPPER_NUM_CASE: {
                        this._deleteRange = MIX_LOWER_UPPER_NUM_CASE;
                        break;
                    }
                    default: {
                        throw new Error('Not support type');
                    }
                }
                break;
            }
            case 'string': {
                this._charsetRange = charsetRange;
                break
            }
            default: {
                throw new Error('Not support type');
            }
        }

        const charset = await this._loadCharset();

        this._validCharsetRangeIndex = {};
        for (let i = 0; i < this._charsetRange.length; i++) {
            const charsetRange = this._charsetRange[i];
            const targetCharSetIndex = charset.findIndex((f) => f == charsetRange);

            this._validCharsetRangeIndex[targetCharSetIndex] = charsetRange;
        }

        for (let i = 0; i < this._deleteRange.length; i++) {
            const deleteRange = this._deleteRange[i];

            if (this._validCharsetRangeIndex[deleteRange]) {
                delete this._validCharsetRangeIndex[deleteRange];
            }
        }

        return this;
    }

    async _run(inputTensor) {
        const ortSession = await this._ortSessionPending;
        const result = await ortSession.run({ input1: inputTensor });

        return result['387'];
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

        const { cpuData, dims } = await this._run(inputTensor);
        const result = [];

        await this._loadCharset();

        const tensor = tf.tensor(cpuData);
        const reshapedTensor = tf.reshape(tensor, dims);
        const argmaxResult = tf.argMax(reshapedTensor, 2);

        const argmaxData = await argmaxResult.data();

        let lastItem = 0;
        for (let i = 0; i < argmaxData.length; i++) {
            if (argmaxData[i] == lastItem) {
                continue;
            } else {
                lastItem = argmaxData[i];
            }

            if (argmaxData[i] != 0 && this._validCharsetRangeIndex[argmaxData[i]]) {
                result.push(this._validCharsetRangeIndex[argmaxData[i]]);
            }
        }

        return result.join('');
    }
}

module.exports = {
    DdddOcr,
    CHARSET_RANGE
};