const path = require('node:path');
const fs = require('node:fs');
const fsm = require('node:fs/promises');

const tf = require('@tensorflow/tfjs');
const ort = require('onnxruntime-node');
const { Jimp, cssColorToHex } = require('jimp');

const { argSort } = require('./array-utils');
const { drawRectangle } = require('./image-utils');

const { tensorflowToImage, arrayToImage } = require('./debug-utils');

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

    _ocrDetectionOnnxPath = path.join(__dirname, './onnx/common_det.onnx');

    _ocrOrtSessionPending = null;
    _ocrBetaOrtSessionPending = null;

    _ocrDetectionOrtSessionPending = null;

    _validCharSet = new Set([]);
    _inValidCharSet = new Set([]);

    _isBetaOcrEnable = false;

    _isDebug = false;

    constructor() {}

    enableBetaOcr(value) {
        this._isBetaOcrEnable = value;

        return this;
    }

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

    _loadDetectionOrtSession() {
        if (!this._ocrDetectionOrtSessionPending) {
            const ocrOnnxPromise = ort.InferenceSession.create(this._ocrDetectionOnnxPath);
            this._ocrDetectionOrtSessionPending = ocrOnnxPromise;
        }

        return this._ocrBetaOrtSessionPending;
    }

    async _loadCharset(charsetPath = this._charsetPath) {
        return fsm.readFile(charsetPath, { encoding: 'utf-8' })
            .then((result) => {
                return JSON.parse(result);
            });
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
        if (this._isBetaOcrEnable) {
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

    async _runDetection(inputTensor) {
        if (!this._ocrDetectionOrtSessionPending) {
            this._loadDetectionOrtSession();
        }

        const ortSession = await this._ocrDetectionOrtSessionPending;
        const result = await ortSession.run({ images: inputTensor });

        return result['output'];
    }

    preProcessImage(image, inputSize) {
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

    demoPostProcess(outputTensor, imageSize) {
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
     * 
     * @param {tf.Tensor} boxes 
     * @param {number} ratio 
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

    _where(ovr, nmsThr) {
        const result = [];

        for (let i = 0; i < ovr.length; i++) {
            if (ovr[i] <= nmsThr) {
                result.push(i);
            }
        }

        return result;
    }

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

    _multiclassNms(boxes, scores, nmsThr, scoreThr) {
        return this._multiclassNmsClassAgnostic(boxes, scores, nmsThr, scoreThr);
    }

    async getBbox(image) {
        const inputSize = [416, 416];

        const { width, height } = image.bitmap;

        const { inputTensor, ratio } = this.preProcessImage(image, inputSize);

        const outputTensor = await this._runDetection(inputTensor);

        const predictions = this.demoPostProcess(outputTensor, inputSize);

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

    async detection(img) {
        const image = await Jimp.read(img);

        const result = await this.getBbox(image.clone());

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