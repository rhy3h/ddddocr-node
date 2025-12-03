import { Jimp } from 'jimp';

import { DdddOcr } from './DdddOcr.js';

import { tf } from './tf/index.js';
import { ort } from './ort/index.js';

import { argSort } from './utils/array-utils.js';
import { tensorflowToImage, arrayToImage } from './utils/debug-utils.js';

class Detection extends DdddOcr {
    /**
     * Path to the ONNX model for standard OCR.
     * @private
     */
    private _ortOnnxPath = '';

    private _ocrDetectionOrtSessionPending!: Promise<ort.InferenceSession>;

    /**
     * Detection
     */
    constructor(onnxPath: string) {
        super();

        this._ortOnnxPath = onnxPath;
    }

    /**
     * Pre-processes an image by resizing and converting it into a tensor format.
     */
    private async _preProcessImage(url: string) {
        const inputSize: [number, number] = [416, 416];

        const image = await Jimp.read(url);

        const grayBgImage = tf.fill([inputSize[1], inputSize[0], 3], 114, 'float32');

        if (this.isDebug) {
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

        const floatData = grayBgImage.arraySync() as number[][][];
        let index = 0;
        for (let h = 0; h < resizedHeight; h++) {
            for (let w = 0; w < resizedWidth; w++) {
                floatData[h][w][0] = image.bitmap.data[index + 2];
                floatData[h][w][1] = image.bitmap.data[index + 1];
                floatData[h][w][2] = image.bitmap.data[index + 0];
                index += 4;
            }
        }

        if (this.isDebug) {
            arrayToImage(floatData, inputSize, 'debug/pre-process-step-2.jpg');
        }

        const backToTensorImg = tf.tensor(floatData, [inputSize[1], inputSize[0], 3], 'float32');

        if (this.isDebug) {
            tensorflowToImage(backToTensorImg, inputSize, 'debug/pre-process-step-3.jpg');
        }

        const transposedImg = backToTensorImg.transpose([2, 0, 1]);
        const inputArray = transposedImg.dataSync();

        return { inputArray, inputSize, width, height, ratio };
    }

    /**
     * Post-processes the output tensor.
     */
    private _demoPostProcess(cpuData: Float32Array, dims: number[], imageSize: number[]) {
        const grids: number[][][] = [];
        const expandedStrides: number[][][] = [];

        const strides = [8, 16, 32];

        const hSizes = strides.map(stride => Math.floor(imageSize[0] / stride));
        const wSizes = strides.map(stride => Math.floor(imageSize[1] / stride));

        for (let i = 0; i < strides.length; i++) {
            const hsize = hSizes[i];
            const wsize = wSizes[i];
            const stride = strides[i];

            const xv = Array.from({ length: wsize }, (_, x) => x);
            const yv = Array.from({ length: hsize }, (_, y) => y);

            const grid: number[][] = [];
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

        const tensor = tf.tensor(cpuData);
        const reshapedTensor = tf.reshape(tensor, dims);

        const outputs = reshapedTensor.arraySync() as number[][][];
        const [h, w, _c] = dims;

        function adjustBoundingBoxCoordinates(output: number, grid: number, expandedStride: number) {
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

    private _calcBbox(boxes: tf.Tensor, ratio: number) {
        const boxesOutput = boxes.arraySync() as number[][];

        const results: number[][] = [];
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
     */
    private _getCurrentBox(boxes: number[][], orders: Array<number> | undefined = undefined) {
        if (orders == undefined) {
            orders = Array.from({ length: boxes.length }, (_, i) => i);
        }

        const x1: number[] = [];
        const y1: number[] = [];
        const x2: number[] = [];
        const y2: number[] = [];

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
     */
    private _where(ovr: number[], nmsThr: number) {
        const result: number[] = [];

        for (let i = 0; i < ovr.length; i++) {
            if (ovr[i] <= nmsThr) {
                result.push(i);
            }
        }

        return result;
    }

    /**
     * Single class NMS implemented in JS.
     */
    private _nms(boxes: number[][], scores: number[], nmsThr: number) {
        let order: number[] = argSort(scores);
        const keep: number[] = [];

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
            
            const inds = this._where(ovr.arraySync() as number[], nmsThr);

            order = tf.gather(orderTensor, tf.tensor(inds, undefined, 'int32').add(1).toInt()).arraySync() as number[];
        }
        
        return keep;
    }

    /**
     * Multiclass NMS implemented in JS. Class-agnostic version.
     */
    private _multiclassNmsClassAgnostic(boxes: tf.Tensor, scores: tf.Tensor, nmsThr: number, scoreThr: number) {
        const clsScores: number[] = scores.flatten().arraySync();
        const clsBoxes = boxes.arraySync() as number[][];

        const validScores: number[] = [];
        const validBoxes: number[][] = [];
        for (let i = 0; i < clsScores.length; i++) {
            if (clsScores[i] > scoreThr) {
                validScores.push(clsScores[i]);
                validBoxes.push(clsBoxes[i]);
            }
        }

        const detections: number[][] = [];

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
     */
    private _multiclassNms(boxes: tf.Tensor, scores: tf.Tensor, nmsThr: number, scoreThr: number) {
        return this._multiclassNmsClassAgnostic(boxes, scores, nmsThr, scoreThr);
    }

    private _parseToXyxy(prediction: number[][], width: number, height: number) {
        const result: number[][] = [];

        let minX: number, maxX: number, minY: number, maxY: number;
        for (let i = 0; i < prediction.length; i++) {
            const [x1, y1, x2, y2] = prediction[i];

            if (x1 < 0) minX = 0;
            else minX = Math.floor(x1);

            if (y1 < 0) minY = 0;
            else minY = Math.floor(y1);

            if (x2 > width) maxX = width;
            else maxX = Math.floor(x2);

            if (y2 > height) maxY = height;
            else maxY = Math.floor(y2);

            result.push([minX, minY, maxX, maxY]);
        }

        return result;
    }

    private _postProcess(cpuData: Float32Array, dims: number[], inputSize: number[], width: number, height: number, ratio: number) {
        const predictions = this._demoPostProcess(cpuData, dims, inputSize);

        const boxes = predictions.slice([0, 0], [-1, 4]);
        const scores = predictions.slice([0, 4], [-1, 1]).mul(predictions.slice([0, 5], [-1, 1]));

        const boxesXyxy = this._calcBbox(boxes, ratio);

        const prediction = this._multiclassNms(boxesXyxy, scores, 0.45, 0.1);

        const result = this._parseToXyxy(prediction, width, height);

        return result;
    }

    private _loadDetectionOrtSession() {
        if (!this._ocrDetectionOrtSessionPending) {
            this._ocrDetectionOrtSessionPending = this.loadModelAsync(this._ortOnnxPath);
        }

        return this._ocrDetectionOrtSessionPending;
    }

    private async _runDetection(inputTensor: ort.Tensor) {
        if (!this._ocrDetectionOrtSessionPending) {
            this._loadDetectionOrtSession();
        }

        const ortSession = await this._ocrDetectionOrtSessionPending;
        const result = await ortSession.run({ images: inputTensor });

        const onnxValue = result['output'];

        return {
            cpuData: onnxValue.data as Float32Array,
            dims: onnxValue.dims as number[],
        };
    }

    /**
     * Detects objects in an image by extracting bounding boxes and optionally 
     * visualizing the detection results on the image.
     * 
     * This method reads the image, retrieves bounding boxes, and then optionally 
     * draws the detected bounding boxes on the image if debugging is enabled.
     */
    public async detection(url: string) {
        const { inputArray, inputSize, width, height, ratio } = await this._preProcessImage(url);

        const inputTensor = new ort.Tensor('float32', inputArray, [1, 3, inputSize[0], inputSize[1]]);

        const { cpuData, dims } = await this._runDetection(inputTensor) as any;

        const result = this._postProcess(cpuData, dims, inputSize, width, height, ratio);

        return result;
    }
}

export { Detection };