import { DdddOcr, CHARSET_RANGE, MODEL_TYPE } from '../dist/esm/index.js';

describe('DdddOcr Source Tests', () => {
    let ddddOcr: DdddOcr;

    beforeAll(() => {
        ddddOcr = new DdddOcr();
    });

    test('Standard OCR Classification', async () => {
        const ocrResult = await ddddOcr.classification('./test/example-en.jpg');
        expect(ocrResult).toBe('8A62N1');
    });

    test('OCR with NUM_CASE range', async () => {
        ddddOcr.setRanges(CHARSET_RANGE.NUM_CASE);
        const result = await ddddOcr.classification('./test/example-en.jpg');
        expect(result).toBe('8621');
    });

    test('OCR with LOWER_CASE range', async () => {
        ddddOcr.setRanges(CHARSET_RANGE.LOWER_CASE);
        const result = await ddddOcr.classification('./test/example-en.jpg');
        expect(result).toBe('');
    });

    test('OCR with UPPER_CASE range', async () => {
        ddddOcr.setRanges(CHARSET_RANGE.UPPER_CASE);
        const result = await ddddOcr.classification('./test/example-en.jpg');
        expect(result).toBe('AN');
    });

    test('OCR with specific string range', async () => {
        ddddOcr.setRanges('N');
        const result = await ddddOcr.classification('./test/example-en.jpg');
        expect(result).toBe('N');
    });

    test('OCR Beta Mode', async () => {
        ddddOcr.setRanges('');
        ddddOcr.setOcrMode(MODEL_TYPE.OCR_BETA);
        const result = await ddddOcr.classification('./test/example-ch.jpg');
        expect(result).toBe('九乘六等于？');
    });

    test('Object Detection', async () => {
        const detectionResult = await ddddOcr.detection('./test/example-ch.jpg');
        expect(detectionResult).toEqual([
            [80, 3, 99, 21],
            [100, 0, 127, 19],
            [2, 2, 22, 22],
            [31, 7, 51, 26],
            [56, 6, 77, 25]
        ]);
    });
});
