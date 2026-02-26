import { DdddOcr } from '../dist/commonjs/index.js';

describe('DdddOcr CJS Distribution Tests', () => {
    test('OCR Classification (CJS)', async () => {
        const ddddOcr = new DdddOcr();
        const ocrResult = await ddddOcr.classification('./test/example-en.jpg');
        expect(ocrResult).toBe('8A62N1');
    });
});
