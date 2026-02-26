import { DdddOcr } from '../dist/esm/index.js';

describe('DdddOcr ESM Distribution Tests', () => {
    test('OCR Classification (ESM)', async () => {
        const ddddOcr = new DdddOcr();
        const ocrResult = await ddddOcr.classification('./test/example-en.jpg');
        expect(ocrResult).toBe('8A62N1');
    });
});
