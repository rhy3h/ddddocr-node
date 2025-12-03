import type tf from '@tensorflow/tfjs';
import { Jimp, rgbaToInt } from 'jimp';

async function tensorflowToImage(tensor: tf.Tensor, inputSize: [number, number], fileName: `${string}.${string}`) {
    const image = new Jimp({ width: inputSize[1], height: inputSize[0] });

    const arr: number[][][] = await tensor.array() as number[][][];
    arr.forEach((row, y) => {
        row.forEach((pixel, x) => {
            const color = rgbaToInt(pixel[0], pixel[1], pixel[2], pixel.length == 4 ? pixel[3] : 255);
            image.setPixelColor(color, x, y);
        });
    });

    await image.write(fileName);
    console.log(`'${fileName}' saved`);
}

async function arrayToImage(arr: number[][][], inputSize: [number, number], fileName: `${string}.${string}`) {
    const image = new Jimp({ width: inputSize[1], height: inputSize[0] });

    arr.forEach((row, y) => {
        row.forEach((pixel, x) => {
            const color = rgbaToInt(pixel[0], pixel[1], pixel[2], pixel.length == 4 ? pixel[3] : 255);
            image.setPixelColor(color, x, y);
        });
    });

    await image.write(fileName);
    console.log(`'${fileName}' saved`);
}

export { tensorflowToImage, arrayToImage};