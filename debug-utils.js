const { Jimp, rgbaToInt } = require('jimp');

function tensorflowToImage(tensor, inputSize, fileName) {
    tensor.array().then(arr => {
        const image = new Jimp({ width: inputSize[1], height: inputSize[0] });

        arr.forEach((row, y) => {
            row.forEach((pixel, x) => {
                const color = rgbaToInt(pixel[0], pixel[1], pixel[2], pixel.length == 4 ? pixel[3] : 255);
                image.setPixelColor(color, x, y);
            });
        });

        image.write(fileName, () => {
            console.log(`'${fileName}' saved`);
        });
    });
}

function arrayToImage(arr, inputSize, fileName) {
    const image = new Jimp({ width: inputSize[1], height: inputSize[0] });

    arr.forEach((row, y) => {
        row.forEach((pixel, x) => {
            const color = rgbaToInt(pixel[0], pixel[1], pixel[2], pixel.length == 4 ? pixel[3] : 255);
            image.setPixelColor(color, x, y);
        });
    });

    image.write(fileName, () => {
        console.log(`'${fileName}' saved`);
    });
}

module.exports = {
    tensorflowToImage,
    arrayToImage
};