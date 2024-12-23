function drawRectangle(image, points, color) {
    points.forEach(point => {
        image.setPixelColor(color, point.x, point.y);
    });

    const drawLine = (x1, y1, x2, y2) => {
        const dx = Math.abs(x2 - x1);
        const dy = Math.abs(y2 - y1);
        const sx = (x1 < x2) ? 1 : -1;
        const sy = (y1 < y2) ? 1 : -1;
        let err = dx - dy;

        while (true) {
            image.setPixelColor(0xFF0000FF, x1, y1);
            if (x1 === x2 && y1 === y2) break;

            const e2 = err * 2;
            if (e2 > -dy) {
                err -= dy;
                x1 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y1 += sy;
            }
        }
    };

    for (let i = 0; i < points.length; i++) {
        const start = points[i];
        const end = points[(i + 1) % points.length];

        drawLine(start.x, start.y, end.x, end.y);
    }
}

module.exports = {
    drawRectangle
}