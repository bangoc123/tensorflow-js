import inference from './inference';

// Set up canvas
const cv = new fabric.Canvas('cv')
cv.isDrawingMode = true;
cv.freeDrawingBrush.width = 20;
cv.freeDrawingBrush.color = "#47494d";
cv.backgroundColor = "#ffffff";
cv.renderAll();


// Add listener to button
const clearBtn = document.getElementById("clear-btn");
clearBtn.addEventListener('click', () => {
    cv.clear();
})

cv.on("mouse:up", () => {
    // Get canvas contents as url
    const scale = (1.) / 10.;
    const url = cv.toDataURL({
        format: 'png',
        multiplier: scale,
    });
    console.log('url', url);
    // FIXME
    const cv28 = document.createElement('canvas');
    cv28.width = 28;
    cv28.height = 28;
    const ctx = cv28.getContext('2d');
    const img = new Image;
    img.src = url;
    img.onload = () => {
        ctx.drawImage(img, 0, 0)
        const imgData = ctx.getImageData(0, 0, 28, 28);
        console.log('data', imgData);
        const imgDateLength = imgData.data.length
        // Loop through each pixel in chunk

        const imgArrayData = new Float32Array(28 * 28); 
        
        for(let j = 0; j < imgDateLength / 4; j +=1) { 
            // All channel has same value -> only need to read red channel
            const red_index = j * 4;
            // Nomarlize pixel value to [0, 1]
            imgArrayData[j] = imgData.data[red_index] / 255;
        }
        
        inference(imgArrayData, imgData);
    }
})

