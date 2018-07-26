import * as tf from "@tensorflow/tfjs";

async function inference(imgArrayData, imgData) {
    const img2d = tf.tensor2d(imgArrayData, [28, 28]);
    const imgToInference = img2d.reshape([1, 28, 28, 1])
    let img = tf.fromPixels(imgData, 1);
    img = img.reshape([1, 28, 28, 1]);
    img = tf.cast(img, 'float32');

    // Load model
    const loadedModel = await tf.loadModel('localstorage://saved-model');

    if (loadedModel){
        const output = loadedModel.predict(imgToInference);
        const axis = 1;
        const predictions = Array.from(output.argMax(axis).dataSync());
        const labels = document.getElementsByClassName("number");
        for(let i=0; i< labels.length; i +=1 ) {
            labels[i].style.backgroundColor = "#fff";
        }
        const label = document.getElementById(`${predictions[0]}`);
        label.style.backgroundColor = "#ffa700";
    } else {
        alert('Can not find any models from storage, please train model before prediction')
    }
}

export default inference;