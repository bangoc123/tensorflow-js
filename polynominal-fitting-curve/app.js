import * as tf from '@tensorflow/tfjs';
import mockData from './data-seed'; 
import { renderCoeff, plotData, plotPred } from './plot';

// Step 1. Initialize Random Variables
const a = tf.variable(tf.scalar(Math.random()))
const b = tf.variable(tf.scalar(Math.random()))
const c = tf.variable(tf.scalar(Math.random()))
const d = tf.variable(tf.scalar(Math.random()))
const e = tf.variable(tf.scalar(Math.random()))

// Step 2. Create An Optimizer
const numOfInterations = 2000;
const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate)

// Step 3. Write Predict Function
function predict(x) {
    return tf.tidy(() => {
        const four = tf.scalar(4, 'int32');
        const three = tf.scalar(3, 'int32');
        return a.mul(x.pow(four))
            .add(b.mul(x.pow(three)))
            .add(c.mul(x.square()))
            .add(d.mul(x))
            .add(e)
    })
}

// Step 4. Define Loss Function: Mean Square Error
function calculateLoss(pred, labels) {
    return pred.sub(labels).square().mean();
}

// Step 5. Implement Training
async function train(x, y, numOfInterations) {
    for (let i =0; i < numOfInterations; i += 1) {
        optimizer.minimize(() => {
            const pred = predict(x);
            const loss = calculateLoss(pred, y);
            console.log(`Epoch ${i} with loss: ${loss}`);
            return loss;

        });
        if ( i%5 == 0) {
            await updateChart();
        }
        // Important: Do not block main thread.
        await tf.nextFrame();
    }
}

const trueCoeff = { a: -0.7, b: -0.8, c: -0.2, d: 0.8, e: 0.4 }
const trainData = mockData(100, trueCoeff);

async function updateChart() {
    // Update realtime coeff
    renderCoeff('#trained .coeff', {
        a: a.dataSync()[0],
        b: b.dataSync()[0],
        c: c.dataSync()[0],
        d: d.dataSync()[0],
        e: e.dataSync()[0],
    });

    const pred = predict(trainData.x);
    await plotPred("#trained .plot", trainData.x, trainData.y, pred);
    // pred.dispose();
}


async function run() {
    // Display data
    console.log('trainData', trainData);
    renderCoeff("#data .coeff", trueCoeff);
    await plotData("#data .plot", trainData.x, trainData.y);
    // Training
    const trainButton = document.getElementsByClassName("train")[0];
    trainButton.addEventListener('click', async () => {
        await train(trainData.x, trainData.y, numOfInterations);
    })
}

run();

