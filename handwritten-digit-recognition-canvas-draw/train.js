import GenerateData from './GenerateData';
import * as tf from "@tensorflow/tfjs";



let data;
async function loadData() {
    data = new GenerateData();
    await data.load();
}

async function run() {
    // Define hyperparameter
    const NUM_EPOCHES = 400;
    const LEARNING_RATE = 0.15;
    const BATCH_SIZE = 100;
    const LINK_SAVE_MODEL = 'localstorage://saved-model';

    // Implement Sequential Neural Network
    const model = tf.sequential()

    // Output ((n + 2p -f) / s + 1) * ((n + 2p -f) / s + 1) * numsFilters
    // = ((28 + 2 * 0 - 5) / 1 + 1) * ((28 + 2 * 0 - 5) / 1 + 1) * 8
    // = (24 * 24 * 8)
    const conv2d_1 = tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    })

    // Add convolution layer 1
    model.add(conv2d_1)

    // Output ((n + 2p -f) / s + 1) * ((n + 2p -f) / s + 1) * numsFilters
    // = ((24 + 2 * 0 - 2) / 2 + 1) * ((24 + 2 * 0 - 2) / 2 + 1) * 8
    // = (12 * 12 * 8)
    const maxPooling2d_1 = tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    })

    // Add max pooling layer  1
    model.add(maxPooling2d_1)

    // Output ((n + 2p -f) / s + 1) * ((n + 2p -f) / s + 1) * numsFilters
    // = ((12 + 2 * 0 - 5) / 1 + 1) * ((12 + 2 * 0 - 5) / 1 + 1) * 16
    // = (8 * 8 * 16)
    const conv2d_2 = tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    })

    // Add convolution layer 2
    model.add(conv2d_2)

    // Output ((n + 2p -f) / s + 1) * ((n + 2p -f) / s + 1) * numsFilters
    // = ((8 + 2 * 0 - 2) / 2 + 1) * ((8 + 2 * 0 - 2) / 2 + 1) * 16
    // = (4 * 4 * 16)
    const maxPooling2d_2 = tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2],
    })

    // Add max pooling layer  2
    model.add(maxPooling2d_2)

    // Flattern output to 1D vector to pass through fully-connected network
    model.add(tf.layers.flatten())

    // Dense layer with softmax function with 10 class {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    const dense = tf.layers.dense({
        units: 10,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    })

    // Add dense layer
    model.add(dense)


    // Define Optimizer
    const optimizer = tf.train.sgd(LEARNING_RATE);

    // Generate Model
    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    })
    for (let i=0; i < NUM_EPOCHES; i++) {
        const [ trainBatch, validationData ] = tf.tidy(() => {
            const trainBatch = data.nextBatch("train", BATCH_SIZE)
            // console.log(trainBatch)
            // Reshape from [BATCH_SIZE, 784] to [BATCH_SIZE, 28, 28, 1]
            trainBatch.images = trainBatch.images.reshape([BATCH_SIZE, 28, 28, 1])
            let validationData;
            // Validation training after each 5 epochs
            if(i % 5 == 0) {
                validationData = data.nextBatch("test", BATCH_SIZE);
                validationData.images =  validationData.images.reshape([BATCH_SIZE, 28, 28, 1]);
            }

            return [trainBatch, validationData]
        });
        
        const history = await model.fit(
            trainBatch.images, trainBatch.labels,
            {
                batchSize: BATCH_SIZE, validationData, epochs: 1,
            }
        )
        const loss = history.history.loss[0]
        const acc = history.history.acc[0]

        if((i+1) % 50 == 0) {
            console.log(`Epoch ${i} with Loss is ${loss} with accuracy: ${acc}`);

            // Update UI.

            const progress = document.getElementById("progress");
            progress.style.width = `${(i+1)/NUM_EPOCHES * 100}%`;
        }

    
        // Dispose to free GPU memory
        tf.dispose([ trainBatch, validationData ])

        // tf.nextFrame() returns a promise that resolves at the next call to
        // requestAnimationFrame(). By awaiting this promise we keep our model
        // training from blocking the main UI thread and freezing the browser.
        await tf.nextFrame();
    }

    console.log('Finish training...');
    const saveResult = await model.save(LINK_SAVE_MODEL);
    console.log(`Saved model to ${LINK_SAVE_MODEL}`)
}

async function boom() {
    await loadData();
    await run();
}

const trainBtn = document.getElementById("train-btn");
trainBtn.addEventListener('click', () => {
    boom();
});





// y = 2 ^ 2 + 1
const y = tf.tidy(() => {
    // a, b, and one will be cleaned up when the tidy ends.
    const one = tf.scalar(1);
    const a = tf.scalar(2);
    const b = a.square();
 
    console.log('numTensors (in tidy): ' + tf.memory().numTensors);
 
    // The value returned inside the tidy function will return
    // through the tidy, in this case to the variable y.
    return b.add(one);
});
 
console.log('numTensors (outside tidy): ' + tf.memory().numTensors);
y.print();



// numTensors (in tidy): 3
// numTensors (outside tidy): 1
// Tensor
//    5