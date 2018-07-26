import * as tf from '@tensorflow/tfjs';

class GenerateData {
    constructor(props) {
        this.FEATURES_LINK = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
        this.LABELS_LINK = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";
        this.CLASSES_NUM = 10;
        this.TOTAL_DATASET_NUM = 65000;
        this.TRAIN_NUM = 55000;
        this.TEST_NUM = 10000; 
        this.IMG_SIZE = 784;
        this.CHUNK_SIZE = 5000;
        this.currentTrainIndex = 0;
        this.currentTestIndex = 0;
    }
    
    extractFeatures() {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = '';
            img.src = this.FEATURES_LINK;
            
            // Create canvas to draw img
            const cv = document.createElement('canvas');
            const ctx = cv.getContext('2d');
            img.onload = () => {
                img.height = img.naturalHeight; // 65000
                img.width = img.naturalWidth; // 784
                // Set width, height to canvas -> 650000 * 5000
                cv.width = img.width;
                cv.height = this.CHUNK_SIZE;

                // 4 is 4 channel like [0, 0, 0, 255]
                const datasetBuff = new ArrayBuffer(this.TOTAL_DATASET_NUM * this.IMG_SIZE * 4) // 65000 * 784 * 4
                
                // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DataView
                // Data View

                for(let i=0; i < this.TOTAL_DATASET_NUM/this.CHUNK_SIZE; i+=1) { // 0 -> 12
                    // Float32Array(3920000)
                    // Create data view to hold value of each pixel.
                    // We will have 13 data view
                    const datasetBytesView = new Float32Array(datasetBuff, 
                        i * this.CHUNK_SIZE * this.IMG_SIZE * 4
                        , this.IMG_SIZE * this.CHUNK_SIZE);
                    
                    // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage
                    // Dataset is a big image with size 784 * 65000 (width * height)
                    // Create a slider with size 784 * 5000 and slide from top to bottom 
                    // -> we will get 13 chunk with size 784 * 5000
                    
                    ctx.drawImage(img, 0, i * this.CHUNK_SIZE, img.width, this.CHUNK_SIZE, 0, 0, img.width, this.CHUNK_SIZE);
                    const imgData = ctx.getImageData(0, 0, img.width, this.CHUNK_SIZE);
                    const imgDateLength = imgData.data.length

                    // Loop through each pixel in chunk
                    for(let j = 0; j < imgDateLength / 4; j +=1) { 
                        // All channel has same value -> only need to read red channel
                        const red_index = j * 4;
                        // Nomarlize pixel value to [0, 1]
                        datasetBytesView[j] = imgData.data[red_index] / 255;
                    }
                    console.log('Done Extracting Labels for chunk: ', i);
                }
                // 784 * 65000 -> 784: flattended image pixels, 65000: number of images
                // each row represent an img 28 * 28
                // each element hold one nomalized pixel data
                this.datasetImgs = new Float32Array(datasetBuff); 
                resolve();
            }
        })
    }

    extractLabels() {
        return new Promise((resolve, reject) => {
            fetch(this.LABELS_LINK).then(res => {
                res.arrayBuffer().then(buff => {
                    const labels = new Uint8Array(buff);
                    this.labels = labels;
                    console.log(labels);
                    resolve();
                }).catch(err => reject(err))
            }).catch(err => reject(err))
        })
    }


    load() {
        return new Promise((resolve, reject) => {
            const promises = [this.extractFeatures(), this.extractLabels()]
            Promise.all(promises).then(() => {
                console.log("Finish extract datas and labels");
                // Generate shuffled train and test indicies
                // Uint32Array(55000) with shuffled indicies
                this.trainIndicies = tf.util.createShuffledIndices(this.TRAIN_NUM);
                this.testIndicies = tf.util.createShuffledIndices(this.TEST_NUM);


                // Generate train and test images
                this.trainImgs = this.datasetImgs.slice(0, this.TRAIN_NUM * this.IMG_SIZE);
                this.testImgs = this.datasetImgs.slice(this.TRAIN_NUM * this.IMG_SIZE);

                // Generate train and test labels
                this.trainLabels = this.labels.slice(0, this.TRAIN_NUM * this.CLASSES_NUM);
                this.testLabels = this.labels.slice(this.TRAIN_NUM * this.CLASSES_NUM);
                resolve();
            }).catch(err => {
                reject(err);
            });
        });
    }

    nextBatch(type, batchSize) {
        let images;
        let labels;
        const batchImgs = new Float32Array(this.IMG_SIZE * batchSize);
        const batchLabels = new Uint8Array(this.CLASSES_NUM * batchSize);
        let idx;
        if(type === "train") {
            [ images, labels ] = [ this.trainImgs, this.trainLabels ];
            const newTrainIndex = this.currentTrainIndex + batchSize;
            idx = this.trainIndicies.slice(this.currentTrainIndex, newTrainIndex);
            this.currentTrainIndex = newTrainIndex;
        } else if (type === "test") {
            [ images, labels ] = [ this.testImgs, this.testLabels ];
            const newTestIndex = this.currentTestIndex + batchSize;
            idx = this.trainIndicies.slice(this.currentTestIndex, newTestIndex);
            this.currentTestIndex = newTestIndex;
        }

        for(let i =0; i < batchSize; i += 1) {
            const index = idx[i];
            const image = images.slice(index * this.IMG_SIZE, (index+1) * this.IMG_SIZE)
            const label = labels.slice(index * this.CLASSES_NUM, (index + 1) * this.CLASSES_NUM)
            batchImgs.set(image, i * this.IMG_SIZE);
            batchLabels.set(label, i * this.CLASSES_NUM);
        }

        return {
            images: tf.tensor2d(batchImgs, [ batchSize, this.IMG_SIZE ]),
            labels: tf.tensor2d(batchLabels, [ batchSize, this.CLASSES_NUM ])
        }
    }
}

export default GenerateData;