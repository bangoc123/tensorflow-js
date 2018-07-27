### Training, Inference CNN on Browser

| Resource | Link |
| ------ | ------ |
| Slide at Google I/O Extended | [Machine learning on Browser and TensorFlow for JavaScript](https://docs.google.com/presentation/d/1ZVO6Ripu0JL6d-aFTsUuDdlfJVKbBZC3xVupGrl_P3k/edit?usp=sharing) |
| Features | [MNIST Data Features](https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png) |
| Labels | [MNIST Data Labels](https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8) |


<img src="https://lh3.googleusercontent.com/9QHeeMU2xKoO7LSzuQ3jqDsTgYMgFEHImUe1o2Xkk_CwKPSg4dv878S1xcPAIAg_XgaFrnwQ-njiryOIAUm0KNY2G4BNNW_sEqe4rootS4UqWBtKLMN2VScXLlR0W6c7KdLKiDaNtnQ" width=500>

We will train a CNN at Browser with 2 hidden layers

  - **Layer Convolution**
    - input **[28, 28, 1]**
    - kernel size: **5**
    - activation Function: **reLu**
    - number of Filters: **8**
    - strides: **1**
  - **Layer MaxPooling**
      - pool Size: **2**
      - strides: **2**
  - **Last Layer**
      - Softmax Layer with **10 classes**

### Installation and Run

```sh
$ cd ./
$ yarn
$ yarn dev
```

```sh
Open 127.0.0.1:1234
```

### Training

![](https://lh3.googleusercontent.com/wfQUynof8T4CXJIPV0FvQvbJHwK9zXeGKm5QmzJq_0A-bqYKzqz4fFRwicGBgPK4-xKRYR1VQFy8WpLvDVWm6t-4bWSi5RxWmzZD1zKDQT3jUfxVszhaNEeJjEbD6XwQ_sBQIWrizXs)
