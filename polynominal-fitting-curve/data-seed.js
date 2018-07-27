import * as tf from '@tensorflow/tfjs';

/**
 * 
 * @param {*} numOfPoints 
 * @param {*} coeff 
 * @param {*} sigma 
 * @returns { x, y: nomalized_y }
 */
function mockData(numOfPoints, coeff, sigma = 0.03) {
    return tf.tidy(() => {
        const [a, b, c, d, e] = [
            tf.scalar(coeff.a),
            tf.scalar(coeff.b),
            tf.scalar(coeff.c),
            tf.scalar(coeff.d),
            tf.scalar(coeff.e)
        ];
        const x = tf.randomUniform([numOfPoints], -1, 1)
        const [four, three] = [
            tf.scalar(4, 'int32'),
            tf.scalar(3, 'int32')
        ]        
        // Polynominal Function: ax^4 + bx^3 + cx^2 + dx + e
        const y = a.mul(x.pow(four))
            .add(b.mul(x.pow(three)))
            .add(c.mul(x.square()))
            .add(d.mul(x))
            .add(e)
            .add(tf.randomNormal([numOfPoints], 0, sigma))
        const y_min = y.min()
        const y_max = y.max()
        const nomalized_y = y.sub(y_min).div(y_max.sub(y_min))
        return {
            x,
            y: nomalized_y,
        };
    });
}

export default mockData;

