import renderChart from 'vega-embed';

export async function plotData(element, x, y) {
    const xvals = await x.data();
    const yvals = await y.data();

    const values = Array.from(yvals).map((y, i) => {
        return {'x': xvals[i], 'y': yvals[i]};
    });
   
    const spec = {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'width': 300,
        'height': 300,
        'data': {'values': values},
        'mark': 'point',
        'encoding': {
            'x': {'field': 'x', 'type': 'quantitative'},
            'y': {'field': 'y', 'type': 'quantitative'}
        }
    };
    return renderChart(element, spec, {actions: false});
}

export async function plotPred(element, x, y, pred) {
    const xvals = await x.data();
    const yvals = await y.data();
    const predVals = await pred.data();
    const values = Array.from(yvals).map((y, i) => {
        return {'x': xvals[i], 'y': yvals[i], pred: predVals[i]};
    });

    const spec = {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'width': 300,
        'height': 300,
        'data': {'values': values},
        'layer': [
          {
            'mark': 'point',
            'encoding': {
              'x': {'field': 'x', 'type': 'quantitative'},
              'y': {'field': 'y', 'type': 'quantitative'}
            }
          },
          {
            'mark': 'line',
            'encoding': {
              'x': {'field': 'x', 'type': 'quantitative'},
              'y': {'field': 'pred', 'type': 'quantitative'},
              'color': {'value': 'tomato'}
            },
          }
        ]
    };

    return renderChart(element, spec, {actions: false});
}

export function renderCoeff(element, coeff) {
    document.querySelector(element).innerHTML =
    `<span>a=${coeff.a.toFixed(3)}, b=${coeff.b.toFixed(3)}, c=${
        coeff.c.toFixed(3)},  d=${coeff.d.toFixed(3)}, e=${coeff.e.toFixed(3)}</span>`;
}