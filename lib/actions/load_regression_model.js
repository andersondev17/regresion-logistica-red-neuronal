import * as tf from '@tensorflow/tfjs';
import { model } from './regression.js';

async function predictSurvival() {
  const age = parseFloat(document.getElementById('ageInput').value);
  const sex = parseInt(document.getElementById('sexInput').value, 10);

  if (Number.isNaN(age)) {
    alert('Ingresa una edad válida');
    return;
  }

  const input = tf.tensor2d([[sex, age]]);
  const prob = model.predict(input).dataSync()[0];
  const survived = prob >= 0.5 ? 'Sí sobrevives' : 'No sobrevives';

  document.getElementById('result').textContent =
    `Probabilidad: ${prob.toFixed(3)} → ${survived}`;
}

document.getElementById('predictBtn').addEventListener('click', predictSurvival);

