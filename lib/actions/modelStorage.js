import * as tf from '@tensorflow/tfjs';
import path from 'node:path';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const MODELS_ROOT = path.join(PROJECT_ROOT, 'public', 'models');

async function ensureDir(dir) {
  await mkdir(dir, { recursive: true });
}

function createFileSaveHandler(targetDir) {
  return tf.io.withSaveHandler(async (modelArtifacts) => {
    await ensureDir(targetDir);

    if (!modelArtifacts.modelTopology) {
      throw new Error('El modelo no contiene topologia para serializar.');
    }

    const json = {
      modelTopology: modelArtifacts.modelTopology,
      format: 'layers-model',
      generatedBy: `TensorFlow.js v${tf.version.tfjs}`,
      convertedBy: null,
    };

    if (modelArtifacts.trainingConfig) {
      json.trainingConfig = modelArtifacts.trainingConfig;
    }

    if (modelArtifacts.weightSpecs?.length) {
      const weightsPath = path.join(targetDir, 'weights.bin');
      if (!modelArtifacts.weightData) {
        throw new Error('Los pesos no incluyen datos binarios.');
      }
      await writeFile(weightsPath, Buffer.from(modelArtifacts.weightData));
      json.weightsManifest = [
        {
          paths: ['weights.bin'],
          weights: modelArtifacts.weightSpecs,
        },
      ];
    }

    const modelPath = path.join(targetDir, 'model.json');
    await writeFile(modelPath, JSON.stringify(json, null, 2), 'utf-8');

    return {
      modelArtifactsInfo: {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        modelTopologyBytes: Buffer.byteLength(JSON.stringify(modelArtifacts.modelTopology)),
        weightSpecsBytes: modelArtifacts.weightSpecs
          ? Buffer.byteLength(JSON.stringify(modelArtifacts.weightSpecs))
          : 0,
        weightDataBytes: modelArtifacts.weightData ? modelArtifacts.weightData.byteLength : 0,
      },
      responses: [],
    };
  });
}

export async function saveModelToPublic(model, subfolder) {
  const targetDir = path.join(MODELS_ROOT, subfolder);
  const handler = createFileSaveHandler(targetDir);
  await model.save(handler);
  console.log(`Modelo guardado en ${targetDir}`);
}

export async function savePreprocessingStats(stats) {
  await ensureDir(MODELS_ROOT);
  const filePath = path.join(MODELS_ROOT, 'preprocessing.json');
  await writeFile(filePath, JSON.stringify(stats, null, 2), 'utf-8');
  console.log(`Preprocesamiento guardado en ${filePath}`);
}
