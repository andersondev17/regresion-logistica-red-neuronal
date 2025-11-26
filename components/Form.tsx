"use client";

import { FormEvent, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { Button } from "./ui/Button";

type Stats = {
  ageMin: number;
  ageMax: number;
  ageRange: number;
};

type PredictionResult = {
  logisticProb: number;
  nnProb: number;
  avgProb: number;
};

const MODELS_BASE = "/models";

function normalizeAge(age: number, stats: Stats) {
  const min = stats.ageMin ?? 0;
  const max = stats.ageMax ?? min + stats.ageRange;
  const range = stats.ageRange || Math.max(max - min, 1e-6);
  const clamped = Math.min(Math.max(age, min), max);
  return (clamped - min) / range;
}

export default function PredictionForm() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [models, setModels] = useState<{ logistic: tf.LayersModel | null; nn: tf.LayersModel | null }>({
    logistic: null,
    nn: null,
  });

  const modelsReady = Boolean(models.logistic && models.nn && stats);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const [logistic, nn, statsResponse] = await Promise.all([
          tf.loadLayersModel(`${MODELS_BASE}/logistic/model.json`),
          tf.loadLayersModel(`${MODELS_BASE}/nn/model.json`),
          fetch(`${MODELS_BASE}/preprocessing.json`).then((res) => {
            if (!res.ok) throw new Error('No se pudieron obtener las estadisticas');
            return res.json();
          }),
        ]);

        if (cancelled) return;
        setModels({ logistic, nn });
        setStats(statsResponse);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        console.error(err);
        setError('No se pudieron cargar los modelos. Ejecuta los scripts de entrenamiento para generarlos.');
      }
    }

    bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setResult(null);

    if (!modelsReady || !stats || !models.logistic || !models.nn) {
      setError('Modelos cargando, intentalo nuevamente en unos segundos.');
      return;
    }

    const formData = new FormData(event.currentTarget);
    const sexValue = formData.get('sex') as string;
    const ageValue = parseFloat(formData.get('age') as string);

    if (!sexValue) {
      setError('Selecciona el sexo.');
      return;
    }

    if (Number.isNaN(ageValue)) {
      setError('Ingresa una edad valida.');
      return;
    }

    const sex = sexValue === 'male' ? 1 : 0;
    const ageScaled = normalizeAge(ageValue, stats);

    setLoading(true);
    try {
      const [logisticProb, nnProb] = tf.tidy(() => {
        const input = tf.tensor2d([[sex, ageScaled]]);
        const logisticTensor = models.logistic!.predict(input) as tf.Tensor;
        const nnTensor = models.nn!.predict(input) as tf.Tensor;
        const logisticValue = logisticTensor.dataSync()[0];
        const nnValue = nnTensor.dataSync()[0];
        return [logisticValue, nnValue];
      });

      const avgProb = (logisticProb + nnProb) / 2;
      setResult({ logisticProb, nnProb, avgProb });
    } catch (predictionError) {
      console.error(predictionError);
      setError('Ocurrio un error calculando la prediccion.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-10">
      <form className="space-y-4" onSubmit={handleSubmit}>
        <div className="space-y-1">
          <label htmlFor="sex" className="text-zinc-900 dark:text-zinc-100 leading-8">
            Sexo *
          </label>
          <select
            id="sex"
            name="sex"
            className="w-full rounded-xl border border-zinc-300 bg-white px-4 py-3 text-zinc-900 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          >
            <option value="">Seleccione el sexo</option>
            <option value="male">Masculino</option>
            <option value="female">Femenino</option>
          </select>
        </div>

        <div className="space-y-1">
          <label htmlFor="age" className="text-zinc-900 dark:text-zinc-100 leading-8">
            Edad *
          </label>
          <input
            id="age"
            name="age"
            type="number"
            placeholder="Ingresa tu edad"
            className="w-full rounded-xl border border-zinc-300 bg-white px-4 py-3 text-zinc-900 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            min="0"
            max="120"
            required
          />
        </div>

        <Button
          type="submit"
          fullWidth
          disabled={loading || !modelsReady}
          variant="primary"
        >
          {modelsReady ? (loading ? "Calculando..." : "Calcular") : "Cargando modelos..."}
        </Button>
      </form>

      {error && (
        <div className="p-4 rounded-xl border border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20 text-center">
          <h3 className="text-lg font-semibold text-red-900 dark:text-red-100">Error</h3>
          <p className="text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}

      {result && (
        <div className="p-4 rounded-xl border border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20 space-y-2">
          <h3 className="text-lg font-semibold text-green-900 dark:text-green-100 text-center">
            {result.avgProb >= 0.5 ? "Sobrevive" : "No sobrevive"}
          </h3>
          <div className="text-green-700 dark:text-green-300">
            <p>Regresion logistica: <strong>{(result.logisticProb * 100).toFixed(1)}%</strong></p>
            <p>Red neuronal: <strong>{(result.nnProb * 100).toFixed(1)}%</strong></p>
            <p>Promedio: <strong>{(result.avgProb * 100).toFixed(1)}%</strong></p>
          </div>
        </div>
      )}
    </div>
  );
}
