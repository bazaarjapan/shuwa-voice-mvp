'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';

type Phase = 'idle' | 'loading' | 'ready' | 'error';
type Sample = { label: string; vector: number[] };
type TrainingStage = 'idle' | 'countdown' | 'capturing';

const DEFAULT_LABELS = ['こんにちは', 'ありがとう', '大丈夫', '助けて', 'もう一度'];
const STORAGE_KEY = 'shuwa-voice-samples-v1';
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

function makeVector(result: HandLandmarkerResult): number[] | null {
  if (!result.landmarks.length) return null;
  const hands = result.landmarks
    .map((landmarks, index) => ({
      landmarks,
      side: result.handedness[index]?.[0]?.categoryName ?? `hand-${index}`,
    }))
    .sort((a, b) => a.side.localeCompare(b.side))
    .slice(0, 2);
  const wrists = hands.map((hand) => hand.landmarks[0]);
  const center = wrists.reduce(
    (sum, point) => ({ x: sum.x + point.x / wrists.length, y: sum.y + point.y / wrists.length }),
    { x: 0, y: 0 },
  );
  const points = hands.flatMap((hand) => hand.landmarks);
  const scale = Math.max(0.08, ...points.map((point) => Math.hypot(point.x - center.x, point.y - center.y)));
  const vector: number[] = [hands.length];
  for (let slot = 0; slot < 2; slot += 1) {
    const hand = hands[slot];
    vector.push(hand ? (hand.side === 'Left' ? -1 : 1) : 0);
    for (let point = 0; point < 21; point += 1) {
      const landmark = hand?.landmarks[point];
      vector.push(
        landmark ? (landmark.x - center.x) / scale : 0,
        landmark ? (landmark.y - center.y) / scale : 0,
        landmark ? landmark.z / scale : 0,
      );
    }
  }
  return vector;
}

function vectorDistance(a: number[], b: number[]) {
  if (a.length !== b.length || a[0] !== b[0]) return Number.POSITIVE_INFINITY;
  let sum = 0;
  for (let i = 1; i < a.length; i += 1) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum / (a.length - 1));
}

function averageVectors(vectors: number[][]): number[] | null {
  if (!vectors.length) return null;
  const frequencies = new Map<number, number>();
  vectors.forEach((vector) => frequencies.set(vector[0], (frequencies.get(vector[0]) ?? 0) + 1));
  const [commonHandCount, consistentFrameCount] = [...frequencies.entries()].sort((a, b) => b[1] - a[1])[0];
  if (consistentFrameCount < 6 || consistentFrameCount / vectors.length < 0.7) return null;
  const matching = vectors.filter((vector) => vector[0] === commonHandCount);
  return matching[0].map((_, index) => {
    if (index === 0) return commonHandCount;
    return matching.reduce((sum, vector) => sum + vector[index], 0) / matching.length;
  });
}

function drawHands(canvas: HTMLCanvasElement, video: HTMLVideoElement, hands: NormalizedLandmark[][]) {
  const width = video.videoWidth || 1280;
  const height = video.videoHeight || 720;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext('2d');
  if (!context) return;
  context.clearRect(0, 0, width, height);
  context.lineWidth = Math.max(4, width / 240);
  context.lineCap = 'round';
  context.strokeStyle = '#f8bd24';
  context.fillStyle = '#13251d';
  hands.forEach((landmarks) => {
    CONNECTIONS.forEach(([start, end]) => {
      context.beginPath();
      context.moveTo((1 - landmarks[start].x) * width, landmarks[start].y * height);
      context.lineTo((1 - landmarks[end].x) * width, landmarks[end].y * height);
      context.stroke();
    });
    landmarks.forEach((point) => {
      context.beginPath();
      context.arc((1 - point.x) * width, point.y * height, Math.max(5, width / 180), 0, Math.PI * 2);
      context.fill();
    });
  });
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameRef = useRef<number | null>(null);
  const latestVectorRef = useRef<number[] | null>(null);
  const samplesRef = useRef<Sample[]>([]);
  const lastVideoTimeRef = useRef(-1);
  const recentLabelsRef = useRef<string[]>([]);
  const lastSpokenRef = useRef({ label: '', time: 0 });
  const trainingRef = useRef<{ stage: TrainingStage; label: string; frames: number[][] }>({ stage: 'idle', label: '', frames: [] });
  const trainingTimersRef = useRef<number[]>([]);

  const [phase, setPhase] = useState<Phase>('idle');
  const [status, setStatus] = useState('カメラを始めると、ここに案内が出ます');
  const [labels, setLabels] = useState(DEFAULT_LABELS);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedLabel, setSelectedLabel] = useState(DEFAULT_LABELS[0]);
  const [recognized, setRecognized] = useState('まだ認識していません');
  const [confidence, setConfidence] = useState(0);
  const [history, setHistory] = useState<string[]>([]);
  const [autoSpeak, setAutoSpeak] = useState(true);
  const [handsVisible, setHandsVisible] = useState(0);
  const [trainingStage, setTrainingStage] = useState<TrainingStage>('idle');
  const [trainingLabel, setTrainingLabel] = useState('');
  const [countdown, setCountdown] = useState(3);
  const [capturedFrames, setCapturedFrames] = useState(0);

  useEffect(() => {
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (!saved) return;
        const parsed = JSON.parse(saved) as { labels?: string[]; samples?: Sample[] };
        if (parsed.labels?.length === 5) {
          setLabels(parsed.labels);
          setSelectedLabel(parsed.labels[0]);
        }
        if (Array.isArray(parsed.samples)) {
          setSamples(parsed.samples);
          samplesRef.current = parsed.samples;
        }
      } catch {
        localStorage.removeItem(STORAGE_KEY);
      }
    });
    return () => { cancelled = true; };
  }, []);

  const speak = useCallback((text: string) => {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ja-JP';
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
  }, []);

  const runRecognition = useCallback((vector: number[]) => {
    const counts = new Map<string, number>();
    samplesRef.current.forEach((sample) => counts.set(sample.label, (counts.get(sample.label) ?? 0) + 1));
    const allSamples = samplesRef.current.filter((sample) => (counts.get(sample.label) ?? 0) >= 3);
    if (!allSamples.length) {
      setRecognized('まずは1つ、3回覚えさせてね');
      setConfidence(0);
      return;
    }
    let best: { label: string; distance: number } | null = null;
    for (const sample of allSamples) {
      const distance = vectorDistance(vector, sample.vector);
      if (!best || distance < best.distance) best = { label: sample.label, distance };
    }
    if (!best) return;
    const score = Math.max(0, Math.min(99, Math.round((1 - best.distance / 0.32) * 100)));
    setConfidence(score);
    if (best.distance > 0.24) {
      recentLabelsRef.current = [];
      setRecognized('もう一度、ゆっくり見せてね');
      return;
    }
    recentLabelsRef.current = [...recentLabelsRef.current.slice(-7), best.label];
    const stable = recentLabelsRef.current.filter((label) => label === best.label).length >= 6;
    setRecognized(stable ? best.label : '考えています…');
    if (!stable) return;
    const now = Date.now();
    if (lastSpokenRef.current.label === best.label && now - lastSpokenRef.current.time < 3000) return;
    lastSpokenRef.current = { label: best.label, time: now };
    setHistory((previous) => [...previous.slice(-5), best.label]);
    if (autoSpeak) speak(best.label);
  }, [autoSpeak, speak]);

  const runRecognitionRef = useRef(runRecognition);
  useEffect(() => {
    runRecognitionRef.current = runRecognition;
  }, [runRecognition]);

  const detectFrame = useCallback(function frame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !canvas || !landmarker || video.readyState < 2) {
      frameRef.current = requestAnimationFrame(frame);
      return;
    }
    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;
      const result = landmarker.detectForVideo(video, performance.now());
      setHandsVisible(result.landmarks.length);
      drawHands(canvas, video, result.landmarks);
      const vector = makeVector(result);
      latestVectorRef.current = vector;
      if (vector) {
        if (trainingRef.current.stage === 'capturing') {
          trainingRef.current.frames.push([...vector]);
          setCapturedFrames(trainingRef.current.frames.length);
          setStatus(`「${trainingRef.current.label}」の形を記録しています。手を止めてね`);
        } else if (trainingRef.current.stage === 'countdown') {
          // カウントダウン中は手を追跡するだけで、通常認識は行わない。
        } else {
          setStatus(`${result.landmarks.length}つの手を見つけました`);
          runRecognitionRef.current(vector);
        }
      } else {
        recentLabelsRef.current = [];
        setStatus('手を四角の中に見せてね');
        setConfidence(0);
      }
    }
    frameRef.current = requestAnimationFrame(frame);
  }, []);

  const cancelTraining = useCallback((message = '学習をキャンセルしました') => {
    trainingTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    trainingTimersRef.current = [];
    trainingRef.current = { stage: 'idle', label: '', frames: [] };
    setTrainingStage('idle');
    setTrainingLabel('');
    setCapturedFrames(0);
    setCountdown(3);
    recentLabelsRef.current = [];
    setRecognized('まだ認識していません');
    setConfidence(0);
    setStatus(message);
  }, []);

  const stopCamera = useCallback(() => {
    cancelTraining('カメラを止めました');
    if (frameRef.current) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    setPhase('idle');
    setHandsVisible(0);
  }, [cancelTraining]);

  useEffect(() => () => {
    trainingTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    if (frameRef.current) cancelAnimationFrame(frameRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    landmarkerRef.current?.close();
    window.speechSynthesis?.cancel();
  }, []);

  async function startCamera() {
    try {
      setPhase('loading');
      setStatus('AIとカメラを準備しています…');
      if (!landmarkerRef.current) {
        const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm');
        landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.6,
          minTrackingConfidence: 0.6,
        });
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      });
      streamRef.current = stream;
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setPhase('ready');
      setStatus('準備できました。手を見せてね！');
      frameRef.current = requestAnimationFrame(detectFrame);
    } catch (error) {
      console.error(error);
      setPhase('error');
      setStatus('カメラを使えませんでした。許可を確認して、もう一度ためしてね');
    }
  }

  function finishTraining() {
    const session = trainingRef.current;
    const vector = averageVectors(session.frames);
    if (!vector || session.frames.length < 6) {
      cancelTraining('手を十分に見つけられませんでした。もう一度ためしてね');
      return;
    }
    const next = [...samplesRef.current, { label: session.label, vector }];
    samplesRef.current = next;
    setSamples(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ labels, samples: next }));
    const count = next.filter((sample) => sample.label === session.label).length;
    cancelTraining(`「${session.label}」を覚えました（${Math.min(count, 3)}/3回）`);
    speak(count >= 3 ? '学習が完成しました' : '覚えました');
  }

  function startTraining() {
    if (!latestVectorRef.current) {
      setStatus('手が見えていません。手を映してから押してね');
      return;
    }
    window.speechSynthesis?.cancel();
    recentLabelsRef.current = [];
    setRecognized('学習中です');
    setConfidence(0);
    setCountdown(3);
    setCapturedFrames(0);
    setTrainingStage('countdown');
    setTrainingLabel(selectedLabel);
    trainingRef.current = { stage: 'countdown', label: selectedLabel, frames: [] };
    setStatus('手を見せたまま待ってね。あと3秒');
    [2, 1].forEach((number, index) => {
      trainingTimersRef.current.push(window.setTimeout(() => {
        setCountdown(number);
        setStatus(`手を見せたまま待ってね。あと${number}秒`);
      }, (index + 1) * 1000));
    });
    trainingTimersRef.current.push(window.setTimeout(() => {
      trainingRef.current.stage = 'capturing';
      setTrainingStage('capturing');
      setCapturedFrames(0);
      setStatus(`「${selectedLabel}」を記録しています。手を止めてね`);
    }, 3000));
    trainingTimersRef.current.push(window.setTimeout(finishTraining, 4200));
  }

  function resetSelectedLabel() {
    if (trainingRef.current.stage !== 'idle') return;
    const next = samplesRef.current.filter((sample) => sample.label !== selectedLabel);
    samplesRef.current = next;
    setSamples(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ labels, samples: next }));
    setStatus(`「${selectedLabel}」の学習を消しました`);
  }

  function updateLabel(index: number, value: string) {
    if (trainingRef.current.stage !== 'idle') return;
    const next = labels.map((label, labelIndex) => labelIndex === index ? value : label);
    const old = labels[index];
    const renamedSamples = samplesRef.current.map((sample) => sample.label === old ? { ...sample, label: value } : sample);
    setLabels(next);
    setSelectedLabel(value);
    setSamples(renamedSamples);
    samplesRef.current = renamedSamples;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ labels: next, samples: renamedSamples }));
  }

  function resetLearning() {
    cancelTraining('覚えた手の形を消しました');
    samplesRef.current = [];
    setSamples([]);
    setHistory([]);
    setRecognized('まだ認識していません');
    setConfidence(0);
    localStorage.removeItem(STORAGE_KEY);
  }

  const trainedCount = (label: string) => samples.filter((sample) => sample.label === label).length;

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="しゅわボイス ホーム"><span className="brand-mark" aria-hidden="true">手</span><span>しゅわボイス</span></a>
        <span className="mvp-badge">MVP・おためし版</span>
      </header>
      <div id="top" className="page-shell">
        <section className="intro">
          <p className="eyebrow">手のかたちを、ことばに。</p>
          <h1>カメラに手話を<br /><em>見せてみよう。</em></h1>
          <p className="lead">手の形をAIが見つけて、文字と声で伝えます。まずは自分の手話を3回ずつ覚えさせてね。</p>
        </section>
        <section className="workspace" aria-label="手話認識カメラ">
          <div className="camera-card">
            <div className="camera-head"><div><span className={`live-dot ${phase === 'ready' ? 'on' : ''}`} />{phase === 'ready' ? 'カメラ動作中' : 'カメラ停止中'}</div><span>{handsVisible ? `手を ${handsVisible}つ 発見` : '手を探しています'}</span></div>
            <div className="camera-stage">
              <video ref={videoRef} playsInline muted aria-label="カメラの映像" /><canvas ref={canvasRef} aria-hidden="true" />
              <div className="guide-frame" aria-hidden="true"><span>この中に手を見せてね</span></div>
              {trainingStage !== 'idle' && <div className={`training-overlay ${trainingStage}`} role="status" aria-live="assertive"><span>{trainingStage === 'countdown' ? countdown : capturedFrames}</span><strong>{trainingStage === 'countdown' ? 'この形のまま待ってね' : '記録しています'}</strong><small>{trainingStage === 'countdown' ? `「${trainingLabel}」の学習を始めます` : '手を止めて、カメラを見てね'}</small></div>}
              {phase !== 'ready' && <div className="camera-empty"><span className="hand-symbol" aria-hidden="true">✋</span><strong>カメラはまだ止まっています</strong><small>映像は保存も送信もしません</small></div>}
            </div>
            <div className="camera-actions">
              {phase === 'ready' ? <button className="button secondary" onClick={stopCamera}>■ カメラを止める</button> : <button className="button primary" onClick={startCamera} disabled={phase === 'loading'}>{phase === 'loading' ? '準備中…' : '▶ カメラを始める'}</button>}
              <p role="status" aria-live="polite">{status}</p>
            </div>
          </div>
          <aside className="result-card" aria-label="認識結果">
            <span className="card-label">認識したことば</span><div className="result-word" aria-live="polite">{recognized}</div>
            <div className="confidence-row"><span>AIの自信</span><strong>{confidence}%</strong></div><div className="meter"><span style={{ width: `${confidence}%` }} /></div>
            <button className="speak-button" onClick={() => recognized && speak(recognized)} disabled={confidence < 1}><span aria-hidden="true">●)))</span> もう一度よみあげる</button>
            <label className="toggle-row"><input type="checkbox" checked={autoSpeak} onChange={(event) => setAutoSpeak(event.target.checked)} /><span>認識したら自動でよみあげる</span></label>
            <div className="history"><div><span>ことばの記録</span><button onClick={() => setHistory([])}>消す</button></div><p>{history.length ? history.join('　') : '認識したことばがここに並びます'}</p></div>
          </aside>
        </section>
        <section className="training" aria-labelledby="training-title">
          <div className="section-heading"><div><span className="step-number">1</span><p className="eyebrow">さいしょにすること</p></div><h2 id="training-title">5つの手話を<br />AIに覚えさせよう</h2><p>ことばを選んで学習を始めると、3秒後に約1秒間だけ手の形を記録します。角度を少し変えて、1つにつき3回覚えさせてね。</p></div>
          <div className="training-panel">
            <div className="label-grid">
              {labels.map((label, index) => {
                const count = trainedCount(label);
                return <label key={index} className={`label-card ${selectedLabel === label ? 'selected' : ''} ${trainingStage !== 'idle' ? 'locked' : ''}`}><input type="radio" name="selected-label" checked={selectedLabel === label} disabled={trainingStage !== 'idle'} onChange={() => setSelectedLabel(label)} /><span className="label-number">0{index + 1}</span><input className="label-input" value={label} disabled={trainingStage !== 'idle'} aria-label={`${index + 1}番目のことば`} onFocus={() => setSelectedLabel(label)} onChange={(event) => updateLabel(index, event.target.value)} /><span className={`count-badge ${count >= 3 ? 'done' : ''}`}>{Math.min(count, 3)}/3</span></label>;
              })}
            </div>
            <div className="learn-box"><div><span>いま覚えることば</span><strong>{selectedLabel || 'ことばを入力してね'}</strong></div>{trainingStage === 'idle' ? <button className="button learn-button" onClick={startTraining} disabled={phase !== 'ready' || !selectedLabel.trim() || trainedCount(selectedLabel) >= 3}>{trainedCount(selectedLabel) >= 3 ? '✓ 3回の学習が完成' : '＋ ガイド付きで覚える'}</button> : <button className="button cancel-button" onClick={() => cancelTraining()}>× 学習をやめる</button>}<small>写真ではなく、約1秒分の手の点を平均した数字だけをこのPCに保存します。3回そろったことばだけを認識します。</small></div>
            <div className="reset-actions">{trainedCount(selectedLabel) > 0 && <button className="reset-button" disabled={trainingStage !== 'idle'} onClick={resetSelectedLabel}>「{selectedLabel}」だけ学び直す</button>}<button className="reset-button" onClick={resetLearning}>覚えた形をぜんぶ消す</button></div>
          </div>
        </section>
        <section className="how-it-works" aria-labelledby="how-title"><p className="eyebrow">しくみはかんたん</p><h2 id="how-title">3つのステップで伝わる</h2><div className="steps"><article><span>01</span><b aria-hidden="true">✋</b><h3>見つける</h3><p>カメラで手の関節を見つけます。</p></article><article><span>02</span><b aria-hidden="true">⌁</b><h3>考える</h3><p>覚えた手の形とくらべます。</p></article><article><span>03</span><b aria-hidden="true">あ</b><h3>伝える</h3><p>ことばを文字と声で出します。</p></article></div></section>
        <section className="notice"><span aria-hidden="true">!</span><div><h2>このMVPでできること</h2><p>今は、止めた手の形を一つずつ見分ける練習版です。手の動き・顔の表情・日本手話の文法を使う文章の翻訳は、次のステップで加えます。大切な会話では、結果を相手と確認してください。</p></div></section>
      </div>
      <footer><span>しゅわボイス MVP</span><span>カメラ映像は保存しません</span></footer>
    </main>
  );
}
