import { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, RefreshCcw } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const PIXELS_PER_METER = 1.5;
const DETECTION_INTERVAL = 500; // ms

const MedidorObjeto = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number>();
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isVideoOn, setIsVideoOn] = useState(false);
  const [isMeasuring, setIsMeasuring] = useState(false);
  const [measures, setMeasures] = useState<{
    width: number;
    height: number;
    object: string;
  } | null>(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  const stopDetection = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  }, []);

  const detectFrame = useCallback(async () => {
    if (!model || !videoRef.current || !canvasRef.current || !isMeasuring) return;

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Match canvas size to video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      const predictions = await model.detect(video);
      
      if (predictions && predictions.length > 0) {
        const obj = predictions[0];
        const [x, y, width, height] = obj.bbox;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(x, y, width, height);
        
        ctx.strokeStyle = '#FFF';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Draw measurements
        const widthMeters = (width / (canvas.width / PIXELS_PER_METER)).toFixed(2);
        const heightMeters = (height / (canvas.width / PIXELS_PER_METER)).toFixed(2);

        ctx.font = '16px Arial';
        ctx.fillStyle = '#FFF';
        
        // Background for text
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(x, y - 20, 140, 20);
        ctx.fillRect(x - 45, y + height/2 - 10, 40, 20);

        // Text
        ctx.fillStyle = '#FFF';
        ctx.fillText(`${widthMeters}m × ${heightMeters}m`, x + 5, y - 5);

        setMeasures({
          width: Number(widthMeters),
          height: Number(heightMeters),
          object: obj.class
        });
      }
    } catch (error) {
      console.error('Erro na detecção:', error);
      setError('Erro na detecção de objetos');
    }
  }, [model, isMeasuring]);

  const startDetection = useCallback(() => {
    stopDetection();
    intervalRef.current = window.setInterval(detectFrame, DETECTION_INTERVAL);
  }, [detectFrame, stopDetection]);

  useEffect(() => {
    loadModel();
    return () => stopDetection();
  }, [stopDetection]);

  useEffect(() => {
    if (isMeasuring) {
      startDetection();
    } else {
      stopDetection();
    }
  }, [isMeasuring, startDetection, stopDetection]);

  const loadModel = async () => {
    try {
      await tf.ready();
      const loadedModel = await cocoSsd.load({
        base: 'lite_mobilenet_v2'
      });
      setModel(loadedModel);
      setIsLoading(false);
    } catch {
      setError('Erro ao carregar modelo');
      setIsLoading(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { max: 30 }
        }
      });
      
      if (videoRef.current) {
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsVideoOn(true);
        setError('');
        setMeasures(null);
        setIsMeasuring(true);
      }
    } catch {
      setError('Erro ao acessar câmera');
    }
  };

  const stopCamera = () => {
    stopDetection();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    streamRef.current = null;
    setIsVideoOn(false);
    setIsMeasuring(false);
    setMeasures(null);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[300px]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-md mx-auto p-2 sm:p-4">
      {error && (
        <div className="mb-4 p-2 bg-red-100 text-red-700 rounded">{error}</div>
      )}

      <div className="relative bg-gray-100 mb-4 rounded overflow-hidden" style={{ aspectRatio: '4/3' }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
          style={{ objectFit: 'cover' }}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
        />

        {measures && (
          <div className="absolute bottom-4 left-4 right-4 bg-black/50 text-white p-3 rounded text-sm">
            <p className="text-center mb-1">Objeto: {measures.object}</p>
            <div className="flex justify-around">
              <span>L: {measures.width}m</span>
              <span>A: {measures.height}m</span>
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-center gap-4">
        {!isVideoOn ? (
          <button
            onClick={startCamera}
            className="flex gap-2 items-center bg-blue-500 text-white px-3 py-2 text-sm rounded"
          >
            <Camera size={18} />
            Iniciar Câmera
          </button>
        ) : (
          <button
            onClick={stopCamera}
            className="flex gap-2 items-center bg-red-500 text-white px-3 py-2 text-sm rounded"
          >
            <RefreshCcw size={18} />
            Parar
          </button>
        )}
      </div>
    </div>
  );
};

export default MedidorObjeto;