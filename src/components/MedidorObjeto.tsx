import { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, RefreshCcw } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const PIXELS_PER_METER = 1.5;

const MedidorObjeto = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectionRef = useRef<number>();
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
    if (detectionRef.current) {
      cancelAnimationFrame(detectionRef.current);
    }
  }, []);

  const startDetection = useCallback(async () => {
    if (!model || !videoRef.current || !canvasRef.current) return;

    const detectFrame = async () => {
      if (!videoRef.current || !canvasRef.current || !isMeasuring) return;

      try {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = video.videoWidth * window.devicePixelRatio;
        canvas.height = video.videoHeight * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
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

          ctx.strokeStyle = '#FFF';
          ctx.fillStyle = '#FFF';
          ctx.font = 'bold 18px Arial';
          
          ctx.beginPath();
          ctx.moveTo(x, y - 20);
          ctx.lineTo(x + width, y - 20);
          ctx.stroke();
          
          ctx.moveTo(x, y - 25);
          ctx.lineTo(x, y - 15);
          ctx.moveTo(x + width, y - 25);
          ctx.lineTo(x + width, y - 15);
          ctx.stroke();
          
          ctx.beginPath();
          ctx.moveTo(x - 20, y);
          ctx.lineTo(x - 20, y + height);
          ctx.stroke();
          
          ctx.moveTo(x - 25, y);
          ctx.lineTo(x - 15, y);
          ctx.moveTo(x - 25, y + height);
          ctx.lineTo(x - 15, y + height);
          ctx.stroke();

          const widthMeters = (width / (canvas.width / PIXELS_PER_METER)).toFixed(2);
          const heightMeters = (height / (canvas.width / PIXELS_PER_METER)).toFixed(2);

          const measurementBg = (text: string, textX: number, textY: number) => {
            const metrics = ctx.measureText(text);
            const padding = 4;
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(
              textX - padding, 
              textY - metrics.actualBoundingBoxAscent - padding,
              metrics.width + padding * 2,
              metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent + padding * 2
            );
            ctx.fillStyle = '#FFF';
          };

          const widthText = `${widthMeters}m`;
          const heightText = `${heightMeters}m`;
          const widthX = x + width/2 - 20;
          const widthY = y - 25;
          const heightX = x - 45;
          const heightY = y + height/2;

          measurementBg(widthText, widthX, widthY);
          measurementBg(heightText, heightX, heightY);
          
          ctx.fillText(widthText, widthX, widthY);
          ctx.fillText(heightText, heightX, heightY);

          setMeasures({
            width: Number(widthMeters),
            height: Number(heightMeters),
            object: obj.class
          });
        }

        detectionRef.current = requestAnimationFrame(detectFrame);
      } catch (error) {
        console.error('Erro na detecção:', error);
        setError('Erro na detecção de objetos');
      }
    };

    detectFrame();
  }, [model, isMeasuring]);

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
          height: { ideal: 720 }
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

      <div className="relative aspect-[3/4] bg-gray-100 mb-4 rounded overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
        />

        {measures && (
          <div className="absolute bottom-4 left-4 right-4 bg-black/50 text-white p-4 rounded">
            <p className="text-center mb-2">Objeto Detectado: {measures.object}</p>
            <div className="flex justify-around">
              <span>Largura: {measures.width}m</span>
              <span>Altura: {measures.height}m</span>
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-center gap-4">
        {!isVideoOn ? (
          <button
            onClick={startCamera}
            className="flex gap-2 items-center bg-blue-500 text-white px-3 py-2 sm:px-4 text-sm sm:text-base rounded"
          >
            <Camera size={20} />
            Iniciar Câmera
          </button>
        ) : (
          <button
            onClick={stopCamera}
            className="flex gap-2 items-center bg-red-500 text-white px-3 py-2 sm:px-4 text-sm sm:text-base rounded"
          >
            <RefreshCcw size={20} />
            Parar Câmera
          </button>
        )}
      </div>
    </div>
  );
};

export default MedidorObjeto;