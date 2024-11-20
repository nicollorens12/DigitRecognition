import { useRef, useState } from 'react';
import styles from '../styles/index.module.css';
import { FaGithub } from "react-icons/fa";

export default function Home() {
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState(null);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setPrediction(null); // Clear the prediction as well
  };

  const predictDigit = async () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
  
    // Obtener los datos en bruto de la imagen
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  
    try {
      // Realizar el request a la API
      const response = await fetch("http://127.0.0.1:8000/model/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: Array.from(imageData.data) }),
      });
  
      const data = await response.json();
      if (response.ok) {
        setPrediction(data.prediction); // Mostrar la predicción
      } else {
        console.error(data.error);
        setPrediction("Error: " + data.error);
      }
    } catch (error) {
      console.error("Error al conectar con la API:", error);
      setPrediction("Error al conectar con la API");
    }
  };
  

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.lineWidth = 20; // Set the line width to 5
    ctx.beginPath();
    ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    canvas.isDrawing = true;
  };

  const draw = (e) => {
    if (!canvasRef.current.isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    ctx.stroke();
  };

  const stopDrawing = () => {
    const canvas = canvasRef.current;
    canvas.isDrawing = false;
  };

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Digit Recognition App</h1>
      <h2 className={styles.subtitle}>Made by Nico Llorens</h2>
      <a
        href="https://github.com/nicollorens12/DigitRecognition"
        target="_blank"
        rel="noopener noreferrer"
        className={styles.githubLink}
      >
        <FaGithub />
         Source Code
      </a>
      <div className={styles.canvasContainer}>
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          className={styles.drawingCanvas}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
        ></canvas>
      </div>
      <div className={styles.buttonContainer}>
        <button onClick={clearCanvas} className={styles.button}>
          Erase
        </button>
        <button onClick={predictDigit} className={styles.button}>
          Predict
        </button>
      </div>
      {prediction !== null && (
        <p className={styles.predictionText}>Prediction: {prediction}</p>
      )}
    </div>
  );
}
