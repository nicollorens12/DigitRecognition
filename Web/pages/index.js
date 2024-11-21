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
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    try {
      let API_URL = "";
      const isDebug = process.env.NEXT_PUBLIC_DEBUG === 'true';
      if (isDebug) {
        API_URL = "http://localhost:8000";
      } else {
        API_URL = "https://digitrecognition-mhek.onrender.com";
      }

      const response = await fetch(`${API_URL}/model/predict/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: Array.from(imageData.data) }),
      });

      const data = await response.json();
      if (response.ok) {
        setPrediction(data.prediction);
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
    ctx.lineWidth = 5; // Set the line width to 5
    ctx.beginPath();

    const offsetX = e.nativeEvent.offsetX || e.touches[0].clientX - canvas.getBoundingClientRect().left;
    const offsetY = e.nativeEvent.offsetY || e.touches[0].clientY - canvas.getBoundingClientRect().top;
    ctx.moveTo(offsetX, offsetY);

    canvas.isDrawing = true;
  };

  const draw = (e) => {
    if (!canvasRef.current.isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const offsetX = e.nativeEvent?.offsetX || e.touches[0].clientX - canvas.getBoundingClientRect().left;
    const offsetY = e.nativeEvent?.offsetY || e.touches[0].clientY - canvas.getBoundingClientRect().top;

    ctx.lineTo(offsetX, offsetY);
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
      <div className={styles.instructionsContainer}>
        <a href='https://github.com/nicollorens12/DigitRecognition' className={styles.link}>
          <FaGithub size={32} />
        </a>
        <a href='https://github.com/nicollorens12/DigitRecognition' className={styles.link}>
          Source Code
        </a>
      </div>
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
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
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
      {prediction !== null ? (
        <div className={styles.predictionContainer}>
          <p className={styles.predictionText}>Prediction: {prediction}</p>
        </div>
      ) : (
        <div className={styles.blankSpace}></div>
      )}
    </div>
  );
}
