import { useRef, useState, useEffect } from 'react';
import styles from '../styles/index.module.css';
import { FaGithub } from "react-icons/fa";
import LoadingSpinner from '../components/LoadingSpinner';  // Importa el componente

export default function Home() {
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);  // Nuevo estado para controlar la carga

  useEffect(() => {
    // Desactivar gestos de desplazamiento en el área del canvas
    const canvas = canvasRef.current;

    const preventTouchScroll = (e) => {
      e.preventDefault(); // Previene el desplazamiento del navegador
    };

    canvas.addEventListener('touchstart', preventTouchScroll, { passive: false });
    canvas.addEventListener('touchmove', preventTouchScroll, { passive: false });
    canvas.addEventListener('touchend', preventTouchScroll, { passive: false });

    return () => {
      // Limpia los event listeners cuando el componente se desmonte
      canvas.removeEventListener('touchstart', preventTouchScroll);
      canvas.removeEventListener('touchmove', preventTouchScroll);
      canvas.removeEventListener('touchend', preventTouchScroll);
    };
  }, []);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
  };

  const predictDigit = async () => {
    setLoading(true);
    const canvas = canvasRef.current;
    
    try {
        // Convierte el canvas en una imagen base64
        const base64Image = canvas.toDataURL("image/png").split(",")[1]; // Elimina el prefijo 'data:image/png;base64,'

        const response = await fetch("https://aqac1jeaca.execute-api.eu-west-3.amazonaws.com/default/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ image: base64Image }),
        });

        const data = await response.json();
        if (response.ok) {
            setPrediction(data.Prediction.replace(/[ \[\]]/g, ''));
        } else {
            console.error(data.Error);
            setPrediction("Error: " + data.Error);
        }
    } catch (error) {
        console.error("Error al conectar con la API:", error);
        setPrediction("Error al conectar con la API");
    } finally {
        setLoading(false);
    }
};

  

  const startDrawing = (e) => {
    e.preventDefault(); // Evitar el comportamiento predeterminado
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.lineWidth = 20;
    ctx.beginPath();

    const offsetX = e.nativeEvent?.offsetX || e.touches[0].clientX - canvas.getBoundingClientRect().left;
    const offsetY = e.nativeEvent?.offsetY || e.touches[0].clientY - canvas.getBoundingClientRect().top;
    ctx.moveTo(offsetX, offsetY);

    canvas.isDrawing = true;
  };

  const draw = (e) => {
    if (!canvasRef.current.isDrawing) return;
    e.preventDefault(); // Evitar el desplazamiento de la página mientras se dibuja
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const offsetX = e.nativeEvent?.offsetX || e.touches[0].clientX - canvas.getBoundingClientRect().left;
    const offsetY = e.nativeEvent?.offsetY || e.touches[0].clientY - canvas.getBoundingClientRect().top;

    ctx.lineTo(offsetX, offsetY);
    ctx.stroke();
  };

  const stopDrawing = (e) => {
    e.preventDefault(); // Evitar el comportamiento predeterminado
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
          style={{ touchAction: 'none' }} // Evitar gestos predeterminados como desplazamiento
        ></canvas>
      </div>
      <div className={styles.buttonContainer}>
        <button onClick={clearCanvas} className={styles.button}>
          Erase
        </button>
        <button onClick={predictDigit} className={styles.button}>
          {loading ? <LoadingSpinner /> : 'Predict'}  {/* Muestra el spinner o el texto */}
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
