// components/LoadingSpinner.js
import styles from "../styles/LoadingSpinner.module.css";
export default function LoadingSpinner() {
    return (
      <div className={styles.threeBody}>
        <div className={styles.threeBodyDot}></div>
        <div className={styles.threeBodyDot}></div>
        <div className={styles.threeBodyDot}></div>
      </div>
    );
  }
  