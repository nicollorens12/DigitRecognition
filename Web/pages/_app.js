import '@styles/global.css'; // Importa tu archivo de estilos globales

function MyApp({ Component, pageProps }) {
  return <Component {...pageProps} />; // Esto renderiza la página actual
}

export default MyApp;