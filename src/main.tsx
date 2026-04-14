import { render } from '@solidjs/web';
import { Errored } from 'solid-js';
import App from './components/App';
import './styles/global.css';

const root = document.getElementById('root');
if (!root) throw new Error('#root element not found');

render(() => (
  <Errored fallback={(e) => <p class="error">Error: {String(e)}</p>}>
    <App />
  </Errored>
), root);
