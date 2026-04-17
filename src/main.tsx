import { render } from '@solidjs/web';
import App from './components/App';
import './styles/global.css';

console.log('[build] v30 - check buildTagGrid result');

const root = document.getElementById('root');
if (!root) throw new Error('#root element not found');

render(() => <App />, root);