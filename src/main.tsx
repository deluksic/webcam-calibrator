import { render } from '@solidjs/web';
import App from './components/App';
import './styles/global.css';

export const VERSION = 'v53';
console.log(`[build] ${VERSION} - debug view shows all extent candidates, no containment filter`);

const root = document.getElementById('root');
if (!root) throw new Error('#root element not found');

render(() => <App />, root);