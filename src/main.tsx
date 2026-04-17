import { render } from '@solidjs/web';
import App from './components/App';
import { VERSION } from './version';
import './styles/global.css';

console.log(`[build] ${VERSION} - debug view shows all extent candidates, no containment filter`);

const root = document.getElementById('root');
if (!root) throw new Error('#root element not found');

render(() => <App />, root);