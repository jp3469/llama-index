import Header from './components/Header';
// import DocumentTools from './components/DocumentTools';
import IndexQuery from './components/IndexQuery';
import './style.scss';

function App() {
  return (
    <div className='app'>
      <Header />
      <div className='content'>
        <IndexQuery />
      </div>
    </div>
  );
}

export default App;
