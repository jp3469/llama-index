import { useState } from 'react';
import { CircleLoader } from 'react-spinners';
import classNames from 'classnames';
import queryIndex from '../apis/queryIndex';

const IndexQuery = () => {
  const [isLoading, setLoading] = useState(false);
  const [responseText, setResponseText] = useState('');

  const handleQuery = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key == 'Enter') {
      setLoading(true);
      queryIndex(e.currentTarget.value).then((response) => {
        setLoading(false);
        setResponseText(response.text);
      });
    }
  };

  return (
    <div className='query'>
      <div className='query__input'>
        <label htmlFor='query-text'>Ask me what to eat!</label>
        <input
          type='text'
          name='query-text'
          placeholder="I'm hungry!"
          onKeyDown={handleQuery}
        ></input>
      </div>

      {/* <CircleLoader
        className={classNames('query__loader', {
          'query__loader--loading': isLoading,
        })}
        color='#00f596'
      /> */}

      <div
        className={classNames('query__results', {
          'query__results--loading': isLoading,
        })}
      >
        <div className='query__sources__item'>
          <p className='query__sources__item__id'>Response</p>
        </div>
        {responseText}
      </div>
    </div>
  );
};

export default IndexQuery;
