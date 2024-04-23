export type ResponseSources = {
  text: string;
  doc_id: string;
  start: number;
  end: number;
  similarity: number;
};

export type QueryResponse = {
  text: string;
  sources: ResponseSources[];
};

/*
Got it! Could you provide more details on what you'd like to test within the app?

-> {text: 'Got it! Could you provide more details on what you'd like to test within the app?'}
*/

const queryIndex = async (query: string): Promise<QueryResponse> => {
  const queryURL = new URL('https://llama-index-1nqj.onrender.com/query?');
  queryURL.searchParams.append('text', query);

  const response = await fetch(queryURL, { mode: 'cors' });
  if (!response.ok) {
    return { text: 'Error in query', sources: [] };
  }

  const queryResponse = (await response.json()) as QueryResponse;

  return queryResponse;
};

export default queryIndex;
