import json
import requests
from requests.adapters import HTTPAdapter


def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    session = session or requests.Session()
    retry = requests.packages.urllib3.util.retry.Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


request_adapter = requests_retry_session()


def get_predicted_labels(text: str):
    data = dict()
    data['samples'] = [{"seq": text}]

    response = request_adapter.post(url='http://localhost:5005/classify',
                                    headers={'Content-Type': 'application/json'},
                                    params=(('model', 'Affected_persons'),),
                                    data=json.dumps(data))
    result = json.loads(response.text)
    print(result)
    return 1 if 'non-citizens' in str(result['samples'][0]['predicted_labels']) else 0


if __name__ == "__main__":
    class_name = get_predicted_labels('non-citizens pink people')
    print(class_name)