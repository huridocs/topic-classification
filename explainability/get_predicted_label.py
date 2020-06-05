import json
import numpy
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


def get_predicted_labels(texts: str):
    data = dict()
    data['samples'] = [{"seq": s} for s in texts]

    response = request_adapter.post(url='http://localhost:5005/classify',
                                    headers={'Content-Type': 'application/json'},
                                    params=(('model', 'affected_persons'),),
                                    data=json.dumps(data))
    result = json.loads(response.text)

    full_results = numpy.zeros((len(data), 2))
    for index, sample in enumerate(result['samples']):
        if 'non_citizens' in str(sample['predicted_labels']):
            full_results[index][1] = 1
        else:
            full_results[index][0] = 1

    return full_results


if __name__ == "__main__":
    class_name = get_predicted_labels(['non-citizens pink people'])
    print(class_name)