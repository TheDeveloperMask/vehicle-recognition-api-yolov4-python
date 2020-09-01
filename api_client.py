import requests

filename = 'cars.jpg'
file = open(filename, 'rb')
files = {"image": file}
res = requests.post(url = 'http://127.0.0.1:6000',
                    files = files)

if res.status_code == 200:
    for car in res.json()['cars']:
        print(car)
