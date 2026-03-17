import requests

def test_api():
    files = {"file": open("data/contracts/contract_001.pdf", "rb")}
    res = requests.post("http://localhost:8000/extract", files=files)

    assert res.status_code == 200
