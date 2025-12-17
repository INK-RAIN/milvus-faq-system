import requests
import time
import sys

BASE_URL = "http://localhost:8001"

def wait_for_server():
    print("Waiting for server to start...")
    for _ in range(120):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("Server is up!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("Server failed to start.")
    return False

def test_ingest():
    print("Testing ingestion...")
    faqs = [
        {"question": "如何退货？", "answer": "请在订单页面点击退货按钮，并填写退货原因。"},
        {"question": "退款多久到账？", "answer": "退款通常在商家确认收货后1-3个工作日内原路返回。"},
        {"question": "支持哪些支付方式？", "answer": "我们支持支付宝、微信支付和银行卡支付。"}
    ]
    response = requests.post(f"{BASE_URL}/ingest", json={"faqs": faqs})
    print(f"Ingest response: {response.json()}")
    if response.status_code == 200 and response.json().get("count") == 3:
        print("Ingestion successful.")
    else:
        print("Ingestion failed.")
        sys.exit(1)

def test_search():
    print("Testing search...")
    query = "怎么退货"
    response = requests.post(f"{BASE_URL}/search", json={"query": query, "top_k": 1})
    results = response.json()
    print(f"Search results for '{query}':")
    for res in results:
        print(f"  Q: {res['question']}")
        print(f"  A: {res['answer']}")
        print(f"  Score: {res['score']}")
    
    if len(results) > 0 and "退货" in results[0]["question"]:
        print("Search successful.")
    else:
        print("Search failed.")
        sys.exit(1)

if __name__ == "__main__":
    if wait_for_server():
        test_ingest()
        time.sleep(2) # Wait for index to be ready/persisted (Milvus is fast but just in case)
        test_search()
    else:
        sys.exit(1)
