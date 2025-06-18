import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import defaultdict
import pytest
import redis
import gymnasium as gym


import browsergym.core
from browsergym.utils.obs import flatten_axtree_to_str


# 使用一个不同的 Redis 数据库来进行测试，避免污染主数据库
TEST_REDIS_URL = "redis://localhost:6380/1"
# 标记，用于检查 redis 是否可用
redis_available = False
try:
    r = redis.from_url(TEST_REDIS_URL)
    r.ping()
    redis_available = True
except (redis.exceptions.ConnectionError, ImportError):
    pass


# 模拟一个简单的网站服务器，它会记录每个路径的请求次数
class RequestCounterHandler(BaseHTTPRequestHandler):
    # 使用类级别的字典来存储所有实例的请求计数
    request_counts = defaultdict(int)
    # 用于模拟不完整响应的标志
    simulate_incomplete_response = False
    simulate_error_response = False

    def do_GET(self):
        self.request_counts[self.path] += 1
        
        # 模拟错误响应
        if self.simulate_error_response:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Internal Server Error")
            return
        
        self.send_response(200)
        if self.path == "/":
            content = b'<html><head><link rel="stylesheet" href="/style.css"></head><body><h1>Hello</h1></body></html>'
            
            # 模拟不完整响应
            if self.simulate_incomplete_response:
                content = content[:len(content)//2]  # 只发送一半内容
            
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            
        elif self.path == "/style.css":
            content = b"h1 { color: blue; }"
            self.send_header("Content-type", "text/css")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            
        elif self.path == "/incomplete.html":
            # 故意返回不完整的HTML
            content = b'<html><head><title>Test'  # 缺少结束标签
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            
        elif self.path == "/json":
            content = b'{"message": "hello", "status": "ok"}'
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            
        elif self.path == "/invalid-json":
            content = b'{"message": "hello", "status":'  # 不完整的JSON
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            
        else:
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")

    @classmethod
    def reset_counts(cls):
        cls.request_counts.clear()
        cls.simulate_incomplete_response = False
        cls.simulate_error_response = False


@pytest.fixture(scope="module")
def http_server_and_redis():
    """
    一个 Pytest Fixture，它在后台线程中启动一个HTTP服务器，
    并提供一个Redis客户端，同时在测试前后负责清理工作。
    """
    if not redis_available:
        pytest.skip("Redis server not available at " + TEST_REDIS_URL)

    # 启动 HTTP 服务器
    server = HTTPServer(("localhost", 0), RequestCounterHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    server_url = f"http://localhost:{server.server_port}"

    # 初始化 Redis 客户端
    redis_client = redis.from_url(TEST_REDIS_URL, decode_responses=True)

    yield server_url, redis_client

    # 清理
    server.shutdown()
    server_thread.join()


# 在每个测试函数之前重置计数和Redis数据库
@pytest.fixture(autouse=True)
def reset_before_each_test(http_server_and_redis):
    _, redis_client = http_server_and_redis
    RequestCounterHandler.reset_counts()
    redis_client.flushdb()


def test_cache_disabled(http_server_and_redis):
    """测试当 enable_context_cache=False 时，不发生缓存"""
    server_url, _ = http_server_and_redis
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": server_url},
        enable_context_cache=False,
    )
    obs, info = env.reset()
    assert RequestCounterHandler.request_counts["/"] == 1
    obs, info = env.reset()
    assert RequestCounterHandler.request_counts["/"] == 2
    env.close()


def test_cache_hit_for_document(http_server_and_redis):
    """测试主文档的缓存命中"""
    server_url, _ = http_server_and_redis
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": server_url},
        enable_context_cache=True,
        context_cache_kwargs={"redis_url": TEST_REDIS_URL},
    )
    # 第一次访问，缓存未命中
    env.reset()
    assert RequestCounterHandler.request_counts["/"] == 1
    # 第二次访问，应该命中缓存
    env.reset()
    assert RequestCounterHandler.request_counts["/"] == 1
    env.close()


def test_cache_with_css_resources(http_server_and_redis):
    """测试CSS资源的缓存"""
    server_url, _ = http_server_and_redis
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": server_url},
        enable_context_cache=True,
        context_cache_kwargs={"redis_url": TEST_REDIS_URL},
    )
    # 第一次访问，主文档和CSS都应该被请求
    env.reset()
    time.sleep(1)  # 等待CSS加载
    assert RequestCounterHandler.request_counts["/"] == 1
    assert RequestCounterHandler.request_counts["/style.css"] == 1
    
    # 第二次访问，两者都应该从缓存中获取
    env.reset()
    time.sleep(1)
    assert RequestCounterHandler.request_counts["/"] == 1
    assert RequestCounterHandler.request_counts["/style.css"] == 1
    env.close()


def test_cache_integrity_validation(http_server_and_redis):
    """测试缓存完整性验证"""
    server_url, redis_client = http_server_and_redis
    
    # 手动在Redis中插入一个不完整的缓存条目
    import json
    import base64
    import hashlib
    
    url = f"{server_url}/"
    cache_key = hashlib.md5(url.encode("utf-8")).hexdigest()
    
    # 创建一个不完整的缓存条目（缺少必要字段）
    incomplete_cache = {
        "status": 200,
        "headers": {"content-type": "text/html"},
        # 缺少 body_b64 字段
        "timestamp": int(time.time() * 1000)
    }
    
    redis_client.set(cache_key, json.dumps(incomplete_cache), ex=3600)
    
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": server_url},
        enable_context_cache=True,
        context_cache_kwargs={"redis_url": TEST_REDIS_URL},
    )
    
    # 由于缓存验证失败，应该重新请求
    env.reset()
    assert RequestCounterHandler.request_counts["/"] == 1
    env.close()


def test_content_length_mismatch_handling(http_server_and_redis):
    """测试内容长度不匹配的处理"""
    server_url, redis_client = http_server_and_redis
    
    # 创建一个内容长度不匹配的恶意缓存条目
    import json
    import base64
    import hashlib
    
    url = f"{server_url}/"
    cache_key = hashlib.md5(url.encode("utf-8")).hexdigest()
    
    # 创建内容长度不匹配的缓存条目
    malicious_cache = {
        "status": 200,
        "headers": {
            "content-type": "text/html",
            "content-length": "1000"  # 错误的长度
        },
        "body_b64": base64.b64encode(b"<html><body>Short content</body></html>").decode(),
        "timestamp": int(time.time() * 1000)
    }
    
    redis_client.set(cache_key, json.dumps(malicious_cache), ex=3600)
    
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": server_url},
        enable_context_cache=True,
        context_cache_kwargs={"redis_url": TEST_REDIS_URL},
    )
    
    # 由于内容长度验证失败，应该重新请求
    env.reset()
    assert RequestCounterHandler.request_counts["/"] == 1
    env.close()


def test_cache_hit_statistics(http_server_and_redis):
    """测试缓存命中统计"""
    server_url, _ = http_server_and_redis
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": server_url},
        enable_context_cache=True,
        context_cache_kwargs={"redis_url": TEST_REDIS_URL},
    )
    
    # 第一次访问
    obs, info = env.reset()
    assert "browser_cache_hit_stats" in info
    
    # 第二次访问，应该有缓存命中
    obs, info = env.reset()
    cache_stats = info.get("browser_cache_hit_stats", {})
    # 应该有至少一个主机的缓存命中
    assert len(cache_stats) > 0
    env.close()


def test_page_snapshot_functionality(http_server_and_redis):
    """测试页面快照功能"""
    server_url, _ = http_server_and_redis
    
    # 修改服务器以返回包含动态内容的页面
    original_do_get = RequestCounterHandler.do_GET
    
    def dynamic_page_handler(self):
        if self.path == "/":
            # 返回一个包含动态内容的页面
            content = f'''
            <html>
            <head><title>Dynamic Page</title></head>
            <body>
                <h1>Static Content</h1>
                <div id="dynamic">Loading...</div>
                <script>
                    setTimeout(() => {{
                        document.getElementById('dynamic').innerHTML = 'Dynamic content loaded at {time.time()}';
                    }}, 500);
                </script>
            </body>
            </html>
            '''.encode()
            
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            RequestCounterHandler.request_counts[self.path] += 1
        else:
            original_do_get(self)
    
    RequestCounterHandler.do_GET = dynamic_page_handler
    
    try:
        # 使用页面快照功能
        env = gym.make(
            "browsergym/openended",
            task_kwargs={"start_url": server_url},
            enable_context_cache=True,
            context_cache_kwargs={
                "redis_url": TEST_REDIS_URL,
                "enable_page_snapshot": True,
                "snapshot_wait_time": 2000
            },
        )
        
        # 第一次访问 - 创建快照
        print("First visit - creating snapshot...")
        env.reset()
        assert RequestCounterHandler.request_counts["/"] == 1
        
        # 第二次访问 - 使用快照
        print("Second visit - using snapshot...")
        env.reset() 
        # 页面计数不应该增加（使用快照）
        assert RequestCounterHandler.request_counts["/"] == 1
        
        print("Page snapshot test passed!")
        env.close()
        
    finally:
        RequestCounterHandler.do_GET = original_do_get


def test_snapshot_mode_user_interaction_requests(http_server_and_redis):
    """测试页面快照模式下用户交互是否能正常触发新请求"""
    server_url, _ = http_server_and_redis
    
    # 模拟一个包含用户交互的页面
    original_do_get = RequestCounterHandler.do_GET
    
    def interactive_page_handler(self):
        if self.path == "/":
            # 返回包含交互按钮的页面
            content = f'''
            <html>
            <head><title>Interactive Page</title></head>
            <body>
                <h1>Interactive Dashboard</h1>
                <div id="initial-data">Initial: Loaded at page start</div>
                <div id="dynamic-data">Click button to load</div>
                <button id="load-btn">Load Data</button>
                
                <script>
                    // 页面加载时的请求（在快照中不会重复执行）
                    fetch('/api/initial-data')
                        .then(response => response.text())
                        .then(data => {{
                            document.getElementById('initial-data').innerHTML = data;
                        }});
                    
                    // 用户交互触发的请求（应该正常执行）
                    document.getElementById('load-btn').onclick = function() {{
                        fetch('/api/dynamic-data')
                            .then(response => response.text())
                            .then(data => {{
                                document.getElementById('dynamic-data').innerHTML = data;
                            }});
                    }};
                </script>
            </body>
            </html>
            '''.encode()
            
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            RequestCounterHandler.request_counts[self.path] += 1
            
        elif self.path == "/api/initial-data":
            content = "Initial data loaded"
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content.encode())
            RequestCounterHandler.request_counts[self.path] += 1
            
        elif self.path == "/api/dynamic-data":
            content = f"Dynamic data - Request #{RequestCounterHandler.request_counts[self.path] + 1}"
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content.encode())
            RequestCounterHandler.request_counts[self.path] += 1
        else:
            original_do_get(self)
    
    RequestCounterHandler.do_GET = interactive_page_handler
    
    try:
        # 使用页面快照模式
        env = gym.make(
            "browsergym/openended",
            task_kwargs={"start_url": server_url},
            enable_context_cache=True,
            context_cache_kwargs={
                "redis_url": TEST_REDIS_URL,
                "enable_page_snapshot": True,
                "snapshot_wait_time": 2000,
                "cacheable_resource_types": [
                    "document", "stylesheet", "script", "image", "font", "xhr", "fetch"
                ],
            },
        )
        
        # 第一次访问 - 创建快照
        print("First visit - creating snapshot...")
        env.reset()
        time.sleep(2)  # 等待页面完全加载
        
        initial_page_requests = RequestCounterHandler.request_counts["/"]
        initial_api_requests = RequestCounterHandler.request_counts["/api/initial-data"]
        
        print(f"After first visit: page={initial_page_requests}, api={initial_api_requests}")
        
        # 第二次访问 - 使用快照
        print("Second visit - using snapshot...")
        env.reset()
        time.sleep(1)  # 快照加载很快
        
        # 页面应该从快照加载（计数不增加）
        assert RequestCounterHandler.request_counts["/"] == initial_page_requests
        # 初始API也不应该重复调用（因为数据在快照中）
        assert RequestCounterHandler.request_counts["/api/initial-data"] == initial_api_requests
        # 模拟用户点击按钮（这应该触发新的API请求）
        print("Simulating user button click...")
        obs, reward, done, truncated, info = env.step("click('7')") # TODO 这是 som 打标后的结果, 需要从下面这行结果才能看到具体的 bid 标记。 
        print(flatten_axtree_to_str(obs.get("axtree_object", {}))) 
        time.sleep(1)  # 等待请求完成
        
        # 动态API应该被调用（新的用户交互）
        assert RequestCounterHandler.request_counts["/api/dynamic-data"] >= 1
        
        print("✅ 页面快照模式下用户交互正常工作！")
        env.close()
        
    finally:
        RequestCounterHandler.do_GET = original_do_get
