import copy
import logging
import re
import time
from abc import ABC
from pathlib import Path
from typing import Literal, Optional
import hashlib
import json
import base64

try:
    import redis
except ImportError:
    redis = None

import gymnasium as gym
import numpy as np
import playwright.sync_api

from . import _get_global_playwright
from .action.base import execute_python_code
from .action.highlevel import HighLevelActionSet
from .chat import Chat
from .constants import BROWSERGYM_ID_ATTRIBUTE, EXTRACT_OBS_MAX_TRIES, TEXT_MAX_LENGTH
from .observation import (
    MarkingError,
    _post_extract,
    _pre_extract,
    extract_dom_extra_properties,
    extract_dom_snapshot,
    extract_focused_element_bid,
    extract_merged_axtree,
    extract_screenshot,
)
from .spaces import AnyBox, AnyDict, Unicode
from .task import AbstractBrowserTask

logger = logging.getLogger(__name__)


def _try_to_extract_legacy_goal(goal: list):
    legacy_goal_strings = []
    for message in goal:
        if message["type"] == "text":
            legacy_goal_strings.append(message["text"])
        else:
            logger.debug(
                f"Message type {repr(message['type'])} present in the goal, cannot be converted to legacy text-only format."
            )
            legacy_goal_strings.append(
                'WARNING: This goal cannot be converted to a text-only goal format. Use the new goal format instead ("goal_object" field). Any agent reading this should abort immediately.'
            )
            break
    legacy_goal = "\n".join(legacy_goal_strings)

    return legacy_goal


class BrowserEnv(gym.Env, ABC):
    """The main BrowserGym class, which encapsulates instruction-following Web browsing into a Gymnasium environment."""

    # gym metadata
    metadata = {"render_modes": None}
    stealth_js_script = None

    def __init__(
        self,
        # task-related arguments
        task_entrypoint: type[AbstractBrowserTask],
        task_kwargs: dict = {},
        viewport: Optional[dict] = None,  # will override the task's viewport
        slow_mo: Optional[int] = None,  # will override the task's slow_mo
        timeout: Optional[int] = None,  # will override the task's timeout
        tags_to_mark: Literal["all", "standard_html"] = "standard_html",
        # interactive / debugging arguments
        headless: bool = True,
        wait_for_user_message: bool = False,
        terminate_on_infeasible: bool = True,
        resizeable_window: bool = False,
        record_video_dir: Optional[str] = None,
        pw_chromium_kwargs: dict = {},
        pw_context_kwargs: dict = {},
        # agent-related arguments
        action_mapping: Optional[callable] = HighLevelActionSet().to_python_code,
        enable_context_cache: bool = False,
        context_cache_kwargs: dict = {},
    ):
        """
        Instantiate a ready to use BrowserEnv gym environment.

        Args:
            task_entrypoint: a callable that returns a new task object from a seed. Used for creating a new task during `reset()`.
            task_kwargs: additional arguments passed to `task_entrypoint`.
            viewport: desired viewport size. This will override the value defined by the task, which might change its behaviour and difficulty. Should only be set for debugging/testing.
            slow_mo: desired slow_mo value for Playwright. This will override the value defined by the task, which might change its behaviour and difficulty. Should only be set for debugging/testing.
            timeout: desired timeout value for Playwright. This will override the value defined by the task, which might change its behaviour and difficulty. Should only be set for debugging/testing.
            tags_to_mark: which HTML tags should be marked by BrowserGym and receive a bid. Value "all" will mark every element in the page, while "standard_html" (default) will only mark standard html tags.
            headless: whether the browser should run in headless mode or not. This will affect the viewport size, which might change the behaviour and difficulty of the task. Headless mode should only be disabled for debugging/testing.
            wait_for_user_message: whether the environment should pause and wait for a user message in the chat after a new message is sent by the agent. Useful for running agents in interactive mode.
            resizeable_window: whether the browser window should be resizeable or not. This will affect the viewport size, which might change the behaviour and difficulty of the task. Should only be set for debugging/testing.
            record_video_dir: if set, indicates a directory to which viewport videos will be recorded.
            pw_chromium_kwargs: extra parameters for the playwright Browser. Should only be used for debugging/testing.
            pw_context_kwargs: extra parameters for the playwright BrowserContext. Should only be used for debugging/testing.
            action_mapping: if set, the environment will use this function to map every received action to executable Python code.
            enable_context_cache: if set, the environment will use redis to cache network requests.
            context_cache_kwargs: additional arguments for redis caching.
        """
        super().__init__()
        self.task_entrypoint = task_entrypoint
        self.task_kwargs = dict(**task_kwargs)
        self.viewport = viewport
        self.slow_mo = slow_mo
        self.timeout = timeout
        self.tags_to_mark = tags_to_mark
        self.headless = headless
        self.wait_for_user_message = wait_for_user_message
        self.terminate_on_infeasible = terminate_on_infeasible
        self.resizeable_window = resizeable_window
        self.record_video_dir = record_video_dir
        self.pw_chromium_kwargs = pw_chromium_kwargs
        self.pw_context_kwargs = pw_context_kwargs
        self.action_mapping = action_mapping
        self.enable_context_cache = enable_context_cache
        self.context_cache_kwargs = context_cache_kwargs

        # check argument values
        assert tags_to_mark in ("all", "standard_html")

        # caching
        self.redis_client = None
        self.browser_cache_hit_stats = {}
        if self.enable_context_cache:
            if redis is None:
                raise ImportError(
                    "redis package not installed. Please install it with `pip install redis`"
                )
            redis_url = self.context_cache_kwargs.get("redis_url", "redis://localhost:6379/0")
            self.redis_client = redis.from_url(
                redis_url, encoding="utf-8", decode_responses=True
            )
            self.cache_ttl = self.context_cache_kwargs.get("ttl", 3600)

        # task
        self.task = None

        # playwright
        self.browser: playwright.sync_api.Browser = None
        self.context: playwright.sync_api.BrowserContext = None
        self.page: playwright.sync_api.Page = None
        self.page_history: dict = {}

        # chat
        self.chat: Chat = None

        # observation space
        self.observation_space = gym.spaces.Dict(
            {
                "chat_messages": gym.spaces.Sequence(
                    gym.spaces.Dict(
                        {
                            "role": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                            "message": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                        }
                    )
                ),
                "goal": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                "goal_object": gym.spaces.Sequence(AnyDict()),
                "open_pages_urls": gym.spaces.Sequence(
                    Unicode(min_length=0, max_length=TEXT_MAX_LENGTH)
                ),
                "open_pages_titles": gym.spaces.Sequence(
                    Unicode(min_length=0, max_length=TEXT_MAX_LENGTH)
                ),
                "active_page_index": gym.spaces.Box(low=0, high=255, dtype=int),
                "url": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                "screenshot": AnyBox(
                    low=0,
                    high=255,
                    shape=(-1, -1, 3),
                    dtype=np.uint8,
                ),  # swapped axes (height, width, RGB)
                "dom_object": AnyDict(),
                "axtree_object": AnyDict(),
                "extra_element_properties": AnyDict(),
                "focused_element_bid": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                "last_action": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                "last_action_error": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                "elapsed_time": gym.spaces.Box(low=0, high=np.inf, dtype=float),
            }
        )

        # action space
        self.action_space = Unicode(min_length=0, max_length=TEXT_MAX_LENGTH)

        # 设置可缓存的资源类型
        self.cacheable_resource_types = self.context_cache_kwargs.get(
            'cacheable_resource_types', 
            ["document", "stylesheet", "image", "font", "script"]
        )
        
        # 页面快照功能配置
        self.enable_page_snapshot = self.context_cache_kwargs.get('enable_page_snapshot', False)
        self.snapshot_wait_time = self.context_cache_kwargs.get('snapshot_wait_time', 5000)  # ms
        if self.enable_page_snapshot:
            # 如果开启页面快照，则需要确保缓存xhr和fetch请。虽然快照模式缓存了最完整的document内容，但还是会请求动态内容。
            if "xhr" not in self.cacheable_resource_types:
                self.cacheable_resource_types.append("xhr")
            if "fetch" not in self.cacheable_resource_types:
                self.cacheable_resource_types.append("fetch")

        # 添加缓存性能统计
        self.cache_performance_stats = {
            "cache_read_times": [],  # Redis读取时间
            "snapshot_read_times": [],  # 快照读取时间
            "cache_hit_count": 0,
            "cache_miss_count": 0,
            "snapshot_hit_count": 0,
            "snapshot_miss_count": 0
        }

    def close(self):
        if self.task:
            # stop the task
            self.task.teardown()
            # close the chat
            self.chat.close()
            # close the browser context
            self.context.close()
            # close the browser
            self.browser.close()
            self.task = None
        if self.enable_context_cache:
            self.redis_client.close()
            self.browser_cache_hit_stats = {}
            self.redis_client = None

    def reset(self, seed=None, *args, **kwargs):
        super().reset(seed=seed, *args, **kwargs)
        self.np_random = None  # make sure all randomness is handled by the task

        if self.task:
            self.task.teardown()
            self.context.close()
            self.chat.close()
            self.browser.close()

        # create a new task
        self.task = self.task_entrypoint(seed=seed, **self.task_kwargs)

        def override_property(task, env, property):
            """Extract property value from env if not None, otherwise from task."""
            env_value = getattr(env, property)
            task_value = getattr(task, property)
            if env_value is None:
                return task_value
            else:
                logger.warning(
                    f"Overriding the task's {property} parameter ({repr(task_value)} => {repr(env_value)}). This might change the task's behaviour and difficulty."
                )
                return env_value

        # fetch task's desired parameters for browser setup
        viewport = override_property(self.task, self, "viewport")
        slow_mo = override_property(self.task, self, "slow_mo")
        timeout = override_property(self.task, self, "timeout")

        # use the global Playwright instance
        pw: playwright.sync_api.Playwright = _get_global_playwright()
        # important: change playwright's test id attribute from "data-testid" to "bid"
        pw.selectors.set_test_id_attribute(BROWSERGYM_ID_ATTRIBUTE)

        # create a new browser
        self.browser = pw.chromium.launch(
            headless=self.headless,
            slow_mo=slow_mo,
            args=(
                [f"--window-size={viewport['width']},{viewport['height']}"]
                if self.resizeable_window
                else None
            ),
            # will raise an Exception if above args are overriden
            **self.pw_chromium_kwargs,
        )

        if self.stealth_js_script is None:
            try:
                with open(Path(__file__).parent / "javascript/stealth.min.js", "r") as f:
                    self.stealth_js_script = f.read()
            except FileNotFoundError:
                raise FileNotFoundError("stealth.min.js not found. Please check the file path.")

        # create a new browser context for pages
        self.context = self.browser.new_context(
            no_viewport=True if self.resizeable_window else None,
            viewport=viewport,
            record_video_dir=(
                Path(self.record_video_dir) / "task_video" if self.record_video_dir else None
            ),
            record_video_size=viewport,
            ignore_https_errors=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36", # 示例UA
            # will raise an Exception if above args are overriden
            **self.pw_context_kwargs,
        )

        if self.enable_context_cache:
            self.context.route("**/*", self._handle_route)

        self.context.add_init_script(self.stealth_js_script)

        # set default timeout
        self.context.set_default_timeout(timeout)

        # hack: keep track of the active page with a javascript callback
        # there is no concept of active page in playwright
        # https://github.com/microsoft/playwright/issues/2603
        self.context.expose_binding(
            "browsergym_page_activated", lambda source: self._activate_page_from_js(source["page"])
        )
        self.context.add_init_script(
            r"""
window.browsergym_page_activated();
window.addEventListener("focus", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("focusin", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("load", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("pageshow", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("mousemove", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("mouseup", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("mousedown", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("wheel", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("keyup", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("keydown", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("input", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("touchstart", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("touchend", () => {window.browsergym_page_activated();}, {capture: true});
document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
        window.browsergym_page_activated();
    }
}, {capture: true});
"""
        )

        # create the chat
        self.chat = Chat(
            headless=self.headless,
            chat_size=(500, max(viewport["height"], 800)),
            record_video_dir=self.record_video_dir,
        )

        # create a new page
        self.page = self.context.new_page()
        recording_start_time = time.time()

        # setup the task
        task_goal, task_info = self.task.setup(page=self.page)

        # process the task goal

        # no goal specified
        if task_goal is None:
            self.goal_object = []
        # convert text-only goal (legacy) to new format
        elif isinstance(task_goal, str):
            self.goal_object = [{"type": "text", "text": task_goal}]
        # new format goal with multiple texts and images (OpenAI style)
        elif isinstance(task_goal, list):
            self.goal_object = task_goal
        else:
            raise ValueError(f"task_goal should be of type str or list, got {task_goal.__class__}")

        # initialize the chat
        self.chat.add_message(
            role="assistant",
            msg="Hi! I am your UI assistant, I can perform web tasks for you. What can I help you with?",
        )

        # send task goal (if any) to the chat
        for message in self.goal_object:
            match message["type"]:
                case "text":
                    self.chat.add_message(role="user", msg=message["text"])
                case "image_url":
                    image_src = message["image_url"]
                    if isinstance(image_src, dict):
                        image_src = image_src["url"]
                    self.chat.add_message(role="user_image", msg=image_src)
                case _:
                    raise ValueError(
                        f"Unknown message type {repr(message['type'])} in the task goal."
                    )

        self._wait_dom_loaded()

        # after the task's setup, the active page might have changed
        # perform a safety check
        self._active_page_check()

        # 如果启用了页面快照，捕获当前页面状态
        if self.enable_page_snapshot and self.enable_context_cache:
            self._capture_page_snapshot(self.page, self.page.url)

        # init start time
        self.start_time = time.time()

        # no action yet
        self.last_action = ""
        self.last_action_error = ""
        self.infeasible_message_received = False

        # if asked, wait for user message
        self._wait_for_user_message()

        # extract obs and info from environment
        obs = self._get_obs()

        info = {}
        info["task_info"] = task_info

        if self.enable_context_cache and self.redis_client:
            # 记录 url 是否被缓存，没命中的话，记录 url 对应的次数为 0，命中就+1
            info["browser_cache_hit_stats"] = self.browser_cache_hit_stats
            
            # 添加缓存性能统计
            cache_stats = self.cache_performance_stats
            info["cache_performance"] = {
                "total_cache_reads": len(cache_stats["cache_read_times"]),
                "avg_cache_read_time": sum(cache_stats["cache_read_times"]) / len(cache_stats["cache_read_times"]) if cache_stats["cache_read_times"] else 0,
                "max_cache_read_time": max(cache_stats["cache_read_times"]) if cache_stats["cache_read_times"] else 0,
                "cache_hit_rate": cache_stats["cache_hit_count"] / (cache_stats["cache_hit_count"] + cache_stats["cache_miss_count"]) if (cache_stats["cache_hit_count"] + cache_stats["cache_miss_count"]) > 0 else 0,
                "total_snapshot_reads": len(cache_stats["snapshot_read_times"]),
                "avg_snapshot_read_time": sum(cache_stats["snapshot_read_times"]) / len(cache_stats["snapshot_read_times"]) if cache_stats["snapshot_read_times"] else 0,
                "max_snapshot_read_time": max(cache_stats["snapshot_read_times"]) if cache_stats["snapshot_read_times"] else 0,
                "snapshot_hit_rate": cache_stats["snapshot_hit_count"] / (cache_stats["snapshot_hit_count"] + cache_stats["snapshot_miss_count"]) if (cache_stats["snapshot_hit_count"] + cache_stats["snapshot_miss_count"]) > 0 else 0,
            }

        # TODO this is a bit hacky, find a better solution to record videos
        if self.record_video_dir:
            info["recording_start_time"] = recording_start_time
            info["recording_file"] = str(self.page.video.path())
            info["chat"] = {
                "recording_start_time": self.chat.recording_start_time,
                "recording_file": str(self.chat.page.video.path()),
            }

        return obs, info

    def step(self, action: str) -> tuple:

        self.last_action = action

        info = {}
        info["action_exec_start"] = time.time()
        info["action_exec_timeout"] = 0

        def send_message_to_user(text: str):
            if not isinstance(text, str):
                raise ValueError(f"Forbidden value: {text} is not a string")
            self.chat.add_message(role="assistant", msg=text)

        def report_infeasible_instructions(reason: str):
            if not isinstance(reason, str):
                raise ValueError(f"Forbidden value: {reason} is not a string")
            self.chat.add_message(role="infeasible", msg=reason)
            self.infeasible_message_received = True

        # try to execute the action
        logger.debug(f"Executing action")
        try:
            if self.action_mapping:
                code = self.action_mapping(action)
            else:
                code = action
            execute_python_code(
                code,
                self.page,
                send_message_to_user=send_message_to_user,
                report_infeasible_instructions=report_infeasible_instructions,
            )
            self.last_action_error = ""
        except Exception as e:
            self.last_action_error = f"{type(e).__name__}: {e}"
            match = re.match("TimeoutError: Timeout ([0-9]+)ms exceeded.", self.last_action_error)
            if match:
                info["action_exec_timeout"] = float(match.groups()[0]) / 1000  # ms to sec
        logger.debug(f"Action executed")
        info["action_exec_stop"] = time.time()

        # wait a bit (for the JavaScript callback to set the active page)
        time.sleep(0.5)  # wait for JS events to be fired (half a second)
        self.context.cookies()  # trigger all waiting Playwright callbacks on the stack (hack, see https://playwright.dev/java/docs/multithreading)

        # wait for the network to idle before extracting the observation, reward etc.
        self._wait_dom_loaded()

        # after the action is executed, the active page might have changed
        # perform a safety check
        self._active_page_check()
        logger.debug(f"Active page checked")

        # if asked, wait for user message
        self._wait_for_user_message()
        logger.debug(f"User message done")

        logger.debug(f"Initiating task validation")
        # extract reward, done, user_message, info (task-specific)
        reward, done, user_message, task_info = self._task_validate()
        info["task_info"] = task_info
        logger.debug(f"Task validation done")

        # add any user message sent by the task to the chat
        if user_message:
            self.chat.add_message(role="user", msg=user_message)

        # extract observation (generic)
        obs = self._get_obs()
        logger.debug(f"Observation extracted")

        # new step API wants a 5-tuple (gymnasium)
        terminated = done or (
            self.terminate_on_infeasible and self.infeasible_message_received
        )  # task or agent can terminate the episode
        truncated = False

        if self.enable_context_cache and self.redis_client:
            # 记录 url 是否被缓存，没命中的话，记录 url 对应的次数为 0，命中就+1
            info["browser_cache_hit_stats"] = self.browser_cache_hit_stats
            
            # 添加缓存性能统计
            cache_stats = self.cache_performance_stats
            info["cache_performance"] = {
                "total_cache_reads": len(cache_stats["cache_read_times"]),
                "avg_cache_read_time": sum(cache_stats["cache_read_times"]) / len(cache_stats["cache_read_times"]) if cache_stats["cache_read_times"] else 0,
                "max_cache_read_time": max(cache_stats["cache_read_times"]) if cache_stats["cache_read_times"] else 0,
                "cache_hit_rate": cache_stats["cache_hit_count"] / (cache_stats["cache_hit_count"] + cache_stats["cache_miss_count"]) if (cache_stats["cache_hit_count"] + cache_stats["cache_miss_count"]) > 0 else 0,
                "total_snapshot_reads": len(cache_stats["snapshot_read_times"]),
                "avg_snapshot_read_time": sum(cache_stats["snapshot_read_times"]) / len(cache_stats["snapshot_read_times"]) if cache_stats["snapshot_read_times"] else 0,
                "max_snapshot_read_time": max(cache_stats["snapshot_read_times"]) if cache_stats["snapshot_read_times"] else 0,
                "snapshot_hit_rate": cache_stats["snapshot_hit_count"] / (cache_stats["snapshot_hit_count"] + cache_stats["snapshot_miss_count"]) if (cache_stats["snapshot_hit_count"] + cache_stats["snapshot_miss_count"]) > 0 else 0,
            }

        return obs, reward, terminated, truncated, info

    def _task_validate(self):
        # back-up these in case validate() navigates pages and messes the history
        prev_active_page = self.page
        prev_page_history = self.page_history.copy()
        # call validate
        reward, done, user_message, info = self.task.validate(self.page, self.chat.messages)

        # safety fix, in case validate() did mess up the active page and/or page history
        if prev_active_page != self.page or prev_page_history != self.page_history:
            logger.info(
                "The active page and / or page history has changed during task.validate(). A recovery fix will be applied."
            )
            self.page = prev_active_page
            self.page_history = prev_page_history

        return reward, done, user_message, info

    def _wait_for_user_message(self):
        # if last message is from the assistant, wait for a user message to continue
        # TODO: be smarter about when to wait for a user message (different action from the assistant?)
        if self.chat.messages[-1]["role"] == "assistant" and self.wait_for_user_message:
            self.chat.wait_for_user_message()

    def _wait_dom_loaded(self):
        for page in self.context.pages:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=3000)
            except playwright.sync_api.Error:
                pass
            for frame in page.frames:
                try:
                    frame.wait_for_load_state("domcontentloaded", timeout=3000)
                except playwright.sync_api.Error:
                    pass

    def _activate_page_from_js(self, page: playwright.sync_api.Page):
        logger.debug(f"_activate_page_from_js(page) called, page={str(page)}")
        if not page.context == self.context:
            raise RuntimeError(
                f"Unexpected: activating a page that belongs to a different browser context ({page})."
            )

        # add the activated page to the page history (or move it to last which is the most recent)
        if page in self.page_history:
            self.page_history[page] = self.page_history.pop(
                page
            )  # move page to the end of dictionnary
        else:
            self.page_history[page] = None  # add page to the end of dictionnary

        self.page = page

    def _active_page_check(self):
        # make sure there is always a page open
        # if all pages have been closed, create a new page
        if len(self.context.pages) == 0:
            logger.warning(f"All pages are closed, opening a new page.")
            self.page = self.context.new_page()

        # if the active page got closed, get the last active page from the history
        while self.page_history and (self.page.is_closed() or self.page not in self.context.pages):
            self.page_history.pop(self.page)  # remove active page from history
            self.page = list(self.page_history.keys())[
                -1
            ]  # set last active page as the active page (most recent)

        # active page should share the same browser context with the environment
        if self.page not in self.context.pages:
            raise RuntimeError(
                f"Unexpected: active page is not part of the browser context's open pages ({self.page})."
            )

        # active page should not be closed
        if self.page.is_closed():
            raise RuntimeError(f"Unexpected: active page has been closed ({self.page}).")

    def _handle_route(self, route: playwright.sync_api.Route):
        # 如果启用了页面快照，使用快照缓存策略
        if self.enable_page_snapshot:
            self._handle_route_with_snapshot(route)
        else:
            self._handle_route_with_http_cache(route)

    def _handle_route_with_snapshot(self, route: playwright.sync_api.Route):
        """使用页面快照的路由处理"""
        request = route.request
        
        # 只对主文档使用快照，其他资源正常处理
        if request.resource_type == "document":
            snapshot_key = self._generate_snapshot_key(request.url)
            
            # 检查是否有页面快照
            snapshot = self._get_page_snapshot(snapshot_key)
            if snapshot:
                logger.debug(f"Serving page from snapshot: {request.url}")
                self._serve_page_snapshot(route, snapshot)
                return
        
        # 对于非文档请求或无快照情况，使用普通HTTP缓存
        self._handle_route_with_http_cache(route)

    def _handle_route_with_http_cache(self, route: playwright.sync_api.Route):
        """原有的HTTP缓存路由处理"""
        request = route.request
        # Only cache GET requests for specific resource types
        if request.method.upper() != "GET" or request.resource_type not in self.cacheable_resource_types:
            route.continue_()
            return

        cache_key = self._generate_cache_key(request.url)
        lock_key = f"lock:{cache_key}"

        # 检查缓存
        cached_response = self._get_from_cache(cache_key)
        if cached_response and self._validate_cached_response(cached_response):
            self._serve_from_cache(route, cached_response)
            # 记录缓存命中统计
            url_host = request.url.split('/')[2] if '/' in request.url else request.url
            self.browser_cache_hit_stats[url_host] = self.browser_cache_hit_stats.get(url_host, 0) + 1
            return

        # 获取分布式锁
        if self._acquire_lock(lock_key):
            try:
                # 双重检查，避免重复请求
                cached_response = self._get_from_cache(cache_key)
                if cached_response and self._validate_cached_response(cached_response):
                    self._serve_from_cache(route, cached_response)
                    return

                # 尝试获取并缓存响应
                success = self._fetch_and_cache_response(route, cache_key)
                if not success:
                    logger.warning(f"Failed to fetch and cache {request.url}, falling back to direct request")
                    route.continue_()
                    
            finally:
                self._release_lock(lock_key)
        else:
            # 如果获取锁失败，等待一小段时间后再次检查缓存
            time.sleep(0.1)
            cached_response = self._get_from_cache(cache_key)
            if cached_response and self._validate_cached_response(cached_response):
                self._serve_from_cache(route, cached_response)
            else:
                # 如果仍然没有缓存，直接请求
                route.continue_()

    def _generate_snapshot_key(self, url: str) -> str:
        """生成页面快照的缓存键"""
        return f"snapshot:{hashlib.md5(url.encode('utf-8')).hexdigest()}"

    def _capture_page_snapshot(self, page: playwright.sync_api.Page, url: str):
        """捕获页面快照"""
        try:
            # 等待页面完全加载（包括动态内容）
            logger.debug(f"Waiting for page to load completely: {url}")
            page.wait_for_load_state("networkidle", timeout=self.snapshot_wait_time)
            
            # 等待额外时间确保所有异步操作完成
            time.sleep(1)
            
            # 获取完整的页面内容
            html_content = page.content()
            
            # 获取页面的所有状态信息
            snapshot_data = {
                "url": url,
                "html_content": html_content,
                "title": page.title(),
                "viewport": page.viewport_size,
                "timestamp": int(time.time() * 1000),
                "user_agent": page.evaluate("navigator.userAgent"),
                # 可以添加更多状态信息
            }
            
            # 保存快照
            snapshot_key = self._generate_snapshot_key(url)
            self._save_page_snapshot(snapshot_key, snapshot_data)
            
            logger.debug(f"Page snapshot captured for: {url}")
            
        except Exception as e:
            logger.warning(f"Failed to capture page snapshot for {url}: {e}")

    def _save_page_snapshot(self, snapshot_key: str, snapshot_data: dict):
        """保存页面快照到Redis"""
        try:
            self.redis_client.set(
                snapshot_key, 
                json.dumps(snapshot_data), 
                ex=self.cache_ttl
            )
        except Exception as e:
            logger.error(f"Failed to save page snapshot: {e}")

    def _get_page_snapshot(self, snapshot_key: str) -> dict | None:
        """从Redis获取页面快照"""
        start_time = time.time()
        try:
            data = self.redis_client.get(snapshot_key)
            read_time = time.time() - start_time
            self.cache_performance_stats["snapshot_read_times"].append(read_time)
            
            if data:
                snapshot = json.loads(data)
                # 检查快照是否过期
                if time.time() * 1000 - snapshot["timestamp"] < self.cache_ttl * 1000:
                    self.cache_performance_stats["snapshot_hit_count"] += 1
                    logger.debug(f"Snapshot cache hit for key {snapshot_key[:8]}... (read time: {read_time:.4f}s)")
                    return snapshot
                self.redis_client.delete(snapshot_key)
            
            self.cache_performance_stats["snapshot_miss_count"] += 1
            logger.debug(f"Snapshot cache miss for key {snapshot_key[:8]}... (read time: {read_time:.4f}s)")
            return None
        except Exception as e:
            read_time = time.time() - start_time
            self.cache_performance_stats["snapshot_read_times"].append(read_time)
            self.cache_performance_stats["snapshot_miss_count"] += 1
            logger.error(f"Failed to get page snapshot (time: {read_time:.4f}s): {e}")
            return None

    def _serve_page_snapshot(self, route: playwright.sync_api.Route, snapshot: dict):
        """使用页面快照响应请求"""
        try:
            html_content = snapshot["html_content"]
            
            # 构造响应头
            headers = {
                "content-type": "text/html; charset=utf-8",
                "cache-control": "no-cache",
                "x-snapshot-served": "true",
                "x-snapshot-timestamp": str(snapshot["timestamp"])
            }
            
            route.fulfill(
                status=200,
                headers=headers,
                body=html_content.encode('utf-8')
            )
            
            # 记录快照命中统计
            url_host = route.request.url.split('/')[2] if '/' in route.request.url else route.request.url
            self.browser_cache_hit_stats[url_host] = self.browser_cache_hit_stats.get(url_host, 0) + 1
            
        except Exception as e:
            logger.error(f"Failed to serve page snapshot: {e}")
            route.continue_()

    def _validate_cached_response(self, cached_response: dict) -> bool:
        """验证缓存响应的完整性"""
        try:
            # 检查必要的字段是否存在
            required_fields = ["status", "headers", "body_b64", "timestamp"]
            if not all(field in cached_response for field in required_fields):
                logger.warning("Cached response missing required fields")
                return False
            
            # 检查响应状态码
            if cached_response["status"] >= 400:
                logger.warning(f"Cached response has error status: {cached_response['status']}")
                return False
            
            # 检查body是否为有效的base64编码
            try:
                body_bytes = base64.b64decode(cached_response["body_b64"])
            except Exception as e:
                logger.warning(f"Invalid base64 in cached response: {e}")
                return False
            
            # 检查内容长度一致性
            headers = cached_response["headers"]
            if "content-length" in headers:
                expected_length = int(headers["content-length"])
                if len(body_bytes) != expected_length:
                    logger.warning(f"Content length mismatch: expected {expected_length}, got {len(body_bytes)}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating cached response: {e}")
            return False

    def _serve_from_cache(self, route: playwright.sync_api.Route, cached_response: dict):
        """从缓存中提供响应"""
        try:
            headers = cached_response["headers"].copy()
            # 移除可能导致问题的headers
            problematic_headers = ("content-security-policy", "content-length", "x-frame-options")
            for header in problematic_headers:
                headers.pop(header, None)
            
            route.fulfill(
                status=cached_response["status"],
                headers=headers,
                body=base64.b64decode(cached_response["body_b64"]),
            )
        except Exception as e:
            logger.error(f"Error serving from cache: {e}")
            route.continue_()

    def _fetch_and_cache_response(self, route: playwright.sync_api.Route, cache_key: str, max_retries: int = 2) -> bool:
        """获取响应并缓存，带重试机制"""
        for attempt in range(max_retries + 1):
            try:
                # 获取响应
                response = route.fetch()
                
                # 验证响应状态
                if response.status >= 400:
                    logger.warning(f"HTTP error {response.status} for {route.request.url}")
                    if attempt < max_retries:
                        logger.info(f"Retrying... ({attempt + 1}/{max_retries})")
                        time.sleep(0.5 * (attempt + 1))  # 指数退避
                        continue
                    else:
                        return False
                
                # 获取响应体
                body_bytes = response.body()
                
                # 验证内容完整性
                if not self._validate_response_integrity(response, body_bytes):
                    if attempt < max_retries:
                        logger.info(f"Content integrity check failed, retrying... ({attempt + 1}/{max_retries})")
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    else:
                        logger.warning(f"Content integrity check failed after {max_retries} retries")
                        return False
                
                # 准备缓存数据
                headers = response.headers.copy()
                # 移除可能影响缓存的编码头
                headers.pop("content-encoding", None)
                
                data_to_cache = {
                    "status": response.status,
                    "headers": headers,
                    "body_b64": base64.b64encode(body_bytes).decode("ascii"),
                    "url": route.request.url,  # 添加URL用于调试
                    "fetch_time": int(time.time() * 1000),  # 添加获取时间
                }
                
                # 保存到缓存
                self._save_to_cache(cache_key, data_to_cache)
                
                # 提供响应
                serve_headers = headers.copy()
                problematic_headers = ("content-security-policy", "content-length", "x-frame-options")
                for header in problematic_headers:
                    serve_headers.pop(header, None)
                
                route.fulfill(
                    status=response.status,
                    headers=serve_headers,
                    body=body_bytes
                )
                
                logger.debug(f"Successfully cached and served {route.request.url}")
                return True
                
            except playwright.sync_api.TimeoutError as e:
                logger.warning(f"Timeout fetching {route.request.url}: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying due to timeout... ({attempt + 1}/{max_retries})")
                    time.sleep(1.0 * (attempt + 1))
                    continue
                else:
                    return False
                    
            except Exception as e:
                logger.error(f"Error fetching {route.request.url} (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying due to error... ({attempt + 1}/{max_retries})")
                    time.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    return False
        
        return False

    def _validate_response_integrity(self, response: playwright.sync_api.Response, body_bytes: bytes) -> bool:
        """验证响应完整性"""
        try:
            # 检查Content-Length头部
            content_length_header = response.headers.get("content-length")
            if content_length_header:
                expected_length = int(content_length_header)
                actual_length = len(body_bytes)
                if actual_length != expected_length:
                    logger.warning(
                        f"Content length mismatch for {response.url}: "
                        f"expected {expected_length}, got {actual_length}"
                    )
                    return False
            
            # 对于HTML文档，检查是否有基本的HTML结构
            if response.headers.get("content-type", "").startswith("text/html"):
                body_text = body_bytes.decode("utf-8", errors="ignore").lower()
                if not (body_text.strip().startswith("<!doctype") or body_text.strip().startswith("<html")):
                    logger.warning(f"HTML document appears incomplete for {response.url}")
                    return False
            
            # 对于JSON，检查是否是有效的JSON
            elif response.headers.get("content-type", "").startswith("application/json"):
                try:
                    json.loads(body_bytes.decode("utf-8"))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON response for {response.url}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating response integrity: {e}")
            return True  # 如果验证失败，默认认为是有效的，避免阻止正常流程

    def _generate_cache_key(self, url: str) -> str:
        """Generate MD5 hash based on URL as cache key"""
        return hashlib.md5(url.encode("utf-8")).hexdigest()

    def _save_to_cache(self, cache_key: str, data: dict):
        """Save data to Redis cache"""
        data["timestamp"] = int(time.time() * 1000)
        self.redis_client.set(cache_key, json.dumps(data), ex=self.cache_ttl)

    def _get_from_cache(self, cache_key: str) -> dict | None:
        """Read data from Redis cache"""
        start_time = time.time()
        try:
            data = self.redis_client.get(cache_key)
            read_time = time.time() - start_time
            self.cache_performance_stats["cache_read_times"].append(read_time)
            
            if data:
                parsed = json.loads(data)
                # cache_ttl is in seconds, timestamp is in ms
                if time.time() * 1000 - parsed["timestamp"] < self.cache_ttl * 1000:
                    self.cache_performance_stats["cache_hit_count"] += 1
                    logger.debug(f"Cache hit for key {cache_key[:8]}... (read time: {read_time:.4f}s)")
                    return parsed
                self.redis_client.delete(cache_key)  # Delete expired cache
            
            self.cache_performance_stats["cache_miss_count"] += 1
            logger.debug(f"Cache miss for key {cache_key[:8]}... (read time: {read_time:.4f}s)")
            return None
        except (json.JSONDecodeError, TypeError) as e:
            read_time = time.time() - start_time
            self.cache_performance_stats["cache_read_times"].append(read_time)
            self.cache_performance_stats["cache_miss_count"] += 1
            logger.debug(f"Cache read error for key {cache_key[:8]}... (time: {read_time:.4f}s): {e}")
            return None

    def _acquire_lock(self, lock_key: str, timeout: int = 10) -> bool:
        """Acquire Redis distributed lock"""
        return self.redis_client.set(lock_key, "locked", ex=timeout, nx=True)

    def _release_lock(self, lock_key: str):
        """Release Redis distributed lock"""
        self.redis_client.delete(lock_key)

    def _get_obs(self):

        for retries_left in reversed(range(EXTRACT_OBS_MAX_TRIES)):
            try:
                # pre-extraction, mark dom elements (set bid, set dynamic attributes like value and checked)
                _pre_extract(self.page, self.tags_to_mark)

                dom = extract_dom_snapshot(self.page)
                axtree = extract_merged_axtree(self.page)
                focused_element_bid = extract_focused_element_bid(self.page)
                extra_properties = extract_dom_extra_properties(dom)
            except (playwright.sync_api.Error, MarkingError) as e:
                err_msg = str(e)
                # try to add robustness to async events (detached / deleted frames)
                if retries_left > 0 and (
                    "Frame was detached" in err_msg
                    or "Frame with the given frameId is not found" in err_msg
                    or "Execution context was destroyed" in err_msg
                    or "Frame has been detached" in err_msg
                    or "Cannot mark a child frame without a bid" in err_msg
                ):
                    logger.warning(
                        f"An error occured while extracting the dom and axtree. Retrying ({retries_left}/{EXTRACT_OBS_MAX_TRIES} tries left).\n{repr(e)}"
                    )
                    # post-extract cleanup (ARIA attributes)
                    _post_extract(self.page)
                    time.sleep(0.5)
                    continue
                else:
                    raise e
            break

        # post-extraction cleanup of temporary info in dom
        _post_extract(self.page)

        # obs is generic to all tasks
        obs = {
            "chat_messages": copy.deepcopy(self.chat.messages),
            "goal": _try_to_extract_legacy_goal(self.goal_object),  # legacy goal, deprecated
            "goal_object": self.goal_object,  # new goal format, list of messages openai style
            "open_pages_urls": [page.url for page in self.context.pages],
            "open_pages_titles": [page.title() for page in self.context.pages],
            "active_page_index": np.asarray([self.context.pages.index(self.page)]),
            "url": self.page.url,  # redundant with "open_pages_urls" and "active_page_index"
            "screenshot": extract_screenshot(self.page),
            "dom_object": dom,
            "axtree_object": axtree,
            "extra_element_properties": extra_properties,
            "focused_element_bid": focused_element_bid,
            "last_action": self.last_action,
            "last_action_error": self.last_action_error,
            "elapsed_time": np.asarray([time.time() - self.start_time]),
        }

        return obs
