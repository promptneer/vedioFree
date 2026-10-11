"""AI 指挥模块测试：用假客户端打桩，不调真实 DeepSeek API。"""
import json

import pytest
import torch

from tensor_editor import TensorEditor
from tensor_editor.ai_agent import VideoAIAgent, TOOLS, AgentError


class _FakeMessage:
    def __init__(self, tool_calls=None, content=None):
        self._tool_calls = tool_calls
        self._content = content

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant"}
        if self._content is not None:
            d["content"] = self._content
        if self._tool_calls is not None:
            d["tool_calls"] = self._tool_calls
        return d


class _FakeChoice:
    def __init__(self, msg):
        self.message = msg


class _FakeResp:
    def __init__(self, msg):
        self.choices = [_FakeChoice(msg)]


class _FakeCompletions:
    def __init__(self, script):
        self.script = script
        self.i = 0

    def create(self, **kwargs):
        assert "tools" in kwargs and kwargs["tools"] == TOOLS
        msg = self.script[self.i]
        self.i += 1
        return _FakeResp(msg)


class _FakeChat:
    def __init__(self, script):
        self.completions = _FakeCompletions(script)


class _FakeClient:
    def __init__(self, script):
        self.chat = _FakeChat(script)


def _tc(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _agent(script):
    frames = torch.full((10, 32, 48, 3), 100, dtype=torch.uint8)
    agent = VideoAIAgent(client=_FakeClient(script))
    agent.register_video("main", TensorEditor(frames))
    return agent


def test_tools_schema_valid():
    names = [t["function"]["name"] for t in TOOLS]
    assert names == ["cut_frames", "modify_region", "concatenate_videos",
                     "apply_color", "apply_transition", "export_video"]
    for t in TOOLS:
        fn = t["function"]
        assert t["type"] == "function"
        assert fn["parameters"]["type"] == "object"
        assert set(fn["parameters"]["required"]) <= set(fn["parameters"]["properties"])


def test_agent_loop_apply_color_and_cut():
    script = [
        _FakeMessage(tool_calls=[_tc("c1", "apply_color",
                                     {"video_id": "main", "brightness": 2.0})]),
        _FakeMessage(tool_calls=[_tc("c2", "cut_frames",
                                     {"video_id": "main", "start": 0, "end": 4})]),
        _FakeMessage(content="done"),
    ]
    agent = _agent(script)
    result = agent.run("调亮并切出前4帧")
    assert result == "done"
    assert agent.get_video("main").frames.max().item() == 200
    assert agent.get_video("main_cut1").num_frames == 4


def test_agent_unknown_video_id_reports_error():
    script = [
        _FakeMessage(tool_calls=[_tc("c1", "apply_color", {"video_id": "nope"})]),
        _FakeMessage(content="ok"),
    ]
    agent = _agent(script)
    agent.run("x")
    tool_msg = agent.messages[-2]
    assert tool_msg["role"] == "tool"
    payload = json.loads(tool_msg["content"])
    assert payload["ok"] is False and "未知 video_id" in payload["error"]


def test_agent_requires_key_without_client():
    import os

    os.environ.pop("DEEPSEEK_API_KEY", None)
    with pytest.raises(AgentError):
        VideoAIAgent()


def test_provider_registry_complete():
    from tensor_editor.providers import PROVIDERS, list_providers

    assert set(PROVIDERS) == {"deepseek", "minimax", "qwen", "kimi", "doubao"}
    for name, info in PROVIDERS.items():
        assert info["label"] and info["base_url"] and info["env_key"], name
    assert list_providers()["qwen"] == "通义千问"


def test_unknown_provider_rejected():
    with pytest.raises(AgentError) as exc:
        VideoAIAgent(provider="notexist", client=_FakeClient([]))
    assert "notexist" in str(exc.value)


def test_doubao_requires_explicit_model():
    with pytest.raises(AgentError) as exc:
        VideoAIAgent(provider="doubao", client=_FakeClient([]))
    assert "--model" in str(exc.value)


def test_provider_defaults_and_overrides():
    agent = VideoAIAgent(client=_FakeClient([]))
    assert agent.provider == "deepseek"
    assert agent.model == "deepseek-chat"
    assert agent.base_url == "https://api.deepseek.com"

    agent2 = VideoAIAgent(provider="qwen", model="qwen-max",
                          base_url="https://x.example/v1",
                          client=_FakeClient([]))
    assert agent2.provider == "qwen"
    assert agent2.model == "qwen-max"
    assert agent2.base_url == "https://x.example/v1"

    agent3 = VideoAIAgent(provider="kimi", api_key="sk-dummy",
                          client=_FakeClient([]))
    assert agent3.model == "kimi-k2-0905-preview"


def _without_proxy_env():
    """清掉代理环境变量：某些沙箱的 no_proxy 写法会让 httpx 解析炸，
    与被测代码无关。返回清理函数（测试后恢复）。"""
    import os

    saved = {}
    for key in list(os.environ):
        if "proxy" in key.lower():
            saved[key] = os.environ.pop(key)
    return saved


def test_provider_env_key_resolution():
    import os

    saved_proxy = _without_proxy_env()
    os.environ["DASHSCOPE_API_KEY"] = "sk-test"
    try:
        agent = VideoAIAgent(provider="qwen")  # 无 client，走真实构造（不联网）
        assert agent.model == "qwen-plus"
        assert agent.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    finally:
        del os.environ["DASHSCOPE_API_KEY"]
        os.environ.update(saved_proxy)

    saved_proxy = _without_proxy_env()
    try:
        os.environ.pop("MOONSHOT_API_KEY", None)
        with pytest.raises(AgentError) as exc:
            VideoAIAgent(provider="kimi")
        assert "MOONSHOT_API_KEY" in str(exc.value)
    finally:
        os.environ.update(saved_proxy)
