from unittest.mock import Mock, patch

import pytest
import requests

from trl.experimental.async_grpo.vllm_client import VLLMClient


def _response(payload=None, status_code=200):
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = payload
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code}")
    return response


def test_control_request_failure_is_not_silently_acknowledged():
    client = VLLMClient("http://rollout.internal:8000", server_timeout=17)

    with patch("requests.post", return_value=_response(status_code=503)) as post:
        with pytest.raises(requests.HTTPError, match="HTTP 503"):
            client.pause()

    post.assert_called_once_with(
        "http://rollout.internal:8000/pause",
        params={"mode": "keep"},
        timeout=17,
    )


def test_weight_update_request_uses_explicit_timeout_and_checks_status():
    client = VLLMClient("http://rollout.internal:8000")
    response = _response()

    with patch("requests.post", return_value=response) as post:
        client.update_weights({"names": ["model.norm.weight"]}, timeout=41)

    post.assert_called_once_with(
        "http://rollout.internal:8000/update_weights",
        json={"update_info": {"names": ["model.norm.weight"]}},
        timeout=41,
    )
    response.raise_for_status.assert_called_once_with()


def test_server_introspection_request_checks_status():
    client = VLLMClient("http://rollout.internal:8000", server_timeout=23)

    with patch("requests.get", return_value=_response(status_code=401)):
        with pytest.raises(requests.HTTPError, match="HTTP 401"):
            client.get_world_size()
