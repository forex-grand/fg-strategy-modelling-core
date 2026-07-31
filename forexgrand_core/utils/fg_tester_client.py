"""Optional fg-tester client stub."""

from __future__ import annotations

import json
import logging
from urllib import request

from forexgrand_core.settings import Settings

LOGGER = logging.getLogger(__name__)


def request_live_test(model_gcs_path: str, symbol: str, config: Settings) -> dict:
    """
    Request a live-test run from fg-tester service.

    This is optional and should never block training/deployment flow.
    """
    pass
    # payload = json.dumps({"model_gcs_path": model_gcs_path, "symbol": symbol}).encode("utf-8")
    # req = request.Request(
    #     config.fg_tester_api_url,
    #     data=payload,
    #     headers={"Content-Type": "application/json"},
    #     method="POST",
    # )
    # with request.urlopen(req, timeout=10) as response:
    #     body = response.read().decode("utf-8")
    # parsed = json.loads(body) if body else {}
    # LOGGER.info("fg-tester response for %s: %s", symbol, parsed)
    # return parsed

