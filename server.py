
import os
from urllib.parse import urlparse

import requests
from fastmcp import FastMCP

mcp = FastMCP("Music Analysis")


@mcp.tool()
def test_audio_download(audio_url: str) -> dict:
    """测试音频下载，返回HTTP状态、文件类型和文件大小，不进行音频分析。"""

    # 仅允许 HTTPS 的 TempFile 链接
    parsed = urlparse(audio_url)

    if (
        parsed.scheme != "https"
        or parsed.hostname not in ["tempfile.org", "www.tempfile.org"]
    ):
        return {
            "success": False,
            "error": "目前只允许测试 tempfile.org 的 HTTPS 链接"
        }

    max_size = 25 * 1024 * 1024
    downloaded = 0

    try:
        with requests.get(
            audio_url,
            stream=True,
            timeout=(15, 45),
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"}
        ) as response:

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", "unknown"
            )

            content_length = response.headers.get("Content-Length")

            if content_length:
                if int(content_length) > max_size:
                    return {
                        "success": False,
                        "error": "文件超过25MB测试上限",
                        "content_length": content_length
                    }

            if "text/html" in content_type.lower():
                return {
                    "success": False,
                    "error": "下载到的是网页，不是音频文件",
                    "http_status": response.status_code,
                    "content_type": content_type,
                    "final_url": response.url
                }

            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue

                downloaded += len(chunk)

                if downloaded > max_size:
                    return {
                        "success": False,
                        "error": "文件超过25MB测试上限",
                        "downloaded_bytes": downloaded
                    }

            return {
                "success": downloaded > 0,
                "http_status": response.status_code,
                "content_type": content_type,
                "downloaded_bytes": downloaded,
                "downloaded_MB": round(downloaded / 1024 / 1024, 2),
                "final_url": response.url,
                "message": "下载读取完成，未进行音频分析"
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        stateless_http=True
    )
