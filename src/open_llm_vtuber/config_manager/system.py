# config_manager/system.py
from typing import ClassVar, Dict

from pydantic import Field, model_validator

from .i18n import Description, I18nMixin


class VideoAnalysisConfig(I18nMixin):
    """Configuration for post-meeting video analysis using a Visual LLM."""

    enabled: bool = Field(False, alias="enabled")
    base_url: str = Field(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        alias="base_url",
    )
    api_key: str = Field("", alias="api_key")
    model: str = Field("gemini-2.0-flash", alias="model")
    max_frames: int = Field(8, alias="max_frames")
    temperature: float = Field(0.7, alias="temperature")
    analysis_prompt: str = Field("", alias="analysis_prompt")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable post-meeting video analysis for conversation feedback",
            zh="启用会议后视频分析以获取对话反馈",
        ),
        "base_url": Description(
            en="Base URL for the Visual LLM API (must support vision/image input)",
            zh="视觉 LLM API 的基础 URL（必须支持视觉/图像输入）",
        ),
        "api_key": Description(
            en="API key for the Visual LLM", zh="视觉 LLM 的 API 密钥"
        ),
        "model": Description(
            en="Model name (must support vision, e.g. gpt-4o, gemini-2.0-flash)",
            zh="模型名称（必须支持视觉，例如 gpt-4o, gemini-2.0-flash）",
        ),
        "max_frames": Description(
            en="Maximum number of frames to extract from the video for analysis",
            zh="从视频中提取用于分析的最大帧数",
        ),
        "temperature": Description(
            en="Sampling temperature for the analysis LLM (0-2)",
            zh="分析 LLM 的采样温度 (0-2)",
        ),
        "analysis_prompt": Description(
            en="Custom system prompt for video analysis. Leave empty to use default IELTS-style prompt",
            zh="自定义视频分析的系统提示词。留空使用默认的雅思风格提示词",
        ),
    }


class SystemConfig(I18nMixin):
    """System configuration settings."""

    conf_version: str = Field(..., alias="conf_version")
    host: str = Field(..., alias="host")
    port: int = Field(..., alias="port")
    config_alts_dir: str = Field(..., alias="config_alts_dir")
    tool_prompts: Dict[str, str] = Field(..., alias="tool_prompts")
    enable_proxy: bool = Field(False, alias="enable_proxy")
    enable_recording: bool = Field(False, alias="enable_recording")
    recording_output_dir: str = Field("recordings", alias="recording_output_dir")
    video_analysis: VideoAnalysisConfig = Field(
        default_factory=VideoAnalysisConfig, alias="video_analysis"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conf_version": Description(en="Configuration version", zh="配置文件版本"),
        "host": Description(en="Server host address", zh="服务器主机地址"),
        "port": Description(en="Server port number", zh="服务器端口号"),
        "config_alts_dir": Description(
            en="Directory for alternative configurations", zh="备用配置目录"
        ),
        "tool_prompts": Description(
            en="Tool prompts to be inserted into persona prompt",
            zh="要插入到角色提示词中的工具提示词",
        ),
        "enable_proxy": Description(
            en="Enable proxy mode for multiple clients",
            zh="启用代理模式以支持多个客户端使用一个 ws 连接",
        ),
        "enable_recording": Description(
            en="Enable audio recording of conversations",
            zh="启用对话录音功能",
        ),
        "recording_output_dir": Description(
            en="Directory for saving audio recordings",
            zh="录音文件保存目录",
        ),
        "video_analysis": Description(
            en="Configuration for post-meeting video analysis feedback",
            zh="会议后视频分析反馈的配置",
        ),
    }

    @model_validator(mode="after")
    def check_port(cls, values):
        port = values.port
        if port < 0 or port > 65535:
            raise ValueError("Port must be between 0 and 65535")
        return values
