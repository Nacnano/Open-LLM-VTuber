import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, File, Form, Request, Response, UploadFile, WebSocket
from loguru import logger
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from .chat_history_manager import get_history
from .proxy_handler import ProxyHandler
from .video_analyzer import analyze_video
from .websocket_handler import WebSocketHandler


def get_duration(file_path: str) -> float | None:
    """Get the duration of a media file using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error getting duration for {file_path}: {e}")
        return None


def init_client_ws_route(ws_handler: WebSocketHandler) -> APIRouter:
    """
    Create and return API routes for handling the `/client-ws` WebSocket connections.

    Args:
        ws_handler: WebSocket handler instance.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()
    # ws_handler is now passed in

    @router.websocket("/client-ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for client connections"""
        await websocket.accept()
        client_uid = str(uuid4())

        try:
            await ws_handler.handle_new_connection(websocket, client_uid)
            await ws_handler.handle_websocket_communication(websocket, client_uid)
        except WebSocketDisconnect:
            await ws_handler.handle_disconnect(client_uid)
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            await ws_handler.handle_disconnect(client_uid)
            raise

    return router


def init_proxy_route(server_url: str) -> APIRouter:
    """
    Create and return API routes for handling proxy connections.

    Args:
        server_url: The WebSocket URL of the actual server

    Returns:
        APIRouter: Configured router with proxy WebSocket endpoint
    """
    router = APIRouter()
    proxy_handler = ProxyHandler(server_url)

    @router.websocket("/proxy-ws")
    async def proxy_endpoint(websocket: WebSocket):
        """WebSocket endpoint for proxy connections"""
        try:
            await proxy_handler.handle_client_connection(websocket)
        except Exception as e:
            logger.error(f"Error in proxy connection: {e}")
            raise

    return router


def init_webtool_routes(ws_handler: WebSocketHandler) -> APIRouter:
    """
    Create and return API routes for handling web tool interactions.

    Args:
        ws_handler: WebSocket handler instance.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """
    default_context_cache = ws_handler.default_context_cache

    router = APIRouter()

    @router.get("/web-tool")
    async def web_tool_redirect():
        """Redirect /web-tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/web_tool")
    async def web_tool_redirect_alt():
        """Redirect /web_tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/live2d-models/info")
    async def get_live2d_folder_info():
        """Get information about available Live2D models"""
        live2d_dir = "live2d-models"
        if not os.path.exists(live2d_dir):
            return JSONResponse(
                {"error": "Live2D models directory not found"}, status_code=404
            )

        valid_characters = []
        supported_extensions = [".png", ".jpg", ".jpeg"]

        for entry in os.scandir(live2d_dir):
            if entry.is_dir():
                folder_name = entry.name.replace("\\", "/")
                model3_file = os.path.join(
                    live2d_dir, folder_name, f"{folder_name}.model3.json"
                ).replace("\\", "/")

                if os.path.isfile(model3_file):
                    # Find avatar file if it exists
                    avatar_file = None
                    for ext in supported_extensions:
                        avatar_path = os.path.join(
                            live2d_dir, folder_name, f"{folder_name}{ext}"
                        )
                        if os.path.isfile(avatar_path):
                            avatar_file = avatar_path.replace("\\", "/")
                            break

                    valid_characters.append(
                        {
                            "name": folder_name,
                            "avatar": avatar_file,
                            "model_path": model3_file,
                        }
                    )
        return JSONResponse(
            {
                "type": "live2d-models/info",
                "count": len(valid_characters),
                "characters": valid_characters,
            }
        )

    @router.post("/upload-video")
    async def upload_video(
        request: Request,
        file: UploadFile = File(...),
        start_timestamp: str | None = Form(None),
        end_timestamp: str | None = Form(None),
    ):
        """
        Endpoint for uploading recorded video from webcam.

        This endpoint:
        1. Saves the uploaded video (webm)
        2. Identifies the session via IP and stops/saves the audio recording
        3. Merges the video and audio into a final mp4

        Args:
            file: The video file uploaded from the client.
            start_timestamp: Unix timestamp (seconds) when video recording started.
            end_timestamp: Unix timestamp (seconds) when video recording ended.

        Returns:
            JSONResponse: Success message with saved file path or error message.
        """
        video_start_time = float(start_timestamp) if start_timestamp else None
        video_end_time = float(end_timestamp) if end_timestamp else None
        logger.info(
            f"📹 Received video file for upload: {file.filename}, "
            f"start_timestamp: {video_start_time}, "
            f"end_timestamp: {video_end_time}"
        )

        try:
            # Create recorded_videos directory if it doesn't exist
            video_dir = Path("recorded_videos")
            video_dir.mkdir(exist_ok=True)

            # Validate file type
            allowed_types = ["video/webm", "video/mp4", "video/mpeg"]
            if file.content_type not in allowed_types:
                logger.warning(
                    f"Invalid file type: {file.content_type}. Allowed: {allowed_types}"
                )
                return JSONResponse(
                    {
                        "error": f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
                    },
                    status_code=400,
                )

            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_ext = file.filename.split(".")[-1] if "." in file.filename else "webm"
            unique_id = str(uuid4())[:8]
            video_filename = f"video_{timestamp}_{unique_id}.{file_ext}"
            video_path = video_dir / video_filename

            # Read and save the video file
            contents = await file.read()
            file_size_mb = len(contents) / (1024 * 1024)

            # Validate file size (max 100MB)
            if file_size_mb > 100:
                logger.warning(f"File too large: {file_size_mb:.2f}MB")
                return JSONResponse(
                    {"error": "File too large. Maximum size is 100MB."},
                    status_code=413,
                )

            # Write video file to disk
            with open(video_path, "wb") as f:
                f.write(contents)

            logger.info(
                f"✅ Video saved successfully: {video_path} (Size: {file_size_mb:.2f}MB)"
            )

            # --- Audio Merging Logic ---
            client_ip = request.client.host
            logger.info(f"🔍 Looking for session with IP: {client_ip}")

            # Try to find matching context
            context = ws_handler.get_context_by_ip(client_ip)
            output_filename = video_filename
            output_path = str(video_path)

            # Try to save and merge audio if we have a context and timestamps
            # Even if has_audio() returns False, we can create a silent audio file
            # if we have video timestamps
            if context and context.audio_recorder:
                has_audio = context.audio_recorder.has_audio()
                has_timestamps = (
                    video_start_time is not None and video_end_time is not None
                )

                if has_audio or has_timestamps:
                    logger.info(
                        f"🎙️ Found active session. "
                        f"Has audio segments: {has_audio}, Has timestamps: {has_timestamps}"
                    )

                    audio_filename = f"audio_{timestamp}_{unique_id}.wav"
                    audio_path = video_dir / audio_filename

                    # Save audio - use trimmed version if we have a start timestamp
                    audio_saved = False
                    if video_start_time is not None:
                        logger.info(
                            f"✂️ Using video timestamps to trim audio: "
                            f"start={video_start_time}, end={video_end_time}"
                        )
                        audio_saved = (
                            await context.audio_recorder.save_recording_from_timestamp(
                                str(audio_path), video_start_time, video_end_time
                            )
                        )
                    else:
                        logger.warning(
                            "⚠️ No video start timestamp provided. "
                            "Audio/video sync may be inaccurate!"
                        )
                        if has_audio:
                            audio_saved = await context.audio_recorder.save_recording(
                                str(audio_path)
                            )

                    await context.audio_recorder.clear()

                    # Only proceed with merge if audio was actually saved
                    if audio_saved and audio_path.exists():
                        # Get durations for logging
                        video_duration = get_duration(str(video_path))
                        audio_duration = get_duration(str(audio_path))

                        if video_duration is not None and audio_duration is not None:
                            logger.info(
                                f"⏱️ Durations - Video: {video_duration:.2f}s, "
                                f"Audio: {audio_duration:.2f}s"
                            )

                        # Merge Video and Audio using ffmpeg
                        # Output file
                        merged_filename = f"recording_{timestamp}_{unique_id}.mp4"
                        merged_path = video_dir / merged_filename

                        logger.info(f"🔄 Merging video and audio to {merged_path}...")

                        try:
                            # Camera/video recording typically has ~200-500ms delay from
                            # when timestamp is captured to when first frame appears.
                            # Add audio delay to compensate for this video capture lag.
                            audio_delay_ms = 0  # milliseconds - adjust if needed

                            logger.info(
                                f"⏱️  Adding {audio_delay_ms}ms audio delay to sync with video"
                            )

                            # ffmpeg command to merge video and audio
                            # Use adelay filter to shift audio forward in time
                            command = [
                                "ffmpeg",
                                "-y",  # Overwrite output
                                "-i",
                                str(video_path),
                                "-i",
                                str(audio_path),
                                "-c:v",
                                "libx264",  # Encode video to H.264 for MP4 compatibility
                                "-preset",
                                "fast",
                                "-crf",
                                "22",
                                "-af",
                                f"adelay=delays={audio_delay_ms}:all=1",  # Delay audio to sync with video
                                "-c:a",
                                "aac",  # Encode audio to AAC
                                "-b:a",
                                "192k",
                                "-shortest",  # Finish when shortest input ends
                                str(merged_path),
                            ]

                            # Run ffmpeg
                            subprocess.run(
                                command,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                            )

                            logger.info(f"✅ Merge successful: {merged_path}")
                            output_filename = merged_filename
                            output_path = str(merged_path)

                            # Optional: Delete intermediates?
                            # Keeping them might be safer for now, or we can delete.
                            # Let's keep them as backups for now.

                        except subprocess.CalledProcessError as e:
                            logger.error(f"❌ ffmpeg merge failed: {e.stderr}")
                            # Fallback to returning just the video (already saved)
                    else:
                        if audio_saved:
                            logger.warning("⚠️ Audio was saved but file doesn't exist")
                        else:
                            logger.warning("⚠️ Audio recording save failed")
                else:
                    logger.warning(
                        "⚠️ No audio segments and no timestamps. Skipping audio merge."
                    )
            else:
                logger.warning(
                    "⚠️ No matching session/audio found for this IP. Returning video only."
                )

            return JSONResponse(
                {
                    "status": "success",
                    "message": "Video uploaded and processed successfully",
                    "filename": output_filename,
                    "path": output_path,
                    "size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2),
                }
            )

        except Exception as e:
            logger.error(f"❌ Error saving/processing video: {e}")
            return JSONResponse(
                {"error": f"Failed to save video: {str(e)}"}, status_code=500
            )

    @router.post("/asr")
    async def transcribe_audio(file: UploadFile = File(...)):
        """
        Endpoint for transcribing audio using the ASR engine
        """
        logger.info(f"Received audio file for transcription: {file.filename}")

        try:
            contents = await file.read()

            # Validate minimum file size
            if len(contents) < 44:  # Minimum WAV header size
                raise ValueError("Invalid WAV file: File too small")

            # Decode the WAV header and get actual audio data
            wav_header_size = 44  # Standard WAV header size
            audio_data = contents[wav_header_size:]

            # Validate audio data size
            if len(audio_data) % 2 != 0:
                raise ValueError("Invalid audio data: Buffer size must be even")

            # Convert to 16-bit PCM samples to float32
            try:
                audio_array = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            except ValueError as e:
                raise ValueError(
                    f"Audio format error: {str(e)}. Please ensure the file is 16-bit PCM WAV format."
                )

            # Validate audio data
            if len(audio_array) == 0:
                raise ValueError("Empty audio data")

            text = await default_context_cache.asr_engine.async_transcribe_np(
                audio_array
            )
            logger.info(f"Transcription result: {text}")
            return {"text": text}

        except ValueError as e:
            logger.error(f"Audio format error: {e}")
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=400,
                media_type="application/json",
            )
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return Response(
                content=json.dumps(
                    {"error": "Internal server error during transcription"}
                ),
                status_code=500,
                media_type="application/json",
            )

    @router.websocket("/tts-ws")
    async def tts_endpoint(websocket: WebSocket):
        """WebSocket endpoint for TTS generation"""
        await websocket.accept()
        logger.info("TTS WebSocket connection established")

        try:
            while True:
                data = await websocket.receive_json()
                text = data.get("text")
                if not text:
                    continue

                logger.info(f"Received text for TTS: {text}")

                # Split text into sentences
                sentences = [s.strip() for s in text.split(".") if s.strip()]

                try:
                    # Generate and send audio for each sentence
                    for sentence in sentences:
                        sentence = sentence + "."  # Add back the period
                        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
                        audio_path = (
                            await default_context_cache.tts_engine.async_generate_audio(
                                text=sentence, file_name_no_ext=file_name
                            )
                        )
                        logger.info(
                            f"Generated audio for sentence: {sentence} at: {audio_path}"
                        )

                        await websocket.send_json(
                            {
                                "status": "partial",
                                "audioPath": audio_path,
                                "text": sentence,
                            }
                        )

                    # Send completion signal
                    await websocket.send_json({"status": "complete"})

                except Exception as e:
                    logger.error(f"Error generating TTS: {e}")
                    await websocket.send_json({"status": "error", "message": str(e)})

        except WebSocketDisconnect:
            logger.info("TTS WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error in TTS WebSocket connection: {e}")
            await websocket.close()

    @router.post("/analyze-video")
    async def analyze_video_endpoint(
        request: Request,
        video_path: str = Form(...),
    ):
        """Analyze a recorded meeting video using a Visual LLM.

        Takes the path to a recorded video file, retrieves the conversation
        transcript from the current session's chat history, and sends both
        to a Visual LLM for feedback on the user's conversational performance.

        Args:
            request: The HTTP request object.
            video_path: Path to the recorded video file.

        Returns:
            JSONResponse: Analysis feedback or error message.
        """
        logger.info(f"📊 Video analysis requested for: {video_path}")

        # Validate video file exists
        if not Path(video_path).exists():
            logger.warning(f"Video file not found: {video_path}")
            return JSONResponse(
                {"error": f"Video file not found: {video_path}"},
                status_code=404,
            )

        # Get video analysis config
        system_config = default_context_cache.system_config
        analysis_config = system_config.video_analysis

        if not analysis_config.enabled:
            logger.warning("Video analysis is not enabled in configuration")
            return JSONResponse(
                {
                    "error": "Video analysis is not enabled. "
                    "Set video_analysis.enabled to true in conf.yaml"
                },
                status_code=400,
            )

        if not analysis_config.api_key:
            logger.warning("Video analysis API key is not configured")
            return JSONResponse(
                {
                    "error": "Video analysis API key is not configured. "
                    "Set video_analysis.api_key in conf.yaml"
                },
                status_code=400,
            )

        # Build transcript from chat history
        transcript = ""
        client_ip = request.client.host
        context = ws_handler.get_context_by_ip(client_ip)

        if context and context.character_config:
            conf_uid = context.character_config.conf_uid
            history_uid = context.history_uid

            if conf_uid and history_uid:
                messages = get_history(conf_uid, history_uid)
                if messages:
                    transcript_lines = []
                    for msg in messages:
                        role_label = "Student" if msg["role"] == "human" else "Examiner"
                        transcript_lines.append(f"{role_label}: {msg['content']}")
                    transcript = "\n\n".join(transcript_lines)
                    logger.info(f"📝 Built transcript with {len(messages)} messages")

        if not transcript:
            logger.warning("No transcript found, analyzing video frames only")
            transcript = "(No transcript available)"

        # Run video analysis
        try:
            feedback = await analyze_video(
                video_path=video_path,
                transcript=transcript,
                base_url=analysis_config.base_url,
                api_key=analysis_config.api_key,
                model=analysis_config.model,
                analysis_prompt=(
                    analysis_config.analysis_prompt
                    if analysis_config.analysis_prompt
                    else None
                ),
                max_frames=analysis_config.max_frames,
                temperature=analysis_config.temperature,
            )

            return JSONResponse(
                {
                    "status": "success",
                    "feedback": feedback,
                    "video_path": video_path,
                    "transcript_length": len(transcript),
                }
            )

        except FileNotFoundError as e:
            logger.error(f"Video file not found: {e}")
            return JSONResponse({"error": str(e)}, status_code=404)
        except RuntimeError as e:
            logger.error(f"Video analysis failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
        except Exception as e:
            logger.error(f"❌ Unexpected error during video analysis: {e}")
            return JSONResponse(
                {"error": f"Video analysis failed: {str(e)}"},
                status_code=500,
            )

    return router
