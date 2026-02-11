import os
import json
from uuid import uuid4
import numpy as np
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, WebSocket, UploadFile, File, Response, Request
from starlette.responses import JSONResponse
import subprocess
from starlette.websockets import WebSocketDisconnect
from loguru import logger
from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler
from .proxy_handler import ProxyHandler


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
    async def upload_video(request: Request, file: UploadFile = File(...)):
        """
        Endpoint for uploading recorded video from webcam.
        
        This endpoint:
        1. Saves the uploaded video (webm)
        2. Identifies the session via IP and stops/saves the audio recording
        3. Merges the video and audio into a final mp4
        
        Args:
            file: The video file uploaded from the client.

        Returns:
            JSONResponse: Success message with saved file path or error message.
        """
        logger.info(f"📹 Received video file for upload: {file.filename}")

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
            
            if context and context.audio_recorder and context.audio_recorder.has_audio():
                logger.info("🎙️ Found active session with audio. Stopping and saving audio...")
                
                # Generate audio filename using the same ID/Timestamp if possible or new one
                # We'll rely on the recorder's generator but try to match the dir
                from .utils.audio_recorder import AudioRecorder
                
                # Use the same IDs if possible, or just generate new one
                conf_uid = context.character_config.conf_uid if context.character_config else "unknown"
                history_uid = context.history_uid
                
                audio_filename = f"audio_{timestamp}_{unique_id}.wav"
                audio_path = video_dir / audio_filename
                
                # Save audio
                await context.audio_recorder.save_recording(str(audio_path))
                await context.audio_recorder.clear()
                
                # Get durations to calculate sync offset
                video_duration = get_duration(str(video_path))
                audio_duration = get_duration(str(audio_path))
                
                offset = 0.0
                if video_duration is not None and audio_duration is not None:
                    # If audio is longer than video, we assume the extra length is at the beginning
                    # (since audio starts at meeting start, video starts at camera open)
                    offset = max(0.0, audio_duration - video_duration)
                    logger.info(f"⏱️ durations - Video: {video_duration:.2f}s, Audio: {audio_duration:.2f}s. Sync offset: {offset:.2f}s")

                # Merge Video and Audio using ffmpeg
                # Output file
                merged_filename = f"recording_{timestamp}_{unique_id}.mp4"
                merged_path = video_dir / merged_filename
                
                logger.info(f"🔄 Merging video and audio to {merged_path}...")
                
                try:
                    # ffmpeg command:
                    # ffmpeg -i video.webm -ss offset -i audio.wav -c:v libx264 ...
                    # -ss before -i applies to the input file that follows it
                    
                    command = [
                        "ffmpeg",
                        "-y", # Overwrite output
                        "-i", str(video_path),
                        "-ss", str(offset), # Seek audio input to sync
                        "-i", str(audio_path),
                        "-c:v", "libx264", # Encode video to H.264 for MP4 compatibility
                        "-preset", "fast",
                        "-crf", "22",
                        "-c:a", "aac",     # Encode audio to AAC
                        "-b:a", "192k",
                        "-shortest",       # Finish when shortest input ends
                        str(merged_path)
                    ]
                    
                    # Run ffmpeg
                    process = subprocess.run(
                        command, 
                        check=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        text=True
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
                logger.warning("⚠️ No matching session/audio found for this IP. Returning video only.")

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

    return router
