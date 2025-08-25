"""
Standalone Remotion Project Renderer with Modal
Complete pipeline in a single file - no external dependencies from your codebase
"""

import os
import tempfile
import uuid
import zipfile
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import boto3
import urllib.parse
from datetime import datetime
from dotenv import load_dotenv

# Modal imports
try:
    import modal
    MODAL_AVAILABLE = True
    modal.enable_output()
except ImportError:
    MODAL_AVAILABLE = False
    print("⚠️ Modal not available. Install with: pip install modal")


# ============= MODAL SETUP =============

if MODAL_AVAILABLE:
    # Define the Modal image with all dependencies
    DEPNDENCIES = Path(__file__).parent / "remotion_package_json"
    
    remotion_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "curl",
            "wget", 
            "zip",
            "unzip",
            "chromium",
            "nodejs",
            "npm",
            "ffmpeg",
            "libasound2",
            "libatk1.0-0",
            "libcairo-gobject2",
            "libgtk-3-0",
            "libgdk-pixbuf2.0-0",
            "libx11-6",
            "libxcomposite1",
            "libxdamage1",
            "libxrandr2",
            # GPU acceleration libraries
            "libgl1-mesa-dev",
            "libgl1-mesa-dri",
            "libegl1-mesa-dev"
        )
        .run_commands(
            "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
            "apt-get update && apt-get install -y nodejs",
            "ln -sf /usr/bin/chromium /usr/bin/google-chrome",
            "npm install -g npm@latest",
            "mkdir -p /assets",
            "node --version",
            "npm --version"
        )
        .pip_install("boto3", "requests", 'dotenv')
        .add_local_dir(
            local_path=str(DEPNDENCIES), 
            remote_path="/assets/remotion_package_json",
            copy=True
            )
        
        .run_commands(
            # Install packages with scripts enabled to ensure dist files are built
            "cd /assets/remotion_package_json && npm ci --no-audit --no-fund",
        )
    )
    
    app = modal.App("standalone-remotion-renderer-images")
    
    @app.function(
        image=remotion_image,
        timeout=2400,
        memory=8192,
        cpu=4,
        gpu="T4",
        secrets=[modal.Secret.from_name("aws-secret")]
    )
    def modal_render_remotion(
        zip_url: str,
        s3_bucket: str,
        aws_region: str = "us-west-2"
    ) -> Dict[str, Any]:
        """Render Remotion project in Modal"""
        import boto3
        import requests
        import zipfile
        import subprocess
        import re
        
        print(f"🎬 Starting Remotion render in Modal...")
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                project_dir = Path(tmpdir) / "project"
                
                # Download and extract zip
                print(f"📥 Downloading project from: {zip_url}")
                response = requests.get(zip_url, timeout=600)
                response.raise_for_status()
                
                zip_path = Path(tmpdir) / "project.zip"
                zip_path.write_bytes(response.content)
                
                print("📂 Extracting project...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(project_dir)
                
                # Find the actual project directory (exact logic from your code)
                subfolders = [
                    item for item in project_dir.iterdir() 
                    if item.is_dir() and not item.name.startswith('_')
                    and not item.name.startswith('__MACOSX')
                ]
                
                actual_project_dir = None
                
                # Handle your specific nested structure
                if len(subfolders) >= 1 and not subfolders[0].name.startswith('__MACOSX'):
                    # Check for videos subdirectory
                    potential_dir = subfolders[0] / 'videos'
                    if potential_dir.exists():
                        project_dir = potential_dir
                        subfolders = [
                            item for item in project_dir.iterdir() 
                            if item.is_dir() and not item.name.startswith('_')
                            and not item.name.startswith('__MACOSX')
                        ]
                        if len(subfolders) >= 1 and not subfolders[0].name.startswith('__MACOSX'):
                            actual_project_dir = subfolders[0]
                    else:
                        # Use the first subfolder directly
                        actual_project_dir = subfolders[0]
                else:
                    # No subfolders, use root
                    actual_project_dir = project_dir
                
                # Ensure we found the right directory with package.json
                if actual_project_dir and not (actual_project_dir / "package.json").exists():
                    # Search for directory with package.json
                    for item in project_dir.iterdir():
                        if item.is_dir() and (item / "package.json").exists():
                            actual_project_dir = item
                            break
                
                if not actual_project_dir or not (actual_project_dir / "package.json").exists():
                    raise Exception(f"Could not find package.json in extracted project")
                
                print(f"📁 Initial project directory: {actual_project_dir}")
                
                # For Motion Canvas projects, might need to parse vite.config.ts
                vite_config_path = actual_project_dir / "vite.config.ts"
                if vite_config_path.exists() and "motionCanvas" in vite_config_path.read_text():
                    print("📝 Found Motion Canvas vite.config.ts, parsing for project location...")
                    vite_content = vite_config_path.read_text(encoding='utf-8')
                    
                    # Look for project paths in motionCanvas plugin configuration
                    import re
                    project_pattern = r'project:\s*\[[\s\n]*[\'"`]([^\'"`]+)[\'"`]'
                    matches = re.findall(project_pattern, vite_content, re.DOTALL)
                    
                    if matches:
                        relative_project_path = matches[0]
                        if relative_project_path.startswith('./'):
                            relative_project_path = relative_project_path[2:]
                        
                        # This is typically pointing to a project.ts file
                        project_ts_path = actual_project_dir / relative_project_path
                        if project_ts_path.exists():
                            print(f"✅ Found project.ts via vite config: {project_ts_path}")
                            # For Motion Canvas, we still run from the root with package.json
                            # but now we know where the project.ts is
                
                print(f"📁 Using project directory: {actual_project_dir}")
                
                # Install dependencies
                print("📦 Installing npm dependencies...")
                install_result = subprocess.run(
                    ["npm", "install"],
                    cwd=actual_project_dir,
                    capture_output=True,
                    text=True
                )
                
                if install_result.returncode != 0:
                    print(f"⚠️ npm install failed: {install_result.stderr}")
                    return {
                        "success": False,
                        "error": f"Failed to install dependencies: {install_result.stderr}"
                    }
                
                print("✅ Dependencies installed")
                
                # Bundle once and reuse for all shards (anti-pattern to bundle every time)
                print("📦 Bundling project once for reuse across shards...")
                bundle_dir = actual_project_dir / "dist"
                bundle_result = subprocess.run(
                    ["npx", "remotion", "bundle", "src/index.ts", str(bundle_dir)],
                    cwd=actual_project_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if bundle_result.returncode != 0:
                    print(f"⚠️ Bundling failed: {bundle_result.stderr}")
                    # Fallback to direct source
                    serve_url = "src/index.ts"
                    print("🔄 Falling back to direct source rendering")
                else:
                    serve_url = str(bundle_dir)
                    print(f"✅ Project bundled successfully at {serve_url}")
                
                # Get total frames for composition
                print("🔍 Getting composition metadata...")
                compositions_result = subprocess.run(
                    ["npx", "remotion", "compositions", "--serve-url", serve_url, "--json"],
                    cwd=actual_project_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                total_frames = None
                composition_fps = 30  # Default fallback
                
                if compositions_result.returncode == 0:
                    try:
                        import json
                        compositions_data = json.loads(compositions_result.stdout)
                        comp_info = next((c for c in compositions_data["compositions"] if c["id"] == first_composition), None)
                        if comp_info:
                            total_frames = comp_info["durationInFrames"]
                            composition_fps = comp_info["fps"]
                            print(f"📊 Composition: {total_frames} frames at {composition_fps} FPS")
                        else:
                            print("⚠️ Could not find composition info")
                            total_frames = 1000  # Fallback estimate
                    except Exception as e:
                        print(f"⚠️ Could not parse composition metadata: {e}")
                        total_frames = 1000  # Fallback estimate
                else:
                    print(f"⚠️ Failed to get composition metadata: {compositions_result.stderr}")
                    total_frames = 1000  # Fallback estimate
                
                # Discover compositions
                # print("🔍 Discovering compositions...")
                # compositions_result = subprocess.run(
                #     ["npx", "remotion", "compositions", "src/index.ts"],
                #     cwd=actual_project_dir,
                #     capture_output=True,
                #     text=True,
                #     timeout=600
                # )
                
                # if compositions_result.returncode != 0:
                #     raise Exception(f"Failed to discover compositions: {compositions_result.stderr}")
                
                # # Parse first composition name
                # compositions_output = compositions_result.stdout.strip()
                # print(f"Raw compositions output:\n{compositions_output}")
                
                first_composition = 'FullVideo'
                print(f"🎯 Using composition: {first_composition}")
                
                # Create output directory
                output_dir = actual_project_dir / "out"
                output_dir.mkdir(exist_ok=True)
                frames_dir = output_dir / "frames"
                frames_dir.mkdir(exist_ok=True)

                start_time = datetime.now()
                
                # Implement 4-shard parallel rendering
                num_shards = 4
                print(f"🧩 Implementing {num_shards}-shard parallel rendering...")
                
                # Calculate frame ranges for each shard (split 0..(total_frames-1) evenly)
                frames_per_shard = total_frames // num_shards
                remainder_frames = total_frames % num_shards
                
                shard_ranges = []
                current_frame = 0
                for i in range(num_shards):
                    # Add extra frame to first few shards if there's a remainder
                    shard_frames = frames_per_shard + (1 if i < remainder_frames else 0)
                    end_frame = current_frame + shard_frames - 1
                    shard_ranges.append((current_frame, end_frame))
                    current_frame += shard_frames
                
                print(f"📊 Frame distribution across {num_shards} shards:")
                for i, (start, end) in enumerate(shard_ranges):
                    print(f"   Shard {i}: frames {start}-{end} ({end - start + 1} frames)")
                
                # Create shard output directories
                shard_dirs = []
                for i in range(num_shards):
                    shard_dir = output_dir / f"shard_{i}"
                    shard_dir.mkdir(exist_ok=True)
                    shard_dirs.append(shard_dir)
                
                # Calculate concurrency per shard (sum across shards ≈ CPU cores)
                total_cpu_cores = 4
                concurrency_per_shard = max(1, total_cpu_cores // num_shards)
                print(f"🔧 Using concurrency {concurrency_per_shard} per shard (total: {concurrency_per_shard * num_shards})")
                
                # Run shards in parallel
                import concurrent.futures
                
                def render_shard(shard_info):
                    shard_idx, start_frame, end_frame, shard_dir = shard_info
                    print(f"🚀 Starting shard {shard_idx}: frames {start_frame}-{end_frame}")
                    
                    shard_cmd = [
                        "npx", "remotion", "render",
                        "--serve-url", serve_url,  # Use pre-bundled version
                        first_composition, str(shard_dir),
                        "--sequence",
                        "--image-format=jpeg",
                        f"--frames={start_frame}-{end_frame}",  # Frame range for this shard
                        f"--concurrency={concurrency_per_shard}",
                        "--chrome-mode=chrome-for-testing",
                        "--gl=angle-egl",
                        "--enable-multiprocess-on-linux",  # Better throughput on Linux
                        "--log=warn"  # Reduce log noise
                    ]
                    
                    result = subprocess.run(
                        shard_cmd,
                        cwd=actual_project_dir,
                        capture_output=True,
                        text=True,
                        timeout=1800  # 30 min timeout per shard
                    )
                    
                    if result.returncode == 0:
                        print(f"✅ Shard {shard_idx} completed successfully")
                        return True, shard_idx, None
                    else:
                        print(f"❌ Shard {shard_idx} failed: {result.stderr}")
                        return False, shard_idx, result.stderr
                
                # Prepare shard tasks
                shard_tasks = []
                for i, (start_frame, end_frame) in enumerate(shard_ranges):
                    shard_tasks.append((i, start_frame, end_frame, shard_dirs[i]))
                
                # Execute all shards in parallel
                print(f"🎬 Launching {num_shards} parallel render processes...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_shards) as executor:
                    futures = [executor.submit(render_shard, task) for task in shard_tasks]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]
                
                # Check for failures
                failed_shards = [r for r in results if not r[0]]
                if failed_shards:
                    error_msgs = [f"Shard {r[1]}: {r[2]}" for r in failed_shards]
                    raise Exception(f"Shard rendering failed: {'; '.join(error_msgs)}")
                
                # Merge all shards into final sequence
                print("🔗 Merging shards into final frame sequence...")
                frame_counter = 0
                for i, shard_dir in enumerate(shard_dirs):
                    shard_frames = sorted(shard_dir.glob("*.jpeg"))
                    print(f"   Merging {len(shard_frames)} frames from shard {i}")
                    
                    for frame_file in shard_frames:
                        # Rename to sequential numbering
                        new_name = f"frame_{frame_counter:05d}.jpeg"
                        target_path = frames_dir / new_name
                        frame_file.rename(target_path)
                        frame_counter += 1
                    
                    # Clean up shard directory
                    shard_dir.rmdir()
                
                print(f"✅ Merged {frame_counter} frames from {num_shards} shards")

                render_time = datetime.now() - start_time
                # Check NVENC support in FFmpeg
                print("🔍 Checking FFmpeg NVENC support...")
                nvenc_check = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                has_h264_nvenc = " h264_nvenc " in nvenc_check.stdout
                has_hevc_nvenc = " hevc_nvenc " in nvenc_check.stdout
                
                print(f"📝 FFmpeg NVENC check results:")
                print(f"   H.264 NVENC: {'✅ Available' if has_h264_nvenc else '❌ Not available'}")
                print(f"   HEVC NVENC: {'✅ Available' if has_hevc_nvenc else '❌ Not available'}")
                
                # Create a video from the frames with ffmpeg
                print("🎬 Creating video from frames...")
                
                if has_h264_nvenc:
                    print("🚀 Using NVIDIA NVENC H.264 encoder with GPU acceleration")
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-y",
                        "-framerate", "30",
                        "-pattern_type", "glob",
                        "-i", str(frames_dir / "*.jpeg"),
                        "-vf", "scale=1920:1080,format=yuv420p",
                        "-c:v", "h264_nvenc",
                        "-preset", "p4",
                        "-rc", "vbr",
                        "-b:v", "10M",
                        "-maxrate", "12M",
                        "-bufsize", "20M",
                        "-r", "30",
                        "-movflags", "+faststart",
                        str(output_dir / "output.mp4")
                    ]
                else:
                    print("🔄 Falling back to CPU encoding (libx264)")
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-framerate",
                        "30",
                        "-i", str(frames_dir / "frame_%04d.jpeg"),
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-crf", "18",
                        "-preset", "medium",
                        "-tune", "film",
                        "-profile:v", "high444",
                        "-y",
                        str(output_dir / "output.mp4")
                    ]
                
                print(f"🔧 FFmpeg command: {' '.join(ffmpeg_cmd)}")
                
                ffmpeg_result = subprocess.run(
                    ffmpeg_cmd,
                    cwd=actual_project_dir,
                    capture_output=True,
                    text=True,
                    timeout=2400
                )
                if ffmpeg_result.returncode != 0:
                    raise Exception(f"FFmpeg failed: {ffmpeg_result.stderr}")

                print("✅ Video created from frames")

                encode_time = datetime.now() - render_time
                print(f"🎬 Render time: {render_time}\n")
                print(f"🎬 Video created from frames in {encode_time}\n")

                # Find the output video
                output_file = output_dir / "output.mp4"
                
                print(f"📹 Video found: {output_file}")
                
                # Get file size
                file_size = output_file.stat().st_size
                print(f"📊 Video size: {file_size / 1024 / 1024:.1f}MB")
                
                # Upload to S3
                if not s3_bucket:
                    s3_bucket = os.environ.get("S3_BUCKET_OUTPUT")
                    aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
                s3_client = boto3.client('s3', region_name=aws_region)
                unique_id = uuid.uuid4()
                s3_key = f"videos/rendered/{unique_id}/video.mp4"
                

                print(f"📤 Uploading to S3: {s3_bucket}/{s3_key}")
                
                with open(output_file, 'rb') as f:
                    s3_client.upload_fileobj(
                        f,
                        s3_bucket,
                        s3_key,
                        ExtraArgs={
                            'ContentType': 'video/mp4',
                            'CacheControl': 'max-age=31536000'
                        }
                    )
                
                # Generate public URL
                public_url = f"https://{s3_bucket}.s3.{aws_region}.amazonaws.com/{urllib.parse.quote(s3_key)}"
                print(f"✅ Upload complete: {public_url}")
                
                return {
                    "success": True,
                    "video_url": public_url,
                    "s3_key": s3_key,
                    "file_size_mb": round(file_size / 1024 / 1024, 2)
                }
                
        except Exception as e:
            print(f"❌ Rendering error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def render_remotion_video_with_url(zip_url: str) -> str:
    """
    Render a Remotion project from a zip URL.
    """
    if not MODAL_AVAILABLE:
        raise Exception("Modal not available. Please install modal-client and set credentials.")
    load_dotenv()
    s3_bucket = os.environ.get("S3_BUCKET_OUTPUT")
    aws_region = os.environ.get("AWS_DEFAULT_REGION")

    if not s3_bucket or not aws_region:
        print(f"S3 bucket {s3_bucket} and AWS region {aws_region}")
        raise Exception("S3 bucket and AWS region are not set")
        return None

    print(f"🎬 Rendering {zip_url} with S3 bucket {s3_bucket} and AWS region {aws_region}")
    
    with app.run():
        result = modal_render_remotion.remote(
            zip_url=zip_url,
            s3_bucket=s3_bucket,
            aws_region=aws_region
        )
    
    if result.get("success"):
        video_url = result.get("video_url")
        file_size_mb = result.get("file_size_mb", 0)
        
        print(f"\n{'='*60}")
        print(f"✅ RENDERING COMPLETE!")
        print(f"📹 Video URL: {video_url}")
        print(f"📊 File size: {file_size_mb} MB")
        print(f"{'='*60}\n")
        
        return video_url
    else:
        error_msg = result.get("error", "Unknown error")
        raise Exception(f"Modal rendering failed: {error_msg}")

# ============= MAIN FUNCTION =============

def render_remotion_video(project_path: str | Path) -> str:
    """
    Takes a Remotion project, zips it, uploads to S3, renders via Modal, returns video URL.
    
    Args:
        project_path: Path to your local Remotion project folder
        
    Returns:
        URL of the rendered video on S3
        
    Example:
        video_url = render_remotion_video("path/to/my-remotion-project")
        print(video_url)  # https://bucket.s3.region.amazonaws.com/videos/...
    """
    
    if not MODAL_AVAILABLE:
        raise Exception("Modal not available. Please install modal-client and set credentials.")
    
    # Get config from environment
    s3_bucket = os.environ.get("S3_BUCKET_OUTPUT")
    aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    
    if not s3_bucket:
        raise ValueError("S3_BUCKET_OUTPUT environment variable not set")
    
    # Setup
    project_path = Path(project_path)
    if not project_path.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")
    
    s3_client = boto3.client("s3", region_name=aws_region)
    
    # Step 1: Create zip file
    print(f"\n📦 Zipping project: {project_path.name}")
    zip_path = Path(tempfile.gettempdir()) / f"{project_path.name}_{uuid.uuid4().hex[:8]}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_path):
            # Skip unnecessary directories
            dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', 'out', 'build', 'dist', '__pycache__']]
            
            for file in files:
                file_path = Path(root) / file
                # Create archive name preserving structure
                arcname = file_path.relative_to(project_path.parent)
                zipf.write(file_path, arcname)
    
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"✅ Zip created: {zip_path.name} ({zip_size_mb:.1f} MB)")
    
    # Step 2: Upload zip to S3
    print(f"📤 Uploading zip to S3...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_key = f"remotion-projects/{timestamp}/{project_path.name}.zip"
    
    try:
        with open(zip_path, 'rb') as f:
            s3_client.upload_fileobj(
                f,
                s3_bucket,
                s3_key,
                ExtraArgs={
                    'ContentType': 'application/zip',
                    'CacheControl': 'max-age=31536000'
                }
            )
        
        zip_url = f"https://{s3_bucket}.s3.{aws_region}.amazonaws.com/{urllib.parse.quote(s3_key)}"
        print(f"✅ Zip uploaded: {zip_url}")
        
    except Exception as e:
        os.remove(zip_path)
        raise Exception(f"Failed to upload zip to S3: {e}")
    
    # Step 3: Clean up local zip
    os.remove(zip_path)
    print("🧹 Local zip file cleaned up")
    
    # Step 4: Render with Modal
    print(f"\n🚀 Starting Modal render...")
    print(f"   This may take several minutes...")
    
    with app.run():
        result = modal_render_remotion.remote(
            zip_url=zip_url,
            s3_bucket=s3_bucket,
            aws_region=aws_region
        )
    
    # Step 5: Check result and return
    if result.get("success"):
        video_url = result.get("video_url")
        file_size_mb = result.get("file_size_mb", 0)
        
        print(f"\n{'='*60}")
        print(f"✅ RENDERING COMPLETE!")
        print(f"📹 Video URL: {video_url}")
        print(f"📊 File size: {file_size_mb} MB")
        print(f"{'='*60}\n")
        
        return video_url
    else:
        error_msg = result.get("error", "Unknown error")
        raise Exception(f"Modal rendering failed: {error_msg}")


# ============= UTILITIES =============

def batch_render(project_paths: list[str]) -> list[str]:
    """
    Render multiple Remotion projects in sequence.
    
    Args:
        project_paths: List of paths to Remotion projects
        
    Returns:
        List of video URLs
    """
    results = []
    
    for i, project_path in enumerate(project_paths, 1):
        print(f"\n{'='*60}")
        print(f"BATCH: Processing {i}/{len(project_paths)}")
        print(f"{'='*60}")
        
        try:
            video_url = render_remotion_video(project_path)
            results.append(video_url)
            print(f"✅ Success: {Path(project_path).name}")
        except Exception as e:
            print(f"❌ Failed: {Path(project_path).name} - {e}")
            results.append(None)
    
    # Summary
    successful = sum(1 for url in results if url)
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {successful}/{len(project_paths)} successful")
    print(f"{'='*60}\n")
    
    return results


# ============= MAIN =============

if __name__ == "__main__":
    import sys
    
    print(sys.argv)
    if len(sys.argv) < 2:
        print("Usage: python render_remotion.py <project_path>")
        print("   or: python render_remotion.py <path1> <path2> ... (for batch)")
        sys.exit(1)
    
    if len(sys.argv) == 2 and not sys.argv[1] == '--url':
        # Single project
        try:
            video_url = render_remotion_video(sys.argv[1])
            print(f"\n🎉 Success! Your video is ready:")
            print(f"   {video_url}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    elif len(sys.argv) >= 3 and sys.argv[1] == '--url':
        video_urls = sys.argv[2:]
        for video_url in video_urls:
            print(f"🎬 Rendering {video_url}")
            video_url = render_remotion_video_with_url(video_url)
            print(f"\n🎉 Success! Your video is ready:")
            print(f"   {video_url}")