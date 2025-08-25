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
            # Additional GPU and graphics libraries
            "libgl1-mesa-dev",
            "libgl1-mesa-dri",
            "libglu1-mesa-dev",
            "libegl1-mesa-dev",
            "mesa-vulkan-drivers",
            "vulkan-tools",
            "libvulkan1",
            "mesa-utils",
            "xvfb"
        )
        .run_commands(
            "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
            "apt-get update && apt-get install -y nodejs",
            "ln -sf /usr/bin/chromium /usr/bin/google-chrome",
            "npm install -g npm@latest",
            "mkdir -p /assets",
            "node --version",
            "npm --version",
            # Create cache directories
            "mkdir -p /tmp/remotion-cache",
            "mkdir -p /root/.cache/remotion"
        )
        .pip_install("boto3", "requests")
        .add_local_dir(
            local_path=str(DEPNDENCIES), 
            remote_path="/assets/remotion_package_json",
            copy=True
            )
        
        .run_commands(
            # Install packages with scripts enabled to ensure dist files are built
            "cd /assets/remotion_package_json && npm ci --no-audit --no-fund",
            # Pre-install Chrome for Testing to avoid download overhead during rendering
            "cd /assets/remotion_package_json && npx --package @remotion/cli remotion browser ensure",
        )
    )
    
    app = modal.App("standalone-remotion-renderer")
    
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
        
        if not s3_bucket:
            s3_bucket = os.environ.get("S3_BUCKET_OUTPUT")

        print(f"🎬 S3 bucket: {s3_bucket}")
        print(f"🎬 AWS region: {aws_region}")
        print(f"🎬 Zip URL: {zip_url}")
        
        def detect_nvenc():
            """Detect available NVENC encoders in FFmpeg"""
            try:
                out = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True, text=True, check=True
                ).stdout
                has_h264_nvenc = " h264_nvenc " in out
                has_hevc_nvenc = " hevc_nvenc " in out
                return {"h264": has_h264_nvenc, "hevc": has_hevc_nvenc}
            except Exception as e:
                print(f"⚠️ FFmpeg check failed: {e}")
                return {"h264": False, "hevc": False}
        
        def get_composition_fps(composition_id, project_dir):
            """Get FPS from composition metadata"""
            try:
                r = subprocess.run(
                    ["npx", "remotion", "compositions", "--json"],
                    cwd=project_dir, capture_output=True, text=True, check=True
                )
                import json
                comps = json.loads(r.stdout)["compositions"]
                fps = next(c["fps"] for c in comps if c["id"] == composition_id)
                print(f"📊 Detected composition FPS: {fps}")
                return fps
            except Exception as e:
                print(f"⚠️ Could not detect FPS, using default 30: {e}")
                return 30
        
        def encode_with_nvenc(frames_dir, output_file, fps, encoders):
            """Encode frames using NVENC or fallback to libx264"""
            ffmpeg_input = os.path.join(frames_dir, "%05d.jpg")  # Remotion uses 5-digit padding
            common = ["-y", "-framerate", str(fps), "-i", ffmpeg_input]
            
            if encoders["h264"]:
                print("🚀 Using NVIDIA NVENC H.264 encoder")
                ffmpeg_cmd = ["ffmpeg", *common,
                              "-c:v", "h264_nvenc",
                              "-preset", "p5",  # Balanced quality/speed
                              "-b:v", "12M",    # 12Mbps for good quality
                              "-pix_fmt", "yuv420p",
                              str(output_file)]
            else:
                print("🔄 Falling back to libx264 (CPU encoding)")
                ffmpeg_cmd = ["ffmpeg", *common,
                              "-c:v", "libx264",
                              "-preset", "veryfast",
                              "-crf", "18",     # High quality
                              "-pix_fmt", "yuv420p",
                              str(output_file)]
            
            print(f"🔧 FFmpeg command: {' '.join(ffmpeg_cmd)}")
            
            start_time = datetime.now()
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            encode_time = datetime.now() - start_time
            
            if result.returncode == 0:
                print(f"✅ Encoding completed in {encode_time}")
                print(f"🎬 Output: {output_file}")
                return encode_time, True
            else:
                print(f"❌ Encoding failed: {result.stderr}")
                return encode_time, False
        
        print(f"🎬 Starting Remotion render in Modal...")
        
        # Check GPU availability and environment
        print("🔍 Checking GPU environment...")
        try:
            gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
            if gpu_check.returncode == 0:
                print("✅ GPU detected via nvidia-smi:")
                print(gpu_check.stdout)
            else:
                print("⚠️ nvidia-smi failed - GPU may not be available")
        except Exception as e:
            print(f"⚠️ Could not run nvidia-smi: {e}")
        
        # Check Chrome installation
        print("🔍 Checking Chrome installation...")
        try:
            chrome_check = subprocess.run(["google-chrome", "--version"], capture_output=True, text=True, timeout=10)
            if chrome_check.returncode == 0:
                print(f"✅ Chrome version: {chrome_check.stdout.strip()}")
            else:
                print("⚠️ google-chrome command failed")
        except Exception as e:
            print(f"⚠️ Could not check Chrome version: {e}")
            
        # Check Vulkan support
        print("🔍 Checking Vulkan support...")
        try:
            vulkan_check = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10)
            if vulkan_check.returncode == 0:
                print("✅ Vulkan support detected")
                print(vulkan_check.stdout[:500] + "..." if len(vulkan_check.stdout) > 500 else vulkan_check.stdout)
            else:
                print("⚠️ Vulkan not available")
        except Exception as e:
            print(f"⚠️ Could not check Vulkan: {e}")
            
        # Check GL libraries
        print("🔍 Checking OpenGL libraries...")
        try:
            gl_check = subprocess.run(["glxinfo", "-B"], capture_output=True, text=True, timeout=10)
            if gl_check.returncode == 0:
                print("✅ OpenGL libraries available")
                print(gl_check.stdout[:300] + "..." if len(gl_check.stdout) > 300 else gl_check.stdout)
            else:
                print("⚠️ glxinfo failed")
        except Exception as e:
            print(f"⚠️ Could not check OpenGL: {e}")
            
        # Check NVENC support
        print("🔍 Checking NVENC support...")
        nvenc_encoders = detect_nvenc()
        if nvenc_encoders["h264"]:
            print("✅ NVENC H.264 encoder available")
        if nvenc_encoders["hevc"]:
            print("✅ NVENC HEVC encoder available")
        if not any(nvenc_encoders.values()):
            print("⚠️ No NVENC encoders found - will use CPU encoding")
        
        # Set display for headless GPU rendering
        print("🔧 Setting up display for GPU rendering...")
        os.environ['DISPLAY'] = ':99'
        try:
            # Start virtual display in background
            xvfb_process = subprocess.Popen(
                ["Xvfb", ":99", "-screen", "0", "1920x1080x24", "-ac"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Give it a moment to start
            import time
            time.sleep(2)
            print("✅ Virtual display started")
        except Exception as e:
            print(f"⚠️ Could not start virtual display: {e}")
        
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
                
                
                first_composition = 'FullVideo'
                print(f"🎯 Using composition: {first_composition}")
                
                # Create output directory
                output_dir = actual_project_dir / "out"
                output_dir.mkdir(exist_ok=True)
                
                # Test GPU capabilities first
                print("🔍 Testing GPU capabilities...")
                gpu_cmd = [
                    "npx", "remotion", "gpu", 
                    "--chrome-mode=chrome-for-testing", 
                    "--gl=angle-egl"
                ]
                gpu_result = subprocess.run(
                    gpu_cmd,
                    cwd=actual_project_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                print(f"📝 GPU test output:\n{gpu_result.stdout}")
                if gpu_result.stderr:
                    print(f"📝 GPU test stderr:\n{gpu_result.stderr}")
                
                # Prewarm the browser and bundle (exclude from timing)
                print("🔥 Prewarming browser and bundle...")
                prewarm_cmd = [
                    "npx", "remotion", "compositions", "src/index.ts",
                    "--chrome-mode=chrome-for-testing",
                    "--quiet"
                ]
                subprocess.run(prewarm_cmd, cwd=actual_project_dir, capture_output=True, timeout=120)
                print("✅ Prewarm complete")
                
                # Get composition FPS for proper encoding
                composition_fps = get_composition_fps(first_composition, actual_project_dir)
                
                # A/B Test Matrix: Test different configurations
                test_configs = [
                    {"name": "CPU-Direct", "type": "direct", "flags": [], "file": "cpu_direct.mp4"},
                    {"name": "GPU-Direct", "type": "direct", "flags": ["--chrome-mode=chrome-for-testing", "--gl=angle-egl"], "file": "gpu_direct.mp4"},
                    {"name": "JPEG+CPU", "type": "frames", "flags": ["--chrome-mode=chrome-for-testing", "--gl=angle-egl"], "file": "jpeg_cpu.mp4", "encoder": "cpu"},
                    {"name": "JPEG+NVENC", "type": "frames", "flags": ["--chrome-mode=chrome-for-testing", "--gl=angle-egl"], "file": "jpeg_nvenc.mp4", "encoder": "nvenc"},
                ]
                
                results = []
                
                for config in test_configs:
                    print(f"\n{'='*60}")
                    print(f"🧪 Testing: {config['name']}")
                    print(f"{'='*60}")
                    
                    output_file_test = output_dir / config['file']
                    
                    # Start GPU monitoring in background
                    gpu_monitor = None
                    try:
                        gpu_monitor = subprocess.Popen([
                            "nvidia-smi", 
                            "--query-gpu=utilization.gpu,memory.used", 
                            "--format=csv,noheader,nounits", 
                            "-l", "2"
                        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    except:
                        print("⚠️ Could not start GPU monitoring")
                    
                    # Time the entire pipeline
                    total_start_time = datetime.now()
                    success = False
                    render_time = None
                    encode_time = None
                    
                    if config['type'] == 'direct':
                        # Direct video rendering (original method)
                        render_cmd = [
                            "npx", "remotion", "render", 
                            first_composition, 
                            str(output_file_test),
                            "--concurrency=4",
                            "--log=warn"
                        ] + config['flags']
                        
                        print(f"🔧 Direct render command: {' '.join(render_cmd)}")
                        
                        render_start = datetime.now()
                        render_result = subprocess.run(
                            render_cmd,
                            cwd=actual_project_dir,
                            capture_output=True,
                            text=True,
                            timeout=2400
                        )
                        render_time = datetime.now() - render_start
                        success = render_result.returncode == 0
                        
                        if success:
                            print(f"✅ Direct render completed in {render_time}")
                        else:
                            print(f"❌ Direct render failed: {render_result.stderr}")
                    
                    elif config['type'] == 'frames':
                        # JPEG frames + separate encoding
                        frames_dir = output_dir / f"frames_{config['name'].lower().replace('+', '_')}"
                        frames_dir.mkdir(exist_ok=True)
                        
                        # Step 1: Render frames
                        frames_cmd = [
                            "npx", "remotion", "render",
                            first_composition,
                            str(frames_dir),
                            "--sequence",
                            "--image-format=jpeg",
                            "--concurrency=4",
                            "--log=warn"
                        ] + config['flags']
                        
                        print(f"🔧 Frames render command: {' '.join(frames_cmd)}")
                        
                        render_start = datetime.now()
                        render_result = subprocess.run(
                            frames_cmd,
                            cwd=actual_project_dir,
                            capture_output=True,
                            text=True,
                            timeout=2400
                        )
                        render_time = datetime.now() - render_start
                        
                        if render_result.returncode == 0:
                            print(f"✅ Frames rendered in {render_time}")
                            
                            # Step 2: Encode with appropriate encoder
                            if config['encoder'] == 'nvenc':
                                encode_time, encode_success = encode_with_nvenc(
                                    str(frames_dir), output_file_test, composition_fps, nvenc_encoders
                                )
                            else:  # CPU encoding
                                # Force CPU encoding even if NVENC is available
                                cpu_encoders = {"h264": False, "hevc": False}
                                encode_time, encode_success = encode_with_nvenc(
                                    str(frames_dir), output_file_test, composition_fps, cpu_encoders
                                )
                            
                            success = encode_success
                            
                            # Clean up frames to save space
                            try:
                                import shutil
                                shutil.rmtree(frames_dir, ignore_errors=True)
                                print("🧹 Cleaned up frame files")
                            except:
                                pass
                        else:
                            print(f"❌ Frames render failed: {render_result.stderr}")
                            success = False
                    
                    total_time = datetime.now() - total_start_time
                    
                    # Stop GPU monitoring
                    gpu_stats = "N/A"
                    if gpu_monitor:
                        try:
                            gpu_monitor.terminate()
                            gpu_output, _ = gpu_monitor.communicate(timeout=5)
                            if gpu_output:
                                lines = gpu_output.decode().strip().split('\n')
                                if lines:
                                    # Calculate average GPU utilization
                                    gpu_utils = []
                                    for line in lines[-10:]:  # Last 10 readings
                                        if ',' in line:
                                            util = line.split(',')[0].strip()
                                            if util.isdigit():
                                                gpu_utils.append(int(util))
                                    if gpu_utils:
                                        avg_gpu = sum(gpu_utils) / len(gpu_utils)
                                        gpu_stats = f"{avg_gpu:.1f}%"
                        except:
                            pass
                    
                    # Record results
                    result = {
                        "config": config['name'],
                        "success": success,
                        "time": total_time,
                        "seconds": total_time.total_seconds(),
                        "render_time": render_time,
                        "encode_time": encode_time,
                        "gpu_util": gpu_stats,
                        "file_size": 0
                    }
                    
                    if success:
                        print(f"✅ {config['name']} completed in {total_time}")
                        if render_time and encode_time:
                            print(f"   📊 Render: {render_time}, Encode: {encode_time}")
                        print(f"🎮 Average GPU utilization: {gpu_stats}")
                        
                        # Check file size
                        if output_file_test.exists():
                            file_size = output_file_test.stat().st_size / (1024 * 1024)  # MB
                            result["file_size"] = file_size
                            print(f"📊 Output file size: {file_size:.1f} MB")
                    else:
                        print(f"❌ {config['name']} failed")
                    
                    results.append(result)
                
                # Print benchmark summary
                print(f"\n{'='*80}")
                print("📊 BENCHMARK RESULTS")
                print(f"{'='*80}")
                
                successful_results = [r for r in results if r['success']]
                if successful_results:
                    # Sort by time (fastest first)
                    successful_results.sort(key=lambda x: x['seconds'])
                    
                    print(f"{'Config':<15} {'Total':<10} {'Render':<10} {'Encode':<10} {'GPU Util':<10} {'Size (MB)':<10}")
                    print("-" * 75)
                    
                    for result in successful_results:
                        total_str = f"{result['seconds']:.1f}s"
                        render_str = f"{result['render_time'].total_seconds():.1f}s" if result['render_time'] else "N/A"
                        encode_str = f"{result['encode_time'].total_seconds():.1f}s" if result['encode_time'] else "N/A"
                        size_str = f"{result['file_size']:.1f}" if result['file_size'] > 0 else "N/A"
                        print(f"{result['config']:<15} {total_str:<10} {render_str:<10} {encode_str:<10} {result['gpu_util']:<10} {size_str:<10}")
                    
                    # Determine winner and show comparison
                    fastest = successful_results[0]
                    print(f"\n🏆 WINNER: {fastest['config']} ({fastest['seconds']:.1f}s total)")
                    
                    if len(successful_results) > 1:
                        slowest = successful_results[-1]
                        speedup = slowest['seconds'] / fastest['seconds']
                        time_saved = slowest['seconds'] - fastest['seconds']
                        print(f"💨 {fastest['config']} is {speedup:.1f}x faster than {slowest['config']}")
                        print(f"⏱️  Time saved: {time_saved:.1f} seconds")
                        
                        # Show NVENC vs CPU comparison if both exist
                        nvenc_result = next((r for r in successful_results if 'NVENC' in r['config']), None)
                        cpu_result = next((r for r in successful_results if 'CPU' in r['config'] and 'JPEG' in r['config']), None)
                        
                        if nvenc_result and cpu_result:
                            nvenc_encode = nvenc_result['encode_time'].total_seconds() if nvenc_result['encode_time'] else 0
                            cpu_encode = cpu_result['encode_time'].total_seconds() if cpu_result['encode_time'] else 0
                            if nvenc_encode > 0 and cpu_encode > 0:
                                encode_speedup = cpu_encode / nvenc_encode
                                print(f"🚀 NVENC encoding is {encode_speedup:.1f}x faster than CPU encoding")
                    
                    # Use the fastest result as our output
                    best_file = output_dir / [c['file'] for c in test_configs if c['name'] == fastest['config']][0]
                    if best_file.exists():
                        output_file = best_file
                        print(f"📁 Using output from fastest config: {output_file}")
                else:
                    print("❌ All rendering attempts failed!")
                    raise Exception("All rendering configurations failed")
                
                print(f"{'='*80}")
                
                # Find the output video
                if not output_file.exists():
                    # Search for any mp4 file
                    video_files = list(output_dir.glob("*.mp4"))
                    if not video_files:
                        video_files = list(actual_project_dir.glob("*.mp4"))
                    
                    if video_files:
                        output_file = video_files[0]
                    else:
                        raise Exception("No video file found after rendering")
                
                print(f"📹 Video found: {output_file}")
                
                # Get file size
                file_size = output_file.stat().st_size
                print(f"📊 Video size: {file_size / 1024 / 1024:.1f}MB")
                
                # Upload to S3
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
    
    s3_bucket = os.environ.get("S3_BUCKET_OUTPUT")
    aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    print(f"S3 bucket {s3_bucket} and AWS region {aws_region}")
    
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
   
    # else:
    #     # Batch render
    #     project_paths = sys.argv[1:]
    #     results = batch_render(project_paths)
        
    #     print("\n📊 Results:")
    #     for path, url in zip(project_paths, results):
    #         if url:
    #             print(f"   ✅ {Path(path).name}: {url}")
    #         else:
    #             print(f"   ❌ {Path(path).name}: Failed")
