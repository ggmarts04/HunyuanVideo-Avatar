import json
import os
import subprocess
import uuid
import base64
import requests
import tempfile
import shutil

# Define the path to the inference script
# This might be 'hymm_sp/sample_gpu_poor.py' if we modify it directly,
# or a new script like 'runpod_infer.py'
INFERENCE_SCRIPT_PATH = "/app/hymm_sp/sample_gpu_poor.py" # Assuming modification of existing script for now
DEFAULT_PROMPT = "A person speaks." # Default prompt if not provided
DEFAULT_FPS = 25

def download_file(url, local_filename):
    """Downloads a file from a URL to a local path."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

def handler(event):
    """
    RunPod Serverless handler function.
    Expects an event with 'input' containing 'image_url' and 'audio_url'.
    Optional: 'prompt' and 'fps'.
    """
    job_input = event.get("input", {})
    image_url = job_input.get("image_url")
    audio_url = job_input.get("audio_url")
    prompt = job_input.get("prompt", DEFAULT_PROMPT)
    fps = job_input.get("fps", DEFAULT_FPS)

    if not image_url or not audio_url:
        return {
            "error": "Missing 'image_url' or 'audio_url' in input."
        }

    # Create a temporary directory for this job
    temp_dir = tempfile.mkdtemp()

    try:
        # Download image and audio
        image_filename = os.path.join(temp_dir, f"input_image_{uuid.uuid4().hex}.png") # Assuming png, might need to be more robust
        audio_filename = os.path.join(temp_dir, f"input_audio_{uuid.uuid4().hex}.wav") # Assuming wav

        downloaded_image_path = download_file(image_url, image_filename)
        downloaded_audio_path = download_file(audio_url, audio_filename)

        if not downloaded_image_path or not downloaded_audio_path:
            return {"error": "Failed to download image or audio file."}

        # Create a temporary CSV file
        csv_filename = os.path.join(temp_dir, f"input_data_{uuid.uuid4().hex}.csv")
        videoid = f"runpod_job_{uuid.uuid4().hex}"

        with open(csv_filename, 'w') as f:
            f.write("videoid,image,audio,prompt,fps\n")
            f.write(f"{videoid},{downloaded_image_path},{downloaded_audio_path},{prompt},{fps}\n")

        # Define paths for output
        # The inference script will save output based on its --save-path and videoid
        # We need to ensure this path is accessible and predictable.
        # Let's assume the script saves to a directory named 'results-single' inside --save-path
        # and the filename is videoid.mp4
        output_base_path = os.path.join(temp_dir, "results")
        os.makedirs(output_base_path, exist_ok=True)

        # Expected output path based on the current sample_gpu_poor.py structure
        # which saves as {save_path}/{videoid}_audio.mp4 after ffmpeg processing
        # The intermediate output is {save_path}/{videoid}.mp4
        # We'll target the final audio-merged file.
        expected_video_filename = f"{videoid}_audio.mp4"
        expected_output_video_path = os.path.join(output_base_path, expected_video_filename)

        # Construct the command for the inference script
        # These arguments should match those in the README/script,
        # especially --ckpt, --save-path, and input CSV.
        # MODEL_BASE is an env var, so ckpt path should be relative to that or absolute if MODEL_BASE is part of it.
        # The original script uses: checkpoint_path=${MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt
        checkpoint_path = os.path.join(os.environ.get("MODEL_BASE", "/app/weights"), "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt")

        command = [
            "python3",
            INFERENCE_SCRIPT_PATH,
            "--input", csv_filename,
            "--ckpt", checkpoint_path,
            "--sample-n-frames", "129",  # Default, consider making configurable
            "--seed", "128",             # Default, consider making configurable
            "--image-size", "704",       # Default, consider making configurable
            "--cfg-scale", "7.5",        # Default, consider making configurable
            "--infer-steps", "50",       # Default, consider making configurable
            "--use-deepcache", "1",
            "--flow-shift-eval-video", "5.0",
            "--save-path", output_base_path, # Direct output to our temp results dir
            "--use-fp8"
            # "--infer-min" # This was in user's example, implies specific audio length handling
            # "--cpu-offload" # If needed for low VRAM, add this
        ]

        # Add --infer-min if it was in the original command example from the user
        # This will be handled by the inference script by setting audio_len to 129
        command.append("--infer-min")


        print(f"Running command: {' '.join(command)}")

        # Execute the command
        # Ensure PYTHONPATH is set in Dockerfile, or add it to env here
        process = subprocess.run(command, capture_output=True, text=True, env={**os.environ})

        if process.returncode != 0:
            print(f"Inference script error: {process.stderr}")
            return {
                "error": "Inference script failed.",
                "stdout": process.stdout,
                "stderr": process.stderr,
            }

        print(f"Inference script stdout: {process.stdout}")
        print(f"Inference script stderr: {process.stderr}") # Good to log stderr even on success

        if not os.path.exists(expected_output_video_path):
            # Fallback: check if the non-audio version exists, maybe ffmpeg failed
            intermediate_video_path = os.path.join(output_base_path, f"{videoid}.mp4")
            if os.path.exists(intermediate_video_path):
                 print(f"Final audio-merged video not found at {expected_output_video_path}, but intermediate video found at {intermediate_video_path}. This might indicate an ffmpeg issue in the inference script.")
                 # For now, proceed with the intermediate if final is missing
                 expected_output_video_path = intermediate_video_path
            else:
                print(f"Output video not found at {expected_output_video_path} or {intermediate_video_path}")
                # List files in output_base_path for debugging
                print(f"Contents of {output_base_path}: {os.listdir(output_base_path)}")
                return {"error": f"Output video not found after inference. Expected: {expected_output_video_path}"}

        # Read the generated video and encode it in base64
        with open(expected_output_video_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

        return {"video_base64": video_base64}

    except Exception as e:
        print(f"Handler error: {e}")
        return {"error": str(e)}
    finally:
        # Clean up: remove the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# Example usage (for local testing, not part of RunPod execution)
if __name__ == "__main__":
    # This section is for local testing only.
    # You would need to have the model weights and Docker environment set up.
    # Create dummy files and URLs for testing

    # Mock event
    mock_event = {
        "input": {
            "image_url": "https://zockto.b-cdn.net/imgtest.jpg", # Replace with a valid direct image URL
            "audio_url": "https://zockto.b-cdn.net/fizawav.wav", # Replace with a valid direct audio URL
            "prompt": "A test person speaking.",
            "fps": 25
        }
    }

    # Set MODEL_BASE if running locally and it's not set (adjust path as needed for your local setup)
    if "MODEL_BASE" not in os.environ:
        os.environ["MODEL_BASE"] = "./weights" # Or your actual local weights path

    # Ensure the inference script exists (for local test)
    if not os.path.exists(INFERENCE_SCRIPT_PATH):
        print(f"Warning: Inference script {INFERENCE_SCRIPT_PATH} not found. Local test may fail.")
        # Create a dummy script for basic testing of the handler flow
        os.makedirs(os.path.dirname(INFERENCE_SCRIPT_PATH), exist_ok=True)
        with open(INFERENCE_SCRIPT_PATH, "w") as f:
            f.write("#!/usr/bin/env python3\n")
            f.write("import sys; import os; import shutil; \n")
            f.write("print('Mock inference script called with:', sys.argv)\n")
            f.write("args = sys.argv\n")
            f.write("save_path_idx = args.index('--save-path') + 1\n")
            f.write("input_csv_idx = args.index('--input') + 1\n")
            f.write("save_path = args[save_path_idx]\n")
            f.write("input_csv = args[input_csv_idx]\n")
            f.write("with open(input_csv, 'r') as cf: content = cf.readlines()\n")
            f.write("videoid = content[1].split(',')[0]\n")
            f.write("output_vid = os.path.join(save_path, f'{videoid}_audio.mp4')\n")
            f.write("shutil.copyfile('/app/assets/material/teaser.png', output_vid) # Copy dummy file as output\n") # Use a small file
            f.write("print(f'Mock script: Created dummy video at {output_vid}')\n")
            f.write("sys.exit(0)\n")
        os.chmod(INFERENCE_SCRIPT_PATH, 0o755)


    print("Running local test of handler...")
    result = handler(mock_event)

    if "video_base64" in result and result["video_base64"]:
        print(f"Handler test successful. Video base64 length: {len(result['video_base64'])}")
        # To save the video:
        # with open("test_output.mp4", "wb") as f:
        #     f.write(base64.b64decode(result["video_base64"]))
        # print("Saved test_output.mp4")
    else:
        print(f"Handler test failed. Result: {result}")
