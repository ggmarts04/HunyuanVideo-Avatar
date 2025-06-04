import os
import numpy as np
from pathlib import Path
from loguru import logger
import imageio
import torch
# torch.distributed is not needed for single inference
# from torch.utils.data.distributed import DistributedSampler # Remove
from torch.utils.data import DataLoader # Still needed for dataset wrapping
from hymm_sp.config import parse_args
from hymm_sp.sample_inference_audio import HunyuanVideoSampler
from hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal
from hymm_sp.data_kits.face_align import AlignImage
from einops import rearrange # Added for rearrange
import shutil # Added for shutil.copy
import subprocess # Added for subprocess.run

from transformers import WhisperModel
from transformers import AutoFeatureExtractor

MODEL_OUTPUT_PATH = os.environ.get('MODEL_BASE', '/app/weights') # Ensure default for MODEL_BASE

def main():
    args = parse_args()
    models_root_path = Path(args.ckpt)

    if not models_root_path.exists():
        # Try to resolve with MODEL_BASE if it's a relative path in ckpt
        if not models_root_path.is_absolute() and Path(MODEL_OUTPUT_PATH, args.ckpt).exists():
            args.ckpt = str(Path(MODEL_OUTPUT_PATH, args.ckpt))
            models_root_path = Path(args.ckpt)
        else:
            logger.error(f"`models_root` (checkpoint path) not exists: {args.ckpt}")
            raise ValueError(f"`models_root` (checkpoint path) not exists: {args.ckpt}")

    # Create save folder to save the samples - This is handled by handler.py creating output_base_path
    save_path_dir = args.save_path # The handler passes a specific dir like /tmp/job_id/results
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir, exist_ok=True)
    logger.info(f"Using save path: {save_path_dir}")


    # Load models
    # rank = 0 # Not needed, no distributed setup
    vae_dtype = torch.float16 # Keep as is
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)
    args = hunyuan_video_sampler.args # Get the updated args

    if args.cpu_offload:
        from diffusers.hooks import apply_group_offloading
        onload_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        apply_group_offloading(hunyuan_video_sampler.pipeline.transformer, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=1)
        logger.info("CPU offloading enabled for transformer.")

    # Construct full paths for model components based on MODEL_BASE
    whisper_path = os.path.join(MODEL_OUTPUT_PATH, "ckpts/whisper-tiny/")
    det_align_base_dir = os.path.join(MODEL_OUTPUT_PATH, 'ckpts/det_align/')
    det_path = os.path.join(det_align_base_dir, 'detface.pt')

    if not os.path.exists(whisper_path):
        logger.warning(f"Whisper model path not found: {whisper_path}. Attempting to load by name assuming it's in cache or pre-downloaded.")
        whisper_path = "openai/whisper-tiny" # Fallback, though Dockerfile should handle this.

    if not os.path.exists(det_path):
         logger.error(f"Face detection model path not found: {det_path}")
         raise FileNotFoundError(f"Face detection model path not found: {det_path}")


    wav2vec = WhisperModel.from_pretrained(whisper_path).to(device=device, dtype=torch.float32)
    wav2vec.requires_grad_(False)
    logger.info("Whisper model loaded.")
    
    align_instance = AlignImage(str(device), det_path=det_path) # Ensure device is string
    logger.info("AlignImage instance created.")
    
    # Use the same whisper_path for feature_extractor, or a direct HF identifier
    feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_path)
    logger.info("Feature extractor loaded.")

    kwargs = {
            "text_encoder": hunyuan_video_sampler.text_encoder, 
            "text_encoder_2": hunyuan_video_sampler.text_encoder_2, 
            "feature_extractor": feature_extractor, 
        }
    video_dataset = VideoAudioTextLoaderVal(
            image_size=args.image_size,
            meta_file=args.input, # This is the CSV file path from handler
            **kwargs,
        )
    logger.info(f"VideoAudioTextLoaderVal dataset created with meta_file: {args.input}")

    # Simplified data loader for a single item
    # The CSV will contain only one data row for the serverless function
    if len(video_dataset) == 0:
        logger.error("No data found in the input CSV file.")
        raise ValueError("Input CSV file is empty or invalid.")

    # No sampler needed, just load the first (and only) item
    json_loader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0) # num_workers=0 for simplicity in serverless
    logger.info("DataLoader created.")


    # Process only the first batch (which is the only item)
    # Loop is kept for structure, but will only run once.
    for batch_index, batch in enumerate(json_loader, start=1):
        if batch_index > 1: # Should not happen with batch_size=1 and single item CSV
            logger.warning("Processing more than one batch, expected single item for serverless.")
            break

        fps = batch["fps"]
        videoid = batch['videoid'][0]
        # Audio path is already absolute path to temp file from handler
        audio_path = str(batch["audio_path"][0])

        # Save path is the directory passed by handler (e.g., /tmp/job_id/results/)
        # Output filenames will be based on this path and videoid
        output_path_raw_video = os.path.join(save_path_dir, f"{videoid}.mp4")
        output_path_final_video = os.path.join(save_path_dir, f"{videoid}_audio.mp4")

        logger.info(f"Processing videoid: {videoid}")
        logger.info(f"Input audio path: {audio_path}")
        logger.info(f"Raw video output path: {output_path_raw_video}")
        logger.info(f"Final video output path (with audio): {output_path_final_video}")


        if args.infer_min:
            # This logic was in the original script, seems to cap audio length
            # For serverless, ensure this makes sense or if actual audio length should be used
            batch["audio_len"][0] = 129
            logger.info(f"Using --infer-min, audio_len set to {batch['audio_len'][0]}")
            
        samples = hunyuan_video_sampler.predict(args, batch, wav2vec, feature_extractor, align_instance)
        logger.info("Prediction completed.")

        # samples['samples'][0] is (c, t, h, w)
        # sample = samples['samples'][0].unsqueeze(0) # Original code adds batch dim, but it's already (1, c, t, h, w) if pipeline returns it that way
        sample = samples['samples'] # Assuming pipeline returns (bs, c, t, h, w) and bs=1
        if sample.dim() == 5 and sample.shape[0] == 1: # Should be (1, C, F, H, W)
             sample = sample[0] # Remove batch dim to get (C, F, H, W) for rearrange
        elif sample.dim() != 4: # (C,F,H,W)
            logger.error(f"Unexpected sample dimensions: {sample.shape}")
            raise ValueError(f"Unexpected sample dimensions from predictor: {sample.shape}")

        # Ensure we only take up to audio_len frames
        # sample shape is (C, F, H, W) after potential unsqueeze and index
        num_frames_from_sample = sample.shape[1]
        frames_to_take = min(num_frames_from_sample, batch["audio_len"][0])
        
        sample = sample[:, :frames_to_take, :, :] # (C, audio_len, H, W)
        logger.info(f"Sampled frames trimmed to audio_len: {frames_to_take}")
        
        video = rearrange(sample, "c f h w -> f h w c") # (audio_len, H, W, C)
        video = (video.float().clamp(0,1) * 255.).data.cpu().numpy().astype(np.uint8)
        
        torch.cuda.empty_cache()
        logger.info("CUDA cache emptied.")

        # The original script had a loop for final_frames, which is not needed if video is already a numpy array
        # final_frames = []
        # for frame in video:
        #     final_frames.append(frame)
        # final_frames = np.stack(final_frames, axis=0)
        # This is equivalent to:
        final_frames = video # Already in (f, h, w, c) numpy format

        # rank == 0 check is not needed as we are not in distributed mode
        logger.info(f"Saving raw video to {output_path_raw_video} with fps {fps.item()}")
        imageio.mimsave(output_path_raw_video, final_frames, fps=fps.item(), quality=8) # Added quality
        logger.info(f"Raw video saved. Merging audio using ffmpeg.")
        
        # Ensure audio_path is valid and accessible
        if not os.path.exists(audio_path):
            logger.error(f"Audio file for ffmpeg merging not found: {audio_path}")
            # As a fallback, copy the raw video to the final video path so handler can find something
            shutil.copy(output_path_raw_video, output_path_final_video)
            logger.warning(f"Copied raw video to {output_path_final_video} due to missing audio.")
        else:
            ffmpeg_command = f"ffmpeg -i '{output_path_raw_video}' -i '{audio_path}' -c:v libx264 -c:a aac -shortest '{output_path_final_video}' -y -loglevel error"
            logger.info(f"Executing ffmpeg command: {ffmpeg_command}")
            try:
                subprocess.run(ffmpeg_command, shell=True, check=True)
                logger.info(f"FFmpeg processing complete. Final video at: {output_path_final_video}")
                if os.path.exists(output_path_raw_video):
                    os.remove(output_path_raw_video) # Clean up raw video
                    logger.info(f"Removed raw video file: {output_path_raw_video}")
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg command failed: {e}")
                # If ffmpeg fails, copy the raw video to the final path so something is returned
                if not os.path.exists(output_path_final_video):
                     shutil.copy(output_path_raw_video, output_path_final_video)
                     logger.warning(f"Copied raw video to {output_path_final_video} due to ffmpeg error.")

        logger.info(f"Processing for videoid {videoid} complete.")
        # Since this is a single run, we can break after the first item.
        break
    
    logger.info("Inference script finished.")

if __name__ == "__main__":
    main()