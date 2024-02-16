# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/HypoX64/deep3d_v1.0.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading cuda models")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

    @torch.inference_mode()
    def predict(
        self,
        video: Path = Input(description="Input video"),
        model: str = Input(description="Model size", default="deep3d_v1.0_640x360", choices=["deep3d_v1.0_640x360", "deep3d_v1.0_1280x720"]),
    ) -> Path:
        """Run a single prediction on the model"""
        output_path = "/tmp/output.mp4"
        video_path = str(video)
        model = "checkpoints/" + model + "_cuda.pt"
        # Run the inference script
        subprocess.run(["python", "inference.py", "--model", model, "--video", video_path, "--out", output_path])
        while not os.path.exists(output_path):
            time.sleep(1)
        return Path(output_path)