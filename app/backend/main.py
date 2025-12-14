"""
FastAPI backend server for running scene generation sampling.
"""
import asyncio
import os
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Scene Generation API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the project root directory (parent of app folder)
PROJECT_ROOT = Path(__file__).parent.parent.parent


class SamplingRequest(BaseModel):
    """Request model for sampling scenes."""
    num_scenes: int = Field(default=10, ge=1, le=1000, description="Number of scenes to generate")
    existing_output_dir: Optional[str] = Field(default=None, description="Path to existing output directory to skip sampling")
    load: str = Field(default="rrudae6n", description="Model checkpoint to load")
    dataset: str = Field(default="custom_scene", description="Dataset name")
    dataset_processed_scene_data_path: str = Field(
        default="data/metadatas/custom_scene_metadata.json",
        description="Path to processed scene data"
    )
    dataset_max_num_objects_per_scene: int = Field(
        default=12,
        description="Maximum number of objects per scene"
    )
    algorithm: str = Field(default="scene_diffuser_midiffusion", description="Algorithm to use")
    algorithm_trainer: str = Field(default="ddpm", description="Trainer type")
    experiment_find_unused_parameters: bool = Field(default=True, description="Find unused parameters")
    algorithm_classifier_free_guidance_use: bool = Field(default=False, description="Use classifier-free guidance")
    algorithm_classifier_free_guidance_use_floor: bool = Field(default=True, description="Use floor in guidance")
    algorithm_classifier_free_guidance_weight: int = Field(default=1, description="Guidance weight")
    algorithm_custom_loss: bool = Field(default=True, description="Use custom loss")
    algorithm_ema_use: bool = Field(default=True, description="Use EMA")
    algorithm_noise_schedule_scheduler: str = Field(default="ddim", description="Noise schedule scheduler")
    algorithm_noise_schedule_ddim_num_inference_timesteps: int = Field(
        default=150,
        description="Number of inference timesteps"
    )


class SamplingResponse(BaseModel):
    """Response model for sampling request."""
    status: str
    message: str
    task_id: Optional[str] = None
    output_dir: Optional[str] = None


class TaskStatus(BaseModel):
    """Task status model."""
    task_id: str
    status: str  # "running", "completed", "failed"
    message: str
    output_dir: Optional[str] = None


# In-memory task storage (in production, use Redis or a database)
tasks: dict[str, TaskStatus] = {}


def build_sampling_command(request: SamplingRequest) -> list[str]:
    """Build the command to run sampling.py with the given parameters."""
    # Use custom_sample_and_render.py from scripts directory
    # Run from project root with PYTHONPATH=. (as per user's exact command)
    sampling_script = PROJECT_ROOT / "scripts" / "custom_sample_and_render.py"
    
    # Ensure the script exists
    if not sampling_script.exists():
        raise FileNotFoundError(f"Sampling script not found at {sampling_script}")
    
    # Process path - convert absolute to relative if needed, or keep relative
    processed_path = request.dataset_processed_scene_data_path
    if os.path.isabs(processed_path):
        # If absolute, try to convert to relative path from project root
        try:
            processed_path = os.path.relpath(processed_path, PROJECT_ROOT)
        except ValueError:
            # If paths are on different drives, keep absolute
            pass
    
    # Build command exactly as user specified - run from project root
    # Order matches user's command exactly
    cmd = [
        "python",
        "-u",  # Unbuffered output
        "scripts/custom_sample_and_render.py",
        f"+num_scenes={request.num_scenes}",
        f"load={request.load}",
        f"dataset={request.dataset}",
        f"dataset.processed_scene_data_path={processed_path}",
        f"dataset.max_num_objects_per_scene={request.dataset_max_num_objects_per_scene}",
        f"algorithm={request.algorithm}",
        f"algorithm.classifier_free_guidance.use={str(request.algorithm_classifier_free_guidance_use).lower()}",
        f"algorithm.ema.use={str(request.algorithm_ema_use).lower()}",
        f"algorithm.trainer={request.algorithm_trainer}",
        "experiment.validation.limit_batch=1",
        "experiment.validation.val_every_n_step=50",
        f"experiment.find_unused_parameters={str(request.experiment_find_unused_parameters).lower()}",
        f"algorithm.custom.loss={str(request.algorithm_custom_loss).lower()}",
        "algorithm.validation.num_samples_to_render=0",
        "algorithm.validation.num_samples_to_visualize=0",
        "algorithm.validation.num_directives_to_generate=0",
        "algorithm.test.num_samples_to_render=0",
        "algorithm.test.num_samples_to_visualize=0",
        "algorithm.test.num_directives_to_generate=0",
        "algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0",
        f"algorithm.classifier_free_guidance.use_floor={str(request.algorithm_classifier_free_guidance_use_floor).lower()}",
        "algorithm.custom.old=False",  # User specified False
        "dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm",
        f"algorithm.noise_schedule.scheduler={request.algorithm_noise_schedule_scheduler}",
        f"algorithm.noise_schedule.ddim.num_inference_timesteps={request.algorithm_noise_schedule_ddim_num_inference_timesteps}",
        "wandb.mode=disabled",
    ]
    return cmd


async def run_sampling_task(request: SamplingRequest, task_id: str):
    """Run the sampling task asynchronously."""
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="running",
        message="Starting sampling process...",
    )
    
    try:
        # Check if using existing output directory
        if request.existing_output_dir:
            output_dir = Path(request.existing_output_dir)
            
            # Handle relative paths
            if not output_dir.is_absolute():
                output_dir = PROJECT_ROOT / output_dir
            
            if not output_dir.exists():
                tasks[task_id].status = "failed"
                tasks[task_id].message = f"Error: Output directory does not exist: {output_dir}"
                return
            
            # Check if directory has PNG and GLB files
            png_files = list(output_dir.glob("*.png"))
            glb_files = list(output_dir.glob("*.glb"))
            
            if not png_files:
                tasks[task_id].status = "failed"
                tasks[task_id].message = f"Error: No PNG files found in {output_dir}"
                return
            
            if not glb_files:
                tasks[task_id].status = "failed"
                tasks[task_id].message = f"Error: No GLB files found in {output_dir}"
                return
            
            # Successfully using existing output
            tasks[task_id].status = "completed"
            tasks[task_id].output_dir = str(output_dir)
            tasks[task_id].message = f"Using existing output directory: {output_dir}\n\nFound {len(png_files)} PNG files and {len(glb_files)} GLB files.\n\nSkipped sampling and rendering."
            return
        
        try:
            cmd = build_sampling_command(request)
        except FileNotFoundError as e:
            tasks[task_id].status = "failed"
            tasks[task_id].message = f"Error: {str(e)}"
            return
        
        # Set up environment - exactly as user specified
        env = os.environ.copy()
        env["PYTHONPATH"] = "."  # User specified PYTHONPATH=.
        env["HYDRA_FULL_ERROR"] = "1"  # Enable full Hydra error messages
        
        # Check if .venv exists and use its python
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        python_cmd = "python"
        if venv_python.exists():
            python_cmd = str(venv_python)
            cmd[0] = python_cmd
        
        tasks[task_id].message = f"Running command: {' '.join(cmd)}"
        
        # Run from project root (as user specified) - not from scripts/
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),  # Run from project root
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # Read output in real-time
        stdout_lines = []
        stderr_lines = []
        
        async def read_stream(stream, lines_list):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded_line = line.decode('utf-8', errors='ignore')
                lines_list.append(decoded_line)
                # Update status with last few lines
                if len(lines_list) > 10:
                    tasks[task_id].message = "\n".join(lines_list[-5:])
        
        await asyncio.gather(
            read_stream(process.stdout, stdout_lines),
            read_stream(process.stderr, stderr_lines),
        )
        
        await process.wait()
        
        if process.returncode == 0:
            # Find the output directory (Hydra creates output directories)
            # Look for the most recent output directory
            outputs_dir = PROJECT_ROOT / "outputs"
            if outputs_dir.exists():
                # Get the most recent output directory
                output_dirs = sorted(
                    outputs_dir.glob("*/*"),
                    key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True
                )
                if output_dirs:
                    output_dir = Path(output_dirs[0])
                    tasks[task_id].output_dir = str(output_dir)
                    
                    # Run render script after sampling completes
                    pkl_path = output_dir / "sampled_scenes_results.pkl"
                    if pkl_path.exists():
                        tasks[task_id].message = f"Sampling completed! Starting rendering...\n\nOutput:\n{''.join(stdout_lines[-20:])}"
                        
                        # Run render script (3D version that generates PNG and GLB)
                        render_cmd = [
                            "python",
                            "../ThreedFront/scripts/render_results_3d.py",
                            str(pkl_path),
                            "--retrieve_by_size",
                        ]
                        
                        # Use venv python if available
                        render_cmd[0] = python_cmd
                        
                        render_process = await asyncio.create_subprocess_exec(
                            *render_cmd,
                            cwd=str(PROJECT_ROOT),
                            env=env,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                        )
                        
                        render_stdout_lines = []
                        async def read_render_stream(stream):
                            while True:
                                line = await stream.readline()
                                if not line:
                                    break
                                decoded_line = line.decode('utf-8', errors='ignore')
                                render_stdout_lines.append(decoded_line)
                                # Update status with render progress
                                if len(render_stdout_lines) > 10:
                                    tasks[task_id].message = f"Rendering...\n{''.join(render_stdout_lines[-5:])}"
                        
                        await read_render_stream(render_process.stdout)
                        await render_process.wait()
                        
                        if render_process.returncode == 0:
                            tasks[task_id].status = "completed"
                            tasks[task_id].message = f"Sampling and rendering completed successfully!\n\nSampling Output:\n{''.join(stdout_lines[-20:])}\n\nRendering Output:\n{''.join(render_stdout_lines[-10:])}"
                        else:
                            tasks[task_id].status = "completed"  # Sampling succeeded, rendering may have failed
                            tasks[task_id].message = f"Sampling completed but rendering had issues.\n\nSampling Output:\n{''.join(stdout_lines[-20:])}\n\nRendering Output:\n{''.join(render_stdout_lines[-10:])}"
                    else:
                        tasks[task_id].status = "completed"
                        tasks[task_id].message = f"Sampling completed successfully!\n\nOutput:\n{''.join(stdout_lines[-20:])}\n\nNote: sampled_scenes_results.pkl not found, skipping rendering."
                else:
                    output_dir = None
                    tasks[task_id].status = "completed"
                    tasks[task_id].message = f"Sampling completed successfully!\n\nOutput:\n{''.join(stdout_lines[-20:])}"
            else:
                output_dir = None
                tasks[task_id].status = "completed"
                tasks[task_id].message = f"Sampling completed successfully!\n\nOutput:\n{''.join(stdout_lines[-20:])}"
        else:
            tasks[task_id].status = "failed"
            error_output = ''.join(stderr_lines[-20:]) if stderr_lines else ''.join(stdout_lines[-20:])
            tasks[task_id].message = f"Sampling failed with return code {process.returncode}\n\nError:\n{error_output}"
    
    except Exception as e:
        tasks[task_id].status = "failed"
        tasks[task_id].message = f"Error running sampling: {str(e)}"


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Scene Generation API", "version": "1.0.0"}


@app.post("/api/sampling/run", response_model=SamplingResponse)
async def run_sampling(request: SamplingRequest):
    """Run the sampling process with the given parameters."""
    import uuid
    task_id = str(uuid.uuid4())
    
    # Start the task asynchronously
    asyncio.create_task(run_sampling_task(request, task_id))
    
    return SamplingResponse(
        status="started",
        message="Sampling task started",
        task_id=task_id,
    )


@app.get("/api/sampling/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a sampling task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.get("/api/sampling/tasks")
async def list_tasks():
    """List all tasks."""
    return {"tasks": list(tasks.values())}


@app.get("/api/images/{task_id}")
async def get_images(task_id: str, limit: int = 50):
    """Get list of rendered images for a task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if not task.output_dir:
        return {"images": [], "count": 0}
    
    output_dir = Path(task.output_dir)
    if not output_dir.exists():
        return {"images": [], "count": 0}
    
    # Find all PNG images in the output directory
    image_files = sorted(
        output_dir.glob("*.png"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True
    )
    
    # Limit the number of images
    image_files = image_files[:limit]
    
    # Return relative paths that can be used to serve the images
    images = [f"/api/images/{task_id}/file/{img.name}" for img in image_files]
    
    return {"images": images, "count": len(image_files), "total": len(list(output_dir.glob("*.png")))}


@app.get("/api/images/{task_id}/file/{filename}")
async def get_image_file(task_id: str, filename: str):
    """Serve an image file."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if not task.output_dir:
        raise HTTPException(status_code=404, detail="Output directory not found")
    
    output_dir = Path(task.output_dir)
    image_path = output_dir / filename
    
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(str(image_path), media_type="image/png")


@app.get("/api/models/{task_id}/file/{filename}")
async def get_model_file(task_id: str, filename: str):
    """Serve a GLB model file."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if not task.output_dir:
        raise HTTPException(status_code=404, detail="Output directory not found")
    
    output_dir = Path(task.output_dir)
    # Replace .png with .glb if filename ends with .png
    if filename.endswith('.png'):
        filename = filename[:-4] + '.glb'
    elif not filename.endswith('.glb'):
        filename = filename + '.glb'
    
    model_path = output_dir / filename
    
    if not model_path.exists() or not model_path.is_file():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(str(model_path), media_type="model/gltf-binary")

