import gradio as gr
import subprocess
import os

DEFAULT_DATASET = "example"
DEFAULT_OUTPUT_DIR = "output"


def run_pipeline(dataset, ball_size, prompt, negative_prompt, model_option, output_dir,
                 denoising_step, control_scale, guidance_scale, use_controlnet,
                 envmap_height, scale, ev_string, gamma, num_iteration, ball_per_iteration,
                 use_torch_compile, algorithm):

    dataset = dataset or DEFAULT_DATASET
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    inpaint_cmd = [
        "python", "inpaint.py",
        "--dataset", dataset,
        "--ball_size", str(ball_size),
        "--prompt", prompt,
        "--negative_prompt", negative_prompt,
        "--model_option", model_option,
        "--output_dir", output_dir,
        "--denoising_step", str(denoising_step),
        "--control_scale", str(control_scale),
        "--guidance_scale", str(guidance_scale),
        "--num_iteration", str(num_iteration),
        "--ball_per_iteration", str(ball_per_iteration),
        "--algorithm", algorithm
    ]
    if not use_controlnet:
        inpaint_cmd.append("--no_controlnet")
    if not use_torch_compile:
        inpaint_cmd.append("--no_torch_compile")

    print(f"Running inpaint.py with command: {' '.join(inpaint_cmd)}")  # 添加这行来打印完整命令
    subprocess.run(inpaint_cmd)

    # Run ball2envmap.py
    ball2envmap_cmd = [
        "python", "ball2envmap.py",
        "--ball_dir", os.path.join(output_dir, "square"),
        "--envmap_dir", os.path.join(output_dir, "envmap"),
        "--envmap_height", str(envmap_height),
        "--scale", str(scale)
    ]

    subprocess.run(ball2envmap_cmd)

    # Run exposure2hdr.py
    exposure2hdr_cmd = [
        "python", "exposure2hdr.py",
        "--input_dir", os.path.join(output_dir, "envmap"),
        "--output_dir", os.path.join(output_dir, "hdr"),
        "--ev_string", ev_string,
        "--gamma", str(gamma)
    ]

    subprocess.run(exposure2hdr_cmd)


    return f"Pipeline completed. Output saved in {output_dir}\n流程完成。输出保存在 {output_dir}"

iface = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Textbox(label="Dataset / 数据集", placeholder=f"Default / 默认: {DEFAULT_DATASET}", info="Input directory containing images / 包含输入图像的目录"),
        gr.Slider(minimum=64, maximum=512, step=32, label="Ball Size / 球体大小",value=256, info="Size of the reflective ball in pixels / 反射球体的像素大小"),
        gr.Textbox(label="Prompt / 提示词", value="a perfect mirrored reflective chrome ball sphere",info="Description for the ball generation / 生成球体的描述"),
        gr.Textbox(label="Negative Prompt / 负面提示词", value="matte, diffuse, flat, dull",info="Attributes to avoid in generation / 生成时要避免的属性"),
        gr.Dropdown(choices=["sd15_old", "sd15_new", "sd21", "sdxl", "sdxl_turbo"], label="Model Option / 模型选项",value="sdxl", info="Choose the Stable Diffusion model / 选择 Stable Diffusion 模型"),
        gr.Textbox(label="Output Directory / 输出目录", placeholder=f"Default / 默认: { DEFAULT_OUTPUT_DIR}", info="Directory to save output files / 保存输出文件的目录"),
        gr.Slider(minimum=10, maximum=100, step=1, label="Denoising Steps / 去噪步骤",value=30, info="Number of denoising steps in diffusion / 扩散过程中的去噪步骤数"),
        gr.Slider(minimum=0, maximum=1, step=0.1, label="Control Scale / 控制尺度",value=0.5, info="ControlNet conditioning scale / ControlNet 条件控制尺度"),
        gr.Slider(minimum=0, maximum=20, step=0.5, label="Guidance Scale / 引导尺度",value=5.0, info="Text-to-image guidance scale / 文本到图像的引导尺度"),
        gr.Checkbox(label="Use ControlNet / 使用 ControlNet", value=True,info="Enable ControlNet for better control / 启用 ControlNet 以获得更好的控制"),
        gr.Slider(minimum=64, maximum=1024, step=32, label="Envmap Height / 环境贴图高度",value=256, info="Height of the output environment map / 输出环境贴图的高度"),
        gr.Slider(minimum=1, maximum=8, step=1, label="Scale / 缩放", value=4,info="Scale factor for environment map generation / 环境贴图生成的缩放因子"),
        gr.Textbox(label="EV String / EV 字符串", value="_ev",info="String used to identify EV in filenames / 用于在文件名中标识 EV 的字符串"),
        gr.Slider(minimum=1, maximum=5, step=0.1, label="Gamma / 伽马值",value=2.4, info="Gamma correction value / 伽马校正值"),
        gr.Slider(minimum=1, maximum=10, step=1, label="Inpainting Iterations / 修复迭代次数",value=2, info="Number of inpainting iterations (default: 2) / 图像修复的迭代次数（默认：2）"),
        gr.Slider(minimum=1, maximum=100, step=1, label="Balls per Iteration / 每次迭代的球数", value=30,info="Number of balls processed in each iteration (default: 30) / 每次迭代处理的球体数量（默认：30）"),
        gr.Checkbox(label="Use Torch Compile / 使用 Torch 编译", value=True,info="Use PyTorch 2.0+ compilation for speed (if available) / 使用 PyTorch 2.0+ 编译以提高速度（如果可用）"),
        gr.Radio(["normal", "iterative"], label="Inpainting Algorithm / 修复算法", value="iterative",info="Choose between normal (single pass) or iterative algorithm (default: iterative) / 选择普通（单次）或迭代算法（默认：迭代）")
    ],
    outputs="text",
    title="HDR Image Generation Pipeline / HDR 图像生成流程",
    description="""
    Generate HDR images from a dataset using inpainting, environment mapping, and HDR conversion.
    使用图像修复、环境贴图和 HDR 转换从数据集生成 HDR 图像。
    
    Inpainting Process Explanation / 修复过程说明:
    - 'Inpainting Iterations' controls how many times the inpainting process is repeated for each image.
    - 'Balls per Iteration' determines how many balls are processed in each iteration.
    - The total number of 'rounds' depends on the number of input images and 'Balls per Iteration'.
    - If you have 100 images and 'Balls per Iteration' is 30, it will take 4 rounds to process all images (3 full rounds of 30 and 1 round of 10).
    - To process all balls in one go, set 'Balls per Iteration' to a large number (e.g., 1000) and 'Inpainting Iterations' to 1.
    
    - "修复迭代次数"控制每个图像重复修复过程的次数。
    - "每次迭代的球数"决定每次迭代处理的球体数量。
    - 总"轮数"取决于输入图像的数量和"每次迭代的球数"。
    - 如果有100张图像，"每次迭代的球数"为30，则需要4轮来处理所有图像（3轮完整的30个和1轮10个）。
    - 要一次性处理所有球体，请将"每次迭代的球数"设置为一个大数字（如1000），并将"修复迭代次数"设置为1。
    
    To speed up the process / 加快处理速度:
    1. Reduce 'Inpainting Iterations' / 减少"修复迭代次数"
    2. Increase 'Balls per Iteration' / 增加"每次迭代的球数"
    3. Reduce 'Denoising Steps' / 减少"去噪步骤"
    4. Disable ControlNet / 禁用 ControlNet
    5. Enable Torch Compile if you have PyTorch 2.0+ / 如果您有 PyTorch 2.0+，启用 Torch 编译
    6. Use 'normal' algorithm instead of 'iterative' / 使用"normal"算法而不是"iterative"
    """
)

iface.launch()
