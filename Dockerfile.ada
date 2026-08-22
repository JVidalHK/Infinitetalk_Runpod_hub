# The base image decides which GPUs this build can run on.
#
# wlsdml1114 publishes one base per architecture — _ada, _h100, _blackwell —
# because the SageAttention inside them is compiled for a single target. A
# worker that lands on a foreign architecture does not fall back to a slower
# kernel: the first attention call raises, ComfyUI produces no output node, and
# the caller sees only "video not found". That is what an H100 run of this
# Blackwell image did, and it is the likely cause of the failures seen whenever
# a 48 GB serverless tier scheduled an Ada card instead of a Blackwell MIG
# slice.
#
# Dockerfile.ada and Dockerfile.h100 are this file with a different default
# below and nothing else; keep the three in step.
ARG BASE_IMAGE=wlsdml1114/engui_genai-base_ada:1.1
FROM ${BASE_IMAGE} as runtime

# wget 설치 (URL 다운로드를 위해)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client librosa

WORKDIR /

# Every clone below is pinned.
#
# These used to track their default branches, so two builds of the same commit
# of this repository could contain different ComfyUI and different custom
# nodes. That makes a regression impossible to attribute and a benchmark
# impossible to repeat — and repeating benchmarks is the entire point of the
# GPU probe added alongside these pins. The versions here are the ones current
# when the endpoint that produced our reference timings was built.
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    git checkout 76135e557da1ec7dcb270160f01e597565e3e003 && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    git checkout f39cbd56fecae0b27a446c0cd450cd591f3a8bea && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && \
    git checkout 6ea2651e7df66d7585f6ffee804b20e92fb38b8a && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    git checkout 3f20054214fec9f9234fd3841ae6f1e4287948f6 && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    git checkout 4ee72c065db22c9d96c2427954dc69e7b908444b && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap && \
    cd ComfyUI-wanBlockswap && \
    git checkout 5fa2ec0fa55879fe43a33e762fff91fc2c553a67

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-MelBandRoFormer && \
    cd ComfyUI-MelBandRoFormer && \
    git checkout 92c86854e6654f4aacc97484471af95c98ea16d4 && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    git checkout 088128b224242e110d3906c6750e9a3a348a659b && \
    pip install -r requirements.txt


# RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf -O /ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk_Single_Q8.gguf
# RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Multi_Q8.gguf -O /ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk_Multi_Q8.gguf
# RUN wget -q https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf -O /ComfyUI/models/diffusion_models/wan2.1-i2v-14b-480p-Q8_0.gguf

RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors -O /ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors
RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors -O /ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors
RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors -O /ComfyUI/models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors

RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors -O /ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors -O /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors    
RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors -O /ComfyUI/models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors
RUN wget -q https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors -O /ComfyUI/models/clip_vision/clip_vision_h.safetensors
RUN wget -q https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors -O /ComfyUI/models/diffusion_models/MelBandRoformer_fp16.safetensors


COPY . .
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
