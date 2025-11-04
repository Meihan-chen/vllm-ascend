# Multi-NPU (Qwen3-Omni-30B-A3B-Thinking)

## Run vllm-ascend on Multi-NPU with Qwen3-Omni-30B-A3B-Thinking

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```

Set up environment variables:

```bash
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

Install system dependencies：

```bash
# If you already have transformers installed, please update transformers version >= 4.57.0.dev0
# pip install transformers -U 
pip install qwen_vl_utils --extra-index-url https://download.pytorch.org/whl/cpu/
```

### Offline Inference on Multi-NPU

Run the following script to execute offline inference on multi-NPU:

```python
import gc
import torch
import os
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel
)
from modelscope import Qwen3OmniMoeProcessor  
from qwen_omni_utils import process_mm_info  


def clean_up():
    """Clean up distributed resources and NPU memory"""
    destroy_model_parallel()  
    destroy_distributed_environment() 
    gc.collect()  # Garbage collection to free up memory
    torch.npu.empty_cache() 


def main():
    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2, 
        distributed_executor_backend="mp",  
        limit_mm_per_prompt={'image': 5, 'video': 2, 'audio': 3},  
        max_model_len=32768, 
    )
    
    sampling_params = SamplingParams(
        temperature=0.6, 
        top_p=0.95,
        top_k=20,
        max_tokens=16384, 
    )
    
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
                {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},
                {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"},
                {"type": "text", "text": "Analyze this audio, image, and video together."}
            ]
        }
    ]
  
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True 
    )
    audios, images, videos = process_mm_info(messages)
    
    inputs = {
        "prompt": text,
        "multi_modal_data": {},
        "mm_processor_kwargs": {"use_audio_in_video": False}  
    }
    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios
 
    outputs = llm.generate([inputs], sampling_params=sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    del llm
    clean_up() 


if __name__ == "__main__":
    main()
```

If you run this script successfully, you can see the info shown below:

```bash
Prompt: '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|audio_start|><|audio_pad|><|audio_end|><|vision_start|><|video_pad|><|vision_end|>Analyze this audio, image, and video together.<|im_end|>\n<|im_start|>assistant\n', Generated text: '<think>\nGot it, let\'s tackle this problem. First, I need to analyze the audio, image, and video together. Let\'s break down each component.\n\nStarting with the audio: there\'s a sound of someone coughing. That\'s a human physiological sound, maybe indicating a person is present or has been there recently.\n\nNext, the image: it\'s a collage of four luxury cars. Let\'s identify them. Top left: Rolls-Royce (white, signature grille, "R" badge). Top right: Mercedes-Benz GLE SUV (dark gray, desert background with dust, reflecting on wet ground). Bottom left: Ferrari Portofino M (red convertible, Ferrari logo, "Portofino M" badge). Bottom right: Porsche 911 (white, classic sports car design). So the image is showcasing high-end automotive brands, emphasizing luxury, performance, and different vehicle types (sedan, SUV, convertible, sports coupe).\n\nThen the video: it\'s a person using a stylus on a tablet to draw a guitar. The sequence shows the drawing process—starting with the guitar shape, then adding details like the strings, body, and neck. The hands are visible, one holding the tablet, the other drawing. The background is a wooden table.\n\nNow, connecting all three. Let\'s think about possible links. The audio (coughing) might be from the person in the video, but the image is cars. Maybe the context is about different interests: the person is drawing a guitar (art/music), while the image shows cars (automotive passion). The coughing could be a natural action during the drawing process.\n\nWait, the user wants analysis of all three together. Let\'s check if there\'s a common theme. Maybe the person in the video is a creative individual who also has an interest in luxury cars (as shown in the image), and the coughing is just a momentary action. But how do they connect?\n\nAlternatively, maybe the image is part of the context for the video. For example, the person might be drawing a guitar but has a passion for cars, so the image is a reference. But the coughing is unrelated? Or maybe the coughing is from the person while they\'re drawing, indicating they\'re in the middle of creating art, and the image of cars is a separate element, but the task is to analyze all three.\n\nLet\'s structure the analysis:\n\n1. Audio: Coughing sound—suggests human presence, possibly the person in the video, indicating a real-world scenario where someone is engaged in an activity (drawing) and has a minor physical action (coughing).\n\n2. Image: Four luxury cars—high-end brands, each representing different segments (luxury sedan, SUV, convertible sports car, sports coupe). This showcases automotive design, engineering, and brand prestige. The cars are presented in a clean, professional manner, typical of marketing or enthusiast content.\n\n3. Video: Digital drawing of a guitar using a stylus on a tablet—demonstrates digital art creation, using modern technology (tablet, stylus). The process shows the evolution from basic outline to detailed drawing, highlighting creativity and skill.\n\nNow, connecting them: The audio (cough) could be from the artist while working on the drawing, placing the video in a real-world context. The image of luxury cars might be a personal interest or inspiration for the artist, though not directly related to the guitar drawing. Alternatively, the image could be part of a broader theme of "passions"—the artist\'s passion for drawing, and the cars representing another passion (automotive). The coughing adds a human element, making the scene relatable (even minor actions like coughing happen during creative work).\n\nAnother angle: the image and video are both examples of visual media (car photography and digital art), while the audio is a sound element that adds context to the video\'s setting. The coughing might indicate the artist is in a physical space, perhaps in a room with the car image as a background or reference, but that\'s speculative.\n\nWait, the problem says "Analyze this audio, image, and video together." So need to find connections or what they collectively represent.\n\nPossible points:\n\n- Human activity: The coughing (audio) is part of the person\'s activity in the video (drawing), so the audio is a sound from the video\'s context. The image of cars might be unrelated, but maybe the user is showing different hobbies: drawing and cars.\n\n- Media types: Audio (sound), image (static visual), video (dynamic visual). Together, they represent different forms of media that can coexist in a person\'s life—listening to music (though no music here), drawing, and admiring cars.\n\n- Contextual elements: The coughing suggests the video is a real-life recording (not a staged animation), making the drawing process authentic. The car image might be a separate interest, showing the person\'s diverse interests (art and automotive).\n\n- Technical aspects: The video uses digital tools (tablet, stylus), which is modern; the car image is high-quality photography; the audio is a natural sound effect.\n\nSo putting it all together: The audio captures a minor human action during the video\'s creative process, the image displays a passion for luxury automobiles, and the video demonstrates digital artistic creation. Together, they illustrate a multifaceted individual engaging in both creative (drawing) and enthusiast (cars) pursuits, with the audio adding realism to the video\'s scene.\n</think>\n\nTo analyze the audio, image, and video together, we examine their **content, context, and interconnections**:  \n\n\n### 1. Audio: Coughing Sound  \n- The audio features a brief, natural human cough. This suggests **real-world presence**—a minor, relatable action that grounds the scene in reality (e.g., the person in the video might be coughing while drawing). It adds authenticity to the video’s context, implying the activity is happening in a physical space rather than a sterile or staged environment.  \n\n\n### 2. Image: Luxury Car Collage  \n- The image showcases four high-end vehicles:  \n  - *Top-left*: Rolls-Royce (luxury sedan, symbolizing opulence and heritage).  \n  - *Top-right*: Mercedes-Benz GLE SUV (off-road-ready, blending luxury with adventure).  \n  - *Bottom-left*: Ferrari Portofino M (convertible sports car, emphasizing speed and style).  \n  - *Bottom-right*: Porsche 911 (iconic sports coupe, representing precision engineering).  \n- This collage reflects **passion for automotive design**, highlighting brand prestige, engineering excellence, and diverse vehicle segments (sedan, SUV, convertible, sports coupe). It likely represents a personal interest or aspirational hobby.  \n\n\n### 3. Video: Digital Guitar Drawing  \n- The video depicts a person using a stylus on a tablet to draw a guitar. The sequence shows:  \n  - Initial outlining of the guitar’s shape.  \n  - Gradual addition of details (strings, body, neck).  \n  - A focus on *creative process* and *digital artistry* (modern tools like tablets/styluses for artistic expression).  \n- The setting (wooden table, hands interacting with technology) emphasizes **hands-on creativity** and the accessibility of digital art.  \n\n\n### Interconnections & Broader Interpretation  \n- **Human Experience**: The cough (audio) bridges the *real-world* context of the video (drawing) with the *imagined* context of the car image. It implies the person in the video is a multi-faceted individual: engaged in creative work (drawing) while also having interests in luxury automobiles (the image).  \n- **Media Diversity**: The three elements represent distinct forms of media:  \n  - *Audio*: A natural sound effect (human physiology).  \n  - *Image*: Static visual marketing/enthusiast content (cars).  \n  - *Video*: Dynamic visual storytelling (art creation).  \n  Together, they illustrate how people engage with multiple forms of media and hobbies in daily life.  \n- **Thematic Unity**: Both the video (art) and the image (cars) reflect **passion-driven pursuits**—the video shows the *process* of creating art, while the image showcases the *object* of admiration (luxury cars). The coughing sound adds a layer of *authenticity* to the creative act, reminding us that even in moments of focus, humans experience mundane, relatable physical actions.  \n\n\nIn summary, the audio, image, and video collectively depict a **real, multifaceted human experience**: someone engaging in creative digital art while also valuing luxury automobiles, with the coughing sound grounding the scene in the messy, relatable reality of daily life.'
```

### Online Inference on Multi-NPU

Run the following script to start the vLLM server on Multi-NPU:

For an Atlas A2 with 64 GB of NPU card memory, tensor-parallel-size should be at least 1, and for 32 GB of memory, tensor-parallel-size should be at least 2.

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking --tensor-parallel-size 2
```

Once your server is started, you can query the model with input prompts.

```bash
curl http://localhost:8000/v1/chat/completions \
-X POST \
-H "Content-Type: application/json" \
-d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
                    }
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"
                    }
                },
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
                    }

                },
                {
                    "type": "text",
                    "text":  "Analyze this audio, image, and video together."
                }
            ]
        }
    ]
}'
```

If you run this script successfully, you can see the info shown below:

```bash
{"id":"chatcmpl-b74594a81b3a417ebe50a3d05a9aa415","object":"chat.completion","created":1762245267,"model":"Qwen/Qwen3-Omni-30B-A3B-Thinking","choices":[{"index":0,"message":{"role":"assistant","content":"<think>\nGot it, let's try to analyze these different elements together. First, let's list out what we have: the initial image shows four luxury cars (Rolls-Royce, Mercedes GLE, Ferrari Portofino, Porsche 911), then there's an audio of someone coughing, and then a video of someone drawing a guitar on a tablet.\n\nHmm, the task is to analyze them together. Maybe look for connections or themes. Let's break down each part.\n\nFirst, the car images: they're all high-end, luxury vehicles, representing different brands and types (sedan, SUV, convertible, sports car). They emphasize status, design, and performance. Then the audio is a cough—maybe indicating a human element, health, or interruption. The video is about digital art creation, using a tablet and stylus, drawing a guitar.\n\nWait, maybe the connection is about different forms of creativity or expression? Cars are a form of industrial design, art (drawing the guitar is artistic), and the cough might be a human moment in the process. Or maybe it's about different mediums: visual (cars), auditory (cough), and digital art (tablet drawing).\n\nLet's check each component:\n\n1. Car images: Visual, luxury, engineering, design.\n2. Audio: Human sound (cough), which is a non-verbal communication, maybe indicating the person is present, maybe tired or sick, but also part of the environment.\n3. Video: Digital art, creativity, technology (tablet), music (guitar).\n\nPossible themes: The intersection of technology and creativity (cars as tech, tablet as tech), human elements (cough as human, drawing as human activity), or different forms of expression (cars as design, guitar as art).\n\nWait, the user might want to see if there's a narrative or connection. Maybe the cough is a moment when someone is creating art, like taking a break or dealing with a minor issue while working. The cars are a separate visual, but maybe the overall theme is about luxury, creativity, and human experience.\n\nAlternatively, maybe it's a test of attention to different media types. Let's think about each:\n\n- Images: Visual art (cars as products of design)\n- Audio: Sound (cough, which is a natural human sound)\n- Video: Dynamic visual (drawing process)\n\nSo the analysis could be about how different media (still images, sound, moving video) convey different aspects of human experience—luxury products, health/physical state, creative process.\n\nLet's structure it:\n\n1. Visual Elements (Car Images): These showcase high-end automotive design, emphasizing brand identity, engineering, and aesthetic appeal. Each car represents a different segment (luxury sedan, premium SUV, exotic convertible, classic sports car), highlighting diversity in luxury transportation.\n\n2. Auditory Element (Cough): This introduces a human, organic element. Coughing is a natural bodily function, indicating presence and physicality, contrasting with the polished, manufactured nature of the cars. It might symbolize the human element behind the creation or consumption of these luxury items.\n\n3. Video (Guitar Drawing): This demonstrates digital creativity and artistic expression. Using a tablet and stylus for illustration shows the intersection of technology and art, with the guitar as a symbol of music and creativity. The act of drawing connects to the design elements seen in the cars (both involve creative design processes).\n\nOverall, the combination explores the relationship between human experience (health, creativity) and the products of human ingenuity (luxury cars, digital art tools). The cough bridges the gap between the sterile, high-end visuals of the cars and the tactile, personal act of drawing, emphasizing that even in the realm of luxury and technology, human elements remain central.\n\nWait, but maybe the cough is just a random sound, and the key is to see the three parts as separate but part of a larger context. Alternatively, maybe the user is testing if we can recognize that these are unrelated but need to describe each.\n\nBut the instruction says \"Analyze this audio, image, and video together,\" so find connections.\n\nLet me check again:\n\n- Image: Four luxury cars (Rolls-Royce Ghost, Mercedes GLE, Ferrari Portofino M, Porsche 911)\n- Audio: Coughing (a single person coughing)\n- Video: Drawing a guitar on a tablet\n\nPossible connections:\n\n- All involve human interaction: cars are driven by humans, coughing is a human action, drawing is a human creative act.\n- Technology: Cars use advanced tech, tablet is tech, cough is biological tech (body).\n- Creativity: Car design is creative engineering, drawing is artistic creativity.\n\nSo the analysis could be that these elements collectively represent different facets of human creativity and interaction with technology—whether through automotive design, digital art creation, or the natural human body. The cough adds a layer of organic humanity to the otherwise polished and engineered worlds of luxury cars and digital tools.\n\nYes, that makes sense. Let's put it all together.\n</think>\n\nTo analyze the **image of luxury cars**, **audio of coughing**, and **video of digital guitar drawing** together, we explore their interplay across themes of **human experience**, **technology**, and **creative expression**:  \n\n\n### 1. Visual: Luxury Automotive Design (Image)  \nThe four cars—Rolls-Royce Ghost, Mercedes GLE SUV, Ferrari Portofino M, and Porsche 911—represent **high-end engineering, brand identity, and aesthetic innovation**. Each vehicle embodies a distinct segment of luxury mobility:  \n- *Rolls-Royce*: Ultra-luxury sedan (status, craftsmanship).  \n- *Mercedes GLE*: Premium SUV (versatility, off-road capability).  \n- *Ferrari Portofino*: Exotic convertible (performance, open-top thrill).  \n- *Porsche 911*: Iconic sports car (precision, heritage).  \n\nThese images highlight **human-driven design**—a blend of artistry, technology, and cultural aspiration. They symbolize the *polished, manufactured world* of consumer goods, where creativity is channeled into engineering and marketing.  \n\n\n### 2. Auditory: Human Organic Presence (Audio)  \nThe coughing sound introduces a **raw, biological human element**. Unlike the sleek, controlled visuals of the cars, this sound is unfiltered and spontaneous—it speaks to *physicality, vulnerability, and everyday reality*.  \n\nThis contrasts with the “perfect” world of luxury products, reminding us that **human experience (health, fatigue, imperfection) underpins even the most refined creations**. The cough bridges the gap between the *manufactured* (cars) and the *organic* (the human body).  \n\n\n### 3. Video: Digital Creativity (Drawing a Guitar)  \nThe video shows a person using a tablet and stylus to draw a guitar—an act of **artistic creation rooted in technology**. This process mirrors the *design ethos* of the luxury cars:  \n- Both involve **iterative creativity**: sketching curves, refining details, and balancing form/function.  \n- Both rely on **digital tools**: car designers use CAD software; the artist uses a tablet.  \n- The guitar itself symbolizes *music and emotional expression*, adding a layer of **cultural and emotional depth** to the “technology” theme.  \n\n\n### Unified Analysis: Humanity at the Intersection of Design, Technology, and Expression  \nTogether, these elements illustrate how **human creativity and experience shape (and are shaped by) technology and luxury**:  \n- **Luxury cars** are products of human ingenuity but exist within a *human context* (e.g., the cough reminds us of the people who drive, design, and consume them).  \n- **Digital art** (guitar drawing) reflects how technology democratizes creativity, while the *act of drawing*—like driving a car—requires human skill and emotion.  \n- **The cough** acts as a “human anchor,” grounding the polished visuals of luxury and art in the messy, real-world reality of human existence.  \n\nIn essence, the combination underscores that **even in realms of high technology and luxury, humanity remains central**: whether through the imperfections of the body, the passion of creation, or the desire to own/express identity through design.","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":17625,"total_tokens":19340,"completion_tokens":1715,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

::::
:::::
