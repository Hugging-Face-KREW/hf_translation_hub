<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 멀티모달 채팅 템플릿 [[multimodal-chat-templates]]

멀티모달 채팅 모델은 텍스트 외에도 이미지, 오디오 또는 비디오와 같은 입력을 받습니다. 멀티모달 채팅 기록의 `content` 키는 서로 다른 타입의 여러 항목을 포함하는 리스트입니다. 이는 `content` 키가 단일 문자열인 텍스트 전용 채팅 모델과는 다릅니다.

[토크나이저](./fast_tokenizer) 클래스가 텍스트 전용 모델의 채팅 템플릿과 토큰화를 처리하는 것과 같은 방식으로,
[Processor](./processors) 클래스는 멀티모달 모델의 전처리, 토큰화 및 채팅 템플릿을 처리합니다. 이들의 [`~ProcessorMixin.apply_chat_template`] 메소드는 거의 동일합니다.

이 가이드는 고수준 [`ImageTextToTextPipeline`]과 [`~ProcessorMixin.apply_chat_template`] 및 [`~GenerationMixin.generate`] 메소드를 사용한 저수준에서 멀티모달 모델과 채팅하는 방법을 보여줍니다.

## ImageTextToTextPipeline [[imagetexttotextpipeline]]

[`ImageTextToTextPipeline`]은 "채팅 모드"가 있는 고수준 이미지 및 텍스트 생성 클래스입니다. 채팅 모드는 대화형 모델이 감지되고 채팅 프롬프트가 [적절히 형식화된](./llm_tutorial#wrong-prompt-format) 경우 활성화됩니다.

채팅 기록의 `content` 키에 이미지와 텍스트 블록을 추가합니다.

```py
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "당신은 항상 해적 스타일로 응답하는 친근한 채팅봇입니다"}],
    },
    {
      "role": "user",
      "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "이것들은 무엇인가요?"},
        ],
    },
]
```

[`ImageTextToTextPipeline`]을 생성하고 채팅을 전달합니다. 대규모 모델의 경우 [device_map="auto"](./models#big-model-inference) 설정이 모델을 더 빠르게 로드하고 사용 가능한 가장 빠른 장치에 자동으로 배치하는 데 도움이 됩니다. 데이터 타입을 [auto](./models#model-data-type)로 설정하는 것도 메모리를 절약하고 속도를 개선하는 데 도움이 됩니다.

```python
import torch
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="Qwen/Qwen2.5-VL-3B-Instruct", device_map="auto", dtype="auto")
out = pipe(text=messages, max_new_tokens=128)
print(out[0]['generated_text'][-1]['content'])
```

```text
Ahoy, me hearty! These be two feline friends, likely some tabby cats, taking a siesta on a cozy pink blanket. They're resting near remote controls, perhaps after watching some TV or just enjoying some quiet time together. Cats sure know how to find comfort and relaxation, don't they?
```

해적 말투에서 현대 미국 영어로 점차 변화하는 것을 제외하면 (결국 3B 모델일 뿐입니다), 이는 정확합니다!

## `apply_chat_template` 사용하기 [[using-applychattemplate]]

[텍스트 전용 모델](./chat_templating)과 마찬가지로 [`~ProcessorMixin.apply_chat_template`] 메소드를 사용하여 멀티모달 모델을 위한 채팅 메시지를 준비합니다.
이 메소드는 이미지와 기타 미디어 타입을 포함하여 채팅 메시지의 토큰화와 형식화를 처리합니다. 결과 입력은 생성을 위해 모델에 전달됩니다.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", device_map="auto", torch_dtype="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

messages = [
    {
      "role": "system",
      "content": [{"type": "text", "text": "당신은 항상 해적 스타일로 응답하는 친근한 채팅봇입니다"}],
    },
    {
      "role": "user",
      "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "이것들은 무엇인가요?"},
        ],
    },
]
```

입력 콘텐츠를 토큰화하기 위해 `messages`를 [`~ProcessorMixin.apply_chat_template`]에 전달합니다. 텍스트 모델과 달리 `apply_chat_template`의 출력은
토큰화된 텍스트 외에도 전처리된 이미지 데이터가 포함된 `pixel_values` 키를 포함합니다.

```py
processed_chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
print(list(processed_chat.keys()))
```

```text
['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']
```

이러한 입력을 [`~GenerationMixin.generate`]에 전달합니다.

```python
out = model.generate(**processed_chat.to(model.device), max_new_tokens=128)
print(processor.decode(out[0]))
```

디코딩된 출력에는 사용자 메시지와 이미지 정보를 포함하는 플레이스홀더 토큰을 포함한 지금까지의 전체 대화가 포함됩니다. 사용자에게 표시하기 전에 출력에서 이전 대화를 잘라내야 할 수도 있습니다.

## 비디오 입력 [[video-inputs]]

일부 비전 모델은 비디오 입력도 지원합니다. 메시지 형식은 [이미지 입력](#image-inputs) 형식과 매우 유사합니다.

- 콘텐츠가 비디오임을 나타내려면 콘텐츠 `"type"`이 `"video"`여야 합니다.
- 비디오의 경우 비디오에 대한 링크(`"url"`)이거나 파일 경로(`"path"`)일 수 있습니다. URL에서 로드된 비디오는 [PyAV](https://pyav.basswood-io.com/docs/stable/) 또는 [Decord](https://github.com/dmlc/decord)로만 디코딩할 수 있습니다.
- URL 또는 파일 경로에서 비디오를 로드하는 것 외에도 디코딩된 비디오 데이터를 직접 전달할 수도 있습니다. 이는 메모리의 다른 곳에서 이미 비디오 프레임을 전처리하거나 디코딩한 경우 (예: OpenCV, decord 또는 torchvision 사용) 유용합니다. 파일에 저장하거나 URL에 저장할 필요가 없습니다.

> [!WARNING]
> `"url"`에서 비디오를 로드하는 것은 PyAV 또는 Decord 백엔드에서만 지원됩니다.

```python
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
      "role": "system",
      "content": [{"type": "text", "text": "당신은 항상 해적 스타일로 응답하는 친근한 채팅봇입니다"}],
    },
    {
      "role": "user",
      "content": [
            {"type": "video", "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4"},
            {"type": "text", "text": "이 비디오에서 무엇을 보시나요?"},
        ],
    },
]
```

### 예시: 디코딩된 비디오 객체 전달하기 [[example-passing-decoded-video-objects]]

```python
import numpy as np

video_object1 = np.random.randint(0, 255, size=(16, 224, 224, 3), dtype=np.uint8),

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "당신은 항상 해적 스타일로 응답하는 친근한 채팅봇입니다"}],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_object1},
            {"type": "text", "text": "이 비디오에서 무엇을 보시나요?"}
        ],
    },
]
```

기존의 (`"load_video()"`) 함수를 사용하여 비디오를 로드하고, 메모리에서 비디오를 편집한 후 메시지에 전달할 수도 있습니다.

```python

# 비디오 백엔드 라이브러리 (pyav, decord, 또는 torchvision)가 사용 가능한지 확인합니다.
from transformers.video_utils import load_video

# 테스트를 위해 비디오 파일을 메모리에 로드합니다
video_object2, _ = load_video(
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4"
)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "당신은 항상 해적 스타일로 응답하는 친근한 채팅봇입니다"}],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_object2},
            {"type": "text", "text": "이 비디오에서 무엇을 보시나요?"}
        ],
    },
]
```

입력 콘텐츠를 토큰화하기 위해 `messages`를 [`~ProcessorMixin.apply_chat_template`]에 전달합니다. 샘플링 프로세스를 제어하는 [`~ProcessorMixin.apply_chat_template`]에 포함할 몇 가지 추가 매개변수가 있습니다.

<hfoptions id="sampling">
<hfoption id="고정 프레임 수">

`num_frames` 매개변수는 비디오에서 균등하게 샘플링할 프레임 수를 제어합니다. 각 체크포인트에는 사전훈련된 최대 프레임 수가 있으며 이 수를 초과하면 생성 품질이 크게 저하될 수 있습니다. 모델 용량과 하드웨어 자원에 모두 적합한 프레임 수를 선택하는 것이 중요합니다. `num_frames`가 지정되지 않으면 프레임 샘플링 없이 전체 비디오가 로드됩니다.

```python
processed_chat = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    num_frames=32,
)
print(processed_chat.keys())
```

이러한 입력은 이제 [`~GenerationMixin.generate`]에서 사용할 준비가 되었습니다.

</hfoption>
<hfoption id="fps">

더 긴 비디오의 경우 `fps` 매개변수를 사용하여 더 나은 표현을 위해 더 많은 프레임을 샘플링하는 것이 좋을 수 있습니다. 이는 초당 추출할 프레임 수를 결정합니다. 예를 들어, 비디오가 10초 길이이고 `fps=2`인 경우 모델은 20개의 프레임을 샘플링합니다. 즉, 10초마다 2개의 프레임이 균등하게 샘플링됩니다.

```py
processed_chat = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    fps=16,
)
print(processed_chat.keys())
```

</hfoption>
<hfoption id="이미지 프레임 리스트">

비디오는 전체 비디오 파일이 아닌 이미지로 저장된 샘플링된 프레임 세트로 존재할 수도 있습니다.

이 경우 이미지 파일 경로의 리스트를 전달하면 프로세서가 자동으로 이를 비디오로 연결합니다. 모든 이미지가 동일한 비디오에서 나온 것으로 가정되므로 모든 이미지가 동일한 크기인지 확인하십시오.

```py
frames_paths = ["/path/to/frame0.png", "/path/to/frame5.png", "/path/to/frame10.png"]
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "당신은 항상 해적 스타일로 응답하는 친근한 채팅봇입니다"}],
    },
    {
      "role": "user",
      "content": [
            {"type": "video", "path": frames_paths},
            {"type": "text", "text": "이 비디오에서 무엇을 보시나요?"},
        ],
    },
]

processed_chat = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
)
print(processed_chat.keys())
```

</hfoption>
</hfoptions>