<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Response 파싱 [[response-parsing]]

채팅 모델이 단일 응답 문자열이 아닌 구조화된 출력을 생성하는 것이 점점 일반적이 되고 있습니다.
구조화된 출력의 가장 일반적인 용도는 [도구 호출](./chat_extras)과 [추론 모델](https://huggingface.co/reasoning-course)입니다.
도구 호출 모델은 호출할 도구의 이름과 전달할 인수를 포함하는 도구 호출을 출력할 수 있으며,
추론 모델은 종종 추론 단계를 "사고의 연쇄"로 출력합니다. 일부 최근 모델들은 이 둘을 모두 사용하기도 하며,
최종 답변 전에 추론 및/또는 하나 이상의 도구 호출을 출력할 수 있습니다.

구조화된 출력을 가진 모델들은 채팅 템플릿화에 도전 과제를 제기합니다. 출력이 채팅에 추가되기 전에 파싱되어야 하기 때문입니다. 구체적인 예를 들면, [GPT-OSS](https://huggingface.co/openai/gpt-oss-120b)에게
날씨가 어떤지 물어보고, 모델이 생각하고 도구를 호출하기로 결정했다고 가정해봅시다. 원시 모델 출력은 다음과 같을 수 있습니다:

```txt
<|start|>analysis<|message|>사용자가 묻습니다: "SF의 날씨는 어떤가요?" 사용자의 위치를 알아야 합니다. 사용자가 명시적으로 SF (샌프란시스코)에 대해 묻습니다.
따라서 캘리포니아 샌프란시스코의 현재 날씨를 알아야 합니다. get_current_weather 함수를 호출해야 합니다. 하지만 날씨 데이터를 얻기 위해 함수를 호출해야 합니다.
따라서 위치 "San Francisco, CA"로 get_current_weather를 호출해야 합니다. 그렇게 하겠습니다.
get_current_weather 함수를 호출하겠습니다.<|end|><|start|>commentary to=functions.get_current_weather<|channel|>commentary <|constrain|>json<|message|>{"location":"San Francisco, CA"}<|call|>
}
```

하지만 이것을 채팅에 추가하려면, 다음과 같이 채팅 메시지 딕셔너리로 포맷해야 합니다:

```json
{
  "role": "assistant",
  "thinking": "사용자가 묻습니다: \"SF의 날씨는 어떤가요?\" 사용자의 위치를 알아야 합니다. 사용자가 명시적으로 SF (샌프란시스코)에 대해 묻습니다. 따라서 캘리포니아 샌프란시스코의 현재 날씨를 알아야 합니다. get_current_weather 함수를 호출해야 합니다. 하지만 날씨 데이터를 얻기 위해 함수를 호출해야 합니다. 위치 \"San Francisco, CA\"로 get_current_weather를 호출해야 합니다. 그렇게 하겠습니다.",
  "tool_calls": [
    {
      "name": "get_current_weather",
      "arguments": {
        "location": "San Francisco, CA"
      }
    }
  ]
}
```

채팅 **템플릿**은 메시지를 모델용 포맷된 입력으로 변환하는 방법을 제공하지만, 모델 출력을 다시 표준 메시지 딕셔너리로 파싱하기 위해서는 다른 것이 필요합니다. 이것이 채팅 **파싱**의 목적입니다.

## [parse_response](~PreTrainedTokenizerBase.parse_response) 메소드 [[the-parseresponsepretrainedtokenizerbaseparseresponse-method]]

이를 지원하는 모델에서 채팅 응답을 파싱하는 것은 간단합니다. [generate](`~generation.GenerationMixin.generate`)에서 원시 디코딩된 출력을 가져와서 토크나이저의 [parse_response](~PreTrainedTokenizerBase.parse_response) 메소드에 전달하면 됩니다:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM3-3B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="auto", device_map="auto")

messages = [
    {
        "role": "user",
        "content": "안녕! 냉전의 종료를 가능한 한 간략하게 요약해줄 수 있나요? 정말 우스꽝스럽게 간단하게요. 관련 정보의 대부분을 실제로 빠뜨려야 합니다."
    }
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(input_ids, max_new_tokens=1024)[0, input_ids.shape[1]:]
out_text = tokenizer.decode(outputs)
parsed = tokenizer.parse_response(out_text)
print(parsed.keys())
```

그러면 다음과 같은 결과를 얻어야 합니다:

```text
dict_keys(['thinking', 'content'])
```

그게 응답 파싱을 시작하는 데 필요한 전부입니다! `parse_response`는 채팅 기록에 추가할 준비가 된 완전한 메시지 딕셔너리를 반환해야 합니다.
토크나이저가 응답 파싱을 지원하지 않을 때, `parse_response`는 오류를 발생시킵니다. 시간이 지남에 따라 더 많은 토크나이저에 지원을 추가할 예정입니다.

## 개발자: 간단한 응답 스키마 이해 [[developers-understanding-a-simple-response-schema]]

내부적으로 `parse_response`는 **JSON 스키마**를 사용하여 모델 출력을 파싱합니다. JSON 스키마는
출력 메시지 딕셔너리의 구조를 나타냅니다. 스키마는 출력 메시지 문자열이 예상된 형식으로 파싱되어야 하는 방법을 나타내는 추가 필드로 확장됩니다. 도구 호출을 제외하고 SmolLM 응답의 스키마를 살펴보겠습니다:

```python
{
    "x-regex": "(?:<think>\n?(?P<thinking>.+?)\n?</think>)?\s*(?P<content>.+?)?\s*(?:<\|im_end\|>|$)",
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "thinking": {"type": "string"}
    }
}
```

스키마가 `role`, `content`, `thinking`이라는 세 개의 키를 가진 JSON "객체"(`딕셔너리`를 의미)를 설명하는 것을 볼 수 있습니다.
모든 assistant 응답은 "assistant" 역할을 가지므로, `role` 키는 `const`(상수)입니다. 다른 두 키는 문자열로서, `x-regex` 필드의 정규식에서 명명된 그룹에서 추출됩니다.

채팅 템플릿과 마찬가지로, 응답 스키마는 토크나이저의 속성으로 설정됩니다. 응답 파싱을 활성화하기 위해서는 `tokenizer.response_schema`를 유효한 스키마 딕셔너리로 설정하기만 하면 되며, `tokenizer.parse_response()`가 작동할 것입니다! 다시, 채팅 템플릿과 마찬가지로, 이 스키마는 프로세서와 함께 저장되므로, 한 번 설정하면 `save_pretrained()` 또는 `push_to_hub()`를 사용하여 스키마를 저장하고 공유할 수 있습니다.

## 개발자: 복잡한 스키마 [[developers-complex-schemas]]

이제 도구 호출을 포함하는 더 복잡한 스키마를 살펴보겠습니다. 파서 내부를 더 이해하기 위해서입니다. 이를 위해 `GPT-OSS` 스키마를 사용하겠습니다. GPT-OSS는 도구 호출과 사고 블록을 모두 출력하며, 모델 응답이 세 가지 "채널" 중 하나로 태그되는 특이한 형식을 사용합니다: 도구 호출 같은 것들을 위한 `commentary`, 사고 연쇄 블록을 위한 `analysis`, 그리고 사용자에게 보내질 메시지를 위한 `final`. 모델이 `get_current_weather`라는 도구를 호출하는 전체 메시지는 명확성을 위해 몇 개의 추가 줄바꿈과 함께 다음과 같을 수 있습니다:

```text
<|channel|>analysis<|message|>
사용자가 묻습니다: "SF의 날씨는 어떤가요?" 따라서 캘리포니아 샌프란시스코의 현재 날씨를 얻어야 합니다. 
get_current_weather 함수를 호출해야 합니다. 따라서 위치 "San Francisco, CA"로 get_current_weather를 호출해야 합니다.
<|end|>
<|start|>assistant<|channel|>commentary 
to=functions.get_current_weather <|constrain|>json<|message|>
{
  "location": "San Francisco, CA"
}
<|call|>
```

파싱은 재귀적으로 진행됩니다. 한 레벨에서 정규식(또는 다른 파서)의 출력이 아래 노드들의 입력이 됩니다.
즉, 하나의 거대한 정규식으로 전체 출력을 파싱해야 한다고 느낄 필요가 없습니다! 대신 스키마부터 시작해서, 진행하면서 관련 묶음을 추출하기 위한 정규식을 추가하세요. 다음은 이를 파싱할 스키마와 몇 가지 설명 주석입니다:

```python
{
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        # "content"와 "thinking"은 모두 이전 예제와 유사하며, 단일 문자열을 추출합니다
        # 하지만 둘 다 추출하기 위해 명명된 그룹을 가진 단일 정규식을 사용하는 대신, 각 하위키에서 정규식을 사용합니다.
        # 객체 노드에 파서/정규식이 없으면, 전체 입력 문자열이 모든 자식에게 전달되므로 
        # 파싱은 객체 레벨에서 명명된 그룹으로 수행되거나 속성 레벨에서 별도의 정규식으로 수행될 수 있습니다.
        "content": {"type": "string", "x-regex": r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)"},
        "thinking": {"type": "string", "x-regex": r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"},
        "tool_calls": {
            # "x-regex-iterator"는 re.findall을 사용하여 여러 가능한 매치를 찾고, 배열/리스트로 반환합니다.
            # 하지만 배열 처리에 대해 걱정할 필요는 없습니다 - 배열의 각 항목은 `items` 스키마에 의해 파싱되므로, 단일 항목에 대한 스키마만 작성하면 됩니다.
            "x-regex-iterator": r"<\|channel\|>commentary (to=functions\..*?<\|message\|>.*?)(?:<\|call\|>|$)",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # const 속성은 고정 값이며, 입력은 이에 영향을 주지 않습니다.
                    "type": {"const": "function"},
                    # 여기서 전체 도구 호출 딕셔너리를 `{"function": ...}` 블록으로 감쌉니다. 입력 문자열은 변경 없이 전달됩니다.
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "x-regex": r"^to=functions\.(\w+)"},
                            "arguments": {
                                "type": "object",
                                "x-regex": "<\|message\|>(.*)",
                                # "x-parser" 필드는 추출된 문자열이 JSON으로 파싱되어야 함을 나타냅니다.
                                # 그런 다음 출력은 아래의 스키마 노드들에 전달되고 재귀적 파싱이 계속됩니다.
                                "x-parser": "json",
                                "additionalProperties": {"type": "any"},
                            },
                        },
                    },
                },
            },
        },
    },
}
```

## 개발자: 파서 로직 이해 [[developers-understanding-the-parser-logic]]

파서는 몇 가지 간단한 규칙을 따릅니다:

1. 스키마의 각 레벨은 위 레벨에서 입력을 받고, 가지고 있는 정규식이나 파서를 적용한 다음, 출력을 자식들에게 전달합니다.
2. 루트 레벨은 디코딩된 전체 모델 출력 문자열을 입력으로 받습니다.
3. 노드가 파싱 후 구조화된 콘텐츠를 가지면 (예를 들어, 정규식에 명명된 그룹이 있어서 딕셔너리를 반환하거나, 파서가 딕셔너리나 리스트를 반환하는 경우), 그 구조화된 콘텐츠가 노드의 자식들에 매핑되고, 각 자식 노드는 해당하는 값을 입력으로 받습니다.
4. `object` (딕셔너리) 노드가 비구조화된 (문자열) 출력을 가지면, 전체 문자열이 모든 자식에게 전달됩니다. 이것은 자식 노드들이 모든 키를 추출하기 위한 단일 부모 정규식을 요구하는 대신 개별적으로 파싱을 처리할 수 있게 합니다.
5. `array` (리스트) 노드가 비구조화된 (문자열) 출력을 가지면, 이는 오류를 발생시킵니다.

각 노드에서 파싱이 어떻게 수행되어야 하는지를 나타내는 허용 가능한 `x-` 키의 작은 집합이 있습니다:
- `x-regex`: 입력에 적용할 정규식 문자열입니다. 정규식에 명명된 그룹이 있으면, 출력은 그룹 이름에서 값으로의 딕셔너리입니다. 명명된 그룹은 `object` 노드에서만 사용되어야 합니다. 그렇지 않으면, 정규식은 정확히 하나의 명명되지 않은 캡처링 그룹을 가져야 하며, 출력은 그 그룹의 값을 문자열로 합니다.
- `x-regex-iterator`: `re.findall()`을 사용하여 입력에 적용할 정규식 문자열입니다. 출력은 모든 매치의 리스트입니다. 이는 `array` 노드에서만 사용되어야 하며, 정규식은 정확히 하나의 명명되지 않은 캡처링 그룹을 가져야 합니다. 출력은 노드의 `items` 스키마에 분산됩니다.
- `x-parser`: 입력에 적용할 내장 파서를 호출합니다. 현재, 지원되는 유일한 파서는 입력 문자열을 JSON으로 파싱하는 `json`입니다. 출력은 추가 파싱을 위해 자식 노드들에 전달됩니다. `json` 파서는 깊게 중첩된 출력을 반환할 수 있다는 점에 유의하세요 - 이 경우, 출력은 자식 노드들을 통과하면서 점진적으로 언래핑됩니다. 자식 노드들은 이 경우 추가적인 `x-parser`나 `x-regex` 필드가 필요하지 않지만, 그들의 구조는 파싱된 JSON의 구조와 일치해야 합니다.
- `x-parser-args`: `x-parser`와 함께만 허용됩니다. 이는 파싱을 제어하는 추가 인수의 딕셔너리입니다. 현재 지원되는 유일한 인수는 출력에 적용할 `jmespath` 변환을 지정하는 `transform`입니다. 이는 JSON 파서가 스키마와 일치하도록 수정이 필요한 구조를 반환할 때 유용합니다.
- `x-regex-key-value`: 이는 거의 필요하지 않지만, 키 이름을 미리 알 수 없는 임의의 인수 이름을 가진 XML 도구 호출을 모델이 출력하는 경우와 같이 비JSON 형식에서 키-값 쌍을 파싱할 때 유용할 수 있습니다. 정규식은 정확히 두 개의 명명된 캡처링 그룹인 `key`와 `value`를 가져야 하며, 출력은 키에서 값으로의 딕셔너리입니다. 이는 `object` 노드에서만 사용되어야 합니다.

일반적으로, 여러 정규식/파서는 같은 레벨에서 결합될 수 없습니다. 예외는 단일 문자열을 반환하는 `x-regex`가 다른 파서와 결합될 수 있다는 것입니다. 이 경우, `x-regex`가 먼저 적용되고, 그 다음 출력이 다른 파서인 `x-regex-iterator`, `x-parser`, 또는 `x-regex-key-value`에 전달됩니다.

이러한 아이디어들을 종합하면, 입력이 스키마를 통해 흐르고, 각 레벨에서 파싱되고 자식 노드들에 분산되는 것을 볼 수 있습니다. 각 레벨은 스키마의 해당 부분에 관련된 입력 콘텐츠만 추출하면 되고, 나머지는 자식 노드들이 처리하도록 할 수 있습니다. 내부적으로, 이는 입력을 받고, 현재 레벨에서 정규식/파서를 적용한 다음, 결과를 자식 노드들에 매핑한 후 각각에 대해 재귀적으로 자신을 호출하는 파서 함수로 처리됩니다. 재귀는 보통 `string`이나 `number`와 같은 원시 타입인 리프 노드에 도달하면 종료되며, 이들은 단순히 받은 입력을 반환합니다.