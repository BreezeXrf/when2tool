import ast
import json
import os
import random
import re
import string
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def _try_json_block(s):
    # Try json.loads first (handles valid JSON)
    try:
        data = json.loads(s)
    except Exception:
        # Try single-quote replacement
        try:
            data = json.loads(s.replace("'", '"'))
        except Exception:
            # Fallback: treat the whole thing as a Python expression (handles r"...", True/False, etc.)
            import warnings
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    data = ast.literal_eval(s)
                    # Reject if result contains non-JSON types (e.g., Python sets)
                    try:
                        json.dumps(data)
                    except (TypeError, ValueError):
                        return None
            except Exception:
                return None

    if not isinstance(data, dict):
        return None
    if "name" not in data:
        return None

    if "arguments" not in data:
        if "parameters" in data:
            data["arguments"] = data["parameters"]
        else:
            data["arguments"] = {}

    if isinstance(data["arguments"], str):
        try:
            data["arguments"] = json.loads(data["arguments"])
        except Exception:
            data["arguments"] = {}

    if not isinstance(data["arguments"], dict):
        data["arguments"] = {}

    return data


def _parse_tool_call_from_text(text):
    text = (text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    if "<tool_call>" in text:
        body = text.split("<tool_call>", 1)[1]
        body = body.split("</tool_call>", 1)[0].strip()
        parsed = _try_json_block(body)
        if parsed is not None:
            return parsed

    parsed = _try_json_block(text)
    if parsed is not None:
        return parsed

    # Try all JSON-like blocks from short to long (robust to reasoning text).
    for m in re.finditer(r"\{[\s\S]*?\}", text):
        parsed = _try_json_block(m.group(0))
        if parsed is not None:
            return parsed

    return None


def _normalize_generation_output(decoded):
    parsed = _parse_tool_call_from_text(decoded)
    if parsed is not None:
        return {
            "type": "tool",
            "tool_call_id": "".join(random.sample(string.ascii_letters + string.digits, 9)),
            "tool_name": parsed["name"],
            "arguments": parsed.get("arguments", {}),
            "raw_text": decoded,
        }
    return {"type": "content", "content": decoded, "raw_text": decoded}


class HFAgentBackend:
    def __init__(
        self,
        model_path,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=512,
        device="cuda",
        enable_thinking=None,
    ):
        # honor an explicit --device mps on Apple Silicon; original fell back to CPU without CUDA
        if device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()

        self.generation_config = GenerationConfig.from_pretrained(model_path)
        if temperature is not None:
            self.generation_config.temperature = float(temperature)
        if top_p is not None:
            self.generation_config.top_p = float(top_p)
        if top_k is not None:
            self.generation_config.top_k = int(top_k)

    def _apply_chat_template(self, messages, tools=None, **kwargs):
        call_kwargs = dict(kwargs)
        if tools is not None:
            call_kwargs["tools"] = tools
        if self.enable_thinking is not None:
            call_kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.tokenizer.apply_chat_template(messages, **call_kwargs)
        except TypeError:
            # Some tokenizers/templates do not support enable_thinking.
            call_kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **call_kwargs)

    def render_prompt(self, messages, tools):
        if tools:
            return self._apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
            )
        return self._apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_batch(self, messages_batch, tools_batch, prefills=None):
        outs = []
        for idx, (messages, tools) in enumerate(zip(messages_batch, tools_batch)):
            prompt_text = self.render_prompt(messages, tools)
            pf = (prefills[idx] or "") if prefills else ""

            # Tokenize prompt + prefill together
            full_prompt = prompt_text + pf
            model_inputs = self.tokenizer(
                full_prompt, return_tensors="pt"
            ).to(self.device)

            gen_cfg = deepcopy(self.generation_config)
            output = self.model.generate(
                **model_inputs,
                generation_config=gen_cfg,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            decoded = self.tokenizer.decode(
                output[0][len(model_inputs["input_ids"][0]) :],
                skip_special_tokens=True,
            )
            # Prepend prefill so downstream parsing sees the full response
            full_text = pf + decoded
            out = _normalize_generation_output(full_text)
            out["prompt_text"] = prompt_text
            out["finish_reason"] = "length_or_eos"
            outs.append(out)
        return outs


class VLLMAgentBackend:
    def __init__(
        self,
        model_path,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=512,
        tensor_parallel_size=1,
        max_model_len=4096,
        dtype="bfloat16",
        enable_thinking=None,
    ):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(model_path)

        if temperature is not None:
            self.generation_config.temperature = float(temperature)
        if top_p is not None:
            self.generation_config.top_p = float(top_p)
        if top_k is not None:
            self.generation_config.top_k = int(top_k)

        self._max_model_len = max_model_len
        self.llm = LLM(
            model=model_path,
            # VLLM_ENFORCE_EAGER=1 skips torch.compile/CUDA-graph capture (workaround for
            # "Worker proc VllmWorker-N died unexpectedly" after Dynamo compile). Default off.
            enforce_eager=os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
            # VLLM_DISABLE_CUSTOM_AR=1: fall back to NCCL all-reduce (known fix for multi-GPU
            # worker deaths at the first collective in some container/driver combos). Default off.
            disable_custom_all_reduce=os.environ.get("VLLM_DISABLE_CUSTOM_AR", "0") == "1",
            # VLLM_GPU_UTIL: KV-pool sizing (single-80GB 32B needs >=0.95 for batching headroom
            # at max_model_len 16384; vllm requires pool >= max_len). Default = vllm's 0.9.
            gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.9")),
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
        )

    def _sampling_params(self):
        kw = {"n": 1, "max_tokens": self.max_new_tokens}
        if getattr(self.generation_config, "temperature", None) is not None:
            kw["temperature"] = float(self.generation_config.temperature)
        if getattr(self.generation_config, "top_p", None) is not None:
            kw["top_p"] = float(self.generation_config.top_p)
        if getattr(self.generation_config, "top_k", None) is not None:
            kw["top_k"] = int(self.generation_config.top_k)
        if getattr(self.generation_config, "repetition_penalty", None) is not None:
            kw["repetition_penalty"] = float(self.generation_config.repetition_penalty)
        return self.SamplingParams(**kw)

    def _apply_chat_template(self, messages, tools=None, **kwargs):
        call_kwargs = dict(kwargs)
        if tools is not None:
            call_kwargs["tools"] = tools
        if self.enable_thinking is not None:
            call_kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.tokenizer.apply_chat_template(messages, **call_kwargs)
        except TypeError:
            call_kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **call_kwargs)

    def render_prompt(self, messages, tools):
        if tools:
            return self._apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
            )
        return self._apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_batch(self, messages_batch, tools_batch, prefills=None):
        prompts = [self.render_prompt(m, t) for m, t in zip(messages_batch, tools_batch)]
        if prefills:
            prompts = [p + (pf or "") for p, pf in zip(prompts, prefills)]
        params = self._sampling_params()
        # OVERLENGTH GUARD (single-GPU runs cap max_model_len below the 32k default): vllm raises
        # on ANY over-length prompt, killing the whole setting. Pre-filter those requests, emit an
        # empty "overlength" output for them (scored as a failure), keep the batch alive.
        budget = getattr(self, "_max_model_len", 0) - self.max_new_tokens
        keep_idx, skip = [], 0
        if budget > 0:
            for i, p in enumerate(prompts):
                if len(self.tokenizer(p).input_ids) <= budget:
                    keep_idx.append(i)
                else:
                    skip += 1
        else:
            keep_idx = list(range(len(prompts)))
        if skip:
            print(f"[overlength-guard] skipped {skip}/{len(prompts)} prompts > {budget} tokens")
        raw_kept = self.llm.generate(prompts=[prompts[i] for i in keep_idx],
                                     sampling_params=params) if keep_idx else []
        raw = [None] * len(prompts)
        for i, r in zip(keep_idx, raw_kept):
            raw[i] = r

        outs = []
        for idx, (prompt_text, r) in enumerate(zip(prompts, raw)):
            if r is None:
                out = _normalize_generation_output("")
                out["prompt_text"] = prompt_text
                out["finish_reason"] = "overlength"
                outs.append(out)
                continue
            gen_text = r.outputs[0].text if r.outputs else ""
            # Prepend the prefill to the raw output so downstream parsing sees the full response
            pf = (prefills[idx] or "") if prefills else ""
            full_text = pf + gen_text
            out = _normalize_generation_output(full_text)
            out["prompt_text"] = prompt_text
            out["finish_reason"] = r.outputs[0].finish_reason if r.outputs else "empty"
            outs.append(out)
        return outs


class AgentModel:
    def __init__(
        self,
        model_path,
        backend="vllm",
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=512,
        device="cuda",
        tensor_parallel_size=1,
        max_model_len=4096,
        vllm_dtype="bfloat16",
        enable_thinking=None,
        system_prompt_override=None,
    ):
        from utils import get_system_prompt, detect_tool_format
        if system_prompt_override:
            self.system_prompt = system_prompt_override
        else:
            fmt = detect_tool_format(model_path)
            self.system_prompt = get_system_prompt(fmt)

        if backend == "hf":
            self.engine = HFAgentBackend(
                model_path=model_path,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                device=device,
                enable_thinking=enable_thinking,
            )
        elif backend == "vllm":
            self.engine = VLLMAgentBackend(
                model_path=model_path,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                dtype=vllm_dtype,
                enable_thinking=enable_thinking,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def generate_batch(self, messages_batch, tools_batch, prefills=None):
        return self.engine.generate_batch(messages_batch, tools_batch, prefills=prefills)

    def generate(self, messages, tools, prefill=None):
        prefills = [prefill] if prefill is not None else None
        return self.generate_batch([messages], [tools], prefills=prefills)[0]

    def render_prompt(self, messages, tools):
        return self.engine.render_prompt(messages, tools)
