import json
import os
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import gguf  # noqa: E402

from convert_hf_to_gguf import CodeGenModel  # noqa: E402


class TestCodeGenBias(unittest.TestCase):
    def _create_model(self, dir_model: str, with_bias: bool):
        cfg = {
            "architectures": ["CodeGenForCausalLM"],
            "n_layer": 1,
            "n_embd": 4,
            "n_head": 1,
            "n_positions": 4,
            "n_ctx": 4,
            "n_inner": 8,
            "vocab_size": 8,
            "qkv_proj_bias": with_bias,
        }
        with open(os.path.join(dir_model, "config.json"), "w") as f:
            json.dump(cfg, f)
        sd = {
            "transformer.wte.weight": torch.zeros(8, 4),
            "transformer.ln_f.weight": torch.ones(4),
            "transformer.h.0.ln_1.weight": torch.ones(4),
            "transformer.h.0.attn.qkv_proj.weight": torch.zeros(12,4),
            "transformer.h.0.attn.out_proj.weight": torch.zeros(4,4),
            "transformer.h.0.ln_2.weight": torch.ones(4),
            "transformer.h.0.mlp.c_fc.weight": torch.zeros(8,4),
            "transformer.h.0.mlp.c_proj.weight": torch.zeros(4,8),
            "lm_head.weight": torch.zeros(8,4),
        }
        if with_bias:
            sd["transformer.h.0.attn.qkv_proj.bias"] = torch.zeros(12)
        torch.save(sd, os.path.join(dir_model, "pytorch_model.bin"))

    def _convert(self, with_bias: bool):
        with tempfile.TemporaryDirectory() as d:
            self._create_model(d, with_bias)
            out = Path(d) / "out.gguf"
            model = CodeGenModel(Path(d), gguf.LlamaFileType.F32, out)
            model.write()
            reader = gguf.GGUFReader(out)
            key = gguf.KEY_ATTENTION_USE_QKV_BIAS.format(arch="codegen")
            val = reader.get_kv(key)
            has_tensor = "blk.0.attn_qkv.bias" in reader.tensors
            return val, has_tensor

    def test_biasless_codegen(self):
        val, has_tensor = self._convert(False)
        self.assertFalse(val)
        self.assertFalse(has_tensor)

    def test_bias_codegen(self):
        val, has_tensor = self._convert(True)
        self.assertTrue(val)
        self.assertTrue(has_tensor)


if __name__ == "__main__":
    unittest.main()
