import os
import inspect
import transformers
from transformers.cache_utils import DynamicCache

output_path = os.path.join(os.path.dirname(__file__), "qwen3_modeling_inspect.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Transformers Version: {transformers.__version__}\n\n")
    
    # Check Qwen3
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3ForCausalLM
        f.write("=== Qwen3Attention.__init__ Signature & Source ===\n")
        f.write(inspect.getsource(Qwen3Attention.__init__))
        f.write("\n\n=== Qwen3Attention.forward Signature & Source ===\n")
        f.write(inspect.getsource(Qwen3Attention.forward))
        f.write("\n\n=== Qwen3ForCausalLM.forward Signature & Source ===\n")
        f.write(inspect.getsource(Qwen3ForCausalLM.forward))
        print("Successfully extracted Qwen3Attention and Qwen3ForCausalLM sources.")
    except Exception as e:
        f.write(f"Could not import Qwen3Attention: {e}\n")
        print(f"Could not import Qwen3Attention: {e}")
        # Try finding related Qwen classes
        qwen_attrs = [attr for attr in dir(transformers) if "qwen" in attr.lower()]
        f.write(f"Available Qwen attributes in transformers: {qwen_attrs}\n")
        print(f"Available Qwen attributes: {qwen_attrs}")
        
        # Also check Qwen2 as fallback
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
            f.write("\n=== Fallback: Qwen2Attention.forward ===\n")
            f.write(inspect.getsource(Qwen2Attention.forward))
        except Exception as e2:
            f.write(f"Could not import Qwen2Attention: {e2}\n")

    # Inspect DynamicCache
    f.write("\n\n=== DynamicCache Class & Methods ===\n")
    try:
        f.write(inspect.getsource(DynamicCache))
        print("Successfully extracted DynamicCache source.")
    except Exception as e:
        f.write(f"Could not inspect DynamicCache: {e}\n")
        print(f"Could not inspect DynamicCache: {e}")

print(f"Inspection written to {output_path}")
