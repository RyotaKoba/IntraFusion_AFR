from importlib.util import module_for_loader
from timm.models import create_model
import torch.nn as nn

def rm_modules(model):
    num_layers = len(model.model.layers)
    
    rm_modules = [(model.model.layers[n].mlp.gate_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].mlp.up_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].mlp.down_proj,'weight') for n in range(num_layers)]
    
    return tuple(rm_modules)

def all_rm_modules(model):
    num_layers = len(model.model.layers)
    
    rm_modules = [(model.model.layers[n].self_attn.q_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].self_attn.k_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].self_attn.v_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].self_attn.o_proj,'weight') for n in range(num_layers)]
    
    rm_modules = rm_modules + [(model.model.layers[n].mlp.gate_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].mlp.up_proj,'weight') for n in range(num_layers)]
    rm_modules = rm_modules + [(model.model.layers[n].mlp.down_proj,'weight') for n in range(num_layers)]
    
    return tuple(rm_modules)