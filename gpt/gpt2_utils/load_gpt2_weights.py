import tensorflow as tf
import numpy as np
import json
import os
import torch


def load_gpt2_model_settings_and_params(model_dir):
    # finds the latest checkpoint file in the model directory
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    # load the model settings from the hparams.json file
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))

    # load the model parameters from the checkpoint file, layer by layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        # load the variable of the layer and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        # print(name, variable_array)   # uncomment to see the variable names and arrays

        # process the variable name (e.g. "model/wte", "model/ln_f/g", "model/h2/mlp/c_proj/w") to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # skip the "model/" prefix

        # for the n layers, identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return settings, params


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shapes do not match: Left: {left.shape}, Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


def load_gpt2_weights_into_model(gpt, params):
    # token and position embeddings
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # Masked Multi Head Attention weights
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.transformer_blocks[b].att.W_q.weight = assign(
            gpt.transformer_blocks[b].att.W_q.weight, q_w.T
        )
        gpt.transformer_blocks[b].att.W_k.weight = assign(
            gpt.transformer_blocks[b].att.W_k.weight, k_w.T
        )
        gpt.transformer_blocks[b].att.W_v.weight = assign(
            gpt.transformer_blocks[b].att.W_v.weight, v_w.T
        )
        # Masked Multi Head Attention biases
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.transformer_blocks[b].att.W_q.bias = assign(
            gpt.transformer_blocks[b].att.W_q.bias, q_b
        )
        gpt.transformer_blocks[b].att.W_k.bias = assign(
            gpt.transformer_blocks[b].att.W_k.bias, k_b
        )
        gpt.transformer_blocks[b].att.W_v.bias = assign(
            gpt.transformer_blocks[b].att.W_v.bias, v_b
        )
        # Masked Multi Head Attention output projection weights and biases
        gpt.transformer_blocks[b].att.out_proj.weight = assign(
            gpt.transformer_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[b].att.out_proj.bias = assign(
            gpt.transformer_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        # Feed Forward weights and biases
        gpt.transformer_blocks[b].ff.layers[0].weight = assign(
            gpt.transformer_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.transformer_blocks[b].ff.layers[0].bias = assign(
            gpt.transformer_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )
        gpt.transformer_blocks[b].ff.layers[2].weight = assign(
            gpt.transformer_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.transformer_blocks[b].ff.layers[2].bias = assign(
            gpt.transformer_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        # 2 Layer Normalization weights and biases
        gpt.transformer_blocks[b].norm1.scale = assign(
            gpt.transformer_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.transformer_blocks[b].norm1.shift = assign(
            gpt.transformer_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.transformer_blocks[b].norm2.scale = assign(
            gpt.transformer_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.transformer_blocks[b].norm2.shift = assign(
            gpt.transformer_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )
    # Final layer norm
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

    # Output projection weights (reuses token embedding)
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
