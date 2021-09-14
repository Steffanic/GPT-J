import os
class GPT_J:
    def __init__(self):
        if os.path.exists("already_installed_GPT_J.txt"):
            pass
        else:
            self.install()
            with open("already_installed_GPT_J.txt", "w"):
                pass
    def install(self):          
        ########### get the repos and install dependencies #####
        os.system("apt install zstd")
        
        # the "slim" version contain only bf16 weights and no optimizer parameters, which minimizes bandwidth and memory
        os.system("wget -c https://the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd")
        
        os.system("time tar -I zstd -xf step_383500_slim.tar.zstd")
        
        os.system("git clone https://github.com/kingoflolz/mesh-transformer-jax.git")
        
        os.system("pip install -r mesh-transformer-jax/requirements.txt")
        # jax 0.2.12 is required due to a regression with xmap in 0.2.13
        os.system("pip install mesh-transformer-jax/ jax==0.2.12 tensorflow==2.5.0")
        
        ################  setup model #########
        import requests
        from jax.config import config
        
        colab_tpu_addr = os.environ['COLAB_TPU_ADDR'].split(':')[0]
        url = f'http://{colab_tpu_addr}:8475/requestversion/tpu_driver0.1_dev20210607'
        requests.post(url)
        
        # The following is required to use TPU Driver as JAX's backend.
        config.FLAGS.jax_xla_backend = "tpu_driver"
        config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']
        
        ###############
        
        import time
        
        import jax
        from jax.experimental import maps
        import numpy as np
        import optax
        import transformers
        
        from mesh_transformer.checkpoint import read_ckpt
        from mesh_transformer.sampling import nucleaus_sample
        from mesh_transformer.transformer_shard import CausalTransformer
        
        #################
        
        params = {
          "layers": 28,
          "d_model": 4096,
          "n_heads": 16,
          "n_vocab": 50400,
          "norm": "layernorm",
          "pe": "rotary",
          "pe_rotary_dims": 64,
        
          "seq": 2048,
          "cores_per_replica": 8,
          "per_replica_batch": 1,
        }
        
        per_replica_batch = params["per_replica_batch"]
        cores_per_replica = params["cores_per_replica"]
        seq = params["seq"]
        
        
        params["sampler"] = nucleaus_sample
        
        # here we "remove" the optimizer parameters from the model (as we don't need them for inference)
        params["optimizer"] = optax.scale(0)
        
        mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
        devices = np.array(jax.devices()).reshape(mesh_shape)
        
        maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))
        
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
        
        ###################
        
        total_batch = per_replica_batch * jax.device_count() // cores_per_replica
        
        network = CausalTransformer(params)
        
        network.state = read_ckpt(network.state, "step_383500/", devices.shape[1])
        
        network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))
        
        ######### RUN Model ############
        
        # allow text wrapping in generated output: https://stackoverflow.com/a/61401455
        from IPython.display import HTML, display
        
        def set_css():
          display(HTML('''
          <style>
            pre {
                white-space: pre-wrap;
            }
          </style>
          '''))
        get_ipython().events.register('pre_run_cell', set_css)
        
        ##################

    def infer(self, context, top_p=0.9, temp=1.0, gen_len=512):
        tokens = tokenizer.encode(context)
        
        provided_ctx = len(tokens)
        pad_amount = seq - provided_ctx
        
        padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
        batched_tokens = np.array([padded_tokens] * total_batch)
        length = np.ones(total_batch, dtype=np.uint32) * len(tokens)
        
        start = time.time()
        output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(total_batch) * top_p, "temp": np.ones(total_batch) * temp})
        
        samples = []
        decoded_tokens = output[1][0]
        
        for o in decoded_tokens[:, :, 0]:
          samples.append(f"\033[1m{context}\033[0m{tokenizer.decode(o)}")
        
        print(f"completion done in {time.time() - start:06}s")
        return samples


############ HERE you can use GPT_J

if __name__ =="__main__":
    gpt_j = GPT_J()
    #@title  { form-width: "300px" }
    top_p = 0.9 #@param {type:"slider", min:0, max:1, step:0.1}
    temp = 1 #@param {type:"slider", min:0, max:1, step:0.1}
    
    context = """
    import tensorflow
    # 4 layer CNN with a softmax output
    # test on MNIST data set
    """
    
    print(gpt_j.infer(top_p=top_p, temp=temp, gen_len=1000, context=context)[0])