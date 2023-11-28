from tvm.relay import testing


def resnet152():
    mod, params = testing.resnet.get_workload(num_layers=152, batch_size=1, image_shape=(3, 224, 224))
    return mod, params, (1, 3, 224, 224)

def densenet201():
    mod, params = testing.densenet.get_workload(densenet_size=201, batch_size=1, image_shape=(3, 224, 224))
    return mod, params, (1, 3, 224, 224)

# def bert():
#     import torch
#     import tvm
#     import tvm.relay as relay
#     from transformers import BertTokenizer, BertModel

#     # Load the exported PyTorch model

#     # Define the input shape and data type
#     # input_shape = (1, 3, 224, 224)
#     # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


#     # # # Example input text
#     # text = "Hello, how are you?"

#     # # Tokenize the input text
#     # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     # input = tokenizer.encode_plus(
#     #     text,
#     #     add_special_tokens=True,
#     #     max_length=128,  # Specify the desired sequence length
#     #     padding='max_length',
#     #     truncation=True,
#     #     return_tensors='pt'  # Return PyTorch tensors
#     # )
#     # input_tensor = torch.tensor([input])
#     # input_dtype = 'float32'

#     # traced_model = torch.jit.trace(model, input_shape)

#     # Mock input tensors (example)
#     model = BertModel.from_pretrained('bert-base-uncased',torchscript=True, return_dict=False)
#     input_ids = torch.tensor([[101, 2054, 2003, 1037, 2518, 102]])  # Shape: (1, 6)
#     attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])  # Shape: (1, 6)
#     input_shape=[('input0',(1,6)),('input1',(1,6))]
#     scripted_model = torch.jit.trace(model, (input_ids, attention_mask),strict=False)
#     print(scripted_model)
#     # Convert the PyTorch model to TVM Relay format
#     relay_model,_ = relay.frontend.from_pytorch(scripted_model, input_infos=input_shape)

#     # Specify the target hardware as CUDA
#     target = 'cuda'

#     # Specify the target device, such as 'cuda' or 'cuda -libs=cudnn' for CUDA with cuDNN
#     target_host = 'llvm -target=x86_64-linux-gnu'

#     # Perform further optimizations or transformations on the relay_model as needed

#     # Compile the relay_model using TVM
#     mod, params = relay.frontend.compile(relay_model, target=target, target_host=target_host)

#     return mod,params,input_shape
#     # Save the compiled module and parameters for future use
#     # output_path = 'compiled_module.tar'
#     # tvm.runtime.save_module(output_path, mod, params)

# bert()
db = {
    "resnet152": resnet152,
    "densenet201": densenet201,
    "bert": bert
}
