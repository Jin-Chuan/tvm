import json
import sys
from tvm import relay
import tvm
from tvm.contrib import graph_executor
import numpy as np
import os
import pdb; 
# from models import db


def get_info(mod, params, data_shape):
    # target = "cuda -arch=sm_72"
    # target_host = "llvm -mtriple=x86_64-linux-gnu"
    # target = tvm.target.Target(target=target, host=target_host)
    # target = tvm.target.Target("cuda -arch=sm_75")
    # target = tvm.target.Target("cuda", options=["-arch=sm_75"])
    with tvm.transform.PassContext(opt_level=3):
        # lib = relay.build(mod, target, params=params)
        # pdb.set_trace()
        lib = relay.build(mod, tvm.target.cuda(), params=params) #bs大，加上不知道为什么在cpu上执行 所以超时被kill了？
    dev = tvm.cuda()
    # module = graph_runtime.GraphModule(lib["default"](dev))
    module = graph_executor.GraphModule(lib["default"](dev))
    data = np.ones(data_shape).astype("float32")
    data = data * 10
    # print(module.get_input_info())
    module.set_input("data", data)
    module.run()
    return (lib.get_lib().imported_modules[0].get_source()).splitlines(keepends=True), json.loads(module.module["get_schedule_json"]()), json.loads(lib.get_graph_json())

def generate_final_schedule(source_code_lines, schedule_raw, graph):
    def split_function_declaration(line):#func_name
        parts = line.split("(")
        parameters_str = parts[2].split(")")[0]
        left_parts = parts[1].split(" ")
        name = left_parts[-1]
        # pdb.set_trace()
        return_type = left_parts[-2]
        header = " ".join(left_parts[:-2])
        parameter_str_list = parameters_str.split(", ")
        parameters = []
        for param_str in parameter_str_list:
            parts = param_str.split(" ")
            param_name = parts[-1]
            param_type = " ".join(parts[:-1])
            parameters.append({"name": param_name, "type": param_type})
        return header, return_type, name, parameters
    # 1. storage info from graph_json
    storage_id = graph["attrs"]["storage_id"][1]
    ## FIXME: a hack here
    ## to avoid buffer reuse, we replace storage_id to itself.
    for i in range(len(storage_id)):
        storage_id[i] = i
    
    storage = []
    for i in range(max(storage_id) + 1):
        storage.append({"name": "null", "size": 0, "stype": "null"})

    arg_idx = []

    for i in range(len(storage_id)):
        shape = graph["attrs"]["shape"][1][i]
        t = graph["attrs"]["dltype"][1][i]
        size = 1
        for j in shape:
            size = size * j
        sid = storage_id[i]
        if storage[sid]["size"] < size:# always true
            storage[sid]["size"] = size
            storage[sid]["stype"] = t

    for a in graph["arg_nodes"]:
        sid = storage_id[a] # 不就恒等映射吗，storage_id其实没啥用
        name = graph["nodes"][a]["name"]
        storage[sid]["name"] = name
        arg_idx.append(sid)

    # 2. append dynamic allocated storage
    temp_storage_begin = len(storage)
    for temp_arg in schedule_raw["temp_args"]:
        storage.append({"name": "temp_arg", "size": temp_arg, "stype": "byte"})

    # 3. remap the kernel args
    i = 0
    kernels = []
    node_row_ptr = graph["node_row_ptr"]
    for j in range(len(graph["nodes"])):
        node = graph["nodes"][j]
        if node["op"] == "null":
            continue
        if node["attrs"]["func_name"] == "__nop":
            continue

        schedule_func = schedule_raw["funcs"][i]
        while len(schedule_func["kernels"]) == 0:
            i = i + 1
            schedule_func = schedule_raw["funcs"][i]

        if schedule_func["name"] != node["attrs"]["func_name"]:
            raise Exception("schedule name != node name, %s != %s" % (schedule_func["name"],node["name"]))
        # if node["attrs"]["num_outputs"] != "1":
        #     print(node["attrs"]["num_outputs"])
        #     raise Exception("node output != 1")
        host_inputs = []
        
        # pdb.set_trace()
        for inp in node["inputs"]:
            host_inputs.append(node_row_ptr[inp[0]]+inp[1])
        for idx in range(int(node["attrs"]["num_outputs"])):
            host_inputs.append(node_row_ptr[j]+idx)
        for kernel in schedule_func["kernels"]:
            new_args = []
            for arg in kernel["args"]:
                if arg < 0:
                    new_args.append(temp_storage_begin-arg-1)
                else:
                    new_args.append(storage_id[host_inputs[arg]])
            kernels.append({"name": kernel["name"], "launch_params": kernel["launch_params"], "args": new_args})
        i = i+1

    output_idx = graph["heads"][0][0]
    storage[storage_id[output_idx]]["name"] = "output"

    schedule = {
        "storage": storage,
        "kernels": kernels,
        "args": arg_idx
    }

    # 4. generate shared memory usage

    func_name = ""
    shared_memory = 0

    result = {}

    for line in source_code_lines:
        if line.find("void") != -1:
            # save old values
            if func_name != "":
                if shared_memory < 4:
                    shared_memory = 4
                result[func_name] = shared_memory

            _, _, curr_func_name, _ = split_function_declaration(line)
            func_name = curr_func_name
            shared_memory = 4
        if line.find("__shared__") != -1:
            # __shared__ float x[123];
            size = line.split("[")[1].split("]")[0]
            shared_memory = shared_memory + int(size) * 4

    if func_name != "":
        if shared_memory < 4:
            shared_memory = 4
        result[func_name] = shared_memory

    schedule["shared_memory"] = result
    return schedule

def resnet():
    from tvm.relay import testing
    mod, params = testing.resnet.get_workload(num_layers=152, batch_size=1, image_shape=(3, 224, 224))
    return mod, params, (1, 3, 224, 224)

def bert():
    from transformers import BertTokenizer, BertConfig, BertModel
    import tvm
    from tvm import relay
    import torch


    class AutoModelAMP(BertModel):
        def __init__(self, *args, **keyargs):
            super().__init__(*args, **keyargs)

        def forward(self, *args, **keyargs):
            return super().forward(*args, **keyargs)


    config = BertConfig.from_pretrained("bert-base-uncased")
    config.return_dict = False
    config.torchscript = True
    model = AutoModelAMP(config).eval()

    batch = 1#64
    seq_len = 128
    input = torch.zeros([batch, seq_len], dtype=torch.int).long()
    inputs = {
        "data": input,
        "attention_mask": input,
        "token_type_ids": input,
    }
    input_list = []
    for k, v in inputs.items():
        # print(k)
        input_list.append(v)

    shape_list = []
    for k, v in inputs.items():
        shape_list.append((k, v.shape))

    traced_model = torch.jit.trace(
        model, input_list, strict=False).eval()
    mod, params = relay.frontend.from_pytorch(traced_model, shape_list)
    return mod,params,(batch,seq_len)



if __name__ == "__main__":
    network_dir = os.path.join("resource/")
    if not os.path.exists(network_dir):
        os.mkdir(network_dir)
    # mod, params, data_shape = resnet()
    mod, params, data_shape = bert()
    source_code_lines, schedule_raw, graph = get_info(mod, params, data_shape)
    with open(os.path.join(network_dir, "graph_raw.json"), "w") as f:
        json.dump(graph, f, indent=4)
    schedule = generate_final_schedule(source_code_lines, schedule_raw, graph)
    with open(os.path.join(network_dir, "source.cu"), "w") as f:
        f.writelines(source_code_lines)
    with open(os.path.join(network_dir, "schedule_raw.json"), "w") as f:
        json.dump(schedule_raw, f, indent=4)
    with open(os.path.join(network_dir, "schedule.json"), "w") as f:
        json.dump(schedule, f, indent=4)
    with open(os.path.join(network_dir, "graph.json"), "w") as f:
        json.dump(graph, f, indent=4)
    DAG = {}
    funcs = {}
    for node in schedule_raw["funcs"]:
        funcs[node['name']] = []
        for f in node["kernels"]:
            funcs[node['name']].append(f['name'])

    for node in graph["nodes"]:
        if node["op"] == 'null':
            continue
        if node["attrs"]["func_name"]=="__nop":  # name reshape_nop不唯一
            continue
        
        DAG[node["name"]] = {}
        DAG[node["name"]]["prenode"] = []
        DAG[node["name"]]["funcs"] = funcs[node["attrs"]["func_name"]]
        # print(node["attrs"]["func_name"], funcs[node["attrs"]["func_name"]])
        # if node["attrs"]["func_name"]!="__nop": 
        #     DAG[node["name"]]["funcs"] = funcs[node["attrs"]["func_name"]]
        # else:
        #     DAG[node["name"]]["funcs"] = []
        for pre, _, __ in node["inputs"]:
            if graph["nodes"][pre]["op"] == 'null':
                continue
            DAG[node["name"]]["prenode"].append(graph["nodes"][pre]["name"])

    #next没用到 
    for name in DAG:
        node = DAG[name]
        DAG[name]["next"] = []
        for name1 in DAG:
            node1 = DAG[name1]
            if name in node1["prenode"]:
                DAG[name]["next"].append(name1)

    i = 0
    for name in DAG:
        node = DAG[name]
        DAG[name]["args"] = []
        DAG[name]["launch_params"] = []
        DAG[name]["shared"] = []
        for f in node["funcs"]:
            if f != schedule["kernels"][i]["name"]:
                raise Exception("schedule name != node name, %s != %s" % (f, schedule["kernels"][i]["name"]))
            DAG[name]["args"].append(schedule["kernels"][i]["args"])
            DAG[name]["launch_params"].append(schedule["kernels"][i]["launch_params"])
            DAG[name]["shared"].append(schedule["shared_memory"][f]-4 if (schedule["shared_memory"][f]-4)%16==0 else ((schedule["shared_memory"][f]-4)//16 + 1)*16)#除去reef中初始为4
            i += 1

    #prenode只用于判断度为0的节点
    runtime = {"head":[]}
    for name in DAG:
        node = DAG[name]
        if len(node['prenode']) == 0:
            runtime["head"].append(name)
            if "nop" in name:
                raise Exception("nop")
            
    runtime["storage"] = schedule["storage"]
    runtime["args"] = schedule["args"]
    runtime["dag"] = DAG
    runtime["offset"] = []
    runtime["weight"] = 0
    runtime["data"] = 0
    for i in range(len(runtime["storage"])):
        if i in runtime["args"]:
            runtime["offset"].append([True, runtime["weight"]])
            runtime["weight"] += runtime["storage"][i]["size"]
        else:
            runtime["offset"].append([False, runtime["data"]])
            runtime["data"] += runtime["storage"][i]["size"]
    # for i in range(len(runtime["storage"])):
    #     runtime["offset"].append(runtime["offset"][i]+runtime["storage"][i]["size"])
    with open(os.path.join(network_dir, "runtime.json"), "w") as f:
        json.dump(runtime, f, indent=4)
    with open(os.path.join(network_dir, "source.cu"), "w") as f:
        f.writelines(source_code_lines)