import json
def graph():
    with open('resource/graph_raw.json','r') as f:
        graph=json.load(f)
        # print(len(graph['arg_nodes']))
        # for x in graph['nodes']:
        #     if x.get("attrs") is not None and x["attrs"].get("func_name") is not None and type(x["attrs"]["func_name"])!=str:
        #         print(x)

        # for x in graph['arg_nodes']:
        #     print(graph["nodes"][x])
        # print(graph['nodes'][24])   
        # print(graph['nodes'][513])   

        for x in graph['nodes']:
            flag=False
            for inp in  x["inputs"]:
                if inp[0]==514:
                    flag=True
            if flag:
                print(x)

        # storage_id = graph["attrs"]["storage_id"][1]
        # x,y=0,0
        # for i in range(len(storage_id)):
        #     for j in range(len(storage_id)):
        #         if i!=j and storage_id[i]==storage_id[j]:
        #             print(i,j)
        #             x,y=i,j
        #             break
        # print(x,y)
        # print(graph["attrs"]["dltype"][1][x], graph["attrs"]["dltype"][1][y])
        # print(graph["attrs"]["shape"][1][x], graph["attrs"]["shape"][1][y])

        # print(graph['nodes'][x])

def nop():
    with open("resource/bert/graph.json","r") as f:
        graph=json.load(f) 
        nops=[]
        idx=0
        cnts={}

        for node in  graph['nodes']:
            if node['name']=='reshape_nop':
                print(node["inputs"])
        #     if cnts.get(node['name']) is None:
        #         cnts[node['name']]=1
        #     else :
        #         cnts[node['name']]+=1
        #     if "nop" in node['name']:
        #         nops.append(idx)
        #     for input in node["inputs"]:
        #         if input[0] in nops:
        #             if input[0]>= idx:
        #                 raise Exception("error")

        #     idx+=1
        # idx=0
        # for x in cnts:
        #     if cnts[x]>1:
        #         print(x,cnts[x])
    #     for node in graph['nodes']:
    #         for input in node["inputs"]:
    #             if input[0] in nops:
    #                 print(idx,node,nops.index(input[0]))
    #         idx+=1
    # print(graph["nodes"][487],nops[11])

def schedule():
    with open('resource/schedule_raw.json','r') as f:
        schedule=json.load(f)
        for node in schedule["funcs"]:
            if len(node["kernels"])>1:
                print(node)

nop()
# graph()
# schedule()