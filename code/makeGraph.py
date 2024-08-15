'''构图：
在makeGraph13的基础上，由全序列特征 改为 DxDD为中心，10A为半径 的层次化特征

ProteinBert局部特征 + DxDD为中心的10A pdb
'''

import glob
import re
import pandas as pd
from Bio.PDB import PDBParser
import networkx as nx
import torch
from torch import combinations
from torch_geometric.data import Data, Batch
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import numpy as np


# 定义氨基酸类型
amino_acid_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

three_to_one = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V'
}

def parse_pdb(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)
    # print("structure=", structure, file_path)
    amino_acids = []
    coordinates = {}
    residue_numbers = []  # 存放所有氨基酸的序号的数组
    residue_names = []  # 存放所有氨基酸名称的数组
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id('CA'):
                    amino_acids.append(residue)
                    residue_number = residue.get_id()[1]
                    residue_numbers.append(residue_number)  # 将序号添加到数组中
                    residue_name = residue.get_resname()  # 获取氨基酸名称
                    residue_names.append(three_to_one[residue_name])  # 将名称添加到数组中
                    a = len(amino_acids)
                    coordinates[a] = residue['CA'].coord
                    #residue_numbers是构图的pdb中氨基酸的位置 == fasta中位置
                    #residue_names 是构图的pdb中氨基酸 三位英语的名称 转为一位缩写 = fasta中的一位名称
    return amino_acids, coordinates, np.array(residue_numbers), residue_names


def print_graph_info(graph):
    # 打印节点数量和边数量
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    print("graph.edges", graph.edges)

    # # 打印每个节点的特征
    # for node in graph.nodes(data=True):
    #     print("Node:", node[0])
    #     print("Features:", node[1]['features'])
    #
    # # 打印图的邻接信息
    # print("Adjacency list:")
    # for node, neighbors in graph.adjacency():
    #     print(node, ":", neighbors)



def create_graph(coordinates, DDXX_position, residue_names, threshold_distance=10.0):
    # 创建一个无向图
    max_edge_distance= 8
    graph = nx.Graph()
    start_pos, middle_pos, end_pos = DDXX_position
    # 计算DxDD位置的坐标
    start_coord = coordinates[start_pos + 1]  # 因为索引是从 0 开始的，所以要加 1
    middle_coord = coordinates[middle_pos + 1]
    end_coord = coordinates[end_pos + 1]

    num = []
    name = []
    ''' 遍历每个氨基酸， 添加所有与 DxDD 位置小于阈值的氨基酸作为图的节点【aa1是氨基酸名称】 '''
    for i, aa1 in enumerate(residue_names):
        dist_to_start = calculate_distance(coordinates[i + 1], start_coord)
        dist_to_middle = calculate_distance(coordinates[i + 1], middle_coord)
        dist_to_end = calculate_distance(coordinates[i + 1], end_coord)

        if dist_to_start < threshold_distance or dist_to_middle < threshold_distance or dist_to_end < threshold_distance:
            graph.add_node(i)
            num.append(i+1)#对应于pdb中的编号
            name.append(aa1)

        # 添加边，仅连接距离小于8A的节点
    for u in graph.nodes():
        for v in graph.nodes():
            if u < v:  # 避免重复添加边
                distance_uv = calculate_distance(coordinates[u + 1], coordinates[v + 1])
                if distance_uv < max_edge_distance:
                    graph.add_edge(u, v)

    return graph,num,name



def calculate_distance(coord1, coord2):
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2) ** 0.5




def read_excel(file_path):
    # 使用pandas读取Excel文件
    df = pd.read_excel(file_path)
    # Excel 文件中的第一列数据。
    pdb_file_names = df.iloc[:, 0].tolist()
    # 读取酶标签； Excel 文件中第二列数据的列表
    enzyme_labels = df.iloc[:, 1].tolist()
    return pdb_file_names, enzyme_labels


def extract_proteinBert_features(df):
    # 使用 ProteinBERT 提取特征
    pretrained_model_generator, input_encoder = load_pretrained_model()
    Pmodel = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(832))
    # 对序列进行标记化
    encoded_xs = [input_encoder.encode_X([seq], 832) for seq in df['Sequence']]
    pfeature = []
    for i, encoded_x in enumerate(encoded_xs):
        print("第", i, "个:")
        local_representations, global_representations = Pmodel.predict(encoded_x, batch_size=8)
        print("local_representations=", local_representations)
        pfeature.append(local_representations[0])

    pfeature = np.array(pfeature)
    print("pfeature", pfeature.shape, type(pfeature))
    return pfeature

def find_DxDD_positions(df):
    DxDD_positions = []
    for sequence in df['Sequence']:
        for i in range(len(sequence) - 3):
            if sequence[i] == 'D' and sequence[i + 2] == 'D' and sequence[i + 3] == 'D':
                DxDD_positions.append((i, i + 2, i + 3))  # 添加每个模式的位置（起始、中间、结束）
                break
    return DxDD_positions




def create_data():
    # 指定Excel文件路径
    excel_file_path =""
    # 从Excel中读取PDB文件名
    pdb_file_names, enzyme_labels = read_excel(excel_file_path)

    file_path =""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = {'Reaction': [], 'Sequence': []}
    for line in lines:
        parts = line.strip().split()
        reaction = int(parts[1])
        sequence = parts[2]  # 提取酶的序列
        data['Reaction'].append(reaction)
        data['Sequence'].append(sequence)
        # print("Sequence=",sequence)
        # print("Sequence[1]=",sequence[1])

    df = pd.DataFrame(data)
    # df中序列的下标是从0开始的，序列中的氨基酸下标也是从0开始的
    print("df[0][0]=第0条序列的第0个氨基酸",df.iloc[0]['Sequence'][0])

    # 找每个序列中DxDD 的每个D的位置
    DxDD_positions = find_DxDD_positions(df)
    print("DxDD pattern positions:", DxDD_positions)



    # 提取 ProteinBERT 特征
    pfeature = extract_proteinBert_features(df)
    data_list = []
    x_p_list = []
    coordinates_list = []
    DxDD_positions_list = []
    residue_names_list = []
    data_list = []
    # print("pdb_file_names", len(pdb_file_names))
    # print("pfeature", len(pfeature))
    spositive_param = torch.nn.Parameter(torch.tensor(10.0))
    print("学习的参数", spositive_param)

    # print("pdb_file_names", len(pdb_file_names))
    # print("pfeature", len(pfeature))

    for i, (pdb_file_name, pfea) in enumerate(zip(pdb_file_names, pfeature)):
        # 构造PDB文件的完整路径
        print("pdb_file_name=", pdb_file_name)
        pdb_file_path = f"{pdb_file_name}.pdb"
        # print("pdb_file_path=", pdb_file_path)

        # 检查PDB文件是否存在
        if not glob.glob(pdb_file_path):
            print(f"未找到{pdb_file_name}的PDB文件")
            continue

        # 从pdb文件中获得 所有氨基酸类型、坐标
        amino_acids, coordinates, residue_numbers, residue_names = parse_pdb(pdb_file_path)

        # 创建一个列表，存放保留的氨基酸特征
        filtered_pfea = []

        # 构图 得到层次化的 对应pdb中的编号 和 对应氨基酸缩写
        coordinates_list.append(coordinates)
        DxDD_positions_list.append(DxDD_positions[i])
        residue_names_list.append(residue_names)

        # protein_graph, residue_numbers, residue_names = create_graph(amino_acids, coordinates, DxDD_positions[i],residue_names,residue_numbers)

        # 遍历每个氨基酸的编号
        for residue_number, residue_name in zip(residue_numbers, residue_names):
            # 将 pfea 中对应位置的特征加入 filtered_pfea
            filtered_pfea.append(pfea[residue_number - 1])
        # 将 filtered_pfea 转换为 NumPy 数组
        filtered_pfea = np.array(filtered_pfea)
        # print("filtered_pfea=",filtered_pfea.shape)

        x_p = torch.tensor(filtered_pfea, dtype=torch.float32)
        x_p_list.append(x_p)

    return x_p_list, coordinates_list, DxDD_positions_list, residue_names_list, enzyme_labels


def main():
        data_list = create_data()
        # print("data_list",data_list)



if __name__ == "__main__":
    main()
