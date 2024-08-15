import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GATConv, GNNExplainer
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from makeGraph import create_data, create_graph
from torch_geometric.nn import global_mean_pool

# 定义GCN模型类
class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels * 2)
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels)
        self.conv3 = GATConv(hidden_channels, out_channels)
        self.threshold_distance = nn.Parameter(torch.tensor(10.0))
        self.cansu = nn.Parameter(torch.tensor(0.75))

    def forward(self, x, edge_index, batch=None, **kwargs):
        coord = kwargs.get('coord')
        DxDD = kwargs.get('DxDD')
        residue = kwargs.get('residue')

        threshold_distance = self.threshold_distance
        cansu = self.cansu
        threshold_distance = threshold_distance * cansu

        # 创建图
        protein_graph, residue_numbers, residue_names = create_graph(coord, DxDD, residue, threshold_distance=threshold_distance)
        edges = protein_graph.edges
        nodes = sorted(set(sum(edges, ())))
        node_map = {node: idx for idx, node in enumerate(nodes)}
        new_edges = [(node_map[u], node_map[v]) for u, v in edges]
        edge_index = torch.tensor(list(new_edges)).t().contiguous()

        # 通过卷积层
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index) * threshold_distance

        # 处理 batch
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)

        return x


def read_excel(file_path):
    df = pd.read_excel(file_path)
    pdb_file_names = df.iloc[:, 0].tolist()
    enzyme_labels = df.iloc[:, 1].tolist()
    return pdb_file_names, enzyme_labels


def find_optimal_threshold(train_data_xp, train_data_coord, train_data_DxDD, train_data_residue, train_labels, model,
                           optimizer, epochs):
    all_losses = []
    all_thresholds = []

    for epoch in range(epochs):
        model.train()
        for xp, coord, DxDD, residue, label in zip(train_data_xp, train_data_coord, train_data_DxDD, train_data_residue,
                                                   train_labels):
            optimizer.zero_grad()
            output = model(xp, None, coord=coord, DxDD=DxDD, residue=residue)  # 传递所有必要的参数
            loss = F.cross_entropy(output, torch.tensor([label], dtype=torch.long))
            loss.backward()
            optimizer.step()
            current_threshold = model.threshold_distance.item() * model.cansu.item()
            all_losses.append(loss.item())
            all_thresholds.append(current_threshold)
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Threshold: {current_threshold:.4f}")

    min_loss = min(all_losses)
    optimal_thresholds = [thresh for loss, thresh in zip(all_losses, all_thresholds) if loss <= min_loss + 0.001]
    optimal_threshold = np.mean(optimal_thresholds)

    print(f"最优半径: {optimal_threshold:.4f}")
    return optimal_threshold


def evaluate_on_test(test_data_xp, test_data_coord, test_data_DxDD, test_data_residue, test_labels, model,
                     optimal_threshold):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for xp, coord, DxDD, residue, label in zip(test_data_xp, test_data_coord, test_data_DxDD, test_data_residue,
                                                   test_labels):
            model.threshold_distance = nn.Parameter(torch.tensor(optimal_threshold))
            model.cansu = nn.Parameter(torch.tensor(1.0))  # 使最终的阈值等于optimal_threshold
            output = model(xp, None, coord=coord, DxDD=DxDD, residue=residue)  # 传递所有必要的参数
            _, predicted = torch.max(output, 1)
            all_predictions.append(predicted.item())

    accuracy = accuracy_score(test_labels, all_predictions)
    print(f"使用最优半径测试后的准确率: {accuracy:.4f}")
    return accuracy


def explain_model(model, data_xp, data_coord, data_DxDD, data_residue):
    explainer = GNNExplainer(model, epochs=100, feat_mask_type='scalar', return_type='log_prob')
    for xp, coord, DxDD, residue in zip(data_xp, data_coord, data_DxDD, data_residue):
        # 构建 edge_index
        protein_graph, residue_numbers, residue_names = create_graph(coord, DxDD, residue,
                                                                     model.threshold_distance.item())
        edges = protein_graph.edges
        nodes = sorted(set(sum(edges, ())))
        node_map = {node: idx for idx, node in enumerate(nodes)}
        new_edges = [(node_map[u], node_map[v]) for u, v in edges]
        edge_index = torch.tensor(list(new_edges)).t().contiguous()

        # 确保传递给 explainer 的参数与 forward 方法中的一致
        kwargs = {
            'coord': coord,
            'DxDD': DxDD,
            'residue': residue
        }

        # 调用 explainer 进行解释
        node_feat_mask, edge_mask = explainer.explain_graph(xp, edge_index=edge_index, **kwargs)

        threshold = 0.5  # 设定一个阈值
        important_nodes = [i for i, score in enumerate(node_feat_mask) if score > threshold]
        print("Important Nodes:", important_nodes)
        print("Node Feature Mask:", node_feat_mask)
        print("Edge Mask:", edge_mask)



def main():
    excel_file_path = ""
    pdb_file_names, enzyme_labels = read_excel(excel_file_path)
    x_p_list, coordinates_list, DxDD_positions_list, residue_names_list, enzyme_labels = create_data()

    # 划分训练集和测试集
    train_xp, test_xp, train_coord, test_coord, train_DxDD, test_DxDD, train_residue, test_residue, train_labels, test_labels = train_test_split(
        x_p_list, coordinates_list, DxDD_positions_list, residue_names_list, enzyme_labels, test_size=0.3,
        random_state=42)

    model = GCNModel(in_channels=1562, hidden_channels=64, out_channels=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 100

    # 在训练集上找到最优半径
    optimal_threshold = find_optimal_threshold(train_xp, train_coord, train_DxDD, train_residue, train_labels, model,
                                               optimizer, epochs)

    # 使用最优半径在测试集上进行评估
    evaluate_on_test(test_xp, test_coord, test_DxDD, test_residue, test_labels, model, optimal_threshold)

    # 添加可解释性分析
    explain_model(model, test_xp, test_coord, test_DxDD, test_residue)


if __name__ == "__main__":
    main()
